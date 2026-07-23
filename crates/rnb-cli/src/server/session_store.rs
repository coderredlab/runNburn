use super::http::ApiError;
use super::response_config::normalize_metadata;
use super::response_input::{normalize_input, ResponseInput};
use super::responses::ResponseRequest;
use rnb_llm::EngineSequenceState;
use rnb_runtime::memory::ByteLruPolicy;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const RESPONSE_TTL_SECONDS: u64 = 30 * 24 * 60 * 60;
static NEXT_CONVERSATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CacheKey {
    Response(String),
    Conversation(String),
}

#[derive(Clone)]
pub(super) struct ResolvedResponseContext {
    pub history_items: Vec<Value>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub resume_state: Option<Arc<EngineSequenceState>>,
}

struct StoredResponse {
    response: Value,
    model_input_items: Vec<Value>,
    history_items: Vec<Value>,
    sequence_state: Option<Arc<EngineSequenceState>>,
    expires_at: u64,
    bytes: u64,
}

struct ConversationRecord {
    created_at: u64,
    metadata: Value,
    history_items: Vec<Value>,
    sequence_state: Option<Arc<EngineSequenceState>>,
    bytes: u64,
}

pub(super) struct ResponseStore {
    responses: HashMap<String, StoredResponse>,
    conversations: HashMap<String, ConversationRecord>,
    max_bytes: u64,
    records_lru: ByteLruPolicy<CacheKey>,
    snapshots_lru: ByteLruPolicy<CacheKey>,
}

impl ResponseStore {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            responses: HashMap::new(),
            conversations: HashMap::new(),
            max_bytes,
            records_lru: ByteLruPolicy::new(u64::MAX),
            snapshots_lru: ByteLruPolicy::new(u64::MAX),
        }
    }

    pub fn snapshot_fits(
        &self,
        history_items: &[Value],
        conversation_id: Option<&str>,
        store: bool,
        response: &Value,
        output_items: &[Value],
        snapshot_bytes: u64,
    ) -> bool {
        let next_history_bytes =
            values_bytes(history_items).saturating_add(values_bytes(output_items));
        let response_bytes = store.then(|| {
            value_bytes(response)
                .saturating_add(values_bytes(history_items))
                .saturating_add(next_history_bytes)
        });
        let conversation_bytes = conversation_id.and_then(|id| {
            self.conversations.get(id).map(|conversation| {
                value_bytes(&conversation.metadata).saturating_add(next_history_bytes)
            })
        });
        let snapshot_copies = u64::from(store) + u64::from(conversation_id.is_some());
        response_bytes
            .unwrap_or(0)
            .saturating_add(conversation_bytes.unwrap_or(0))
            .saturating_add(snapshot_bytes.saturating_mul(snapshot_copies))
            <= self.max_bytes
    }

    pub fn resolve(
        &mut self,
        request: &mut ResponseRequest,
        now: u64,
    ) -> Result<ResolvedResponseContext, ApiError> {
        self.prune_expired(now);
        let previous_response_id = optional_id(
            request.previous_response_id.as_ref(),
            "previous_response_id",
        )?;
        let conversation_id = optional_conversation_id(request.conversation.as_ref())?;
        if previous_response_id.is_some() && conversation_id.is_some() {
            return Err(ApiError::invalid(
                "previous_response_id and conversation cannot be used together",
                Some("previous_response_id"),
                Some("invalid_value"),
            ));
        }

        let current_input_items = request
            .input
            .take()
            .map(ResponseInput::into_items)
            .unwrap_or_default();
        if !current_input_items.is_empty() {
            normalize_input(ResponseInput::Items(current_input_items.clone()), None)?;
        }

        let (mut history_items, resume_state) = if let Some(response_id) = &previous_response_id {
            self.response_context(response_id)?
        } else if let Some(conversation_id) = &conversation_id {
            self.conversation_context(conversation_id)?
        } else {
            (Vec::new(), None)
        };
        history_items.extend(current_input_items.iter().cloned());
        if history_items.is_empty() {
            return Err(ApiError::invalid(
                "input is required when no previous response or conversation items exist",
                Some("input"),
                Some("missing_required_parameter"),
            ));
        }

        Ok(ResolvedResponseContext {
            history_items,
            previous_response_id,
            conversation_id,
            resume_state,
        })
    }

    pub fn commit(
        &mut self,
        history_items: &[Value],
        conversation_id: Option<&str>,
        store: bool,
        response: Value,
        output_items: &[Value],
        sequence_state: Option<EngineSequenceState>,
        now: u64,
    ) -> Result<(), ApiError> {
        let response_id = response["id"].as_str().unwrap_or_default().to_string();
        if response_id.is_empty() {
            return Err(ApiError::internal("generated response has no id"));
        }
        if !store && conversation_id.is_none() {
            return Ok(());
        }

        let mut next_history_items = history_items.to_vec();
        next_history_items.extend(output_items.iter().cloned());
        let sequence_state = sequence_state.map(Arc::new);
        let snapshot_bytes = sequence_state.as_ref().map_or(0, |state| state.byte_size());
        let response_bytes = store.then(|| {
            value_bytes(&response)
                .saturating_add(values_bytes(history_items))
                .saturating_add(values_bytes(&next_history_items))
        });
        let conversation_bytes = conversation_id
            .map(|id| {
                self.conversations
                    .get(id)
                    .map(|conversation| {
                        value_bytes(&conversation.metadata)
                            .saturating_add(values_bytes(&next_history_items))
                    })
                    .ok_or_else(|| conversation_not_found(id))
            })
            .transpose()?;
        let required_bytes = response_bytes
            .unwrap_or(0)
            .saturating_add(conversation_bytes.unwrap_or(0));
        if required_bytes > self.max_bytes {
            return Err(ApiError::invalid(
                "response history exceeds the configured session memory budget",
                Some("input"),
                Some("context_length_exceeded"),
            ));
        }

        if let Some(conversation_id) = conversation_id {
            let key = CacheKey::Conversation(conversation_id.to_string());
            self.records_lru.remove(&key);
            self.snapshots_lru.remove(&key);
            let conversation = self
                .conversations
                .get_mut(conversation_id)
                .ok_or_else(|| conversation_not_found(conversation_id))?;
            conversation.history_items = next_history_items.clone();
            conversation.sequence_state = sequence_state.clone();
            conversation.bytes = conversation_bytes.unwrap_or(0);
        }

        if let Some(bytes) = response_bytes {
            self.responses.insert(
                response_id.clone(),
                StoredResponse {
                    response,
                    model_input_items: stored_input_items(history_items, &response_id),
                    history_items: next_history_items,
                    sequence_state: sequence_state.clone(),
                    expires_at: now.saturating_add(RESPONSE_TTL_SECONDS),
                    bytes,
                },
            );
            let key = CacheKey::Response(response_id.clone());
            self.records_lru.touch(key.clone(), bytes);
            if snapshot_bytes != 0 {
                self.snapshots_lru.touch(key, snapshot_bytes);
            }
        }
        if let Some(conversation_id) = conversation_id {
            if let Some(conversation) = self.conversations.get(conversation_id) {
                let key = CacheKey::Conversation(conversation_id.to_string());
                self.records_lru.touch(key.clone(), conversation.bytes);
                if snapshot_bytes != 0 {
                    self.snapshots_lru.touch(key, snapshot_bytes);
                }
            }
        }
        self.enforce_budget();

        if store && !self.responses.contains_key(&response_id) {
            return Err(ApiError::internal(
                "stored response was evicted during commit",
            ));
        }
        if let Some(conversation_id) = conversation_id {
            if !self.conversations.contains_key(conversation_id) {
                return Err(ApiError::internal(
                    "conversation was evicted during response commit",
                ));
            }
        }
        Ok(())
    }

    pub fn get_response(&mut self, id: &str, now: u64) -> Result<Value, ApiError> {
        self.prune_expired(now);
        let (bytes, response) = self
            .responses
            .get(id)
            .map(|entry| (entry.bytes, entry.response.clone()))
            .ok_or_else(|| response_not_found(id))?;
        self.records_lru
            .touch(CacheKey::Response(id.to_string()), bytes);
        if !self.responses.contains_key(id) {
            return Err(response_not_found(id));
        }
        Ok(response)
    }

    pub fn get_input_items(
        &mut self,
        id: &str,
        now: u64,
        descending: bool,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Value, ApiError> {
        self.prune_expired(now);
        let (bytes, mut items) = self
            .responses
            .get(id)
            .map(|entry| (entry.bytes, entry.model_input_items.clone()))
            .ok_or_else(|| response_not_found(id))?;
        self.records_lru
            .touch(CacheKey::Response(id.to_string()), bytes);
        if !self.responses.contains_key(id) {
            return Err(response_not_found(id));
        }

        if descending {
            items.reverse();
        }
        let start = match after {
            Some(cursor) => items
                .iter()
                .position(|item| item["id"].as_str() == Some(cursor))
                .map(|index| index + 1)
                .ok_or_else(|| {
                    ApiError::invalid(
                        format!("after cursor '{cursor}' was not found"),
                        Some("after"),
                        Some("invalid_value"),
                    )
                })?,
            None => 0,
        };
        let remaining = items.len().saturating_sub(start);
        let take = remaining.min(limit);
        let data = items.into_iter().skip(start).take(take).collect::<Vec<_>>();
        let first_id = data.first().and_then(|item| item["id"].as_str());
        let last_id = data.last().and_then(|item| item["id"].as_str());
        Ok(json!({
            "object": "list",
            "data": data,
            "first_id": first_id,
            "last_id": last_id,
            "has_more": remaining > take
        }))
    }

    pub fn delete_response(&mut self, id: &str) -> Result<Value, ApiError> {
        if !self.responses.contains_key(id) {
            return Err(response_not_found(id));
        }
        self.remove_response(id);
        Ok(json!({"id": id, "object": "response", "deleted": true}))
    }

    pub fn create_conversation(
        &mut self,
        request: ConversationCreateRequest,
        now: u64,
    ) -> Result<Value, ApiError> {
        if request.items.len() > 20 {
            return Err(ApiError::invalid(
                "items may contain at most 20 entries",
                Some("items"),
                Some("array_above_max_length"),
            ));
        }
        if !request.items.is_empty() {
            normalize_input(ResponseInput::Items(request.items.clone()), None)?;
        }
        let metadata = normalize_metadata(request.metadata)?;
        let id = format!(
            "conv_{}_{:x}",
            now,
            NEXT_CONVERSATION_ID.fetch_add(1, Ordering::Relaxed)
        );
        let bytes = values_bytes(&request.items).saturating_add(value_bytes(&metadata));
        if bytes > self.max_bytes {
            return Err(ApiError::invalid(
                "conversation exceeds the configured session memory budget",
                Some("items"),
                Some("context_length_exceeded"),
            ));
        }
        self.conversations.insert(
            id.clone(),
            ConversationRecord {
                created_at: now,
                metadata: metadata.clone(),
                history_items: request.items,
                sequence_state: None,
                bytes,
            },
        );
        self.records_lru
            .touch(CacheKey::Conversation(id.clone()), bytes);
        self.enforce_budget();
        if !self.conversations.contains_key(&id) {
            return Err(ApiError::invalid(
                "conversation exceeds the configured session memory budget",
                Some("items"),
                Some("context_length_exceeded"),
            ));
        }
        Ok(conversation_value(&id, now, metadata))
    }

    pub fn get_conversation(&mut self, id: &str) -> Result<Value, ApiError> {
        let (created_at, metadata, bytes) = self
            .conversations
            .get(id)
            .map(|conversation| {
                (
                    conversation.created_at,
                    conversation.metadata.clone(),
                    conversation.bytes,
                )
            })
            .ok_or_else(|| conversation_not_found(id))?;
        self.records_lru
            .touch(CacheKey::Conversation(id.to_string()), bytes);
        if !self.conversations.contains_key(id) {
            return Err(conversation_not_found(id));
        }
        Ok(conversation_value(id, created_at, metadata))
    }

    pub fn delete_conversation(&mut self, id: &str) -> Result<Value, ApiError> {
        self.conversations
            .remove(id)
            .ok_or_else(|| conversation_not_found(id))?;
        let key = CacheKey::Conversation(id.to_string());
        self.records_lru.remove(&key);
        self.snapshots_lru.remove(&key);
        Ok(json!({"id": id, "object": "conversation.deleted", "deleted": true}))
    }

    fn response_context(
        &mut self,
        response_id: &str,
    ) -> Result<(Vec<Value>, Option<Arc<EngineSequenceState>>), ApiError> {
        let (bytes, history, state) = self
            .responses
            .get(response_id)
            .map(|entry| {
                (
                    entry.bytes,
                    entry.history_items.clone(),
                    entry.sequence_state.clone(),
                )
            })
            .ok_or_else(|| response_not_found(response_id))?;
        let key = CacheKey::Response(response_id.to_string());
        self.records_lru.touch(key.clone(), bytes);
        if let Some(state) = &state {
            self.snapshots_lru.touch(key, state.byte_size());
        }
        if !self.responses.contains_key(response_id) {
            return Err(response_not_found(response_id));
        }
        Ok((history, state))
    }

    fn conversation_context(
        &mut self,
        conversation_id: &str,
    ) -> Result<(Vec<Value>, Option<Arc<EngineSequenceState>>), ApiError> {
        let (history_items, state, bytes) = self
            .conversations
            .get(conversation_id)
            .map(|conversation| {
                (
                    conversation.history_items.clone(),
                    conversation.sequence_state.clone(),
                    conversation.bytes,
                )
            })
            .ok_or_else(|| conversation_not_found(conversation_id))?;
        let key = CacheKey::Conversation(conversation_id.to_string());
        self.records_lru.touch(key.clone(), bytes);
        if let Some(state) = &state {
            self.snapshots_lru.touch(key, state.byte_size());
        }
        if !self.conversations.contains_key(conversation_id) {
            return Err(conversation_not_found(conversation_id));
        }
        Ok((history_items, state))
    }

    fn prune_expired(&mut self, now: u64) {
        let expired = self
            .responses
            .iter()
            .filter_map(|(id, entry)| (entry.expires_at <= now).then_some(id.clone()))
            .collect::<Vec<_>>();
        for id in expired {
            self.remove_response(&id);
        }
    }

    fn enforce_budget(&mut self) {
        while self
            .records_lru
            .resident_bytes()
            .saturating_add(self.snapshots_lru.resident_bytes())
            > self.max_bytes
        {
            if let Some(key) = self.snapshots_lru.pop_oldest() {
                self.drop_snapshot(&key);
                continue;
            }
            let Some(key) = self.records_lru.pop_oldest() else {
                break;
            };
            self.remove_record(&key);
        }
    }

    fn drop_snapshot(&mut self, key: &CacheKey) {
        match key {
            CacheKey::Response(id) => {
                if let Some(response) = self.responses.get_mut(id) {
                    response.sequence_state = None;
                }
            }
            CacheKey::Conversation(id) => {
                if let Some(conversation) = self.conversations.get_mut(id) {
                    conversation.sequence_state = None;
                }
            }
        }
    }

    fn remove_record(&mut self, key: &CacheKey) {
        self.snapshots_lru.remove(key);
        match key {
            CacheKey::Response(id) => {
                self.responses.remove(id);
            }
            CacheKey::Conversation(id) => {
                self.conversations.remove(id);
            }
        }
    }

    fn remove_response(&mut self, id: &str) {
        let key = CacheKey::Response(id.to_string());
        self.responses.remove(id);
        self.records_lru.remove(&key);
        self.snapshots_lru.remove(&key);
    }
}

fn stored_input_items(items: &[Value], response_id: &str) -> Vec<Value> {
    let suffix = response_id.strip_prefix("resp_").unwrap_or(response_id);
    items
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, mut item)| {
            if let Some(object) = item.as_object_mut() {
                let kind = object.get("type").and_then(Value::as_str);
                let is_message = kind.is_none() || kind == Some("message");
                let prefix = match kind {
                    None | Some("message") => "msg",
                    Some("function_call") | Some("function_call_output") => "fc",
                    _ => "item",
                };
                object
                    .entry("id")
                    .or_insert_with(|| json!(format!("{prefix}_{suffix}_{index}")));
                if is_message {
                    object.entry("type").or_insert_with(|| json!("message"));
                    if let Some(text) = object.get("content").and_then(Value::as_str) {
                        object.insert(
                            "content".to_string(),
                            json!([{"type": "input_text", "text": text}]),
                        );
                    }
                    object.entry("status").or_insert_with(|| json!("completed"));
                }
            }
            item
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConversationCreateRequest {
    #[serde(default)]
    items: Vec<Value>,
    metadata: Option<Value>,
}

fn optional_id(value: Option<&Value>, param: &'static str) -> Result<Option<String>, ApiError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_str()
        .filter(|id| !id.is_empty())
        .map(|id| Some(id.to_string()))
        .ok_or_else(|| {
            ApiError::invalid(
                format!("{param} must be a non-empty string"),
                Some(param),
                Some("invalid_value"),
            )
        })
}

fn optional_conversation_id(value: Option<&Value>) -> Result<Option<String>, ApiError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let id = value
        .as_str()
        .or_else(|| value.get("id").and_then(Value::as_str));
    id.filter(|id| !id.is_empty())
        .map(|id| Some(id.to_string()))
        .ok_or_else(|| {
            ApiError::invalid(
                "conversation must be a non-empty ID or an object containing id",
                Some("conversation"),
                Some("invalid_value"),
            )
        })
}

fn response_not_found(id: &str) -> ApiError {
    ApiError::resource_not_found(
        format!("Response '{id}' was not found"),
        Some("response_id"),
    )
}

fn conversation_not_found(id: &str) -> ApiError {
    ApiError::resource_not_found(
        format!("Conversation '{id}' was not found"),
        Some("conversation_id"),
    )
}

fn conversation_value(id: &str, created_at: u64, metadata: Value) -> Value {
    json!({
        "id": id,
        "object": "conversation",
        "created_at": created_at,
        "metadata": metadata
    })
}

fn values_bytes(values: &[Value]) -> u64 {
    values.iter().map(value_bytes).fold(
        values.len().saturating_mul(std::mem::size_of::<Value>()) as u64,
        u64::saturating_add,
    )
}

fn value_bytes(value: &Value) -> u64 {
    serde_json::to_vec(value).map_or(std::mem::size_of::<Value>() as u64, |bytes| {
        (std::mem::size_of::<Value>() as u64).saturating_add((bytes.len() as u64).saturating_mul(2))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: Value) -> ResponseRequest {
        serde_json::from_value(value).unwrap()
    }

    fn assistant(text: &str) -> Value {
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        })
    }

    #[test]
    fn stored_response_chains_input_and_output_then_deletes() {
        let mut store = ResponseStore::new(1_000_000);
        let mut first = request(json!({"model": "m", "input": "hello"}));
        let first_context = store.resolve(&mut first, 10).unwrap();
        store
            .commit(
                &first_context.history_items,
                None,
                true,
                json!({"id": "resp_1", "object": "response", "output": [assistant("hi")]}),
                &[assistant("hi")],
                None,
                10,
            )
            .unwrap();
        assert_eq!(store.get_response("resp_1", 11).unwrap()["id"], "resp_1");

        let mut next = request(json!({
            "model": "m",
            "previous_response_id": "resp_1",
            "input": "again"
        }));
        let next_context = store.resolve(&mut next, 11).unwrap();
        assert_eq!(next_context.history_items.len(), 3);
        assert_eq!(next_context.history_items[1]["role"], "assistant");
        assert_eq!(next_context.history_items[2]["content"][0]["text"], "again");

        assert!(store.delete_response("resp_1").unwrap()["deleted"]
            .as_bool()
            .unwrap());
        let mut missing = request(json!({
            "model": "m",
            "previous_response_id": "resp_1",
            "input": "again"
        }));
        let error = match store.resolve(&mut missing, 12) {
            Ok(_) => panic!("deleted response unexpectedly resolved"),
            Err(error) => error,
        };
        assert_eq!(error.status, 404);
    }

    #[test]
    fn conversation_keeps_internal_store_false_history() {
        let mut store = ResponseStore::new(1_000_000);
        let created = store
            .create_conversation(
                serde_json::from_value(json!({
                    "items": [{"type": "message", "role": "user", "content": "remember"}],
                    "metadata": {"case": "conversation"}
                }))
                .unwrap(),
                20,
            )
            .unwrap();
        let conversation_id = created["id"].as_str().unwrap();
        let mut first = request(json!({"model": "m", "conversation": conversation_id}));
        let first_context = store.resolve(&mut first, 20).unwrap();
        store
            .commit(
                &first_context.history_items,
                Some(conversation_id),
                false,
                json!({"id": "resp_internal", "object": "response"}),
                &[assistant("remembered")],
                None,
                20,
            )
            .unwrap();
        assert_eq!(
            store.get_response("resp_internal", 21).unwrap_err().status,
            404
        );

        let mut next = request(json!({
            "model": "m",
            "conversation": conversation_id,
            "input": "recall"
        }));
        let next_context = store.resolve(&mut next, 21).unwrap();
        assert_eq!(next_context.history_items.len(), 3);
        assert_eq!(next_context.history_items[1]["role"], "assistant");
        store
            .commit(
                &next_context.history_items,
                Some(conversation_id),
                true,
                json!({"id": "resp_public", "object": "response"}),
                &[assistant("recalled")],
                None,
                21,
            )
            .unwrap();
        store.delete_response("resp_public").unwrap();
        let mut after_delete = request(json!({"model": "m", "conversation": conversation_id}));
        assert_eq!(
            store
                .resolve(&mut after_delete, 22)
                .unwrap()
                .history_items
                .len(),
            4
        );
    }

    #[test]
    fn input_items_have_ids_and_paginate_descending() {
        let mut store = ResponseStore::new(1_000_000);
        let mut request = request(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "user", "content": "two"},
                {"type": "message", "role": "user", "content": "three"}
            ]
        }));
        let context = store.resolve(&mut request, 30).unwrap();
        store
            .commit(
                &context.history_items,
                None,
                true,
                json!({"id": "resp_page", "object": "response"}),
                &[],
                None,
                30,
            )
            .unwrap();

        let first = store
            .get_input_items("resp_page", 31, true, None, 2)
            .unwrap();
        assert_eq!(first["data"].as_array().unwrap().len(), 2);
        assert_eq!(first["data"][0]["content"][0]["text"], "three");
        assert_eq!(first["first_id"], first["data"][0]["id"]);
        assert_eq!(first["last_id"], first["data"][1]["id"]);
        assert_eq!(first["has_more"], true);

        let cursor = first["last_id"].as_str().unwrap();
        let second = store
            .get_input_items("resp_page", 32, true, Some(cursor), 2)
            .unwrap();
        assert_eq!(second["data"].as_array().unwrap().len(), 1);
        assert_eq!(second["data"][0]["content"][0]["text"], "one");
        assert_eq!(second["has_more"], false);
    }

    #[test]
    fn oversized_response_state_is_rejected_before_commit() {
        let mut store = ResponseStore::new(1);
        let mut request = request(json!({"model": "m", "input": "hello"}));
        let context = store.resolve(&mut request, 40).unwrap();
        let error = store
            .commit(
                &context.history_items,
                None,
                true,
                json!({"id": "resp_too_large", "object": "response"}),
                &[],
                None,
                40,
            )
            .unwrap_err();
        assert_eq!(error.code, Some("context_length_exceeded"));
        assert_eq!(
            store.get_response("resp_too_large", 41).unwrap_err().status,
            404
        );
    }

    #[test]
    fn conversation_create_rejects_unknown_fields() {
        assert!(serde_json::from_value::<ConversationCreateRequest>(json!({
            "items": [],
            "unknown": true
        }))
        .is_err());
    }

    #[test]
    fn expired_response_and_oversized_conversation_are_evicted() {
        let mut store = ResponseStore::new(1_000_000);
        let mut request = request(json!({"model": "m", "input": "hello"}));
        let context = store.resolve(&mut request, 0).unwrap();
        store
            .commit(
                &context.history_items,
                None,
                true,
                json!({"id": "resp_expired", "object": "response"}),
                &[],
                None,
                0,
            )
            .unwrap();
        assert_eq!(
            store
                .get_response("resp_expired", RESPONSE_TTL_SECONDS)
                .unwrap_err()
                .status,
            404
        );

        let mut tiny = ResponseStore::new(1);
        let error = tiny
            .create_conversation(
                serde_json::from_value(json!({
                    "items": [{"role": "user", "content": "too large"}]
                }))
                .unwrap(),
                1,
            )
            .unwrap_err();
        assert_eq!(error.code, Some("context_length_exceeded"));
    }
}
