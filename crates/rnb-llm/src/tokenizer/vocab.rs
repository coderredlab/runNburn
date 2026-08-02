use std::collections::HashMap;

/// 특수 토큰 ID 모음
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: Option<u32>,
}

/// token_id → token string 매핑 + 역방향 맵
#[derive(Debug, Clone)]
pub struct Vocab {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub special: SpecialTokens,
    token_types: Vec<u32>,
    tokenization_special_ids: Vec<u32>,
    pub unknown: Option<u32>,
    pub separator: Option<u32>,
}

impl Vocab {
    pub fn new(tokens: Vec<String>, special: SpecialTokens) -> Self {
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self {
            id_to_token: tokens,
            token_to_id,
            special,
            token_types: Vec::new(),
            tokenization_special_ids: Vec::new(),
            unknown: None,
            separator: None,
        }
    }

    pub fn set_token_metadata(
        &mut self,
        token_types: Vec<u32>,
        unknown: Option<u32>,
        separator: Option<u32>,
        mut added_token_ids: Vec<u32>,
    ) {
        let inferred_unknown = token_types
            .iter()
            .position(|token_type| *token_type == 2)
            .map(|id| id as u32);
        let mut tokenization_special_ids = token_types
            .iter()
            .enumerate()
            .filter_map(|(id, token_type)| matches!(token_type, 2 | 3 | 4).then_some(id as u32))
            .collect::<Vec<_>>();
        tokenization_special_ids.append(&mut added_token_ids);
        tokenization_special_ids.sort_unstable();
        tokenization_special_ids.dedup();

        self.token_types = token_types;
        self.tokenization_special_ids = tokenization_special_ids;
        self.unknown = unknown.or(inferred_unknown);
        self.separator = separator;
    }

    pub fn has_explicit_token_types(&self) -> bool {
        !self.token_types.is_empty()
    }

    pub fn tokenization_special_ids(&self) -> &[u32] {
        &self.tokenization_special_ids
    }

    pub fn unknown_id(&self) -> Option<u32> {
        self.unknown
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    // plan의 get/find/len 인터페이스도 지원
    pub fn get(&self, id: u32) -> Option<&str> {
        self.token_str(id)
    }

    pub fn find(&self, token: &str) -> Option<u32> {
        self.token_id(token)
    }

    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab() -> Vocab {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "o".to_string(),
            "hello".to_string(),
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        Vocab::new(tokens, special)
    }

    #[test]
    fn test_vocab_size() {
        let v = make_vocab();
        assert_eq!(v.size(), 7);
    }

    #[test]
    fn test_token_str_roundtrip() {
        let v = make_vocab();
        assert_eq!(v.token_str(3), Some("he"));
        assert_eq!(v.token_str(6), Some("hello"));
        assert_eq!(v.token_str(99), None);
    }

    #[test]
    fn test_token_id_lookup() {
        let v = make_vocab();
        assert_eq!(v.token_id("he"), Some(3));
        assert_eq!(v.token_id("missing"), None);
    }

    #[test]
    fn test_special_tokens() {
        let v = make_vocab();
        assert_eq!(v.special.bos, 1);
        assert_eq!(v.special.eos, 2);
        assert!(v.special.pad.is_none());
    }

    #[test]
    fn token_metadata_derives_unknown_and_atomic_special_ids() {
        let mut vocab = make_vocab();
        vocab.set_token_metadata(vec![2, 3, 4, 1, 1, 1, 1], None, Some(6), vec![5]);

        assert_eq!(vocab.unknown_id(), Some(0));
        assert_eq!(vocab.separator, Some(6));
        assert_eq!(vocab.tokenization_special_ids(), &[0, 1, 2, 5]);
    }
}
