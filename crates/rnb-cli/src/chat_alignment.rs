use rnb_llm::{ChatMessage, ChatTemplateOptions, Tokenizer};
use serde_json::Value;

pub(crate) fn render_chat_resume_alignment(
    tokenizer: &Tokenizer,
    messages_before_assistant: &[ChatMessage],
    assistant_content: &str,
    options: ChatTemplateOptions,
    tool_definitions: &[Value],
) -> Result<(String, String), String> {
    let options = ChatTemplateOptions {
        add_generation_prompt: false,
        ..options
    };
    let existing = tokenizer
        .render_chat_prompt_with_tools(messages_before_assistant, options, tool_definitions)
        .map_err(|error| error.to_string())?;
    let mut assistant_sentinel = "__RNB_CHAT_ASSISTANT_CONTENT_7F43A9C2__".to_string();
    let mut user_sentinel = "__RNB_CHAT_NEXT_USER_CONTENT_51D8E604__".to_string();
    while existing.contains(&assistant_sentinel)
        || existing.contains(&user_sentinel)
        || assistant_content.contains(&assistant_sentinel)
        || assistant_content.contains(&user_sentinel)
    {
        assistant_sentinel.push('_');
        user_sentinel.push('_');
    }

    let mut probe_messages = messages_before_assistant.to_vec();
    probe_messages.push(ChatMessage::new("assistant", assistant_sentinel.clone()));
    probe_messages.push(ChatMessage::new("user", user_sentinel.clone()));
    let probe = tokenizer
        .render_chat_prompt_with_tools(&probe_messages, options, tool_definitions)
        .map_err(|error| error.to_string())?;
    let (_, probe_tail) = probe
        .split_once(&assistant_sentinel)
        .ok_or_else(|| "chat template omitted the assistant response content".to_string())?;
    let (append_text, _) = probe_tail
        .split_once(&user_sentinel)
        .ok_or_else(|| "chat template omitted the next user content".to_string())?;

    let mut completed_messages = messages_before_assistant.to_vec();
    completed_messages.push(ChatMessage::new("assistant", assistant_content));
    completed_messages.push(ChatMessage::new("user", user_sentinel.clone()));
    let completed = tokenizer
        .render_chat_prompt_with_tools(&completed_messages, options, tool_definitions)
        .map_err(|error| error.to_string())?;
    let (prompt_prefix, _) = completed
        .split_once(&user_sentinel)
        .ok_or_else(|| "chat template omitted the next user content".to_string())?;

    Ok((prompt_prefix.to_string(), append_text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_llm::tokenizer::vocab::{SpecialTokens, Vocab};

    fn tokenizer() -> Tokenizer {
        let vocab = Vocab::new(
            vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()],
            SpecialTokens {
                bos: 1,
                eos: 2,
                pad: None,
            },
        );
        let mut tokenizer =
            Tokenizer::new_sentencepiece_with_config(vocab, Vec::new(), Vec::new(), false, true);
        tokenizer.set_chat_template(Some(
            "{% for message in messages %}<{{ message.role }}>{{ message.content | trim }}</{{ message.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
                .to_string(),
        ));
        tokenizer
    }

    #[test]
    fn alignment_uses_template_normalized_assistant_content() {
        let tokenizer = tokenizer();
        let (prompt_prefix, append_text) = render_chat_resume_alignment(
            &tokenizer,
            &[ChatMessage::new("user", "Name the colors")],
            " Green Pink ",
            ChatTemplateOptions::default(),
            &[],
        )
        .unwrap();

        assert_eq!(
            prompt_prefix,
            "<user>Name the colors</user><assistant>Green Pink</assistant><user>"
        );
        assert_eq!(append_text, "</assistant><user>");
    }
}
