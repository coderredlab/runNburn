use crate::error::{LlmError, Result};
use crate::tokenizer::Tokenizer;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum GenerationConstraint {
    JsonSchema(Value),
    Lark(String),
}

impl GenerationConstraint {
    fn grammar(&self) -> TopLevelGrammar {
        match self {
            Self::JsonSchema(schema) => TopLevelGrammar::from_json_schema(schema.clone()),
            Self::Lark(grammar) => TopLevelGrammar::from_lark(grammar.clone()),
        }
    }

    pub fn validate(&self, tokenizer: &Tokenizer) -> Result<()> {
        let mut decoder = StructuredDecoder::new(tokenizer, self)?;
        decoder.compute_mask().map(|_| ())
    }
}

pub(crate) struct StructuredDecoder {
    matcher: Matcher,
}

impl StructuredDecoder {
    pub(crate) fn new(tokenizer: &Tokenizer, constraint: &GenerationConstraint) -> Result<Self> {
        let factory = tokenizer.structured_decoder_factory()?;
        Ok(Self {
            matcher: Matcher::new(factory.create_parser(constraint.grammar())),
        })
    }

    fn compute_mask(&mut self) -> Result<llguidance::toktrie::SimpleVob> {
        self.matcher
            .compute_mask_or_eos()
            .map_err(|error| LlmError::InvalidChatRequest(error.to_string()))
    }

    pub(crate) fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
        let mask = self.compute_mask()?;
        if mask.len() != logits.len() {
            return Err(LlmError::Forward(format!(
                "structured decoder vocabulary mismatch: mask has {} tokens, logits have {}",
                mask.len(),
                logits.len()
            )));
        }
        if mask.num_set() == 0 {
            return Err(LlmError::Forward(
                "structured decoder produced an empty token mask".to_string(),
            ));
        }
        mask.iter_unset_entries(|index| logits[index] = f32::NEG_INFINITY);
        Ok(())
    }

    pub(crate) fn consume_token(&mut self, token: u32) -> Result<()> {
        self.matcher.consume_token(token).map_err(|error| {
            LlmError::Forward(format!("structured decoder rejected token: {error}"))
        })
    }

    pub(crate) fn is_stopped(&self) -> bool {
        self.matcher.is_stopped()
    }
}

pub(crate) fn build_parser_factory(
    tokenizer: &Tokenizer,
) -> std::result::Result<ParserFactory, String> {
    let vocab_size = tokenizer.vocab.size();
    let mut words = (0..vocab_size)
        .map(|token| tokenizer.decoded_token_bytes(token as u32))
        .collect::<Vec<_>>();
    let special = &tokenizer.vocab.special;
    words[special.eos as usize] = b"\xff<eos>".to_vec();
    if special.bos != special.eos {
        words[special.bos as usize] = b"\xff<bos>".to_vec();
    }
    if let Some(pad) = special.pad {
        if pad != special.eos && pad != special.bos {
            words[pad as usize] = b"\xff<pad>".to_vec();
        }
    }

    let info = TokRxInfo {
        vocab_size: vocab_size as u32,
        tok_eos: special.eos,
        tok_bos: Some(special.bos),
        tok_pad: special.pad,
        tok_unk: tokenizer.vocab.token_id("<unk>"),
        tok_end_of_turn: None,
    };
    let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    let mut factory = ParserFactory::new_simple(&tok_env).map_err(|error| error.to_string())?;
    factory.quiet();
    Ok(factory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SpecialTokens, Vocab};

    fn ascii_tokenizer() -> Tokenizer {
        let mut tokens = (32u8..=126)
            .map(|byte| (byte as char).to_string())
            .collect::<Vec<_>>();
        let bos = tokens.len() as u32;
        tokens.push("<bos>".to_string());
        let eos = tokens.len() as u32;
        tokens.push("<eos>".to_string());
        Tokenizer::new_sentencepiece_with_config(
            Vocab::new(
                tokens,
                SpecialTokens {
                    bos,
                    eos,
                    pad: None,
                },
            ),
            Vec::new(),
            Vec::new(),
            false,
            false,
        )
    }

    #[test]
    fn json_schema_masks_invalid_tokens_and_accepts_valid_object() {
        let tokenizer = ascii_tokenizer();
        let constraint = GenerationConstraint::JsonSchema(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }));
        let mut decoder = StructuredDecoder::new(&tokenizer, &constraint).unwrap();
        let x = tokenizer.vocab.token_id("x").unwrap() as usize;
        let mut logits = vec![0.0; tokenizer.vocab.size()];
        decoder.mask_logits(&mut logits).unwrap();
        assert_eq!(logits[x], f32::NEG_INFINITY);

        for character in r#"{"city":"Seoul"}"#.chars() {
            let token = tokenizer.vocab.token_id(&character.to_string()).unwrap();
            let mut logits = vec![0.0; tokenizer.vocab.size()];
            decoder.mask_logits(&mut logits).unwrap();
            assert!(
                logits[token as usize].is_finite(),
                "token {character:?} was masked"
            );
            decoder.consume_token(token).unwrap();
        }

        let mut logits = vec![0.0; tokenizer.vocab.size()];
        decoder.mask_logits(&mut logits).unwrap();
        assert!(logits[tokenizer.vocab.special.eos as usize].is_finite());
    }
}
