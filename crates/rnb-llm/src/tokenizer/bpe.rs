use crate::tokenizer::vocab::Vocab;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
struct SpmMergeSymbol {
    text: String,
    n: usize,
    prev: isize,
    next: isize,
}

#[derive(Clone, Debug, PartialEq)]
struct SpmBigram {
    left: usize,
    right: usize,
    score: f32,
    size: usize,
}

impl Eq for SpmBigram {}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BpeBigram {
    rank: usize,
    left: usize,
    right: usize,
}

impl Ord for BpeBigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for BpeBigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct BpeMergeSymbol {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

/// GPT-2 byte-level BPE에서 사용하는 byte↔unicode 매핑 테이블.
/// Python의 openai/tiktoken bytes_to_unicode()와 동일.
fn build_byte_unicode_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut n = 0u32;

    // 직접 매핑 바이트 범위: printable ASCII + Latin-1 supplement
    let direct_ranges: &[(u8, u8)] = &[
        (b'!', b'~'), // 33-126
        (0xA1, 0xAC), // 161-172
        (0xAE, 0xFF), // 174-255
    ];

    let mut is_direct = [false; 256];
    for &(start, end) in direct_ranges {
        for b in start..=end {
            byte_to_char[b as usize] = b as char;
            is_direct[b as usize] = true;
        }
    }

    // 나머지 바이트 → 256+ 유니코드 영역으로 매핑
    for b in 0u8..=255 {
        if !is_direct[b as usize] {
            byte_to_char[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }

    let unicode_to_byte: HashMap<char, u8> = byte_to_char
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect();

    (byte_to_char, unicode_to_byte)
}

pub enum TokenizerMode {
    SentencePiece {
        add_space_prefix: bool,
    },
    Gemma4Bpe,
    Gpt2Bpe {
        byte_to_unicode: [char; 256],
        unicode_to_byte: HashMap<char, u8>,
    },
}

pub struct Tokenizer {
    pub vocab: Vocab,
    merge_rank: HashMap<(u32, u32), usize>,
    mode: TokenizerMode,
    sentencepiece_scores: Vec<f32>,
    add_bos_token: bool,
    chat_template: Option<String>,
    max_token_chars: usize,
    structured_decoder_factory: OnceLock<std::result::Result<llguidance::ParserFactory, String>>,
}

impl Tokenizer {
    /// SentencePiece BPE 모드 (LLaMA, TinyLlama 등)
    pub fn new(vocab: Vocab, merges: Vec<(u32, u32)>) -> Self {
        Self::new_sentencepiece(vocab, merges)
    }

    pub fn new_sentencepiece(vocab: Vocab, merges: Vec<(u32, u32)>) -> Self {
        Self::new_sentencepiece_with_config(vocab, merges, vec![], true, true)
    }

    pub fn new_sentencepiece_with_scores(
        vocab: Vocab,
        merges: Vec<(u32, u32)>,
        scores: Vec<f32>,
    ) -> Self {
        Self::new_sentencepiece_with_config(vocab, merges, scores, true, true)
    }

    pub fn new_sentencepiece_with_config(
        vocab: Vocab,
        merges: Vec<(u32, u32)>,
        scores: Vec<f32>,
        add_bos_token: bool,
        add_space_prefix: bool,
    ) -> Self {
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank))
            .collect();
        let max_token_chars = vocab
            .id_to_token
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            vocab,
            merge_rank,
            mode: TokenizerMode::SentencePiece { add_space_prefix },
            sentencepiece_scores: scores,
            add_bos_token,
            chat_template: None,
            max_token_chars,
            structured_decoder_factory: OnceLock::new(),
        }
    }

    pub fn new_gemma4_bpe(
        vocab: Vocab,
        merges: Vec<(u32, u32)>,
        scores: Vec<f32>,
        add_bos_token: bool,
    ) -> Self {
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank))
            .collect();
        let max_token_chars = vocab
            .id_to_token
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            vocab,
            merge_rank,
            mode: TokenizerMode::Gemma4Bpe,
            sentencepiece_scores: scores,
            add_bos_token,
            chat_template: None,
            max_token_chars,
            structured_decoder_factory: OnceLock::new(),
        }
    }

    /// GPT-2 byte-level BPE 모드 (Qwen, GPT 등)
    pub fn new_gpt2(vocab: Vocab, merges: Vec<(u32, u32)>) -> Self {
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank))
            .collect();
        let (byte_to_unicode, unicode_to_byte) = build_byte_unicode_maps();
        let max_token_chars = vocab
            .id_to_token
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            vocab,
            merge_rank,
            mode: TokenizerMode::Gpt2Bpe {
                byte_to_unicode,
                unicode_to_byte,
            },
            sentencepiece_scores: vec![],
            add_bos_token: true,
            chat_template: None,
            max_token_chars,
            structured_decoder_factory: OnceLock::new(),
        }
    }

    pub fn should_add_bos(&self) -> bool {
        self.add_bos_token
    }

    /// Override the BOS-token policy after construction. Used when GGUF
    /// metadata sets `tokenizer.ggml.add_bos_token=false` (e.g. Qwen3.5)
    /// but the tokenizer was constructed with `new_gpt2` whose default is
    /// `true`.
    pub fn set_add_bos_token(&mut self, add_bos_token: bool) {
        self.add_bos_token = add_bos_token;
    }

    pub fn set_chat_template(&mut self, chat_template: Option<String>) {
        self.chat_template = chat_template;
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub(crate) fn structured_decoder_factory(
        &self,
    ) -> crate::error::Result<&llguidance::ParserFactory> {
        match self
            .structured_decoder_factory
            .get_or_init(|| crate::constrained::build_parser_factory(self))
        {
            Ok(factory) => Ok(factory),
            Err(error) => Err(crate::error::LlmError::Tokenizer(error.clone())),
        }
    }

    /// 텍스트 → token id 시퀀스
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match &self.mode {
            TokenizerMode::SentencePiece { add_space_prefix } => {
                self.initial_tokenize_sp_with_specials(text, *add_space_prefix)
            }
            TokenizerMode::Gemma4Bpe => self.initial_tokenize_gemma4_spm_with_specials(text),
            TokenizerMode::Gpt2Bpe {
                byte_to_unicode, ..
            } => self.initial_tokenize_gpt2_with_specials(text, byte_to_unicode),
        }
    }

    /// token id → 디코딩된 문자열
    pub fn decode_token(&self, id: u32) -> String {
        let raw = self.vocab.token_str(id).unwrap_or("");
        match &self.mode {
            TokenizerMode::SentencePiece { .. } => self.decode_sp(raw),
            TokenizerMode::Gemma4Bpe => self.decode_sp(raw),
            TokenizerMode::Gpt2Bpe {
                unicode_to_byte, ..
            } => self.decode_gpt2(raw, unicode_to_byte),
        }
    }

    pub(crate) fn decoded_token_bytes(&self, id: u32) -> Vec<u8> {
        let raw = self.vocab.token_str(id).unwrap_or("");
        match &self.mode {
            TokenizerMode::SentencePiece { .. } | TokenizerMode::Gemma4Bpe => {
                self.decode_sp(raw).into_bytes()
            }
            TokenizerMode::Gpt2Bpe {
                unicode_to_byte, ..
            } => raw
                .chars()
                .map(|character| unicode_to_byte.get(&character).copied().unwrap_or(b'?'))
                .collect(),
        }
    }

    /// token id 시퀀스 → 문자열
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&id| self.decode_token(id)).collect()
    }

    // =========================================================================
    // BPE merge (공유)
    // =========================================================================

    fn bpe_merge(&self, tokens: Vec<u32>) -> Vec<u32> {
        if tokens.len() < 2 {
            return tokens;
        }

        let token_count = tokens.len();
        let mut symbols = tokens
            .into_iter()
            .enumerate()
            .map(|(idx, id)| BpeMergeSymbol {
                id,
                prev: idx.checked_sub(1),
                next: (idx + 1 < token_count).then_some(idx + 1),
                alive: true,
            })
            .collect::<Vec<_>>();
        let len = symbols.len();
        for (idx, symbol) in symbols.iter_mut().enumerate() {
            symbol.next = (idx + 1 < len).then_some(idx + 1);
        }

        let mut heap = BinaryHeap::new();
        for left in 0..symbols.len() - 1 {
            self.push_bpe_bigram(&symbols, left, left + 1, &mut heap);
        }

        while let Some(candidate) = heap.pop() {
            if !self.bpe_candidate_is_current(&symbols, candidate) {
                continue;
            }

            let left_id = symbols[candidate.left].id;
            let right_id = symbols[candidate.right].id;
            let Some(&rank) = self.merge_rank.get(&(left_id, right_id)) else {
                continue;
            };
            if rank != candidate.rank {
                continue;
            }

            let left_str = self.vocab.token_str(left_id).unwrap_or("");
            let right_str = self.vocab.token_str(right_id).unwrap_or("");
            let merged_str = format!("{}{}", left_str, right_str);
            let Some(merged_id) = self.vocab.token_id(&merged_str) else {
                break;
            };

            let prev = symbols[candidate.left].prev;
            let next = symbols[candidate.right].next;
            symbols[candidate.left].id = merged_id;
            symbols[candidate.left].next = next;
            symbols[candidate.right].alive = false;
            symbols[candidate.right].prev = None;
            symbols[candidate.right].next = None;
            if let Some(next_idx) = next {
                symbols[next_idx].prev = Some(candidate.left);
            }

            if let Some(prev_idx) = prev {
                self.push_bpe_bigram(&symbols, prev_idx, candidate.left, &mut heap);
            }
            if let Some(next_idx) = next {
                self.push_bpe_bigram(&symbols, candidate.left, next_idx, &mut heap);
            }
        }

        let mut out = Vec::new();
        let mut cursor = Some(0usize);
        while let Some(idx) = cursor {
            if symbols[idx].alive {
                out.push(symbols[idx].id);
            }
            cursor = symbols[idx].next;
        }
        out
    }

    fn push_bpe_bigram(
        &self,
        symbols: &[BpeMergeSymbol],
        left: usize,
        right: usize,
        heap: &mut BinaryHeap<BpeBigram>,
    ) {
        if !symbols[left].alive || !symbols[right].alive || symbols[left].next != Some(right) {
            return;
        }
        if let Some(&rank) = self.merge_rank.get(&(symbols[left].id, symbols[right].id)) {
            heap.push(BpeBigram { rank, left, right });
        }
    }

    fn bpe_candidate_is_current(&self, symbols: &[BpeMergeSymbol], candidate: BpeBigram) -> bool {
        let left = candidate.left;
        let right = candidate.right;
        left < symbols.len()
            && right < symbols.len()
            && symbols[left].alive
            && symbols[right].alive
            && symbols[left].next == Some(right)
            && symbols[right].prev == Some(left)
    }

    // =========================================================================
    // SentencePiece 모드
    // =========================================================================

    fn initial_tokenize_sp(&self, text: &str, add_space_prefix: bool) -> Vec<u32> {
        let text = if add_space_prefix {
            format!(" {text}").replace(' ', "\u{2581}")
        } else {
            text.replace(' ', "\u{2581}")
        };
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut best: Vec<Option<(f32, usize, usize, u32)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, 0, 0));

        for i in 0..n {
            let Some((score, pieces, _, _)) = best[i] else {
                continue;
            };

            let mut matched = false;
            let max_j = (i + self.max_token_chars).min(n);
            for j in i + 1..=max_j {
                let piece: String = chars[i..j].iter().collect();
                if let Some(id) = self.vocab.token_id(&piece) {
                    matched = true;
                    let candidate = (score + self.sentencepiece_score(id), pieces + 1, i, id);
                    if self.is_better_sentencepiece_candidate(best[j], candidate) {
                        best[j] = Some(candidate);
                    }
                }
            }

            if !matched {
                let id = self.char_to_token_sp(chars[i]);
                let candidate = (score + self.sentencepiece_score(id), pieces + 1, i, id);
                if self.is_better_sentencepiece_candidate(best[i + 1], candidate) {
                    best[i + 1] = Some(candidate);
                }
            }
        }

        if best[n].is_none() {
            return chars
                .into_iter()
                .map(|c| self.char_to_token_sp(c))
                .collect();
        }

        let mut tokens = Vec::new();
        let mut idx = n;
        while idx > 0 {
            let (_, _, prev, token_id) = best[idx].expect("sentencepiece path should exist");
            tokens.push(token_id);
            idx = prev;
        }
        tokens.reverse();
        tokens
    }

    fn initial_tokenize_sp_with_specials(&self, text: &str, add_space_prefix: bool) -> Vec<u32> {
        let special_tokens = self.special_control_tokens();
        if special_tokens.is_empty() {
            return self.initial_tokenize_sp(text, add_space_prefix);
        }

        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            let matched = special_tokens
                .iter()
                .filter(|(token, _)| text[cursor..].starts_with(token.as_str()))
                .max_by_key(|(token, _)| token.len());

            if let Some((token, token_id)) = matched {
                out.push(*token_id);
                cursor += token.len();
                continue;
            }

            let next_special = special_tokens
                .iter()
                .filter_map(|(token, _)| text[cursor..].find(token).map(|idx| cursor + idx))
                .min()
                .unwrap_or(text.len());

            if next_special > cursor {
                out.extend(self.initial_tokenize_sp(&text[cursor..next_special], add_space_prefix));
                cursor = next_special;
            } else {
                let ch = text[cursor..].chars().next().expect("valid utf-8 cursor");
                out.extend(self.initial_tokenize_sp(&ch.to_string(), add_space_prefix));
                cursor += ch.len_utf8();
            }
        }

        out
    }

    fn special_control_tokens(&self) -> Vec<(String, u32)> {
        let mut out = self
            .vocab
            .id_to_token
            .iter()
            .enumerate()
            .filter_map(|(idx, token)| {
                let enclosed = (token.starts_with('<') && token.ends_with('>'))
                    || (token.starts_with('[') && token.ends_with(']'));
                let looks_special = enclosed && token.len() > 2 && token != "<0x0A>";
                looks_special.then(|| (token.clone(), idx as u32))
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        out
    }

    fn sentencepiece_score(&self, id: u32) -> f32 {
        self.sentencepiece_scores
            .get(id as usize)
            .copied()
            .unwrap_or(0.0)
    }

    fn is_better_sentencepiece_candidate(
        &self,
        current: Option<(f32, usize, usize, u32)>,
        candidate: (f32, usize, usize, u32),
    ) -> bool {
        match current {
            None => true,
            Some((cur_score, cur_pieces, cur_prev, cur_id)) => {
                let (cand_score, cand_pieces, cand_prev, cand_id) = candidate;
                cand_score > cur_score
                    || ((cand_score - cur_score).abs() < f32::EPSILON
                        && (cand_pieces < cur_pieces
                            || (cand_pieces == cur_pieces
                                && (cand_prev > cur_prev
                                    || (cand_prev == cur_prev && cand_id < cur_id)))))
            }
        }
    }

    fn char_to_token_sp(&self, c: char) -> u32 {
        let s = c.to_string();
        if let Some(id) = self.vocab.token_id(&s) {
            return id;
        }
        if c.is_ascii() {
            let hex = format!("<0x{:02X}>", c as u8);
            if let Some(id) = self.vocab.token_id(&hex) {
                return id;
            }
        }
        0
    }

    fn initial_tokenize_gemma4_spm(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let text = text.replace(' ', "\u{2581}");

        let mut symbols = Vec::new();
        for (idx, ch) in text.chars().enumerate() {
            symbols.push(SpmMergeSymbol {
                text: ch.to_string(),
                n: ch.len_utf8(),
                prev: idx as isize - 1,
                next: -1,
            });
        }
        for idx in 0..symbols.len().saturating_sub(1) {
            symbols[idx].next = idx as isize + 1;
        }

        let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();
        let mut work_queue = BinaryHeap::new();

        let try_add_bigram = |left: isize,
                              right: isize,
                              symbols: &Vec<SpmMergeSymbol>,
                              rev_merge: &mut HashMap<String, (usize, usize)>,
                              work_queue: &mut BinaryHeap<SpmBigram>| {
            if left < 0 || right < 0 {
                return;
            }
            let left = left as usize;
            let right = right as usize;
            let text = format!("{}{}", symbols[left].text, symbols[right].text);
            let Some(_token) = self.vocab.token_id(&text) else {
                return;
            };
            // mt89 Stage C — Gemma4 GGUF 의 tokenizer.ggml.scores 가 모두 -1000
            // (default) 라 priority 무의미. 정석 BPE 처럼 merge_rank 기반
            // priority 사용. (left_id, right_id) pair 가 merge rule 에 있어야
            // 만 merge. 없으면 BPE merge skip — HF tokenizer 와 동일 행동.
            let left_id = self.vocab.token_id(&symbols[left].text);
            let right_id = self.vocab.token_id(&symbols[right].text);
            let priority = match (left_id, right_id) {
                (Some(li), Some(ri)) => match self.merge_rank.get(&(li, ri)) {
                    Some(&rank) => -(rank as f32),
                    None => return,
                },
                _ => return,
            };
            work_queue.push(SpmBigram {
                left,
                right,
                score: priority,
                size: text.len(),
            });
            rev_merge.insert(text, (left, right));
        };

        for i in 1..symbols.len() {
            try_add_bigram(
                i as isize - 1,
                i as isize,
                &symbols,
                &mut rev_merge,
                &mut work_queue,
            );
        }

        while let Some(bigram) = work_queue.pop() {
            let left = bigram.left;
            let right = bigram.right;
            if symbols[left].n == 0 || symbols[right].n == 0 {
                continue;
            }
            if symbols[left].n + symbols[right].n != bigram.size {
                continue;
            }

            let right_text = symbols[right].text.clone();
            symbols[left].text.push_str(&right_text);
            symbols[left].n += symbols[right].n;
            symbols[right].n = 0;
            symbols[right].text.clear();

            let right_next = symbols[right].next;
            symbols[left].next = right_next;
            if right_next >= 0 {
                symbols[right_next as usize].prev = left as isize;
            }

            let prev = symbols[left].prev;
            let next = symbols[left].next;
            try_add_bigram(
                prev,
                left as isize,
                &symbols,
                &mut rev_merge,
                &mut work_queue,
            );
            try_add_bigram(
                left as isize,
                next,
                &symbols,
                &mut rev_merge,
                &mut work_queue,
            );
        }

        fn resegment(
            tok: &Tokenizer,
            symbols: &[SpmMergeSymbol],
            rev_merge: &HashMap<String, (usize, usize)>,
            idx: usize,
            output: &mut Vec<u32>,
        ) {
            let text = &symbols[idx].text;
            if let Some(token) = tok.vocab.token_id(text) {
                output.push(token);
                return;
            }
            if let Some(&(left, right)) = rev_merge.get(text) {
                resegment(tok, symbols, rev_merge, left, output);
                resegment(tok, symbols, rev_merge, right, output);
                return;
            }
            for &b in text.as_bytes() {
                let hex = format!("<0x{:02X}>", b);
                output.push(tok.vocab.token_id(&hex).unwrap_or(0));
            }
        }

        let mut out = Vec::new();
        let mut i = 0usize;
        loop {
            if symbols[i].n > 0 {
                resegment(self, &symbols, &rev_merge, i, &mut out);
            }
            let next = symbols[i].next;
            if next < 0 {
                break;
            }
            i = next as usize;
        }
        out
    }

    fn initial_tokenize_gemma4_spm_with_specials(&self, text: &str) -> Vec<u32> {
        let special_tokens = self.special_control_tokens();
        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            let matched = special_tokens
                .iter()
                .filter(|(token, _)| text[cursor..].starts_with(token.as_str()))
                .max_by_key(|(token, _)| token.len());
            if let Some((token, token_id)) = matched {
                out.push(*token_id);
                cursor += token.len();
                continue;
            }
            let next_special = special_tokens
                .iter()
                .filter_map(|(token, _)| text[cursor..].find(token).map(|idx| cursor + idx))
                .min()
                .unwrap_or(text.len());
            let segment = &text[cursor..next_special];
            let mut start = 0usize;
            for (idx, ch) in segment.char_indices() {
                if ch == '\n' {
                    if idx > start {
                        out.extend(self.initial_tokenize_gemma4_spm(&segment[start..idx]));
                    }
                    let end = idx + ch.len_utf8();
                    out.extend(self.initial_tokenize_gemma4_spm(&segment[idx..end]));
                    start = end;
                }
            }
            if start < segment.len() {
                out.extend(self.initial_tokenize_gemma4_spm(&segment[start..]));
            }
            cursor = next_special;
        }
        out
    }

    fn decode_sp(&self, raw: &str) -> String {
        if raw.starts_with("<0x") && raw.ends_with('>') && raw.len() == 6 {
            if let Ok(byte) = u8::from_str_radix(&raw[3..5], 16) {
                return String::from(byte as char);
            }
        }
        raw.replace('\u{2581}', " ")
    }

    // =========================================================================
    // GPT-2 BPE 모드
    // =========================================================================

    fn initial_tokenize_gpt2(&self, text: &str, byte_to_unicode: &[char; 256]) -> Vec<u32> {
        // UTF-8 bytes → GPT-2 unicode chars → vocab lookup
        text.as_bytes()
            .iter()
            .map(|&b| {
                let c = byte_to_unicode[b as usize];
                let s = c.to_string();
                self.vocab.token_id(&s).unwrap_or(0)
            })
            .collect()
    }

    fn initial_tokenize_gpt2_with_specials(
        &self,
        text: &str,
        byte_to_unicode: &[char; 256],
    ) -> Vec<u32> {
        let special_tokens = self.special_control_tokens();
        if special_tokens.is_empty() {
            return self.bpe_merge(self.initial_tokenize_gpt2(text, byte_to_unicode));
        }

        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            let matched = special_tokens
                .iter()
                .find(|(token, _)| text[cursor..].starts_with(token.as_str()));
            if let Some((token, token_id)) = matched {
                out.push(*token_id);
                cursor += token.len();
                continue;
            }

            let next_special = special_tokens
                .iter()
                .filter_map(|(token, _)| text[cursor..].find(token).map(|idx| cursor + idx))
                .min()
                .unwrap_or(text.len());
            if next_special > cursor {
                out.extend(self.bpe_merge(
                    self.initial_tokenize_gpt2(&text[cursor..next_special], byte_to_unicode),
                ));
                cursor = next_special;
            } else {
                let ch = text[cursor..].chars().next().expect("valid utf-8 cursor");
                out.extend(
                    self.bpe_merge(self.initial_tokenize_gpt2(&ch.to_string(), byte_to_unicode)),
                );
                cursor += ch.len_utf8();
            }
        }
        out
    }

    fn decode_gpt2(&self, raw: &str, unicode_to_byte: &HashMap<char, u8>) -> String {
        // GPT-2 unicode chars → bytes → UTF-8 string
        let bytes: Vec<u8> = raw
            .chars()
            .map(|c| unicode_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::vocab::{SpecialTokens, Vocab};

    fn make_sp_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),    // 0
            "<s>".to_string(),      // 1
            "</s>".to_string(),     // 2
            "\u{2581}".to_string(), // 3  (▁ — SentencePiece space)
            "h".to_string(),        // 4
            "e".to_string(),        // 5
            "l".to_string(),        // 6
            "o".to_string(),        // 7
            "he".to_string(),       // 8  (merge 4+5)
            "ll".to_string(),       // 9  (merge 6+6)
            "hel".to_string(),      // 10 (merge 8+6)
            "hell".to_string(),     // 11 (merge 10+6)
            "hello".to_string(),    // 12 (merge 11+7)
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let merges = vec![
            (4, 5),  // h + e → he   (rank 0)
            (6, 6),  // l + l → ll   (rank 1)
            (8, 6),  // he + l → hel (rank 2)
            (10, 6), // hel + l → hell (rank 3)
        ];
        Tokenizer::new_sentencepiece(vocab, merges)
    }

    fn make_sp_scored_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "\u{2581}".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "\u{2581}he".to_string(),
            "llo".to_string(),
            "\u{2581}hello".to_string(),
        ];
        let scores = vec![
            0.0, 0.0, 0.0, -10.0, -5.0, -5.0, -5.0, -5.0, -1.0, -1.0, -0.1,
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        Tokenizer::new_sentencepiece_with_scores(vocab, vec![], scores)
    }

    fn make_sp_tiebreak_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "민".to_string(),
            "국".to_string(),
            "의".to_string(),
            "민국".to_string(),
            "국의".to_string(),
        ];
        let scores = vec![-1000.0; tokens.len()];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        Tokenizer::new_sentencepiece_with_config(vocab, vec![], scores, true, false)
    }

    fn make_sp_no_prefix_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "he".to_string(),
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        Tokenizer::new_sentencepiece_with_config(vocab, vec![], vec![], false, false)
    }

    fn make_sp_special_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "<|turn>".to_string(),
            "<turn|>".to_string(),
            "<|think|>".to_string(),
            "system".to_string(),
            "user".to_string(),
            "model".to_string(),
            "\n".to_string(),
            "assistant".to_string(),
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        Tokenizer::new_sentencepiece_with_config(vocab, vec![], vec![], false, false)
    }

    #[test]
    fn test_encode_simple() {
        let tok = make_sp_tokenizer();
        let ids = tok.encode("he");
        assert_eq!(ids, vec![3, 8]);
    }

    #[test]
    fn test_encode_with_repeat() {
        let tok = make_sp_tokenizer();
        let ids = tok.encode("hell");
        assert_eq!(ids, vec![3, 11]);
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = make_sp_tokenizer();
        let ids = tok.encode("he");
        let text = tok.decode(&ids);
        assert_eq!(text, " he");
    }

    #[test]
    fn test_decode_token() {
        let tok = make_sp_tokenizer();
        assert_eq!(tok.decode_token(8), "he");
        assert_eq!(tok.decode_token(9), "ll");
        assert_eq!(tok.decode_token(0), "<unk>");
    }

    #[test]
    fn test_encode_o() {
        let tok = make_sp_tokenizer();
        let ids = tok.encode("o");
        assert_eq!(ids, vec![3, 7]);
    }

    #[test]
    fn test_encode_hello_partial() {
        let tok = make_sp_tokenizer();
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![3, 12]);
    }

    #[test]
    fn test_sentencepiece_prefers_best_scored_piece_path() {
        let tok = make_sp_scored_tokenizer();
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![10]);
    }

    #[test]
    fn test_sentencepiece_respects_add_space_prefix_false() {
        let tok = make_sp_no_prefix_tokenizer();
        let ids = tok.encode("he");
        assert_eq!(ids, vec![5]);
        assert!(!tok.should_add_bos());
    }

    #[test]
    fn test_sentencepiece_tiebreak_prefers_longer_left_piece() {
        let tok = make_sp_tiebreak_tokenizer();
        let ids = tok.encode("민국의");
        assert_eq!(ids, vec![6, 5]);
    }

    #[test]
    fn test_sentencepiece_preserves_control_tokens_verbatim() {
        let tok = make_sp_special_tokenizer();
        let ids = tok.encode("<|turn>user\n<|think|>assistant<turn|>");
        assert_eq!(ids, vec![3, 7, 9, 5, 10, 4]);
    }

    #[test]
    fn test_gpt2_byte_to_unicode_mapping() {
        let (b2u, u2b) = build_byte_unicode_maps();
        // space(0x20) → Ġ (U+0120)
        assert_eq!(b2u[0x20], '\u{0120}');
        // newline(0x0A) → Ċ (U+010A)
        assert_eq!(b2u[0x0A], '\u{010A}');
        // printable ASCII 'A'(65) → 'A'
        assert_eq!(b2u[b'A' as usize], 'A');
        // roundtrip
        for b in 0u8..=255 {
            let c = b2u[b as usize];
            assert_eq!(*u2b.get(&c).unwrap(), b);
        }
    }

    #[test]
    fn test_gpt2_encode_decode() {
        // GPT-2 style vocab: Ġ = space byte, h, e, l, o as direct chars
        let (b2u, _) = build_byte_unicode_maps();
        let space_char = b2u[b' ' as usize].to_string(); // Ġ

        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            space_char.clone(), // 3 = Ġ (space)
            "h".to_string(),    // 4
            "e".to_string(),    // 5
            "l".to_string(),    // 6
            "o".to_string(),    // 7
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let merges = vec![];
        let tok = Tokenizer::new_gpt2(vocab, merges);

        // "hello" → bytes → GPT-2 unicode → token ids
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![4, 5, 6, 6, 7]); // h, e, l, l, o

        // " hi" → space + h + i (i는 vocab에 없으니 <unk>=0)
        let ids = tok.encode(" hi");
        assert_eq!(ids, vec![3, 4, 0]); // Ġ, h, <unk>(i 없음)

        // decode
        assert_eq!(tok.decode_token(3), " "); // Ġ → space
        assert_eq!(tok.decode_token(4), "h");
    }

    #[test]
    fn test_gpt2_preserves_control_tokens_verbatim() {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "<｜hy_User:opensource｜>".to_string(),
            "<think:opensource>".to_string(),
            "[gMASK]".to_string(),
            "[MASK]".to_string(),
            "[sMASK]".to_string(),
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let tok = Tokenizer::new_gpt2(vocab, vec![]);

        assert_eq!(
            tok.encode("h<｜hy_User:opensource｜>e<think:opensource>"),
            vec![3, 5, 4, 6]
        );
        assert_eq!(tok.encode("[gMASK][MASK][sMASK]"), vec![7, 8, 9]);
    }
    #[test]
    #[ignore = "performance guard for GPT-2 BPE long-prompt merge behavior"]
    fn gpt2_bpe_long_repeated_pair_merge_stays_subquadratic() {
        let a = "a".to_string();
        let aa = "aa".to_string();
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            a.clone(),
            aa,
        ];
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let tok = Tokenizer::new_gpt2(vocab, vec![(3, 3)]);
        let text = "a".repeat(16_384);

        let started = std::time::Instant::now();
        let ids = tok.encode(&text);
        let elapsed = started.elapsed();

        assert_eq!(ids.len(), 8_192);
        assert!(ids.iter().all(|&id| id == 4));
        assert!(
            elapsed.as_millis() < 100,
            "long repeated GPT-2 BPE merge took {elapsed:?}"
        );
    }
}
