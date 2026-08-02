use std::collections::VecDeque;

pub(super) struct CompressorState {
    pub(super) previous: Vec<Vec<f32>>,
    pub(super) current_values: Vec<Vec<f32>>,
    pub(super) current_scores: Vec<Vec<f32>>,
    pub(super) compressed: Vec<Vec<f32>>,
}

impl CompressorState {
    pub(super) fn new(ratio: usize) -> Self {
        Self {
            previous: Vec::with_capacity(ratio),
            current_values: Vec::with_capacity(ratio),
            current_scores: Vec::with_capacity(ratio),
            compressed: Vec::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.previous.clear();
        self.current_values.clear();
        self.current_scores.clear();
        self.compressed.clear();
    }
}

pub(super) struct AttentionState {
    pub(super) window: VecDeque<Vec<f32>>,
    pub(super) compressor: Option<CompressorState>,
    pub(super) indexer_compressor: Option<CompressorState>,
}

impl AttentionState {
    pub(super) fn new(ratio: usize) -> Self {
        Self {
            window: VecDeque::new(),
            compressor: (ratio > 0).then(|| CompressorState::new(ratio)),
            indexer_compressor: (ratio == 4).then(|| CompressorState::new(ratio)),
        }
    }

    pub(super) fn clear(&mut self) {
        self.window.clear();
        if let Some(state) = &mut self.compressor {
            state.clear();
        }
        if let Some(state) = &mut self.indexer_compressor {
            state.clear();
        }
    }
}

pub(super) struct DeepSeek4State {
    pub(super) position: usize,
    pub(super) layers: Vec<AttentionState>,
}

impl DeepSeek4State {
    pub(super) fn new(compress_ratios: &[usize]) -> Self {
        Self {
            position: 0,
            layers: compress_ratios
                .iter()
                .copied()
                .map(AttentionState::new)
                .collect(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}
