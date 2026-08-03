use std::collections::VecDeque;

pub(super) struct CompressorState {
    pub(super) previous: Vec<Vec<f32>>,
    pub(super) current_values: Vec<Vec<f32>>,
    pub(super) current_scores: Vec<Vec<f32>>,
    pub(super) compressed: Vec<Vec<f32>>,
}

#[derive(Clone)]
struct CompressorCheckpoint {
    previous: Vec<Vec<f32>>,
    current_values: Vec<Vec<f32>>,
    current_scores: Vec<Vec<f32>>,
    compressed_len: usize,
}

#[derive(Clone)]
struct AttentionCheckpoint {
    window: VecDeque<Vec<f32>>,
    compressor: Option<CompressorCheckpoint>,
    indexer_compressor: Option<CompressorCheckpoint>,
}

#[derive(Clone)]
pub(crate) struct DeepSeek4StateCheckpoint {
    position: usize,
    layers: Vec<AttentionCheckpoint>,
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

    fn checkpoint(&self) -> CompressorCheckpoint {
        CompressorCheckpoint {
            previous: self.previous.clone(),
            current_values: self.current_values.clone(),
            current_scores: self.current_scores.clone(),
            compressed_len: self.compressed.len(),
        }
    }

    fn restore(&mut self, checkpoint: &CompressorCheckpoint) {
        self.previous.clone_from(&checkpoint.previous);
        self.current_values.clone_from(&checkpoint.current_values);
        self.current_scores.clone_from(&checkpoint.current_scores);
        self.compressed.truncate(checkpoint.compressed_len);
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

    fn checkpoint(&self) -> AttentionCheckpoint {
        AttentionCheckpoint {
            window: self.window.clone(),
            compressor: self.compressor.as_ref().map(CompressorState::checkpoint),
            indexer_compressor: self
                .indexer_compressor
                .as_ref()
                .map(CompressorState::checkpoint),
        }
    }

    fn restore(&mut self, checkpoint: &AttentionCheckpoint) {
        self.window.clone_from(&checkpoint.window);
        match (&mut self.compressor, &checkpoint.compressor) {
            (Some(state), Some(saved)) => state.restore(saved),
            (None, None) => {}
            _ => panic!("DeepSeek4 compressor checkpoint layout mismatch"),
        }
        match (&mut self.indexer_compressor, &checkpoint.indexer_compressor) {
            (Some(state), Some(saved)) => state.restore(saved),
            (None, None) => {}
            _ => panic!("DeepSeek4 indexer checkpoint layout mismatch"),
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

    pub(in crate::engine) fn checkpoint(&self) -> DeepSeek4StateCheckpoint {
        DeepSeek4StateCheckpoint {
            position: self.position,
            layers: self.layers.iter().map(AttentionState::checkpoint).collect(),
        }
    }

    pub(in crate::engine) fn restore(&mut self, checkpoint: &DeepSeek4StateCheckpoint) {
        assert_eq!(
            self.layers.len(),
            checkpoint.layers.len(),
            "DeepSeek4 checkpoint layer count mismatch"
        );
        self.position = checkpoint.position;
        for (state, saved) in self.layers.iter_mut().zip(&checkpoint.layers) {
            state.restore(saved);
        }
    }
}
