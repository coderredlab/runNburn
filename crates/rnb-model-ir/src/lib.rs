#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    DecoderOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub usize);

impl LayerId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub usize);

impl TensorId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertId(pub usize);

impl ExpertId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    TokenEmbedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    FeedForwardGate,
    FeedForwardUp,
    FeedForwardDown,
    GdnQkv,
    GdnGate,
    ExpertGateUp,
    ExpertDown,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId {
    layer: LayerId,
    role: WeightRole,
}

impl WeightId {
    pub const fn new(layer: LayerId, role: WeightRole) -> Self {
        Self { layer, role }
    }

    pub const fn layer(self) -> LayerId {
        self.layer
    }

    pub const fn role(self) -> WeightRole {
        self.role
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Attention,
    FeedForward,
    Gdn,
    Moe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    pub id: LayerId,
    pub kind: LayerKind,
}

impl LayerSpec {
    pub const fn new(id: LayerId, kind: LayerKind) -> Self {
        Self { id, kind }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGraph {
    kind: ModelKind,
    layers: Vec<LayerSpec>,
}

impl ModelGraph {
    pub fn new(kind: ModelKind) -> Self {
        Self {
            kind,
            layers: Vec::new(),
        }
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn push_layer(&mut self, layer: LayerSpec) {
        self.layers.push(layer);
    }

    pub fn layers(&self) -> &[LayerSpec] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_and_weight_ids_are_stable_values() {
        assert_eq!(LayerId(7).index(), 7);
        assert_eq!(
            WeightId::new(LayerId(7), WeightRole::AttentionQ).layer(),
            LayerId(7)
        );
    }

    #[test]
    fn model_graph_records_layers_without_backend_types() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Attention));
        graph.push_layer(LayerSpec::new(LayerId(1), LayerKind::FeedForward));

        assert_eq!(graph.layers().len(), 2);
        assert_eq!(graph.layers()[0].kind, LayerKind::Attention);
    }
}
