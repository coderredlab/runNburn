use crate::common::trace::TraceEvent;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteShapeAnalysis {
    pub max_group: usize,
    pub layers: Vec<RouteShapeLayerSummary>,
    pub total_events: usize,
    pub total_groups: usize,
    pub full_groups: usize,
    pub len_hist: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteShapeLayerSummary {
    pub layer: usize,
    pub events: usize,
    pub unique_experts: usize,
    pub groups: usize,
    pub full_groups: usize,
    pub max_run_len: usize,
    pub len_hist: Vec<usize>,
}

pub fn analyze_route_shape(
    events: &[TraceEvent],
    max_group: usize,
) -> Result<RouteShapeAnalysis, String> {
    if max_group == 0 {
        return Err("max group must be non-zero".to_string());
    }

    let mut by_layer = BTreeMap::<usize, Vec<&TraceEvent>>::new();
    for event in events {
        by_layer.entry(event.layer).or_default().push(event);
    }

    let mut layers = Vec::with_capacity(by_layer.len());
    let mut total_groups = 0usize;
    let mut full_groups = 0usize;
    let mut len_hist = vec![0usize; max_group + 1];

    for (layer, mut layer_events) in by_layer {
        layer_events.sort_unstable_by_key(|event| (event.expert_id, event.step, event.rank));
        let mut layer_hist = vec![0usize; max_group + 1];
        let mut layer_groups = 0usize;
        let mut layer_full_groups = 0usize;
        let mut max_run_len = 0usize;
        let mut unique_experts = 0usize;
        let mut idx = 0usize;
        while idx < layer_events.len() {
            let expert = layer_events[idx].expert_id;
            unique_experts += 1;
            let mut run_len = 1usize;
            while idx + run_len < layer_events.len()
                && layer_events[idx + run_len].expert_id == expert
            {
                run_len += 1;
            }
            max_run_len = max_run_len.max(run_len);

            let full = run_len / max_group;
            let rem = run_len % max_group;
            if full > 0 {
                layer_hist[max_group] += full;
                layer_groups += full;
                layer_full_groups += full;
            }
            if rem > 0 {
                layer_hist[rem] += 1;
                layer_groups += 1;
            }
            idx += run_len;
        }

        for (len, count) in layer_hist.iter().copied().enumerate() {
            len_hist[len] += count;
        }
        total_groups += layer_groups;
        full_groups += layer_full_groups;
        layers.push(RouteShapeLayerSummary {
            layer,
            events: layer_events.len(),
            unique_experts,
            groups: layer_groups,
            full_groups: layer_full_groups,
            max_run_len,
            len_hist: layer_hist,
        });
    }

    Ok(RouteShapeAnalysis {
        max_group,
        layers,
        total_events: events.len(),
        total_groups,
        full_groups,
        len_hist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_shape_groups_events_by_layer_and_expert_after_expert_token_sort() {
        let events = vec![
            TraceEvent {
                step: 0,
                layer: 1,
                expert_id: 7,
                rank: Some(0),
                score: None,
            },
            TraceEvent {
                step: 1,
                layer: 1,
                expert_id: 7,
                rank: Some(0),
                score: None,
            },
            TraceEvent {
                step: 2,
                layer: 1,
                expert_id: 7,
                rank: Some(0),
                score: None,
            },
            TraceEvent {
                step: 3,
                layer: 1,
                expert_id: 7,
                rank: Some(0),
                score: None,
            },
            TraceEvent {
                step: 4,
                layer: 1,
                expert_id: 7,
                rank: Some(0),
                score: None,
            },
            TraceEvent {
                step: 0,
                layer: 1,
                expert_id: 9,
                rank: Some(1),
                score: None,
            },
            TraceEvent {
                step: 1,
                layer: 1,
                expert_id: 9,
                rank: Some(1),
                score: None,
            },
            TraceEvent {
                step: 0,
                layer: 2,
                expert_id: 3,
                rank: Some(0),
                score: None,
            },
        ];

        let analysis = analyze_route_shape(&events, 4).unwrap();

        assert_eq!(analysis.total_events, 8);
        assert_eq!(analysis.total_groups, 4);
        assert_eq!(analysis.len_hist[1], 2);
        assert_eq!(analysis.len_hist[2], 1);
        assert_eq!(analysis.len_hist[4], 1);
        assert_eq!(analysis.full_groups, 1);
        assert_eq!(analysis.layers.len(), 2);
        assert_eq!(analysis.layers[0].layer, 1);
        assert_eq!(analysis.layers[0].groups, 3);
        assert_eq!(analysis.layers[0].max_run_len, 5);
    }

    #[test]
    fn route_shape_rejects_zero_max_group() {
        assert!(analyze_route_shape(&[], 0).is_err());
    }
}
