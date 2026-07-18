use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub step: usize,
    pub layer: usize,
    pub expert_id: usize,
    #[serde(default)]
    pub rank: Option<usize>,
    #[serde(default)]
    pub score: Option<f32>,
}

pub fn parse_jsonl_trace(text: &str) -> Result<Vec<TraceEvent>, String> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event = serde_json::from_str::<TraceEvent>(line)
            .map_err(|e| format!("bad JSONL trace at line {}: {}", idx + 1, e))?;
        out.push(event);
    }
    Ok(out)
}

pub fn parse_rnb_route_csv(text: &str) -> Result<Vec<TraceEvent>, String> {
    let mut layer_steps = HashMap::<usize, usize>::new();
    let mut out = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let values = line
            .split(',')
            .map(str::trim)
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|e| format!("bad route csv at line {}: {}", line_idx + 1, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() < 2 {
            return Err(format!(
                "bad route csv at line {}: expected layer and at least one expert",
                line_idx + 1
            ));
        }
        let layer = values[0];
        let step = layer_steps.entry(layer).or_insert(0);
        for (rank, &expert_id) in values[1..].iter().enumerate() {
            out.push(TraceEvent {
                step: *step,
                layer,
                expert_id,
                rank: Some(rank),
                score: None,
            });
        }
        *step += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_jsonl_trace_events() {
        let text = r#"{"step":0,"layer":3,"expert_id":17,"rank":0,"score":0.42}
{"step":1,"layer":4,"expert_id":2}"#;

        let events = parse_jsonl_trace(text).expect("trace parses");

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            TraceEvent {
                step: 0,
                layer: 3,
                expert_id: 17,
                rank: Some(0),
                score: Some(0.42),
            }
        );
        assert_eq!(events[1].step, 1);
        assert_eq!(events[1].rank, None);
        assert_eq!(events[1].score, None);
    }

    #[test]
    fn parses_rnb_route_csv_as_ordered_events() {
        let text = "3,9,2,5\n4,1,8\n3,7,6\n";

        let events = parse_rnb_route_csv(text).expect("route csv parses");

        assert_eq!(events.len(), 7);
        assert_eq!(events[0].step, 0);
        assert_eq!(events[0].layer, 3);
        assert_eq!(events[0].expert_id, 9);
        assert_eq!(events[0].rank, Some(0));
        assert_eq!(events[3].step, 0);
        assert_eq!(events[3].layer, 4);
        assert_eq!(events[5].step, 1);
        assert_eq!(events[5].layer, 3);
    }
}
