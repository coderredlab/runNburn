use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PredictorTraceLine {
    pub seq: usize,
    pub layer: usize,
    pub selected: Vec<usize>,
    pub top: Vec<PredictorCandidate>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PredictorCandidate {
    pub expert: usize,
    pub prob: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PredictorAnalysisRow {
    pub source: &'static str,
    pub lookahead_groups: usize,
    pub top_n: usize,
    pub samples: usize,
    pub avg_recall: f64,
    pub avg_precision: f64,
    pub avg_false_positive_ratio: f64,
}

pub fn parse_predictor_trace_jsonl(text: &str) -> Result<Vec<PredictorTraceLine>, String> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event = serde_json::from_str::<PredictorTraceLine>(line)
            .map_err(|e| format!("bad predictor JSONL trace at line {}: {}", idx + 1, e))?;
        out.push(event);
    }
    out.sort_by_key(|line| line.seq);
    Ok(out)
}

pub fn analyze_predictor_trace(
    lines: &[PredictorTraceLine],
    lookahead_groups: usize,
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if lookahead_groups == 0 {
        return Err("bad --lookahead-groups value 0: expected at least 1".to_string());
    }
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }
    if lines.len() <= lookahead_groups {
        return Ok(PredictorAnalysisRow {
            lookahead_groups,
            source: "router-top",
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for source_idx in 0..(lines.len() - lookahead_groups) {
        let source = &lines[source_idx];
        let target = &lines[source_idx + lookahead_groups];
        if target.selected.is_empty() {
            continue;
        }

        let predicted = source
            .top
            .iter()
            .take(top_n)
            .map(|candidate| candidate.expert)
            .collect::<HashSet<_>>();
        if predicted.is_empty() {
            continue;
        }

        let actual = target.selected.iter().copied().collect::<HashSet<_>>();
        let hits = predicted.intersection(&actual).count();
        let wrong = predicted.len().saturating_sub(hits);

        recall_sum += hits as f64 / actual.len() as f64;
        precision_sum += hits as f64 / predicted.len() as f64;
        false_positive_ratio_sum += if hits == 0 {
            wrong as f64
        } else {
            wrong as f64 / hits as f64
        };
        samples += 1;
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            lookahead_groups,
            source: "router-top",
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "router-top",
        lookahead_groups,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

pub fn analyze_current_router_trace(
    lines: &[PredictorTraceLine],
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }

    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for line in lines {
        let predicted = line
            .top
            .iter()
            .take(top_n)
            .map(|candidate| candidate.expert)
            .collect::<HashSet<_>>();
        if line.selected.is_empty() || predicted.is_empty() {
            continue;
        }

        let actual = line.selected.iter().copied().collect::<HashSet<_>>();
        let hits = predicted.intersection(&actual).count();
        let wrong = predicted.len().saturating_sub(hits);

        recall_sum += hits as f64 / actual.len() as f64;
        precision_sum += hits as f64 / predicted.len() as f64;
        false_positive_ratio_sum += if hits == 0 {
            wrong as f64
        } else {
            wrong as f64 / hits as f64
        };
        samples += 1;
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            source: "router-current",
            lookahead_groups: 0,
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "router-current",
        lookahead_groups: 0,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

pub fn analyze_prev_same_layer_trace(
    lines: &[PredictorTraceLine],
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }

    let mut previous_by_layer = Vec::<Option<Vec<usize>>>::new();
    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for line in lines {
        if previous_by_layer.len() <= line.layer {
            previous_by_layer.resize_with(line.layer + 1, || None);
        }
        if let Some(previous) = &previous_by_layer[line.layer] {
            if !line.selected.is_empty() && !previous.is_empty() {
                let predicted = previous.iter().copied().take(top_n).collect::<HashSet<_>>();
                let actual = line.selected.iter().copied().collect::<HashSet<_>>();
                let hits = predicted.intersection(&actual).count();
                let wrong = predicted.len().saturating_sub(hits);

                recall_sum += hits as f64 / actual.len() as f64;
                precision_sum += hits as f64 / predicted.len() as f64;
                false_positive_ratio_sum += if hits == 0 {
                    wrong as f64
                } else {
                    wrong as f64 / hits as f64
                };
                samples += 1;
            }
        }
        previous_by_layer[line.layer] = Some(line.selected.clone());
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            source: "prev-same-layer",
            lookahead_groups: 0,
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "prev-same-layer",
        lookahead_groups: 0,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

pub fn analyze_union_prev_same_layer_prev_group_trace(
    lines: &[PredictorTraceLine],
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }

    let mut previous_by_layer = Vec::<Option<Vec<usize>>>::new();
    let mut previous_group = None::<Vec<usize>>;
    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for line in lines {
        if previous_by_layer.len() <= line.layer {
            previous_by_layer.resize_with(line.layer + 1, || None);
        }

        let mut predicted_ordered = Vec::<usize>::new();
        if let Some(previous) = &previous_by_layer[line.layer] {
            predicted_ordered.extend(previous.iter().copied());
        }
        if let Some(previous) = &previous_group {
            predicted_ordered.extend(previous.iter().copied());
        }

        let mut predicted = HashSet::<usize>::new();
        for expert in predicted_ordered {
            predicted.insert(expert);
            if predicted.len() >= top_n {
                break;
            }
        }

        if !line.selected.is_empty() && !predicted.is_empty() {
            let actual = line.selected.iter().copied().collect::<HashSet<_>>();
            let hits = predicted.intersection(&actual).count();
            let wrong = predicted.len().saturating_sub(hits);

            recall_sum += hits as f64 / actual.len() as f64;
            precision_sum += hits as f64 / predicted.len() as f64;
            false_positive_ratio_sum += if hits == 0 {
                wrong as f64
            } else {
                wrong as f64 / hits as f64
            };
            samples += 1;
        }

        previous_by_layer[line.layer] = Some(line.selected.clone());
        previous_group = Some(line.selected.clone());
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            source: "prev-layer-union",
            lookahead_groups: 0,
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "prev-layer-union",
        lookahead_groups: 0,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

pub fn analyze_online_layer_hot_trace(
    lines: &[PredictorTraceLine],
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }

    let mut counts_by_layer = Vec::<Vec<u32>>::new();
    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for line in lines {
        if counts_by_layer.len() <= line.layer {
            counts_by_layer.resize_with(line.layer + 1, Vec::new);
        }
        let counts = &mut counts_by_layer[line.layer];
        let max_expert = line
            .top
            .iter()
            .map(|candidate| candidate.expert)
            .chain(line.selected.iter().copied())
            .max()
            .unwrap_or(0);
        if counts.len() <= max_expert {
            counts.resize(max_expert + 1, 0);
        }

        let mut ranked = counts
            .iter()
            .enumerate()
            .filter_map(|(expert, &count)| (count > 0).then_some((expert, count)))
            .collect::<Vec<_>>();
        ranked.sort_by(|(expert_a, count_a), (expert_b, count_b)| {
            count_b.cmp(count_a).then_with(|| expert_a.cmp(expert_b))
        });
        let predicted = ranked
            .iter()
            .take(top_n)
            .map(|(expert, _)| *expert)
            .collect::<HashSet<_>>();

        if !line.selected.is_empty() && !predicted.is_empty() {
            let actual = line.selected.iter().copied().collect::<HashSet<_>>();
            let hits = predicted.intersection(&actual).count();
            let wrong = predicted.len().saturating_sub(hits);

            recall_sum += hits as f64 / actual.len() as f64;
            precision_sum += hits as f64 / predicted.len() as f64;
            false_positive_ratio_sum += if hits == 0 {
                wrong as f64
            } else {
                wrong as f64 / hits as f64
            };
            samples += 1;
        }

        for &expert in &line.selected {
            if counts.len() <= expert {
                counts.resize(expert + 1, 0);
            }
            counts[expert] = counts[expert].saturating_add(1);
        }
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            source: "online-layer-hot",
            lookahead_groups: 0,
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "online-layer-hot",
        lookahead_groups: 0,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

pub fn analyze_combined_scored_trace(
    lines: &[PredictorTraceLine],
    top_n: usize,
) -> Result<PredictorAnalysisRow, String> {
    if top_n == 0 {
        return Err("bad --top-n value 0: expected at least 1".to_string());
    }

    let mut counts_by_layer = Vec::<Vec<u32>>::new();
    let mut previous_by_layer = Vec::<Option<Vec<usize>>>::new();
    let mut previous_group = None::<Vec<usize>>;
    let mut previous_router_top = None::<Vec<PredictorCandidate>>;
    let mut recall_sum = 0.0;
    let mut precision_sum = 0.0;
    let mut false_positive_ratio_sum = 0.0;
    let mut samples = 0usize;

    for line in lines {
        if counts_by_layer.len() <= line.layer {
            counts_by_layer.resize_with(line.layer + 1, Vec::new);
        }
        if previous_by_layer.len() <= line.layer {
            previous_by_layer.resize_with(line.layer + 1, || None);
        }

        let max_expert = line
            .top
            .iter()
            .map(|candidate| candidate.expert)
            .chain(line.selected.iter().copied())
            .max()
            .unwrap_or(0);
        if counts_by_layer[line.layer].len() <= max_expert {
            counts_by_layer[line.layer].resize(max_expert + 1, 0);
        }

        let predicted = combined_scored_prediction(
            previous_by_layer[line.layer].as_deref(),
            previous_group.as_deref(),
            &counts_by_layer[line.layer],
            previous_router_top.as_deref().unwrap_or(&[]),
            top_n,
        );

        if !line.selected.is_empty() && !predicted.is_empty() {
            let actual = line.selected.iter().copied().collect::<HashSet<_>>();
            let predicted_set = predicted.into_iter().collect::<HashSet<_>>();
            let hits = predicted_set.intersection(&actual).count();
            let wrong = predicted_set.len().saturating_sub(hits);

            recall_sum += hits as f64 / actual.len() as f64;
            precision_sum += hits as f64 / predicted_set.len() as f64;
            false_positive_ratio_sum += if hits == 0 {
                wrong as f64
            } else {
                wrong as f64 / hits as f64
            };
            samples += 1;
        }

        for &expert in &line.selected {
            if counts_by_layer[line.layer].len() <= expert {
                counts_by_layer[line.layer].resize(expert + 1, 0);
            }
            counts_by_layer[line.layer][expert] =
                counts_by_layer[line.layer][expert].saturating_add(1);
        }
        previous_by_layer[line.layer] = Some(line.selected.clone());
        previous_group = Some(line.selected.clone());
        previous_router_top = Some(line.top.clone());
    }

    if samples == 0 {
        return Ok(PredictorAnalysisRow {
            source: "combined-scored",
            lookahead_groups: 0,
            top_n,
            samples: 0,
            avg_recall: 0.0,
            avg_precision: 0.0,
            avg_false_positive_ratio: 0.0,
        });
    }

    Ok(PredictorAnalysisRow {
        source: "combined-scored",
        lookahead_groups: 0,
        top_n,
        samples,
        avg_recall: recall_sum / samples as f64,
        avg_precision: precision_sum / samples as f64,
        avg_false_positive_ratio: false_positive_ratio_sum / samples as f64,
    })
}

fn combined_scored_prediction(
    previous_same_layer: Option<&[usize]>,
    previous_group: Option<&[usize]>,
    online_counts: &[u32],
    router_top: &[PredictorCandidate],
    top_n: usize,
) -> Vec<usize> {
    let mut scores = HashMap::<usize, f64>::new();

    if let Some(previous) = previous_same_layer {
        for (rank, &expert) in previous.iter().enumerate() {
            add_score(&mut scores, expert, 1000.0 - rank as f64);
        }
    }
    for (expert, &count) in online_counts.iter().enumerate() {
        if count > 0 {
            add_score(&mut scores, expert, count as f64 * 100.0);
        }
    }
    for candidate in router_top {
        add_score(&mut scores, candidate.expert, candidate.prob as f64 * 50.0);
    }
    if let Some(previous) = previous_group {
        for (rank, &expert) in previous.iter().enumerate() {
            add_score(&mut scores, expert, 10.0 - rank as f64);
        }
    }

    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(expert_a, score_a), (expert_b, score_b)| {
        score_b
            .total_cmp(score_a)
            .then_with(|| expert_a.cmp(expert_b))
    });
    ranked
        .into_iter()
        .take(top_n)
        .map(|(expert, _)| expert)
        .collect()
}

fn add_score(scores: &mut HashMap<usize, f64>, expert: usize, score: f64) {
    scores
        .entry(expert)
        .and_modify(|current| *current += score)
        .or_insert(score);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_predictor_trace_jsonl() {
        let text = r#"{"seq":1,"layer":4,"selected":[2,5],"top":[{"expert":2,"prob":0.7},{"expert":3,"prob":0.2}]}
{"seq":0,"layer":3,"selected":[1,3],"top":[{"expert":1,"prob":0.6}]}"#;

        let lines = parse_predictor_trace_jsonl(text).expect("trace parses");

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].seq, 0);
        assert_eq!(lines[0].layer, 3);
        assert_eq!(lines[1].selected, vec![2, 5]);
        assert_eq!(lines[1].top[0].expert, 2);
    }

    #[test]
    fn analyzes_top_candidates_against_lookahead_target() {
        let lines = vec![
            PredictorTraceLine {
                seq: 0,
                layer: 0,
                selected: vec![9],
                top: vec![
                    PredictorCandidate {
                        expert: 1,
                        prob: 0.5,
                    },
                    PredictorCandidate {
                        expert: 2,
                        prob: 0.3,
                    },
                    PredictorCandidate {
                        expert: 3,
                        prob: 0.2,
                    },
                ],
            },
            PredictorTraceLine {
                seq: 1,
                layer: 1,
                selected: vec![2, 4],
                top: vec![PredictorCandidate {
                    expert: 4,
                    prob: 0.8,
                }],
            },
        ];

        let row = analyze_predictor_trace(&lines, 1, 3).expect("analysis works");

        assert_eq!(row.samples, 1);
        assert_eq!(row.source, "router-top");
        assert_eq!(row.avg_recall, 0.5);
        assert!((row.avg_precision - (1.0 / 3.0)).abs() < 1e-9);
        assert_eq!(row.avg_false_positive_ratio, 2.0);
    }

    #[test]
    fn analyzes_current_router_candidates_against_current_selection() {
        let lines = vec![PredictorTraceLine {
            seq: 0,
            layer: 0,
            selected: vec![2, 4],
            top: vec![
                PredictorCandidate {
                    expert: 2,
                    prob: 0.9,
                },
                PredictorCandidate {
                    expert: 4,
                    prob: 0.8,
                },
                PredictorCandidate {
                    expert: 7,
                    prob: 0.1,
                },
            ],
        }];

        let row = analyze_current_router_trace(&lines, 2).expect("analysis works");

        assert_eq!(row.source, "router-current");
        assert_eq!(row.samples, 1);
        assert_eq!(row.avg_recall, 1.0);
        assert_eq!(row.avg_precision, 1.0);
        assert_eq!(row.avg_false_positive_ratio, 0.0);
    }

    #[test]
    fn analyzes_previous_same_layer_selection() {
        let lines = vec![
            PredictorTraceLine {
                seq: 0,
                layer: 0,
                selected: vec![1, 2],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 1,
                layer: 1,
                selected: vec![9, 8],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 2,
                layer: 0,
                selected: vec![2, 3],
                top: Vec::new(),
            },
        ];

        let row = analyze_prev_same_layer_trace(&lines, 2).expect("analysis works");

        assert_eq!(row.source, "prev-same-layer");
        assert_eq!(row.samples, 1);
        assert_eq!(row.avg_recall, 0.5);
        assert_eq!(row.avg_precision, 0.5);
        assert_eq!(row.avg_false_positive_ratio, 1.0);
    }

    #[test]
    fn analyzes_union_of_previous_same_layer_and_previous_group() {
        let lines = vec![
            PredictorTraceLine {
                seq: 0,
                layer: 0,
                selected: vec![1, 2],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 1,
                layer: 1,
                selected: vec![9, 8],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 2,
                layer: 0,
                selected: vec![2, 9],
                top: Vec::new(),
            },
        ];

        let row =
            analyze_union_prev_same_layer_prev_group_trace(&lines, 4).expect("analysis works");

        assert_eq!(row.source, "prev-layer-union");
        assert_eq!(row.samples, 2);
        assert!((row.avg_recall - 0.5).abs() < 1e-9);
        assert!((row.avg_precision - 0.25).abs() < 1e-9);
    }

    #[test]
    fn analyzes_online_layer_hot_history() {
        let lines = vec![
            PredictorTraceLine {
                seq: 0,
                layer: 0,
                selected: vec![1, 2],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 1,
                layer: 0,
                selected: vec![1, 3],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 2,
                layer: 0,
                selected: vec![1, 4],
                top: Vec::new(),
            },
        ];

        let row = analyze_online_layer_hot_trace(&lines, 2).expect("analysis works");

        assert_eq!(row.source, "online-layer-hot");
        assert_eq!(row.samples, 2);
        assert!((row.avg_recall - 0.5).abs() < 1e-9);
        assert!((row.avg_precision - 0.5).abs() < 1e-9);
    }

    #[test]
    fn analyzes_combined_scored_prediction() {
        let lines = vec![
            PredictorTraceLine {
                seq: 0,
                layer: 0,
                selected: vec![1, 2],
                top: vec![
                    PredictorCandidate {
                        expert: 7,
                        prob: 0.9,
                    },
                    PredictorCandidate {
                        expert: 4,
                        prob: 0.5,
                    },
                ],
            },
            PredictorTraceLine {
                seq: 1,
                layer: 1,
                selected: vec![9, 8],
                top: Vec::new(),
            },
            PredictorTraceLine {
                seq: 2,
                layer: 0,
                selected: vec![2, 7],
                top: Vec::new(),
            },
        ];

        let row = analyze_combined_scored_trace(&lines, 3).expect("analysis works");

        assert_eq!(row.source, "combined-scored");
        assert_eq!(row.samples, 2);
        assert!((row.avg_recall - 0.25).abs() < 1e-9);
        assert!((row.avg_precision - (1.0 / 6.0)).abs() < 1e-9);
    }
}
