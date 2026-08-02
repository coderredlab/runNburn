use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36PositionSpan {
    Text {
        rows: usize,
    },
    Image {
        grid_width: usize,
        grid_height: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen36PositionPlan {
    pub positions: Vec<[u32; 4]>,
    pub physical_rows: usize,
    pub logical_position_end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen36PositionError(String);

impl Qwen36PositionError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Qwen36PositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Qwen36PositionError {}

pub fn plan_qwen36_multimodal_positions(
    spans: &[Qwen36PositionSpan],
    logical_position_start: u32,
) -> Result<Qwen36PositionPlan, Qwen36PositionError> {
    let mut physical_rows = 0usize;
    let mut image_count = 0usize;
    for span in spans {
        let rows = match *span {
            Qwen36PositionSpan::Text { rows } => rows,
            Qwen36PositionSpan::Image {
                grid_width,
                grid_height,
            } => {
                image_count = image_count.checked_add(1).ok_or_else(|| {
                    Qwen36PositionError::new("Qwen3.6 image span count overflows usize")
                })?;
                if grid_width == 0 || grid_height == 0 {
                    return Err(Qwen36PositionError::new(
                        "Qwen3.6 image grid dimensions must be positive",
                    ));
                }
                grid_width.checked_mul(grid_height).ok_or_else(|| {
                    Qwen36PositionError::new("Qwen3.6 image physical row count overflows usize")
                })?
            }
        };
        physical_rows = physical_rows.checked_add(rows).ok_or_else(|| {
            Qwen36PositionError::new("Qwen3.6 prompt physical row count overflows usize")
        })?;
    }
    if image_count != 1 {
        return Err(Qwen36PositionError::new(format!(
            "Qwen3.6 multimodal prompts require exactly one image span, got {image_count}"
        )));
    }

    let mut positions = Vec::with_capacity(physical_rows);
    let mut logical_position = logical_position_start;
    for span in spans {
        match *span {
            Qwen36PositionSpan::Text { rows } => {
                for _ in 0..rows {
                    positions.push([logical_position; 4]);
                    logical_position = logical_position.checked_add(1).ok_or_else(|| {
                        Qwen36PositionError::new("Qwen3.6 text logical position overflows u32")
                    })?;
                }
            }
            Qwen36PositionSpan::Image {
                grid_width,
                grid_height,
            } => {
                for y in 0..grid_height {
                    let y = u32::try_from(y).map_err(|_| {
                        Qwen36PositionError::new("Qwen3.6 image grid height exceeds u32")
                    })?;
                    let vertical = logical_position.checked_add(y).ok_or_else(|| {
                        Qwen36PositionError::new("Qwen3.6 image vertical position overflows u32")
                    })?;
                    for x in 0..grid_width {
                        let x = u32::try_from(x).map_err(|_| {
                            Qwen36PositionError::new("Qwen3.6 image grid width exceeds u32")
                        })?;
                        let horizontal = logical_position.checked_add(x).ok_or_else(|| {
                            Qwen36PositionError::new(
                                "Qwen3.6 image horizontal position overflows u32",
                            )
                        })?;
                        positions.push([logical_position, vertical, horizontal, 0]);
                    }
                }
                let advance = u32::try_from(grid_width.max(grid_height)).map_err(|_| {
                    Qwen36PositionError::new("Qwen3.6 image logical advance exceeds u32")
                })?;
                logical_position = logical_position.checked_add(advance).ok_or_else(|| {
                    Qwen36PositionError::new("Qwen3.6 image logical position overflows u32")
                })?;
            }
        }
    }

    debug_assert_eq!(positions.len(), physical_rows);
    Ok(Qwen36PositionPlan {
        positions,
        physical_rows,
        logical_position_end: logical_position,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_image_text_positions_separate_physical_and_logical_lengths() {
        let plan = plan_qwen36_multimodal_positions(
            &[
                Qwen36PositionSpan::Text { rows: 2 },
                Qwen36PositionSpan::Image {
                    grid_width: 3,
                    grid_height: 2,
                },
                Qwen36PositionSpan::Text { rows: 1 },
            ],
            0,
        )
        .unwrap();

        assert_eq!(plan.physical_rows, 9);
        assert_eq!(plan.logical_position_end, 6);
        assert_eq!(
            plan.positions,
            vec![
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 0],
                [2, 2, 3, 0],
                [2, 2, 4, 0],
                [2, 3, 2, 0],
                [2, 3, 3, 0],
                [2, 3, 4, 0],
                [5, 5, 5, 5],
            ]
        );
    }

    #[test]
    fn square_24_grid_executes_576_rows_and_advances_24() {
        let plan = plan_qwen36_multimodal_positions(
            &[Qwen36PositionSpan::Image {
                grid_width: 24,
                grid_height: 24,
            }],
            7,
        )
        .unwrap();

        assert_eq!(plan.physical_rows, 576);
        assert_eq!(plan.logical_position_end, 31);
        assert_eq!(plan.positions[0], [7, 7, 7, 0]);
        assert_eq!(plan.positions[575], [7, 30, 30, 0]);
    }

    #[test]
    fn rejects_missing_or_multiple_images() {
        let missing = plan_qwen36_multimodal_positions(&[Qwen36PositionSpan::Text { rows: 1 }], 0)
            .unwrap_err();
        assert!(missing.to_string().contains("exactly one image span"));

        let multiple = plan_qwen36_multimodal_positions(
            &[
                Qwen36PositionSpan::Image {
                    grid_width: 1,
                    grid_height: 1,
                },
                Qwen36PositionSpan::Image {
                    grid_width: 1,
                    grid_height: 1,
                },
            ],
            0,
        )
        .unwrap_err();
        assert!(multiple.to_string().contains("got 2"));
    }
}
