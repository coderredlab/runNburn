use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::compute::MetalContext;
use crate::ffn_chain::{empty_f16_buf, empty_f32_buf};

pub(crate) struct DflashAttentionCarrier {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    query: Retained<ProtocolObject<dyn MTLBuffer>>,
    context_key: Retained<ProtocolObject<dyn MTLBuffer>>,
    context_value: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_key: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_value: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl DflashAttentionCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            query: empty_f32_buf(ctx, seq_len * q_dim),
            context_key: empty_f16_buf(ctx, sliding_window * kv_dim),
            context_value: empty_f16_buf(ctx, sliding_window * kv_dim),
            block_key: empty_f32_buf(ctx, seq_len * kv_dim),
            block_value: empty_f32_buf(ctx, seq_len * kv_dim),
            output: empty_f32_buf(ctx, seq_len * q_dim),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload_context(
        &self,
        context_key: &[u16],
        context_value: &[u16],
        position: usize,
    ) -> Result<usize, String> {
        let kv_dim = self.num_kv_heads * self.head_dim;
        if context_key.len() != context_value.len() || context_key.len() % kv_dim != 0 {
            return Err("Metal DFlash attention context shape mismatch".to_string());
        }
        let context_len = context_key.len() / kv_dim;
        if context_len > self.sliding_window || position < context_len {
            return Err(format!(
                "Metal DFlash context contract mismatch: position={position}, context_len={context_len}, window={}",
                self.sliding_window
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                context_key.as_ptr(),
                self.context_key.contents().as_ptr().cast::<u16>(),
                context_key.len(),
            );
            std::ptr::copy_nonoverlapping(
                context_value.as_ptr(),
                self.context_value.contents().as_ptr().cast::<u16>(),
                context_value.len(),
            );
        }
        Ok(context_len)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_from_buffers(
        &self,
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        query: &ProtocolObject<dyn MTLBuffer>,
        block_key: &ProtocolObject<dyn MTLBuffer>,
        block_value: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        context_len: usize,
        position: usize,
    ) -> Result<(), String> {
        encoder.setComputePipelineState(&ctx.dflash_attention_hd128_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(query), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.context_key), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.context_value), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(block_key), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(block_value), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 5);
        }
        set_u32(encoder, context_len, 6)?;
        set_u32(encoder, self.seq_len, 7)?;
        set_u32(encoder, position, 8)?;
        set_u32(encoder, self.num_heads, 9)?;
        set_u32(encoder, self.num_kv_heads, 10)?;
        set_u32(encoder, self.sliding_window, 11)?;
        set_f32(encoder, (self.head_dim as f32).sqrt().recip(), 12);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.seq_len,
                height: self.num_heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &self,
        ctx: &MetalContext,
        query: &[f32],
        context_key: &[u16],
        context_value: &[u16],
        block_key: &[f32],
        block_value: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        if self.head_dim != 128
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.num_heads % self.num_kv_heads != 0
            || query.len() != self.seq_len * q_dim
            || block_key.len() != self.seq_len * kv_dim
            || block_value.len() != self.seq_len * kv_dim
        {
            return Err("Metal DFlash attention shape mismatch".to_string());
        }
        let context_len = self.upload_context(context_key, context_value, position)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                self.query.contents().as_ptr().cast::<f32>(),
                query.len(),
            );
            std::ptr::copy_nonoverlapping(
                block_key.as_ptr(),
                self.block_key.contents().as_ptr().cast::<f32>(),
                block_key.len(),
            );
            std::ptr::copy_nonoverlapping(
                block_value.as_ptr(),
                self.block_value.contents().as_ptr().cast::<f32>(),
                block_value.len(),
            );
        }

        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| "Metal DFlash command buffer unavailable".to_string())?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| "Metal DFlash compute encoder unavailable".to_string())?;
        self.encode_from_buffers(
            ctx,
            &encoder,
            &self.query,
            &self.block_key,
            &self.block_value,
            &self.output,
            context_len,
            position,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            let error = command
                .error()
                .map(|value| value.localizedDescription().to_string())
                .unwrap_or_else(|| "no NSError attached".to_string());
            return Err(format!(
                "Metal DFlash attention failed status={:?}: {error}",
                command.status()
            ));
        }

        Ok(unsafe {
            std::slice::from_raw_parts(self.output.contents().as_ptr().cast::<f32>(), query.len())
                .to_vec()
        })
    }
}

pub(crate) struct MuseTargetAttentionCarrier {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    query: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_muse_target_attention_f16_hd128(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    query: &ProtocolObject<dyn MTLBuffer>,
    key: &ProtocolObject<dyn MTLBuffer>,
    value: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    sliding_window: Option<usize>,
    scale: f32,
) -> Result<(), String> {
    if seq_len == 0
        || kv_len < seq_len
        || num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || !scale.is_finite()
    {
        return Err("Metal Muse target attention cache contract mismatch".to_string());
    }
    encoder.setComputePipelineState(&ctx.muse_target_attention_f16_hd128_pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(query), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(key), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(value), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
    }
    set_u32(encoder, kv_len, 4)?;
    set_u32(encoder, seq_len, 5)?;
    set_u32(encoder, num_heads, 6)?;
    set_u32(encoder, num_kv_heads, 7)?;
    set_u32(encoder, sliding_window.unwrap_or(0), 8)?;
    set_f32(encoder, scale, 9);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: seq_len * num_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

impl MuseTargetAttentionCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let elements = seq_len * num_heads * 128;
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            query: empty_f32_buf(ctx, elements),
            output: empty_f32_buf(ctx, elements),
        }
    }

    pub(crate) fn upload_query(&self, query: &[f32]) -> Result<(), String> {
        if self.seq_len == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.num_heads % self.num_kv_heads != 0
            || query.len() != self.seq_len * self.num_heads * 128
        {
            return Err("Metal Muse target attention query contract mismatch".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                self.query.contents().as_ptr().cast::<f32>(),
                query.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn output_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.output
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        key: &ProtocolObject<dyn MTLBuffer>,
        value: &ProtocolObject<dyn MTLBuffer>,
        kv_len: usize,
        sliding_window: Option<usize>,
        scale: f32,
    ) -> Result<(), String> {
        encode_muse_target_attention_f16_hd128(
            ctx,
            encoder,
            &self.query,
            key,
            value,
            &self.output,
            self.seq_len,
            kv_len,
            self.num_heads,
            self.num_kv_heads,
            sliding_window,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &self,
        ctx: &MetalContext,
        query: &[f32],
        key: &ProtocolObject<dyn MTLBuffer>,
        value: &ProtocolObject<dyn MTLBuffer>,
        kv_len: usize,
        sliding_window: Option<usize>,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.upload_query(query)?;
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| "Metal Muse target command buffer unavailable".to_string())?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| "Metal Muse target compute encoder unavailable".to_string())?;
        self.encode(ctx, &encoder, key, value, kv_len, sliding_window, scale)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            let error = command
                .error()
                .map(|value| value.localizedDescription().to_string())
                .unwrap_or_else(|| "no NSError attached".to_string());
            return Err(format!(
                "Metal Muse target attention failed status={:?}: {error}",
                command.status()
            ));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(self.output.contents().as_ptr().cast::<f32>(), query.len())
                .to_vec()
        })
    }
}

fn set_u32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: usize,
    index: usize,
) -> Result<(), String> {
    let value =
        u32::try_from(value).map_err(|_| format!("Metal DFlash argument {value} exceeds u32"))?;
    unsafe {
        let pointer = NonNull::new(&value as *const u32 as *mut std::ffi::c_void)
            .expect("u32 argument pointer");
        encoder.setBytes_length_atIndex(pointer, std::mem::size_of::<u32>(), index);
    }
    Ok(())
}

fn set_f32(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: f32, index: usize) {
    unsafe {
        let pointer = NonNull::new(&value as *const f32 as *mut std::ffi::c_void)
            .expect("f32 argument pointer");
        encoder.setBytes_length_atIndex(pointer, std::mem::size_of::<f32>(), index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn metal_matches_noncausal_sliding_window_oracle() {
        let _guard = crate::METAL_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ctx =
            crate::compute::build_metal_context_with_kv_int8(false).expect("Metal test device");
        let seq_len = 3;
        let context_len = 5;
        let position = 5;
        let num_heads = 4;

        let num_kv_heads = 2;
        let head_dim = 128;
        let sliding_window = 6;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let values = |len: usize, mul: usize, offset: usize| {
            (0..len)
                .map(|index| (((index * mul + offset) % 97) as f32 - 48.0) / 64.0)
                .collect::<Vec<_>>()
        };
        let query = values(seq_len * q_dim, 17, 3);
        let context_key_f32 = values(context_len * kv_dim, 13, 5);
        let context_value_f32 = values(context_len * kv_dim, 19, 7);
        let context_key: Vec<u16> = context_key_f32
            .iter()
            .map(|value| f16::from_f32(*value).to_bits())
            .collect();
        let context_value: Vec<u16> = context_value_f32
            .iter()
            .map(|value| f16::from_f32(*value).to_bits())
            .collect();
        let block_key = values(seq_len * kv_dim, 23, 11);
        let block_value = values(seq_len * kv_dim, 29, 13);
        let carrier = DflashAttentionCarrier::new(
            &ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
        );
        let actual = carrier
            .dispatch(
                &ctx,
                &query,
                &context_key,
                &context_value,
                &block_key,
                &block_value,
                position,
            )
            .expect("Metal DFlash dispatch");

        let scale = (head_dim as f32).sqrt().recip();
        let heads_per_kv = num_heads / num_kv_heads;
        let context_start = position - context_len;
        let mut expected = vec![0.0; actual.len()];
        for query_index in 0..seq_len {
            let first_position = (position + query_index + 1).saturating_sub(sliding_window);
            for head in 0..num_heads {
                let kv_head = head / heads_per_kv;
                let output_base = (query_index * num_heads + head) * head_dim;
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0;
                for key_index in 0..context_len + seq_len {
                    if context_start + key_index < first_position {
                        continue;
                    }
                    let committed = key_index < context_len;
                    let row = if committed {
                        key_index
                    } else {
                        key_index - context_len
                    };
                    let key_base = (row * num_kv_heads + kv_head) * head_dim;
                    let mut dot = 0.0;
                    for dim in 0..head_dim {
                        let key = if committed {
                            f16::from_bits(context_key[key_base + dim]).to_f32()
                        } else {
                            block_key[key_base + dim]
                        };
                        dot += query[output_base + dim] * key;
                    }
                    let score = dot * scale;
                    let next_maximum = maximum.max(score);
                    let old_scale = if maximum.is_finite() {
                        (maximum - next_maximum).exp()
                    } else {
                        0.0
                    };
                    let probability = (score - next_maximum).exp();
                    denominator = denominator * old_scale + probability;
                    for dim in 0..head_dim {
                        let value = if committed {
                            f16::from_bits(context_value[key_base + dim]).to_f32()
                        } else {
                            block_value[key_base + dim]
                        };
                        expected[output_base + dim] =
                            expected[output_base + dim] * old_scale + value * probability;
                    }
                    maximum = next_maximum;
                }
                for dim in 0..head_dim {
                    expected[output_base + dim] /= denominator;
                }
            }
        }

        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 2.0e-5, "max_abs={max_abs:e}");
    }

    #[test]
    fn metal_target_matches_cpu_f16_attention_contract() {
        let _guard = crate::METAL_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ctx =
            crate::compute::build_metal_context_with_kv_int8(false).expect("Metal test device");
        let seq_len = 3;
        let kv_len = 67;
        let num_heads = 4;
        let num_kv_heads = 2;
        let kv_dim = num_kv_heads * 128;
        let values = |len: usize, mul: usize, offset: usize| {
            (0..len)
                .map(|index| (((index * mul + offset) % 101) as f32 - 50.0) / 96.0)
                .collect::<Vec<_>>()
        };
        let query = values(seq_len * num_heads * 128, 17, 3);
        let key: Vec<u16> = values(kv_len * kv_dim, 13, 5)
            .into_iter()
            .map(|value| f16::from_f32(value).to_bits())
            .collect();
        let value: Vec<u16> = values(kv_len * kv_dim, 19, 7)
            .into_iter()
            .map(|value| f16::from_f32(value).to_bits())
            .collect();
        let key_buf = empty_f16_buf(&ctx, key.len());
        let value_buf = empty_f16_buf(&ctx, value.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                key.as_ptr(),
                key_buf.contents().as_ptr().cast::<u16>(),
                key.len(),
            );
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                value_buf.contents().as_ptr().cast::<u16>(),
                value.len(),
            );
        }
        let carrier = MuseTargetAttentionCarrier::new(&ctx, seq_len, num_heads, num_kv_heads);
        let actual = carrier
            .dispatch(
                &ctx,
                &query,
                &key_buf,
                &value_buf,
                kv_len,
                Some(48),
                (128.0f32).sqrt().recip(),
            )
            .expect("Metal Muse target attention");
        let expected = rnb_cpu::kernels::attention::attention_batch_f16(
            &query,
            &key,
            &value,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            128,
            (128.0f32).sqrt().recip(),
            Some(48),
            None,
        );
        let (max_index, max_abs) = actual
            .iter()
            .zip(expected.iter())
            .enumerate()
            .map(|(index, (left, right))| (index, (left - right).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("non-empty attention output");
        assert!(
            max_abs <= 2.0e-5,
            "max_abs={max_abs:e} index={max_index} actual={} expected={}",
            actual[max_index],
            expected[max_index]
        );
    }
}
