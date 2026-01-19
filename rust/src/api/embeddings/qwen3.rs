use anyhow::{anyhow, Result};
use flutter_rust_bridge::frb;
use ndarray::{ArrayD, IxDyn};
use ort::{
    tensor::TensorElementType,
    value::{DynTensor, Tensor, ValueType},
};

use crate::api::ort::{build_session_from_file_with_init, OrtInitOptions};
use crate::api::utils::normalize;

const QWEN3_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

#[frb(opaque)]
pub struct Qwen3Embedder {
    tokenizer: tokenizers::Tokenizer,
    session: ort::session::Session,
}

#[frb(sync)]
impl Qwen3Embedder {
    pub fn create(model_path: String, tokenizer_path: String) -> Result<Self> {
        Self::create_with_options(model_path, tokenizer_path, None)
    }

    pub fn create_with_options(
        model_path: String,
        tokenizer_path: String,
        ort_options: Option<OrtInitOptions>,
    ) -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
        let session = build_session_from_file_with_init(model_path, ort_options)?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!(e))?;

        let pad_id = self
            .tokenizer
            .get_padding()
            .map(|p| p.pad_id as i64)
            .unwrap_or(0);

        let mut batch = encodings.len();
        for input in self.session.inputs() {
            if input.name() == "input_ids" {
                if let ValueType::Tensor { shape, .. } = input.dtype() {
                    if let Some(dim) = shape.first() {
                        if *dim > 0 && *dim as usize != batch {
                            return Err(anyhow::anyhow!("Batch size mismatch for input_ids"));
                        }
                        if *dim > 0 {
                            batch = *dim as usize;
                        }
                    }
                }
                break;
            }
        }

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            return Ok(vec![Vec::new(); batch]);
        }

        let mut input_ids_batch = Vec::with_capacity(batch * max_len);
        let mut mask_batch = Vec::with_capacity(batch * max_len);
        let mut masks_u32 = Vec::with_capacity(batch);

        for encoding in encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let pad_len = max_len.saturating_sub(ids.len());

            let mut ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let mut mask_i64: Vec<i64> = mask.iter().map(|&x| x as i64).collect();
            let mut mask_u32: Vec<u32> = mask.to_vec();

            ids_i64.extend(std::iter::repeat(pad_id).take(pad_len));
            mask_i64.extend(std::iter::repeat(0).take(pad_len));
            mask_u32.extend(std::iter::repeat(0).take(pad_len));

            input_ids_batch.extend_from_slice(&ids_i64);
            mask_batch.extend_from_slice(&mask_i64);
            masks_u32.push(mask_u32);
        }

        let position_ids: Vec<i64> = (0..max_len as i64).collect();
        let position_batch = repeat_i64(&position_ids, batch);

        let mut inputs: Vec<(String, DynTensor)> = Vec::new();
        for input in self.session.inputs() {
            let name = input.name();
            match name {
                "input_ids" => {
                    let shape = resolve_shape_with_fallback(input.dtype(), &[batch, max_len])?;
                    let tensor = tensor_from_i64(input.dtype(), &shape, &input_ids_batch)?;
                    inputs.push((name.to_string(), tensor));
                }
                "attention_mask" => {
                    let rank = if let ValueType::Tensor { shape, .. } = input.dtype() {
                        shape.len()
                    } else {
                        2
                    };
                    let (shape, data) = if rank == 1 {
                        if batch > 1 {
                            return Err(anyhow::anyhow!(
                                "attention_mask rank 1 is not batch-compatible"
                            ));
                        }
                        (
                            resolve_shape_with_fallback(input.dtype(), &[max_len])?,
                            mask_batch[..max_len].to_vec(),
                        )
                    } else if rank == 4 {
                        let shape = resolve_shape_with_fallback(
                            input.dtype(),
                            &[batch, 1, max_len, max_len],
                        )?;
                        let mut data = Vec::with_capacity(batch * max_len * max_len);
                        for mask in &masks_u32 {
                            let mask_i64: Vec<i64> =
                                mask.iter().map(|&v| v as i64).collect();
                            for _ in 0..max_len {
                                data.extend_from_slice(&mask_i64);
                            }
                        }
                        (shape, data)
                    } else {
                        (
                            resolve_shape_with_fallback(input.dtype(), &[batch, max_len])?,
                            mask_batch.clone(),
                        )
                    };
                    let tensor = tensor_from_i64(input.dtype(), &shape, &data)?;
                    inputs.push((name.to_string(), tensor));
                }
                "position_ids" | "cache_position" => {
                    let rank = if let ValueType::Tensor { shape, .. } = input.dtype() {
                        shape.len()
                    } else {
                        2
                    };
                    let (shape, data) = if rank == 1 {
                        if batch > 1 {
                            return Err(anyhow::anyhow!(
                                "position_ids rank 1 is not batch-compatible"
                            ));
                        }
                        (
                            resolve_shape_with_fallback(input.dtype(), &[max_len])?,
                            position_ids.clone(),
                        )
                    } else {
                        (
                            resolve_shape_with_fallback(input.dtype(), &[batch, max_len])?,
                            position_batch.clone(),
                        )
                    };
                    let tensor = tensor_from_i64(input.dtype(), &shape, &data)?;
                    inputs.push((name.to_string(), tensor));
                }
                "token_type_ids" => {
                    let shape = resolve_shape_with_fallback(input.dtype(), &[batch, max_len])?;
                    let tensor = zeros_tensor(input.dtype(), &shape)?;
                    inputs.push((name.to_string(), tensor));
                }
                _ if name.starts_with("past_key_values") => {
                    let shape = resolve_past_kv_shape(input.dtype(), batch)?;
                    let tensor = zeros_tensor(input.dtype(), &shape)?;
                    inputs.push((name.to_string(), tensor));
                }
                _ => {
                    let rank = if let ValueType::Tensor { shape, .. } = input.dtype() {
                        shape.len()
                    } else {
                        1
                    };
                    let fallback = match rank {
                        1 => vec![max_len],
                        2 => vec![batch, max_len],
                        _ => vec![1; rank],
                    };
                    let shape = resolve_shape_with_fallback(input.dtype(), &fallback)?;
                    let tensor = zeros_tensor(input.dtype(), &shape)?;
                    inputs.push((name.to_string(), tensor));
                }
            }
        }
        let outputs = self.session.run(inputs)?;
        let (shape, data) = pick_embedding_tensor(&outputs)?;
        if shape.len() == 2 {
            let out_batch = shape[0];
            let hidden_dim = shape[1];
            if out_batch != batch {
                return Err(anyhow::anyhow!("Batch size mismatch in outputs"));
            }
            let mut results = Vec::with_capacity(batch);
            for i in 0..batch {
                let start = i * hidden_dim;
                let end = start + hidden_dim;
                let slice = data
                    .get(start..end)
                    .ok_or(anyhow::anyhow!("Invalid output slice"))?;
                results.push(normalize(slice));
            }
            return Ok(results);
        }
        // Otherwise use last token (Qwen uses last_token pooling).
        let out_batch = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];
        if out_batch != batch {
            return Err(anyhow::anyhow!("Batch size mismatch in outputs"));
        }
        let mut results = Vec::with_capacity(batch);
        for i in 0..batch {
            let mask = fit_mask(&masks_u32[i], seq_len);
            let last_index = mask
                .iter()
                .rposition(|&m| m == 1)
                .unwrap_or(seq_len.saturating_sub(1));
            let start = (i * seq_len + last_index) * hidden_dim;
            let end = start + hidden_dim;
            let slice = data
                .get(start..end)
                .ok_or(anyhow::anyhow!("Invalid last token slice"))?;
            results.push(normalize(slice));
        }

        Ok(results)
    }

    pub fn format_query(query: String) -> String {
        format!("Instruct: {}\nQuery:{}", QWEN3_TASK, query)
    }

    pub fn format_document(text: String) -> String {
        text
    }
}

fn repeat_i64(data: &[i64], times: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() * times);
    for _ in 0..times {
        out.extend_from_slice(data);
    }
    out
}

fn resolve_shape_with_fallback(dtype: &ValueType, fallback: &[usize]) -> Result<Vec<usize>> {
    let ValueType::Tensor { shape, .. } = dtype else {
        return Err(anyhow!("Unsupported input type: {dtype:?}"));
    };
    if shape.len() != fallback.len() {
        return Ok(fallback.to_vec());
    }
    Ok(shape
        .iter()
        .zip(fallback.iter())
        .map(|(dim, fb)| if *dim >= 0 { *dim as usize } else { *fb })
        .collect())
}

fn resolve_past_kv_shape(dtype: &ValueType, batch: usize) -> Result<Vec<usize>> {
    let ValueType::Tensor { shape, .. } = dtype else {
        return Err(anyhow!("Unsupported input type: {dtype:?}"));
    };
    let rank = shape.len();
    let mut resolved = Vec::with_capacity(rank);
    for (idx, dim) in shape.iter().enumerate() {
        if *dim >= 0 {
            resolved.push(*dim as usize);
            continue;
        }
        let value = if idx == 0 {
            batch
        } else if idx == rank.saturating_sub(2) {
            0
        } else {
            1
        };
        resolved.push(value);
    }
    Ok(resolved)
}

fn fit_mask(mask: &[u32], target_len: usize) -> Vec<u32> {
    if mask.len() == target_len {
        return mask.to_vec();
    }
    if mask.len() > target_len {
        return mask[..target_len].to_vec();
    }
    let mut out = mask.to_vec();
    out.extend(std::iter::repeat(0).take(target_len - mask.len()));
    out
}

fn tensor_from_i64(dtype: &ValueType, shape: &[usize], data: &[i64]) -> Result<DynTensor> {
    let ValueType::Tensor { ty, .. } = dtype else {
        return Err(anyhow!("Unsupported input type: {dtype:?}"));
    };
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(anyhow!(
            "Input data length mismatch: expected {expected}, got {}",
            data.len()
        ));
    }
    match ty {
        TensorElementType::Int64 => {
            Ok(Tensor::from_array((shape.to_vec(), data.to_vec()))?.upcast())
        }
        TensorElementType::Int32 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as i32).collect::<Vec<i32>>(),
        ))?
        .upcast()),
        TensorElementType::Int16 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as i16).collect::<Vec<i16>>(),
        ))?
        .upcast()),
        TensorElementType::Int8 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as i8).collect::<Vec<i8>>(),
        ))?
        .upcast()),
        TensorElementType::Uint64 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
        ))?
        .upcast()),
        TensorElementType::Uint32 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as u32).collect::<Vec<u32>>(),
        ))?
        .upcast()),
        TensorElementType::Uint16 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
        ))?
        .upcast()),
        TensorElementType::Uint8 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
        ))?
        .upcast()),
        TensorElementType::Bool => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v != 0).collect::<Vec<bool>>(),
        ))?
        .upcast()),
        TensorElementType::Float32 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
        ))?
        .upcast()),
        TensorElementType::Float64 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        ))?
        .upcast()),
        TensorElementType::Float16 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter()
                .map(|&v| half::f16::from_f32(v as f32))
                .collect::<Vec<half::f16>>(),
        ))?
        .upcast()),
        TensorElementType::Bfloat16 => Ok(Tensor::from_array((
            shape.to_vec(),
            data.iter()
                .map(|&v| half::bf16::from_f32(v as f32))
                .collect::<Vec<half::bf16>>(),
        ))?
        .upcast()),
        _ => Err(anyhow!("Unsupported tensor element type: {ty:?}")),
    }
}

fn zeros_tensor(dtype: &ValueType, shape: &[usize]) -> Result<DynTensor> {
    let ValueType::Tensor { ty, .. } = dtype else {
        return Err(anyhow!("Unsupported input type: {dtype:?}"));
    };
    match ty {
        TensorElementType::Float32 => {
            Ok(Tensor::from_array(ArrayD::<f32>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Float64 => {
            Ok(Tensor::from_array(ArrayD::<f64>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Float16 => {
            Ok(Tensor::from_array(ArrayD::<half::f16>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Bfloat16 => {
            Ok(Tensor::from_array(ArrayD::<half::bf16>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Int64 => {
            Ok(Tensor::from_array(ArrayD::<i64>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Int32 => {
            Ok(Tensor::from_array(ArrayD::<i32>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Int16 => {
            Ok(Tensor::from_array(ArrayD::<i16>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Int8 => {
            Ok(Tensor::from_array(ArrayD::<i8>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Uint64 => {
            Ok(Tensor::from_array(ArrayD::<u64>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Uint32 => {
            Ok(Tensor::from_array(ArrayD::<u32>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Uint16 => {
            Ok(Tensor::from_array(ArrayD::<u16>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Uint8 => {
            Ok(Tensor::from_array(ArrayD::<u8>::zeros(IxDyn(shape)))?.upcast())
        }
        TensorElementType::Bool => {
            Ok(Tensor::from_array(ArrayD::<bool>::from_elem(IxDyn(shape), false))?.upcast())
        }
        _ => Err(anyhow!("Unsupported tensor element type: {ty:?}")),
    }
}
// Shared helper to select output tensor key for embedding models.
fn pick_embedding_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<(Vec<usize>, Vec<f32>)> {
    // Prefer common pooled outputs if present.
    for key in ["sentence_embedding", "pooled_output", "embedding"] {
        if let Some(t) = outputs.get(key) {
            let (shape, data) = t.try_extract_tensor::<f32>()?;
            let shape_usize = shape.iter().map(|d| *d as usize).collect();
            return Ok((shape_usize, data.to_vec()));
        }
    }
    // Fallback to last_hidden_state; caller must pool.
    if let Some(t) = outputs.get("last_hidden_state") {
        let (shape, data) = t.try_extract_tensor::<f32>()?;
        let shape_usize = shape.iter().map(|d| *d as usize).collect();
        Ok((shape_usize, data.to_vec()))
    } else {
        Err(anyhow!("No embedding tensor found in outputs"))
    }
}
