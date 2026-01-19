use anyhow::{anyhow, Result};
use flutter_rust_bridge::frb;
use ndarray::Array2;
use ort::value::Tensor;

use crate::api::ort::{build_session_from_file_with_init, OrtInitOptions};
use crate::api::utils::{mean_pooling_ndarray, normalize};

#[frb(opaque)]
pub struct MiniLmEmbedder {
    tokenizer: tokenizers::Tokenizer,
    session: ort::session::Session,
}

#[frb(sync)]
impl MiniLmEmbedder {
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

        let batch = encodings.len();
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

        let mut inputs = ort::inputs! {
            "input_ids" => Tensor::from_array(([batch, max_len], input_ids_batch))?,
            "attention_mask" => Tensor::from_array(([batch, max_len], mask_batch))?,
        };
        if self
            .session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids")
        {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array(([batch, max_len], vec![0i64; batch * max_len]))?.into(),
            ));
        }

        let outputs = self.session.run(inputs)?;
        if let Some(t) = outputs.get("last_hidden_state") {
            let (shape, data) = t.try_extract_tensor::<f32>()?;
            let shape_usize: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            if shape_usize.len() != 3 {
                return Err(anyhow!("Unexpected output shape: {shape_usize:?}"));
            }
            let out_batch = shape_usize[0];
            let seq_len = shape_usize[1];
            let hidden_dim = shape_usize[2];
            if out_batch != batch {
                return Err(anyhow!("Batch size mismatch in outputs"));
            }

            let mut results = Vec::with_capacity(batch);
            for i in 0..batch {
                let start = i * seq_len * hidden_dim;
                let end = start + seq_len * hidden_dim;
                let slice = data
                    .get(start..end)
                    .ok_or(anyhow!("Invalid output slice"))?;
                let embeddings = Array2::from_shape_vec((seq_len, hidden_dim), slice.to_vec())?;
                let mask = fit_mask(&masks_u32[i], seq_len);
                let pooled = mean_pooling_ndarray(&embeddings, &mask);
                results.push(normalize(&pooled));
            }
            return Ok(results);
        }

        let (shape, data) = pick_embedding_tensor(&outputs)?;
        if shape.len() != 2 {
            return Err(anyhow!("Unexpected output shape: {shape:?}"));
        }
        let out_batch = shape[0];
        let hidden = shape[1];
        if out_batch != batch {
            return Err(anyhow!("Batch size mismatch in outputs"));
        }
        let mut results = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * hidden;
            let end = start + hidden;
            let slice = data
                .get(start..end)
                .ok_or(anyhow!("Invalid output slice"))?;
            results.push(normalize(slice));
        }
        Ok(results)
    }

    pub fn format_query(query: String) -> String {
        query
    }

    pub fn format_document(text: String) -> String {
        text
    }
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

fn pick_embedding_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<(Vec<usize>, Vec<f32>)> {
    for key in ["sentence_embedding", "embedding", "pooled_output", "pooler_output"] {
        if let Some(t) = outputs.get(key) {
            let (shape, data) = t.try_extract_tensor::<f32>()?;
            let shape_usize = shape.iter().map(|d| *d as usize).collect();
            return Ok((shape_usize, data.to_vec()));
        }
    }
    if let Some(t) = outputs.get("last_hidden_state") {
        let (shape, data) = t.try_extract_tensor::<f32>()?;
        let shape_usize = shape.iter().map(|d| *d as usize).collect();
        Ok((shape_usize, data.to_vec()))
    } else {
        Err(anyhow!("No embedding tensor found in outputs"))
    }
}
