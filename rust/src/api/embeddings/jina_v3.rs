use anyhow::Result;
use flutter_rust_bridge::frb;
use ndarray::Array2;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::api::utils::{mean_pooling_ndarray, normalize};

#[frb(opaque)]
pub struct JinaV3Embedder {
    tokenizer: tokenizers::Tokenizer,
    session: ort::session::Session,
}

#[frb(sync)]
impl JinaV3Embedder {
    pub fn create(model_path: String, tokenizer_path: String) -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&mut self, texts: Vec<String>, task_id: i64) -> Result<Vec<Vec<f32>>> {
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

        let inputs = ort::inputs! {
            "input_ids" => Tensor::from_array(([batch, max_len], input_ids_batch))?,
            "attention_mask" => Tensor::from_array(([batch, max_len], mask_batch))?,
            "task_id" => Tensor::from_array(([batch], vec![task_id; batch]))?,
        };
        let outputs = self.session.run(inputs)?;
        let (extracted_shape, extracted_data) = outputs
            .get("last_hidden_state")
            .ok_or(anyhow::anyhow!("Missing last_hidden_state"))?
            .try_extract_tensor::<f32>()?;

        let out_batch = extracted_shape[0] as usize;
        let seq_len = extracted_shape[1] as usize;
        let hidden_dim = extracted_shape[2] as usize;
        if out_batch != batch {
            return Err(anyhow::anyhow!("Batch size mismatch in outputs"));
        }

        let mut results = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * seq_len * hidden_dim;
            let end = start + seq_len * hidden_dim;
            let slice = extracted_data
                .get(start..end)
                .ok_or(anyhow::anyhow!("Invalid output slice"))?;
            let embeddings = Array2::from_shape_vec((seq_len, hidden_dim), slice.to_vec())?;
            let mask = fit_mask(&masks_u32[i], seq_len);
            let pooled = mean_pooling_ndarray(&embeddings, &mask);
            results.push(normalize(&pooled));
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
