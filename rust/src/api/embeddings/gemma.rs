use anyhow::Result;
use flutter_rust_bridge::frb;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

pub const PREFIX_QUERY: &str = "task: search result | query: ";
pub const PREFIX_DOCUMENT: &str = "title: none | text: ";
const HIDDEN_DIM: usize = 768;

#[frb(opaque)]
pub struct GemmaEmbedder {
    tokenizer: tokenizers::Tokenizer,
    session: ort::session::Session,
}

#[frb(sync)]
impl GemmaEmbedder {
    pub fn create(model_path: String, tokenizer_path: String) -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

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

        for encoding in encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let pad_len = max_len.saturating_sub(ids.len());

            let mut ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let mut mask_i64: Vec<i64> = mask.iter().map(|&x| x as i64).collect();

            ids_i64.extend(std::iter::repeat(pad_id).take(pad_len));
            mask_i64.extend(std::iter::repeat(0).take(pad_len));

            input_ids_batch.extend_from_slice(&ids_i64);
            mask_batch.extend_from_slice(&mask_i64);
        }

        let inputs = ort::inputs! {
            "input_ids" => Tensor::from_array(([batch, max_len], input_ids_batch))?,
            "attention_mask" => Tensor::from_array(([batch, max_len], mask_batch))?,
        };
        let outputs = self.session.run(inputs)?;
        let (out_shape, extracted_data) = outputs
            .get("sentence_embedding")
            .ok_or(anyhow::anyhow!("Missing sentence_embedding"))?
            .try_extract_tensor::<f32>()?;
        let out_batch = usize::try_from(out_shape[0])?;
        if out_batch != batch {
            return Err(anyhow::anyhow!("Batch size mismatch in outputs"));
        }

        let mut results = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * HIDDEN_DIM;
            let end = start + HIDDEN_DIM;
            let slice = extracted_data
                .get(start..end)
                .ok_or(anyhow::anyhow!("Invalid output slice"))?;
            results.push(slice.to_vec());
        }

        Ok(results)
    }

    pub fn format_query(query: String) -> String {
        format!("{PREFIX_QUERY}{query}")
    }

    pub fn format_document(text: String) -> String {
        format!("{PREFIX_DOCUMENT}{text}")
    }
}
