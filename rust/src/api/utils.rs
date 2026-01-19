pub use ndarray::Array2;
pub use ndarray::Array2 as FrbArray2Alias;
use ndarray::{Array1, Axis};

#[flutter_rust_bridge::frb(sync)]
pub fn cosine_distance(a: Vec<f32>, b: Vec<f32>) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length".into());
    }

    let mut dot_product = 0.0;
    let mut norm_sq_a = 0.0;
    let mut norm_sq_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_sq_a += a[i] * a[i];
        norm_sq_b += b[i] * b[i];
    }

    if norm_sq_a == 0.0 || norm_sq_b == 0.0 {
        return Err("Cannot compute cosine distance on zero vectors".into());
    }

    let similarity = dot_product / (norm_sq_a.sqrt() * norm_sq_b.sqrt());
    Ok(1.0 - similarity.clamp(-1.0, 1.0))
}

// Internal helper for embedding pipelines that already operate on ndarray.
pub fn mean_pooling_ndarray(embeddings: &Array2<f32>, attention_mask: &[u32]) -> Vec<f32> {
    let (seq_len, hidden_size) = embeddings.dim();
    if seq_len == 0 || hidden_size == 0 {
        return Vec::new();
    }

    let mut mask: Vec<f32> = attention_mask
        .iter()
        .take(seq_len)
        .map(|&v| if v == 0 { 0.0 } else { 1.0 })
        .collect();
    if mask.len() < seq_len {
        mask.extend(std::iter::repeat(0.0).take(seq_len - mask.len()));
    }

    let mask = Array1::from(mask);
    let count = mask.sum();
    if count <= 0.0 {
        return vec![0.0; hidden_size];
    }

    let weighted = embeddings * &mask.insert_axis(Axis(1));
    let pooled = weighted.sum_axis(Axis(0)) / count;
    pooled.to_vec()
}

#[flutter_rust_bridge::frb(sync)]
pub fn mean_pooling(embeddings: &Array2<f32>, attention_mask: &[u32]) -> Vec<f32> {
    mean_pooling_ndarray(embeddings, attention_mask)
}

#[flutter_rust_bridge::frb(sync)]
pub fn mean_pooling_vec(embeddings: Vec<Vec<f32>>, attention_mask: Vec<u32>) -> Vec<f32> {
    let seq_len = embeddings.len();
    if seq_len == 0 {
        return Vec::new();
    }
    let hidden = embeddings[0].len();
    if hidden == 0 {
        return Vec::new();
    }
    if attention_mask.len() != seq_len {
        return Vec::new();
    }
    // Flatten into ndarray for reuse of pooling logic.
    let flat: Vec<f32> = embeddings.into_iter().flatten().collect();
    if let Ok(arr) = Array2::from_shape_vec((seq_len, hidden), flat) {
        mean_pooling_ndarray(&arr, &attention_mask)
    } else {
        Vec::new()
    }
}

#[flutter_rust_bridge::frb(sync)]
pub fn normalize(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm = if norm < 1e-9 { 1e-9 } else { norm };

    embedding.iter().map(|x| x / norm).collect()
}

pub fn take<A>(a: &[A], count: usize) -> Vec<A>
where
    A: Clone,
{
    a.iter().take(count).cloned().collect()
}
