use flutter_embedder::api::embeddings::bge::BgeEmbedder;
use flutter_embedder::api::ort::init_ort;
use ndarray::{Array, Array2};

mod config;
use config::{init_test_config, BGE_EMBEDDING_MODEL_PATH, BGE_TOKENIZER_PATH, ORT_LIB_PATH};

/// @reference https://huggingface.co/onnx-community/bge-small-en-v1.5-ONNX
#[test]
fn bge_embedding_test() {
    init_test_config();
    let tokenizer_path: String = BGE_TOKENIZER_PATH.get().unwrap().into();
    let model_path: String = BGE_EMBEDDING_MODEL_PATH.get().unwrap().into();
    let ort_path: String = ORT_LIB_PATH.get().unwrap().into();

    init_ort("bge_ort".to_string(), Some(ort_path)).unwrap();
    let mut embedder = BgeEmbedder::create(model_path, tokenizer_path).unwrap();

    // Basic embedding example.
    let texts = ["Hello world.".to_string(), "Example sentence.".to_string()];
    let outputs = embedder.embed(texts.to_vec()).unwrap();
    assert_eq!(outputs.len(), 2);
    let embedding_size = outputs[0].len();
    assert_eq!(embedding_size, 384);

    // Retrieval example with query prefix.
    let docs = [
        "Hello world.",
        "The giant panda is a bear species endemic to China.",
        // "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "I love pandas so much!",
    ];
    let doc_inputs = docs
        .iter()
        .map(|s| BgeEmbedder::format_document(s.to_string()))
        .collect::<Vec<_>>();
    let doc_embeddings = embedder.embed(doc_inputs).unwrap();

    let query = BgeEmbedder::format_query("What is a panda?".to_string());
    let query_embedding = embedder.embed(vec![query]).unwrap();

    let query_vec = query_embedding[0].to_vec();
    let doc_matrix: Array2<f32> = Array::from_shape_vec(
        (docs.len(), embedding_size),
        doc_embeddings.into_iter().flatten().collect(),
    )
    .unwrap();
    let query_matrix: Array2<f32> = Array::from_shape_vec((1, embedding_size), query_vec).unwrap();
    let sims = query_matrix.dot(&doc_matrix.t());
    println!("Similarities:\n{sims}");
    let (best_idx, _) = sims
        .row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best_idx, 1);
}
