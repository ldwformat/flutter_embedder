use flutter_embedder::api::embeddings::qwen3::Qwen3Embedder;
use flutter_embedder::api::ort::init_ort;
use ndarray::{array, Array, Array2};

mod config;
use config::{init_test_config, ORT_LIB_PATH, QWEN_EMBEDDING_MODEL_PATH, QWEN_TOKENIZER_PATH};
///
/// @reference https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX
#[test]
fn qwen_embedding() {
    init_test_config();
    let tokenizer_path: String = QWEN_TOKENIZER_PATH.get().unwrap().into();
    let model_path: String = QWEN_EMBEDDING_MODEL_PATH.get().unwrap().into();
    let ort_path: String = ORT_LIB_PATH.get().unwrap().into();

    init_ort("qwen_ort".to_string(), Some(ort_path)).unwrap();
    let mut embedder = Qwen3Embedder::create(model_path, tokenizer_path).unwrap();

    // Each query must come with a one-sentence instruction that describes the task
    let queries = [
        Qwen3Embedder::format_query("What is the capital of China?".to_string()),
        Qwen3Embedder::format_query("Explain gravity".to_string()),
    ];
    let documents = [
        Qwen3Embedder::format_document("The capital of China is Beijing.".to_string()),
        Qwen3Embedder::format_document("Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".to_string()),
    ];
    let inputs = queries
        .into_iter()
        .chain(documents.iter().map(|s| s.to_string()))
        .collect::<Vec<String>>();
    let outputs = embedder.embed(inputs).unwrap();
    let queries: Array2<f32> = Array::from_shape_vec(
        (2, 1024),
        outputs[0..2].to_vec().into_iter().flatten().collect(),
    )
    .unwrap();
    let docs = Array::from_shape_vec(
        (2, 1024),
        outputs[2..4].to_vec().into_iter().flatten().collect(),
    )
    .unwrap();

    let sims = queries.dot(&docs.t());
    let target_sims = array![0.7646, 0.1414, 0.1355, 0.6000];
    assert!(sims.flatten().dot(&target_sims.t()) > 0.95); // cosine similarity
}
