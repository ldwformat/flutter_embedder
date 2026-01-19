use flutter_embedder::api::embeddings::minilm::MiniLmEmbedder;
use flutter_embedder::api::ort::init_ort;
use ndarray::{Array, Array2};

mod config;
use config::{init_test_config, MINILM_EMBEDDING_MODEL_PATH, MINILM_TOKENIZER_PATH, ORT_LIB_PATH};

/// @reference https://huggingface.co/onnx-community/all-MiniLM-L6-v2-ONNX
#[test]
fn minilm_embedding_test() {
    init_test_config();
    let tokenizer_path: String = MINILM_TOKENIZER_PATH.get().unwrap().into();
    let model_path: String = MINILM_EMBEDDING_MODEL_PATH.get().unwrap().into();
    let ort_path: String = ORT_LIB_PATH.get().unwrap().into();

    init_ort("minilm_ort".to_string(), Some(ort_path)).unwrap();
    let mut embedder = MiniLmEmbedder::create(model_path, tokenizer_path).unwrap();

    let sentences = [
        MiniLmEmbedder::format_query("This is an example sentence".to_string()),
        MiniLmEmbedder::format_query("Each sentence is converted".to_string()),
    ];
    let outputs = embedder.embed(sentences.to_vec()).unwrap();

    assert_eq!(outputs.len(), 2);
    let embedding_size = outputs[0].len();
    assert_eq!(embedding_size, 384);

    let embeddings: Array2<f32> = Array::from_shape_vec(
        (2, embedding_size),
        outputs.into_iter().flatten().collect(),
    )
    .unwrap();
    println!("{:?}", embeddings);
}
