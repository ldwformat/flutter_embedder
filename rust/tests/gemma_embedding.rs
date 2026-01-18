use flutter_embedder::api::embeddings::gemma::GemmaEmbedder;
use flutter_embedder::api::ort::init_ort;
use ndarray::{array, Array, Array2};

mod config;
use config::{init_test_config, GEMMA_EMBEDDING_MODEL_PATH, GEMMA_TOKENIZER_PATH, ORT_LIB_PATH};

/// @reference https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX
#[test]
fn gemma_embedding_test() {
    init_test_config();
    let tokenizer_path: String = GEMMA_TOKENIZER_PATH.get().unwrap().into();
    let model_path: String = GEMMA_EMBEDDING_MODEL_PATH.get().unwrap().into();
    let ort_path: String = ORT_LIB_PATH.get().unwrap().into();

    init_ort("gemma_ort".to_string(), Some(ort_path)).unwrap();
    let mut embedder = GemmaEmbedder::create(model_path, tokenizer_path).unwrap();

    // Each query must come with a one-sentence instruction that describes the task
    let query = GemmaEmbedder::format_query("Which planet is known as the Red Planet?".to_string());
    let documents = [
        GemmaEmbedder::format_document(
            "Venus is often called Earth's twin because of its similar size and proximity."
                .to_string(),
        ),
        GemmaEmbedder::format_document(
            "Mars, known for its reddish appearance, is often referred to as the Red Planet."
                .to_string(),
        ),
        GemmaEmbedder::format_document(
            "Jupiter, the largest planet in our solar system, has a prominent red spot."
                .to_string(),
        ),
        GemmaEmbedder::format_document(
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.".to_string(),
        ),
    ];
    let inputs = vec![query]
        .iter()
        .chain(documents.iter())
        .cloned()
        .collect::<Vec<_>>();
    let outputs = embedder.embed(inputs).unwrap();

    assert_eq!(outputs.len(), 5);

    let embedding_size = outputs[2].len();
    assert_eq!(embedding_size, 768);

    let queries: Array2<f32> = Array::from_shape_vec(
        (1, embedding_size),
        outputs[0..1].to_vec().into_iter().flatten().collect(),
    )
    .unwrap();

    let docs = Array::from_shape_vec(
        (4, embedding_size),
        outputs[1..].to_vec().into_iter().flatten().collect(),
    )
    .unwrap();

    let sims = queries.dot(&docs.t());
    let target_sims = array![0.30109745, 0.635883, 0.49304956, 0.48887485];
    assert!(sims.dot(&target_sims.t())[0] > 0.95); // cosine similarity
}
