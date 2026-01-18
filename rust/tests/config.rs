use std::sync::OnceLock;

pub static ORT_LIB_PATH: OnceLock<String> = OnceLock::new();
//
pub static QWEN_EMBEDDING_MODEL_PATH: OnceLock<String> = OnceLock::new();
pub static QWEN_TOKENIZER_PATH: OnceLock<String> = OnceLock::new();
//
pub static GEMMA_EMBEDDING_MODEL_PATH: OnceLock<String> = OnceLock::new();
pub static GEMMA_TOKENIZER_PATH: OnceLock<String> = OnceLock::new();

pub fn init_test_config() {
    let env_vars = dotenvy::dotenv()
        .ok()
        .and_then(|_| {
            dotenvy::vars()
                .collect::<std::collections::HashMap<_, _>>()
                .into()
        })
        .unwrap();

    if let Some(ort_lib_path) = env_vars.get("ORT_LIB_PATH") {
        ORT_LIB_PATH
            .set(ort_lib_path.to_string())
            .expect("Failed to set ORT_LIB_PATH");
    }

    if let Some(model_path) = env_vars.get("EMBEDDING_QWEN3_MODEL_PATH") {
        QWEN_EMBEDDING_MODEL_PATH
            .set(model_path.to_string())
            .expect("Failed to set QWEN_EMBEDDING_MODEL_PATH");
    }

    if let Some(tokenizer_path) = env_vars.get("TOKENIZER_QWEN3_PATH") {
        QWEN_TOKENIZER_PATH
            .set(tokenizer_path.to_string())
            .expect("Failed to set QWEN_TOKENIZER_PATH");
    }

    if let Some(model_path) = env_vars.get("EMBEDDING_GEMMA_MODEL_PATH") {
        GEMMA_EMBEDDING_MODEL_PATH
            .set(model_path.to_string())
            .expect("Failed to set GEMMA_EMBEDDING_MODEL_PATH");
    }

    if let Some(tokenizer_path) = env_vars.get("TOKENIZER_GEMMA_PATH") {
        GEMMA_TOKENIZER_PATH
            .set(tokenizer_path.to_string())
            .expect("Failed to set GEMMA_TOKENIZER_PATH");
    }
}
