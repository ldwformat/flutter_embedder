use std::fs;

use flutter_embedder::api::tokenizer::{
    add_special_tokens, decode, decode_batch, encode, encode_batch,
    load_tokenizer_from_bytes_with_special_tokens, load_tokenizer_from_file,
    load_tokenizer_from_json_with_special_tokens,
};
use flutter_embedder::api::utils::{cosine_distance, mean_pooling_vec, normalize};

mod config;
use config::{init_test_config, QWEN_TOKENIZER_PATH};

#[test]
fn load_from_file_and_encode_decode_smoke() {
    init_test_config();

    let tokenizer_path = QWEN_TOKENIZER_PATH.get().unwrap().into();

    let tokenizer_id = load_tokenizer_from_file(tokenizer_path).unwrap();

    let encoding = encode(tokenizer_id, "hello world".to_string(), None).unwrap();
    assert!(!encoding.ids.is_empty(), "ids should not be empty");
    assert_eq!(encoding.ids.len(), encoding.attention_mask.len());
    let decoded = decode(tokenizer_id, encoding.ids.clone(), None).unwrap();
    assert!(!decoded.is_empty(), "decoded string should not be empty");
}

#[test]
fn load_from_bytes_with_special_tokens_and_batch_encode_decode_smoke() {
    init_test_config();

    let tokenizer_path: String = QWEN_TOKENIZER_PATH.get().unwrap().into();

    let bytes = fs::read(tokenizer_path).unwrap();
    let tokenizer_id =
        load_tokenizer_from_bytes_with_special_tokens(bytes, vec!["[EXTRA]".to_string()]).unwrap();

    let inputs = vec!["hello worlds".to_string(), "test world".to_string()];
    let encodings = encode_batch(tokenizer_id, inputs.clone(), None).unwrap();
    assert_eq!(encodings.len(), 2);
    assert!(encodings.iter().all(|e| !e.ids.is_empty()));

    let decoded = decode_batch(
        tokenizer_id,
        encodings.iter().map(|e| e.ids.clone()).collect(),
        None,
    )
    .unwrap();
    assert_eq!(decoded.len(), 2);
    assert!(decoded.iter().all(|s: &String| !s.is_empty()));
}

#[test]
fn load_from_json_with_special_tokens_and_add_tokens_smoke() {
    init_test_config();
    let tokenizer_path: String = QWEN_TOKENIZER_PATH.get().unwrap().into();

    let json = fs::read_to_string(tokenizer_path).unwrap();
    let tokenizer_id =
        load_tokenizer_from_json_with_special_tokens(json, vec!["[NEW]".to_string()]).unwrap();

    // add_special_tokens is additive and should succeed
    let added = add_special_tokens(tokenizer_id, vec!["[ANOTHER]".to_string()]).unwrap();
    assert!(added >= 1);

    let encoding = encode(tokenizer_id, "hello test".to_string(), None).unwrap();
    assert!(!encoding.ids.is_empty());
}

#[test]
fn vector_utils_smoke() {
    let normalized = normalize(vec![3.0, 4.0].as_slice());
    let norm_sum: f32 = normalized.iter().map(|v| v * v).sum();
    assert!((norm_sum - 1.0).abs() < 1e-3);

    let pooled = mean_pooling_vec(
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        vec![1, 0, 1],
    );
    assert_eq!(pooled.len(), 2);

    let distance = cosine_distance(vec![1.0, 0.0], vec![0.0, 1.0]).unwrap();
    assert!(distance > 0.9);
}
