use std::collections::HashMap;
use std::fs;
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock, RwLock,
};

use tokenizers::{AddedToken, Tokenizer};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenOffsets {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EncodeOutput {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    pub offsets: Vec<TokenOffsets>,
    pub tokens: Vec<String>,
}

impl From<tokenizers::Encoding> for EncodeOutput {
    fn from(value: tokenizers::Encoding) -> Self {
        let offsets = value
            .get_offsets()
            .iter()
            .map(|(start, end)| TokenOffsets {
                start: *start as u32,
                end: *end as u32,
            })
            .collect();

        Self {
            ids: value.get_ids().to_vec(),
            attention_mask: value.get_attention_mask().to_vec(),
            type_ids: value.get_type_ids().to_vec(),
            special_tokens_mask: value.get_special_tokens_mask().to_vec(),
            offsets,
            tokens: value.get_tokens().to_vec(),
        }
    }
}

type TokenizerStore = HashMap<u64, Tokenizer>;

fn store() -> &'static RwLock<TokenizerStore> {
    static STORE: OnceLock<RwLock<TokenizerStore>> = OnceLock::new();
    STORE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn next_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn insert_tokenizer(tokenizer: Tokenizer) -> Result<u64, String> {
    let id = next_id();

    let mut guard = store()
        .write()
        .map_err(|e| format!("Failed to acquire tokenizer store: {e}"))?;
    guard.insert(id, tokenizer);

    Ok(id)
}

fn add_special_tokens_internal(tokenizer: &mut Tokenizer, tokens: Vec<String>) -> u32 {
    if tokens.is_empty() {
        return 0;
    }
    let added_tokens: Vec<AddedToken> = tokens
        .into_iter()
        .map(|t| AddedToken::from(t, true))
        .collect();
    tokenizer.add_special_tokens(&added_tokens) as u32
}

fn with_tokenizer<R, F>(id: u64, f: F) -> Result<R, String>
where
    F: FnOnce(&Tokenizer) -> Result<R, String>,
{
    let guard = store().read().map_err(|e| e.to_string())?;
    let tokenizer = guard
        .get(&id)
        .ok_or_else(|| "Unknown tokenizer id".to_string())?;
    f(tokenizer)
}

fn with_tokenizer_mut<R, F>(id: u64, f: F) -> Result<R, String>
where
    F: FnOnce(&mut Tokenizer) -> Result<R, String>,
{
    let mut guard = store()
        .write()
        .map_err(|e| format!("Failed to acquire tokenizer store: {e}"))?;
    let tokenizer = guard
        .get_mut(&id)
        .ok_or_else(|| "Unknown tokenizer id".to_string())?;
    f(tokenizer)
}

fn load_tokenizer_from_json_inner(
    json: String,
    special_tokens: Option<Vec<String>>,
) -> Result<u64, String> {
    let mut tokenizer =
        Tokenizer::from_str(&json).map_err(|err| format!("Failed to load tokenizer: {err}"))?;

    if let Some(tokens) = special_tokens {
        add_special_tokens_internal(&mut tokenizer, tokens);
    }

    insert_tokenizer(tokenizer)
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_json(json: String) -> Result<u64, String> {
    load_tokenizer_from_json_inner(json, None)
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_json_with_special_tokens(
    json: String,
    special_tokens: Vec<String>,
) -> Result<u64, String> {
    load_tokenizer_from_json_inner(json, Some(special_tokens))
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_bytes(bytes: Vec<u8>) -> Result<u64, String> {
    let json = String::from_utf8(bytes)
        .map_err(|err| format!("Tokenizer bytes are not valid UTF-8 JSON: {err}"))?;
    load_tokenizer_from_json(json)
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_bytes_with_special_tokens(
    bytes: Vec<u8>,
    special_tokens: Vec<String>,
) -> Result<u64, String> {
    let json = String::from_utf8(bytes)
        .map_err(|err| format!("Tokenizer bytes are not valid UTF-8 JSON: {err}"))?;
    load_tokenizer_from_json_inner(json, Some(special_tokens))
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_file(path: String) -> Result<u64, String> {
    let json = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read tokenizer file {path}: {err}"))?;
    load_tokenizer_from_json(json)
}

#[flutter_rust_bridge::frb(sync)]
pub fn load_tokenizer_from_file_with_special_tokens(
    path: String,
    special_tokens: Vec<String>,
) -> Result<u64, String> {
    let json = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read tokenizer file {path}: {err}"))?;
    load_tokenizer_from_json_inner(json, Some(special_tokens))
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_special_tokens(tokenizer_id: u64, tokens: Vec<String>) -> Result<u32, String> {
    with_tokenizer_mut(tokenizer_id, |tokenizer| {
        Ok(add_special_tokens_internal(tokenizer, tokens))
    })
}

#[flutter_rust_bridge::frb(sync)]
pub fn encode(
    tokenizer_id: u64,
    text: String,
    add_special_tokens: Option<bool>,
) -> Result<EncodeOutput, String> {
    with_tokenizer(tokenizer_id, |tokenizer| {
        let encoding = tokenizer
            .encode(text, add_special_tokens.unwrap_or(true))
            .map_err(|err| format!("Encode failed: {err}"))?;
        Ok(EncodeOutput::from(encoding))
    })
}

#[flutter_rust_bridge::frb(sync)]
pub fn encode_batch(
    tokenizer_id: u64,
    texts: Vec<String>,
    add_special_tokens: Option<bool>,
) -> Result<Vec<EncodeOutput>, String> {
    with_tokenizer(tokenizer_id, |tokenizer| {
        let encodings = tokenizer
            .encode_batch(texts, add_special_tokens.unwrap_or(true))
            .map_err(|err| format!("Encode batch failed: {err}"))?;
        Ok(encodings.into_iter().map(EncodeOutput::from).collect())
    })
}

#[flutter_rust_bridge::frb(sync)]
pub fn decode(
    tokenizer_id: u64,
    ids: Vec<u32>,
    skip_special_tokens: Option<bool>,
) -> Result<String, String> {
    with_tokenizer(tokenizer_id, |tokenizer| {
        tokenizer
            .decode(&ids, skip_special_tokens.unwrap_or(true))
            .map_err(|err| format!("Decode failed: {err}"))
    })
}

#[flutter_rust_bridge::frb(sync)]
pub fn decode_batch(
    tokenizer_id: u64,
    batch_ids: Vec<Vec<u32>>,
    skip_special_tokens: Option<bool>,
) -> Result<Vec<String>, String> {
    with_tokenizer(tokenizer_id, |tokenizer| {
        let sentences = batch_ids;
        let sentence_refs: Vec<&[u32]> = sentences.iter().map(|s| s.as_slice()).collect();
        tokenizer
            .decode_batch(
                sentence_refs.as_slice(),
                skip_special_tokens.unwrap_or(true),
            )
            .map_err(|err| format!("Decode batch failed: {err}"))
    })
}

// Async variants (offloaded by flutter_rust_bridge for heavier batch workloads)
#[flutter_rust_bridge::frb]
pub fn encode_batch_async(
    tokenizer_id: u64,
    texts: Vec<String>,
    add_special_tokens: Option<bool>,
) -> Result<Vec<EncodeOutput>, String> {
    encode_batch(tokenizer_id, texts, add_special_tokens)
}

#[flutter_rust_bridge::frb]
pub fn decode_batch_async(
    tokenizer_id: u64,
    batch_ids: Vec<Vec<u32>>,
    skip_special_tokens: Option<bool>,
) -> Result<Vec<String>, String> {
    decode_batch(tokenizer_id, batch_ids, skip_special_tokens)
}
