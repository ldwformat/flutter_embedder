## 0.1.3

* Added Dart-side ModelManager helpers, cache defaults, and embedder factories.
* Example app includes Hugging Face download + BGE quick demo.

## 0.1.2

* Added BGE (bge-small-en-v1.5) and MiniLM (all-MiniLM-L6-v2) embedding support.
* Added ORT init/session options for custom thread and optimization settings.
* Improved mean pooling using ndarray masking + reduction.
* Added Rust smoke tests for BGE and MiniLM models.

## 0.0.1

* Initial release with tokenizers (load/encode/decode, batch APIs, special tokens).
* Embeddings support for Jina V3, Qwen3, and Gemma ONNX models via ORT.
* Vector utils: normalize, mean pooling, cosine distance.
* Android example app for embeddings and tokenizer usage.
