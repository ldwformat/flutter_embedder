# flutter_embedder

Flutter FFI plugin wrapping Hugging Face tokenizers and ONNX embedding runtimes via Rust + ONNX Runtime (ORT).

## Features
- Tokenizers: load from JSON/bytes/file, optional special tokens, encode/decode single & batch with structured outputs.
- Embeddings: Jina V3, Qwen3, and Gemma ONNX models.
- Vector utils: `normalize`, `mean_pooling`, `cosine_distance`.

## Embedding models
All embedding models share the same usage pattern:
```dart
final model = <Model>Embedder.create(
  modelPath: '...',
  tokenizerPath: '...',
);
final embeddings = model.embed(texts: ['...']);
```
Jina V3 requires a task id:
```dart
final embeddings = jina.embed(texts: ['...'], taskId: 0);
```
Formatting helpers are provided as static methods on each embedder type:
```dart
final query = <Model>Embedder.formatQuery(query: '...');
final doc = <Model>Embedder.formatDocument(text: '...');
```

| Model | Hugging Face model card | Notes |
| --- | --- | --- |
| Jina V3 | https://huggingface.co/jinaai/jina-embeddings-v3 | Requires `taskId` input |
| Qwen3 | https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX | Use `formatQuery` / `formatDocument` helpers |
| Gemma | https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX | Use `formatQuery` / `formatDocument` helpers |

## Installation
Add to `pubspec.yaml`:
```yaml
dependencies:
  flutter_embedder: ^0.0.1
```

## Usage
Initialize the Rust runtime once:
```dart
import 'package:flutter_embedder/flutter_embedder.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initFlutterEmbedder();
  runApp(const MyApp());
}
```

Tokenizer example:
```dart
final tokenizer = await HfTokenizer.fromAsset('assets/tokenizer.json');
final encoded = tokenizer.encode('hello world', addSpecialTokens: true);
final decoded = tokenizer.decode(encoded.ids, skipSpecialTokens: true);
```

Embedding example (Qwen3):
```dart
// Optional: pass a custom ORT shared library path.
await initFlutterEmbedder(name: 'flutter_embedder', path: 'libonnxruntime.so');

final qwen = Qwen3Embedder.create(
  modelPath: '/path/to/qwen3-embedding.onnx',
  tokenizerPath: '/path/to/qwen3-tokenizer.json',
);
final query = Qwen3Embedder.formatQuery(query: 'What is the capital of China?');
final embedding = qwen.embed(texts: [query]).first;
```

Embedding example (Gemma):
```dart
final gemma = GemmaEmbedder.create(
  modelPath: '/path/to/gemma-embedding.onnx',
  tokenizerPath: '/path/to/gemma-tokenizer.json',
);
final query = GemmaEmbedder.formatQuery(query: 'Which planet is known as the Red Planet?');
final embedding = gemma.embed(texts: [query]).first;
```

## Android setup
The Android app must include ONNX Runtime’s Java package so the native ORT libraries are bundled:
```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
}
```
Add this in `android/app/build.gradle` or `example/android/app/build.gradle.kts`.

## Models and assets
- Provide your ONNX model files and tokenizer JSON files as local paths.
- On Android, store them in the app’s documents directory and pass those paths to the init functions.
- The `example/` app shows how to copy a tokenizer asset into the app documents directory.

## Platform support
- Android is the primary supported platform for embeddings.
- Other platforms may require additional ORT setup.

## Development
- Regenerate bindings: `flutter_rust_bridge_codegen generate --config-file flutter_rust_bridge.yaml`
- Rust tests: `cargo test --manifest-path rust/Cargo.toml`
