# flutter_embedder

Flutter FFI plugin wrapping Hugging Face tokenizers and ONNX embedding runtimes via Rust + ONNX Runtime (ORT).

## Features
- Tokenizers: load from JSON/bytes/file, optional special tokens, encode/decode single & batch with structured outputs.
- Embeddings: Jina V3, Qwen3, Gemma, BGE, and MiniLM ONNX models.
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
| BGE | https://huggingface.co/onnx-community/bge-small-en-v1.5-ONNX | CLS pooling + query prefix helper |
| MiniLM | https://huggingface.co/onnx-community/all-MiniLM-L6-v2-ONNX | Mean pooling + normalize |

## Installation
Add to `pubspec.yaml`:
```yaml
dependencies:
  flutter_embedder: ^0.1.2
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

Embedding example (BGE):
```dart
final bge = BgeEmbedder.create(
  modelPath: '/path/to/bge-small-en.onnx',
  tokenizerPath: '/path/to/bge-tokenizer.json',
);
final query = BgeEmbedder.formatQuery(query: 'What is a panda?');
final embedding = bge.embed(texts: [query]).first;
```

Embedding example (MiniLM):
```dart
final minilm = MiniLmEmbedder.create(
  modelPath: '/path/to/all-minilm-l6-v2.onnx',
  tokenizerPath: '/path/to/minilm-tokenizer.json',
);
final embedding = minilm.embed(texts: ['This is an example sentence']).first;
```

## Model manager (assets / Hugging Face)
Use `ModelManager` to download models by Hugging Face `modelId` or copy assets
to a local cache directory.

```dart
final manager = await ModelManager.withDefaultCacheDir();

// Download from Hugging Face by modelId.
final files = await manager.fromHuggingFace(
  modelId: 'onnx-community/bge-small-en-v1.5-ONNX',
);
final bge = BgeEmbedder.create(
  modelPath: files.modelPath,
  tokenizerPath: files.tokenizerPath,
);

// Load from bundled assets.
final assetFiles = await manager.fromAssets(
  modelId: 'my-minilm',
  modelAssetPath: 'assets/models/onnx/model.onnx',
  tokenizerAssetPath: 'assets/models/tokenizer.json',
);
final minilm = MiniLmEmbedder.create(
  modelPath: assetFiles.modelPath,
  tokenizerPath: assetFiles.tokenizerPath,
);
```

By default, `withDefaultCacheDir()` uses `getApplicationSupportDirectory()`.
Only use external storage if you want users to manage/cache files manually.

Convenience factories are also available for built-in models:
```dart
final bge = await BgeEmbedderFactory.fromHuggingFace();
final minilm = await MiniLmEmbedderFactory.fromHuggingFace();
final qwen = await Qwen3EmbedderFactory.fromHuggingFace();
final gemma = await GemmaEmbedderFactory.fromHuggingFace();
final jina = await JinaV3EmbedderFactory.fromHuggingFace();
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
