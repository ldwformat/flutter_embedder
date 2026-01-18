library flutter_embedding;

// Public surface: high-level wrapper and runtime initializer.
export 'src/tokenizers/tokenizers.dart'
    show HfTokenizer, normalize, meanPooling, cosineDistance;
export 'src/embeddings/jina_v3.dart' show JinaV3Embedder;
export 'src/embeddings/qwen3.dart' show Qwen3Embedder;
export 'src/embeddings/gemma.dart' show GemmaEmbedder;
// export 'src/rust/frb_generated.dart' show RustLib;

// Advanced/low-level access to generated bindings (optional).
export 'src/rust/api/tokenizer.dart' show EncodeOutput, TokenOffsets;
export 'src/rust/api/ort.dart' show initOrt;

// init wrapper
import 'dart:ffi' as ffi;
import 'dart:io';

import 'src/rust/frb_generated.dart' show RustLib;
import 'src/rust/api/ort.dart' show initOrt;

bool _inited = false;

Future<void> initFlutterEmbedding({
  String name = "FlutterEmbedding",
  String path = "libonnxruntime.so",
}) async {
  if (_inited) return;

  await RustLib.init();

  if (Platform.isAndroid) {
    // If user uses Maven AAR: com.microsoft.onnxruntime:onnxruntime-android
    // the shared object is typically packaged as libonnxruntime.so
    ffi.DynamicLibrary.open(path);
  }

  initOrt(name: name, path: path);

  _inited = true;
}
