import 'package:flutter_embedder/src/rust/api/embeddings/qwen3.dart' as frb;

import 'model_manager.dart';

export 'package:flutter_embedder/src/rust/api/embeddings/qwen3.dart'
    show Qwen3Embedder;

class Qwen3EmbedderFactory {
  static const String defaultModelId =
      'onnx-community/Qwen3-Embedding-0.6B-ONNX';

  static Future<frb.Qwen3Embedder> fromHuggingFace({
    ModelManager? manager,
    String modelId = defaultModelId,
    String revision = 'main',
    String? onnxFile,
    String? tokenizerFile,
    bool force = false,
    String? hfToken,
  }) async {
    final resolved =
        manager ?? await ModelManager.withDefaultCacheDir(hfToken: hfToken);
    final files = await resolved.fromHuggingFace(
      modelId: modelId,
      revision: revision,
      onnxFile: onnxFile,
      tokenizerFile: tokenizerFile,
      force: force,
    );
    return frb.Qwen3Embedder.create(
      modelPath: files.modelPath,
      tokenizerPath: files.tokenizerPath,
    );
  }
}
