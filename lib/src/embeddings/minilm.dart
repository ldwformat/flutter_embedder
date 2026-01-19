import 'package:flutter_embedder/src/rust/api/embeddings/minilm.dart' as frb;

import 'model_manager.dart';

export 'package:flutter_embedder/src/rust/api/embeddings/minilm.dart'
    show MiniLmEmbedder;

class MiniLmEmbedderFactory {
  static const String defaultModelId =
      'onnx-community/all-MiniLM-L6-v2-ONNX';

  static Future<frb.MiniLmEmbedder> fromHuggingFace({
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
    return frb.MiniLmEmbedder.create(
      modelPath: files.modelPath,
      tokenizerPath: files.tokenizerPath,
    );
  }
}
