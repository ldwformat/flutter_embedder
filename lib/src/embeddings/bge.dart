import 'package:flutter_embedder/src/rust/api/embeddings/bge.dart' as frb;

import 'model_manager.dart';

export 'package:flutter_embedder/src/rust/api/embeddings/bge.dart'
    show BgeEmbedder;

class BgeEmbedderFactory {
  static const String defaultModelId = 'onnx-community/bge-small-en-v1.5-ONNX';

  static Future<frb.BgeEmbedder> fromHuggingFace({
    ModelManager? manager,
    String modelId = defaultModelId,
    String revision = 'main',
    String? onnxFile,
    String? tokenizerFile,
    bool includeExternalData = true,
    DownloadProgress? onProgress,
    int maxConnections = 1,
    bool resume = true,
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
      includeExternalData: includeExternalData,
      onProgress: onProgress,
      maxConnections: maxConnections,
      resume: resume,
      force: force,
    );
    return frb.BgeEmbedder.create(
      modelPath: files.modelPath,
      tokenizerPath: files.tokenizerPath,
    );
  }
}
