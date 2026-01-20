import 'package:flutter_embedder/src/rust/api/embeddings/gemma.dart' as frb;

import 'model_manager.dart';

export 'package:flutter_embedder/src/rust/api/embeddings/gemma.dart'
    show GemmaEmbedder;

class GemmaEmbedderFactory {
  static const String defaultModelId =
      'onnx-community/embeddinggemma-300m-ONNX';

  static Future<frb.GemmaEmbedder> fromHuggingFace({
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
    return frb.GemmaEmbedder.create(
      modelPath: files.modelPath,
      tokenizerPath: files.tokenizerPath,
    );
  }
}
