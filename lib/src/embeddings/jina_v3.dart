import 'package:flutter_embedder/src/rust/api/embeddings/jina_v3.dart' as frb;

import 'model_manager.dart';

export 'package:flutter_embedder/src/rust/api/embeddings/jina_v3.dart'
    show JinaV3Embedder;

class JinaV3EmbedderFactory {
  static const String defaultModelId = 'ldwformat/jina-embeddings-v3-Q8-onnx';

  static Future<frb.JinaV3Embedder> fromHuggingFace({
    ModelManager? manager,
    String modelId = defaultModelId,
    String revision = 'main',
    String? onnxFile,
    String? tokenizerFile,
    bool includeExternalData = true,
    DownloadProgress? onProgress,
    int maxConnections = 1,
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
      force: force,
    );
    return frb.JinaV3Embedder.create(
      modelPath: files.modelPath,
      tokenizerPath: files.tokenizerPath,
    );
  }
}
