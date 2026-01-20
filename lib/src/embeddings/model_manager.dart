import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

class EmbeddingModelFiles {
  const EmbeddingModelFiles({
    required this.modelId,
    required this.modelPath,
    required this.tokenizerPath,
    required this.source,
    this.revision,
    this.extraFiles = const {},
  });

  final String modelId;
  final String modelPath;
  final String tokenizerPath;
  final String source;
  final String? revision;
  final Map<String, String> extraFiles;
}

typedef DownloadProgress = void Function(String file, int received, int total);
typedef _Range = ({int start, int end});

class _RemoteFileInfo {
  const _RemoteFileInfo({required this.length, required this.supportsRanges});

  final int length;
  final bool supportsRanges;
}

class ModelManager {
  ModelManager({required this.cacheDir, HttpClient? httpClient, this.hfToken})
    : _client = httpClient ?? HttpClient();

  static Future<ModelManager> withDefaultCacheDir({
    String subdir = "flutter_embedder",
    bool temporary = false,
    String? hfToken,
  }) async {
    final baseDir = temporary
        ? await getTemporaryDirectory()
        : await getApplicationSupportDirectory();
    final path = subdir.isEmpty
        ? baseDir.path
        : "${baseDir.path}${Platform.pathSeparator}$subdir";
    final dir = Directory(path);
    return ModelManager(cacheDir: dir, hfToken: hfToken);
  }

  final Directory cacheDir;
  final HttpClient _client;
  final String? hfToken;

  Future<EmbeddingModelFiles?> getLocalModel(String modelId) async {
    final modelDir = Directory(
      _join(cacheDir.path, _safeModelDirName(modelId)),
    );
    final manifest = await _readManifest(modelDir);
    if (manifest == null) {
      return null;
    }
    final modelPath = _join(modelDir.path, manifest.modelFile);
    final tokenizerPath = _join(modelDir.path, manifest.tokenizerFile);
    if (!File(modelPath).existsSync() || !File(tokenizerPath).existsSync()) {
      return null;
    }
    return EmbeddingModelFiles(
      modelId: manifest.modelId,
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
      source: manifest.source,
      revision: manifest.revision,
      extraFiles: manifest.extraFiles,
    );
  }

  Future<List<String>> listModelIds() async {
    if (!cacheDir.existsSync()) {
      return const [];
    }
    final entries = cacheDir.listSync();
    final ids = <String>[];
    for (final entry in entries.whereType<Directory>()) {
      final manifest = await _readManifest(entry);
      if (manifest != null && manifest.modelId.isNotEmpty) {
        ids.add(manifest.modelId);
      }
    }
    return ids;
  }

  Future<void> deleteModel(String modelId) async {
    final dir = Directory(_join(cacheDir.path, _safeModelDirName(modelId)));
    if (dir.existsSync()) {
      await dir.delete(recursive: true);
    }
  }

  Future<EmbeddingModelFiles> fromAssets({
    required String modelId,
    required String modelAssetPath,
    required String tokenizerAssetPath,
    String? modelFileName,
    String? tokenizerFileName,
    bool overwrite = false,
  }) async {
    final modelDir = await _ensureModelDir(modelId);
    final modelName = modelFileName ?? _basename(modelAssetPath);
    final tokenizerName = tokenizerFileName ?? _basename(tokenizerAssetPath);

    final modelFile = File(_join(modelDir.path, modelName));
    final tokenizerFile = File(_join(modelDir.path, tokenizerName));

    await _writeAssetToFile(modelAssetPath, modelFile, overwrite);
    await _writeAssetToFile(tokenizerAssetPath, tokenizerFile, overwrite);

    final manifest = _ModelManifest(
      modelId: modelId,
      source: "assets",
      modelFile: modelName,
      tokenizerFile: tokenizerName,
    );
    await _writeManifest(modelDir, manifest);

    return EmbeddingModelFiles(
      modelId: modelId,
      modelPath: modelFile.path,
      tokenizerPath: tokenizerFile.path,
      source: manifest.source,
      revision: manifest.revision,
    );
  }

  Future<EmbeddingModelFiles> fromHuggingFace({
    required String modelId,
    String revision = "main",
    String? onnxFile,
    String? tokenizerFile,
    bool includeExternalData = true,
    DownloadProgress? onProgress,
    int maxConnections = 1,
    bool force = false,
  }) async {
    final modelDir = await _ensureModelDir(modelId);
    final files = await _fetchHfFiles(modelId);
    final onnxName = onnxFile ?? _selectOnnxFile(files);
    final tokenizerName = tokenizerFile ?? _selectTokenizerFile(files);
    if (onnxName == null || tokenizerName == null) {
      throw StateError("Missing ONNX model or tokenizer.json for $modelId");
    }

    final modelFile = File(_join(modelDir.path, onnxName));
    final tokenizerJson = File(_join(modelDir.path, tokenizerName));
    final extraFiles = <String, String>{};

    String? onnxDataName;
    if (includeExternalData) {
      onnxDataName = _resolveOnnxDataFile(onnxName, files);
      if (onnxDataName != null) {
        extraFiles["onnx_data"] = onnxDataName;
      }
    }

    await _downloadFile(
      _hfResolveUrl(modelId, revision, onnxName),
      modelFile,
      onProgress,
      maxConnections,
      force,
    );
    await _downloadFile(
      _hfResolveUrl(modelId, revision, tokenizerName),
      tokenizerJson,
      onProgress,
      maxConnections,
      force,
    );
    if (onnxDataName != null) {
      final dataFile = File(_join(modelDir.path, onnxDataName));
      await _downloadFile(
        _hfResolveUrl(modelId, revision, onnxDataName),
        dataFile,
        onProgress,
        maxConnections,
        force,
      );
    }

    final manifest = _ModelManifest(
      modelId: modelId,
      source: "huggingface",
      modelFile: onnxName,
      tokenizerFile: tokenizerName,
      revision: revision,
      extraFiles: extraFiles,
    );
    await _writeManifest(modelDir, manifest);

    return EmbeddingModelFiles(
      modelId: modelId,
      modelPath: modelFile.path,
      tokenizerPath: tokenizerJson.path,
      source: manifest.source,
      revision: revision,
      extraFiles: extraFiles,
    );
  }

  Future<Directory> _ensureModelDir(String modelId) async {
    if (!cacheDir.existsSync()) {
      await cacheDir.create(recursive: true);
    }
    final modelDir = Directory(
      _join(cacheDir.path, _safeModelDirName(modelId)),
    );
    if (!modelDir.existsSync()) {
      await modelDir.create(recursive: true);
    }
    return modelDir;
  }

  Future<void> _writeAssetToFile(
    String assetPath,
    File dest,
    bool overwrite,
  ) async {
    if (dest.existsSync() && !overwrite) {
      return;
    }
    await dest.parent.create(recursive: true);
    final data = await rootBundle.load(assetPath);
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    await dest.writeAsBytes(bytes, flush: true);
  }

  Future<void> _downloadFile(
    Uri url,
    File dest,
    DownloadProgress? onProgress,
    int maxConnections,
    bool overwrite,
  ) async {
    if (dest.existsSync() && !overwrite) {
      return;
    }
    await dest.parent.create(recursive: true);
    if (maxConnections > 1) {
      final info = await _probeRemoteFile(url);
      final ranges = _planRanges(info, maxConnections);
      if (ranges.length > 1) {
        try {
          await _downloadFileParallel(
            url,
            dest,
            info!.length,
            ranges,
            onProgress,
          );
          return;
        } catch (_) {
          await _cleanupParts(dest, ranges.length);
        }
      }
    }
    await _downloadFileSingle(url, dest, onProgress);
  }

  Future<void> _downloadFileSingle(
    Uri url,
    File dest,
    DownloadProgress? onProgress,
  ) async {
    final request = await _client.getUrl(url);
    if (hfToken != null && hfToken!.isNotEmpty) {
      request.headers.set("Authorization", "Bearer $hfToken");
    }
    final response = await request.close();
    if (response.statusCode != 200) {
      throw HttpException(
        "Failed to download $url (${response.statusCode})",
        uri: url,
      );
    }
    final total = response.contentLength;
    var received = 0;
    final sink = dest.openWrite();
    await for (final chunk in response) {
      received += chunk.length;
      sink.add(chunk);
      if (onProgress != null) {
        onProgress(dest.path, received, total);
      }
    }
    await sink.flush();
    await sink.close();
  }

  Future<_RemoteFileInfo?> _probeRemoteFile(Uri url) async {
    final request = await _client.openUrl("HEAD", url);
    if (hfToken != null && hfToken!.isNotEmpty) {
      request.headers.set("Authorization", "Bearer $hfToken");
    }
    final response = await request.close();
    if (response.statusCode < 200 || response.statusCode >= 300) {
      return null;
    }
    final length = response.contentLength;
    if (length <= 0) {
      return null;
    }
    final acceptRanges = response.headers.value("accept-ranges");
    final supportsRanges =
        acceptRanges != null && acceptRanges.contains("bytes");
    return _RemoteFileInfo(length: length, supportsRanges: supportsRanges);
  }

  List<_Range> _planRanges(_RemoteFileInfo? info, int maxConnections) {
    const minPartSize = 4 * 1024 * 1024;
    if (info == null || !info.supportsRanges || maxConnections <= 1) {
      return const [];
    }
    final length = info.length;
    final parts = (length / minPartSize).ceil();
    final partCount = parts.clamp(1, maxConnections);
    if (partCount <= 1) {
      return const [];
    }
    final chunkSize = (length / partCount).ceil();
    final ranges = <_Range>[];
    for (var i = 0; i < partCount; i++) {
      final start = i * chunkSize;
      var end = (i + 1) * chunkSize - 1;
      if (end >= length) {
        end = length - 1;
      }
      if (start > end) {
        break;
      }
      ranges.add((start: start, end: end));
    }
    return ranges;
  }

  Future<void> _downloadFileParallel(
    Uri url,
    File dest,
    int total,
    List<_Range> ranges,
    DownloadProgress? onProgress,
  ) async {
    final partReceived = List<int>.filled(ranges.length, 0);
    final futures = <Future<void>>[];
    for (var i = 0; i < ranges.length; i++) {
      final range = ranges[i];
      final partFile = File("${dest.path}.part$i");
      futures.add(
        _downloadRange(url, partFile, range, (received) {
          partReceived[i] = received;
          if (onProgress != null) {
            final sum = partReceived.fold<int>(0, (a, b) => a + b);
            onProgress(dest.path, sum, total);
          }
        }),
      );
    }
    await Future.wait(futures);

    final sink = dest.openWrite();
    for (var i = 0; i < ranges.length; i++) {
      final partFile = File("${dest.path}.part$i");
      await sink.addStream(partFile.openRead());
      await partFile.delete();
    }
    await sink.flush();
    await sink.close();
  }

  Future<void> _downloadRange(
    Uri url,
    File dest,
    _Range range,
    void Function(int received) onReceived,
  ) async {
    final request = await _client.getUrl(url);
    if (hfToken != null && hfToken!.isNotEmpty) {
      request.headers.set("Authorization", "Bearer $hfToken");
    }
    request.headers.set("Range", "bytes=${range.start}-${range.end}");
    final response = await request.close();
    if (response.statusCode != 206) {
      throw HttpException(
        "Range request failed for $url (${response.statusCode})",
        uri: url,
      );
    }
    var received = 0;
    final sink = dest.openWrite();
    await for (final chunk in response) {
      received += chunk.length;
      sink.add(chunk);
      onReceived(received);
    }
    await sink.flush();
    await sink.close();
  }

  Future<void> _cleanupParts(File dest, int count) async {
    for (var i = 0; i < count; i++) {
      final partFile = File("${dest.path}.part$i");
      if (partFile.existsSync()) {
        await partFile.delete();
      }
    }
  }

  Future<List<String>> _fetchHfFiles(String modelId) async {
    final url = Uri.parse("https://huggingface.co/api/models/$modelId");
    final request = await _client.getUrl(url);
    if (hfToken != null && hfToken!.isNotEmpty) {
      request.headers.set("Authorization", "Bearer $hfToken");
    }
    final response = await request.close();
    if (response.statusCode != 200) {
      throw HttpException(
        "Failed to fetch model metadata for $modelId (${response.statusCode})",
        uri: url,
      );
    }
    final body = await response.transform(utf8.decoder).join();
    final data = jsonDecode(body) as Map<String, dynamic>;
    final siblings = data["siblings"];
    if (siblings is! List) {
      return const [];
    }
    final files = <String>[];
    for (final entry in siblings) {
      if (entry is Map && entry["rfilename"] is String) {
        files.add(entry["rfilename"] as String);
      }
    }
    return files;
  }

  Uri _hfResolveUrl(String modelId, String revision, String filename) {
    final encoded = Uri.encodeFull(filename);
    return Uri.parse(
      "https://huggingface.co/$modelId/resolve/$revision/$encoded",
    );
  }

  String? _selectTokenizerFile(List<String> files) {
    if (files.contains("tokenizer.json")) {
      return "tokenizer.json";
    }
    for (final name in files) {
      if (name.endsWith("tokenizer.json")) {
        return name;
      }
    }
    return null;
  }

  String? _selectOnnxFile(List<String> files) {
    const preferred = ["onnx/model.onnx", "model.onnx"];
    for (final name in preferred) {
      if (files.contains(name)) {
        return name;
      }
    }
    final candidates = files.where((f) => f.endsWith(".onnx")).toList();
    if (candidates.isEmpty) {
      return null;
    }
    candidates.sort((a, b) => _rankOnnx(a).compareTo(_rankOnnx(b)));
    return candidates.first;
  }

  String? _resolveOnnxDataFile(String onnxName, List<String> files) {
    if (!onnxName.endsWith('.onnx')) {
      return null;
    }
    final candidate = "${onnxName}_data";
    if (files.contains(candidate)) {
      return candidate;
    }
    return null;
  }

  int _rankOnnx(String name) {
    var score = 0;
    if (name.contains("onnx/")) score -= 3;
    if (name.contains("model")) score -= 2;
    if (name.contains("fp16")) score += 2;
    if (name.contains("q4") ||
        name.contains("int8") ||
        name.contains("quant")) {
      score += 5;
    }
    return score;
  }

  String _safeModelDirName(String modelId) =>
      modelId.replaceAll(RegExp(r"[^A-Za-z0-9._-]+"), "_");

  String _join(String base, String relative) {
    if (relative.isEmpty) {
      return base;
    }
    final separator = Platform.pathSeparator;
    final cleanBase = base.endsWith(separator) ? base : "$base$separator";
    final cleanRel = relative.startsWith(separator)
        ? relative.substring(1)
        : relative;
    return "$cleanBase$cleanRel";
  }

  String _basename(String path) {
    final lastSlash = path.lastIndexOf("/");
    final lastSep = path.lastIndexOf(Platform.pathSeparator);
    final index = lastSlash > lastSep ? lastSlash : lastSep;
    return index == -1 ? path : path.substring(index + 1);
  }

  Future<_ModelManifest?> _readManifest(Directory modelDir) async {
    final file = File(_join(modelDir.path, _ModelManifest.fileName));
    if (!file.existsSync()) {
      return null;
    }
    final data = jsonDecode(await file.readAsString());
    if (data is! Map<String, dynamic>) {
      return null;
    }
    return _ModelManifest.fromJson(data);
  }

  Future<void> _writeManifest(
    Directory modelDir,
    _ModelManifest manifest,
  ) async {
    final file = File(_join(modelDir.path, _ModelManifest.fileName));
    await file.writeAsString(jsonEncode(manifest.toJson()), flush: true);
  }
}

class _ModelManifest {
  const _ModelManifest({
    required this.modelId,
    required this.source,
    required this.modelFile,
    required this.tokenizerFile,
    this.revision,
    this.extraFiles = const {},
  });

  static const fileName = "model.json";

  final String modelId;
  final String source;
  final String modelFile;
  final String tokenizerFile;
  final String? revision;
  final Map<String, String> extraFiles;

  factory _ModelManifest.fromJson(Map<String, dynamic> json) {
    return _ModelManifest(
      modelId: json["modelId"] as String? ?? "",
      source: json["source"] as String? ?? "",
      modelFile: json["modelFile"] as String? ?? "",
      tokenizerFile: json["tokenizerFile"] as String? ?? "",
      revision: json["revision"] as String?,
      extraFiles:
          (json["extraFiles"] as Map?)?.map(
            (key, value) => MapEntry("$key", "$value"),
          ) ??
          const {},
    );
  }

  Map<String, dynamic> toJson() => {
    "modelId": modelId,
    "source": source,
    "modelFile": modelFile,
    "tokenizerFile": tokenizerFile,
    "revision": revision,
    "extraFiles": extraFiles,
  };
}
