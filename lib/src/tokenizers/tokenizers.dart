import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_embedding/src/rust/api/tokenizer.dart' as ffi_tok;
import 'package:flutter_embedding/src/rust/api/utils.dart' as ffi_utils;

/// Thin Dart wrapper around generated bindings for nicer ergonomics.
class HfTokenizer {
  HfTokenizer._(this._id);

  final BigInt _id;

  /// Load tokenizer from raw JSON string.
  static HfTokenizer fromJson(String json, {List<String>? specialTokens}) {
    final id = specialTokens == null
        ? ffi_tok.loadTokenizerFromJson(json: json)
        : ffi_tok.loadTokenizerFromJsonWithSpecialTokens(
            json: json,
            specialTokens: specialTokens,
          );
    return HfTokenizer._(id);
  }

  /// Load tokenizer from bytes (e.g., network or asset).
  static HfTokenizer fromBytes(List<int> bytes, {List<String>? specialTokens}) {
    final id = specialTokens == null
        ? ffi_tok.loadTokenizerFromBytes(bytes: bytes)
        : ffi_tok.loadTokenizerFromBytesWithSpecialTokens(
            bytes: bytes,
            specialTokens: specialTokens,
          );
    return HfTokenizer._(id);
  }

  /// Load tokenizer from asset path using Flutter's bundle.
  static Future<HfTokenizer> fromAsset(
    String assetPath, {
    List<String>? specialTokens,
  }) async {
    final data = await rootBundle.load(assetPath);
    final bytes = Uint8List.view(
      data.buffer,
      data.offsetInBytes,
      data.lengthInBytes,
    );
    return HfTokenizer.fromBytes(bytes, specialTokens: specialTokens);
  }

  /// Load tokenizer from local file path.
  static HfTokenizer fromFile(String path, {List<String>? specialTokens}) {
    final id = specialTokens == null
        ? ffi_tok.loadTokenizerFromFile(path: path)
        : ffi_tok.loadTokenizerFromFileWithSpecialTokens(
            path: path,
            specialTokens: specialTokens,
          );
    return HfTokenizer._(id);
  }

  /// Add special tokens to an already loaded tokenizer.
  int addSpecialTokens(List<String> tokens) =>
      ffi_tok.addSpecialTokens(tokenizerId: _id, tokens: tokens);

  ffi_tok.EncodeOutput encode(String text, {bool addSpecialTokens = true}) =>
      ffi_tok.encode(
        tokenizerId: _id,
        text: text,
        addSpecialTokens: addSpecialTokens,
      );

  List<ffi_tok.EncodeOutput> encodeBatch(
    List<String> texts, {
    bool addSpecialTokens = true,
  }) =>
      ffi_tok.encodeBatch(
        tokenizerId: _id,
        texts: texts,
        addSpecialTokens: addSpecialTokens,
      );

  Future<List<ffi_tok.EncodeOutput>> encodeBatchAsync(
    List<String> texts, {
    bool addSpecialTokens = true,
  }) =>
      ffi_tok.encodeBatchAsync(
        tokenizerId: _id,
        texts: texts,
        addSpecialTokens: addSpecialTokens,
      );

  String decode(List<int> ids, {bool skipSpecialTokens = true}) => ffi_tok
      .decode(tokenizerId: _id, ids: ids, skipSpecialTokens: skipSpecialTokens);

  List<String> decodeBatch(
    List<Uint32List> batchIds, {
    bool skipSpecialTokens = true,
  }) =>
      ffi_tok.decodeBatch(
        tokenizerId: _id,
        batchIds: batchIds,
        skipSpecialTokens: skipSpecialTokens,
      );

  Future<List<String>> decodeBatchAsync(
    List<Uint32List> batchIds, {
    bool skipSpecialTokens = true,
  }) =>
      ffi_tok.decodeBatchAsync(
        tokenizerId: _id,
        batchIds: batchIds,
        skipSpecialTokens: skipSpecialTokens,
      );
}

// Vector utils passthroughs.
List<double> normalize(List<double> embedding) =>
    ffi_utils.normalize(embedding: embedding);

// Local mean pooling to avoid RustOpaque Array2 coupling in generated code.
List<double> meanPooling(
  List<List<double>> embeddings,
  List<int> attentionMask,
) {
  if (embeddings.isEmpty) return const [];
  final hidden = embeddings.first.length;
  if (hidden == 0 || attentionMask.length != embeddings.length) {
    return const [];
  }
  final pooled = List<double>.filled(hidden, 0);
  var count = 0;
  for (var i = 0; i < embeddings.length; i++) {
    if (attentionMask[i] == 1) {
      final row = embeddings[i];
      for (var j = 0; j < hidden; j++) {
        pooled[j] += row[j];
      }
      count += 1;
    }
  }
  if (count > 0) {
    for (var j = 0; j < hidden; j++) {
      pooled[j] /= count;
    }
  }
  return pooled;
}

double cosineDistance(List<double> a, List<double> b) =>
    ffi_utils.cosineDistance(a: a, b: b);
