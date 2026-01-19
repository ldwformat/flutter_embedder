import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_embedder/flutter_embedder.dart';
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initFlutterEmbedder();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final List<String> _log = [];
  final TextEditingController _qwenModelController = TextEditingController();
  final TextEditingController _qwenTokenizerController =
      TextEditingController();
  final TextEditingController _gemmaModelController = TextEditingController();
  final TextEditingController _gemmaTokenizerController =
      TextEditingController();

  HfTokenizer? _tokenizer;
  Qwen3Embedder? _qwenEmbedder;
  GemmaEmbedder? _gemmaEmbedder;
  ModelManager? _modelManager;
  final TextEditingController _hfModelController = TextEditingController();
  String? _qwenModelPath;
  String? _qwenTokenizerPath;
  String? _gemmaModelPath;
  String? _gemmaTokenizerPath;
  String _docsDir = '';
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _qwenModelController.dispose();
    _qwenTokenizerController.dispose();
    _gemmaModelController.dispose();
    _gemmaTokenizerController.dispose();
    _hfModelController.dispose();
    super.dispose();
  }

  Future<void> _bootstrap() async {
    if (!Platform.isAndroid) {
      _appendLog('This example targets Android only.');
      return;
    }

    final docs = await getApplicationDocumentsDirectory();
    _docsDir = docs.path;

    _modelManager = await ModelManager.withDefaultCacheDir();
    _hfModelController.text = 'onnx-community/bge-small-en-v1.5-ONNX';

    _tokenizer = await HfTokenizer.fromAsset('assets/tokenizer.json');
    final tokenizerPath = await _copyAsset(
      'assets/tokenizer.json',
      '${docs.path}/tokenizer.json',
    );

    _qwenTokenizerController.text = tokenizerPath;
    _gemmaTokenizerController.text = tokenizerPath;
    _qwenModelController.text = '${docs.path}/qwen3-embedding.onnx';
    _gemmaModelController.text = '${docs.path}/gemma-embedding.onnx';

    _appendLog('Documents dir: ${docs.path}');
    _appendLog('Model cache dir: ${_modelManager!.cacheDir.path}');
    _appendLog('Tokenizer asset copied to: $tokenizerPath');
    _appendLog(
      'Place ONNX models at the paths above before running embeddings.',
    );
    if (!mounted) {
      return;
    }
    setState(() {});
  }

  Future<String> _copyAsset(String assetPath, String outputPath) async {
    final data = await rootBundle.load(assetPath);
    final bytes = data.buffer.asUint8List();
    final file = File(outputPath);
    await file.writeAsBytes(bytes, flush: true);
    return file.path;
  }

  void _appendLog(String message) {
    if (!mounted) {
      return;
    }
    setState(() {
      _log.insert(0, message);
    });
  }

  Future<void> _runTokenizerDemo() async {
    if (_tokenizer == null) {
      _appendLog('Tokenizer not ready yet.');
      return;
    }
    final encoding = _tokenizer!.encode('hello worlds', addSpecialTokens: true);
    final decoded = _tokenizer!.decode(encoding.ids, skipSpecialTokens: true);
    _appendLog('Tokenizer IDs: ${encoding.ids}');
    _appendLog('Tokenizer tokens: ${encoding.tokens}');
    _appendLog('Tokenizer decoded: $decoded');
  }

  Future<void> _downloadHfModel() async {
    final manager = _modelManager;
    if (manager == null) {
      _appendLog('Model manager not ready yet.');
      return;
    }
    final modelId = _hfModelController.text.trim();
    if (modelId.isEmpty) {
      _appendLog('Model id is empty.');
      return;
    }
    setState(() => _loading = true);
    try {
      final files = await manager.fromHuggingFace(modelId: modelId);
      _appendLog('Downloaded model: ${files.modelId}');
      _appendLog('ONNX: ${files.modelPath}');
      _appendLog('Tokenizer: ${files.tokenizerPath}');
    } catch (err) {
      _appendLog('Download failed: $err');
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _runQwenDemo() async {
    final modelPath = _qwenModelController.text.trim();
    final tokenizerPath = _qwenTokenizerController.text.trim();
    if (!_requireFile(modelPath, 'Qwen3 model')) return;
    if (!_requireFile(tokenizerPath, 'Qwen3 tokenizer')) return;

    setState(() => _loading = true);
    try {
      if (_qwenEmbedder == null ||
          _qwenModelPath != modelPath ||
          _qwenTokenizerPath != tokenizerPath) {
        _qwenEmbedder = Qwen3Embedder.create(
          modelPath: modelPath,
          tokenizerPath: tokenizerPath,
        );
        _qwenModelPath = modelPath;
        _qwenTokenizerPath = tokenizerPath;
      }
      final embedder = _qwenEmbedder!;
      final inputs = <String>[
        Qwen3Embedder.formatQuery(query: 'What is the capital of China?'),
        Qwen3Embedder.formatQuery(query: 'Explain gravity'),
        Qwen3Embedder.formatDocument(text: 'The capital of China is Beijing.'),
        Qwen3Embedder.formatDocument(
          text:
              'Gravity is a force that attracts two bodies towards each other.',
        ),
      ];
      final embeddings = embedder.embed(texts: inputs);
      final sim = _cosine(embeddings[0], embeddings[2]);
      _appendLog('Qwen3 embeddings: ${embeddings.length} items');
      _appendLog('Qwen3 dim: ${embeddings[0].length}');
      _appendLog('Qwen3 cosine(query0, doc0): ${sim.toStringAsFixed(4)}');
    } catch (err) {
      _appendLog('Qwen3 embedding failed: $err');
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _runGemmaDemo() async {
    final modelPath = _gemmaModelController.text.trim();
    final tokenizerPath = _gemmaTokenizerController.text.trim();
    if (!_requireFile(modelPath, 'Gemma model')) return;
    if (!_requireFile(tokenizerPath, 'Gemma tokenizer')) return;

    setState(() => _loading = true);
    try {
      if (_gemmaEmbedder == null ||
          _gemmaModelPath != modelPath ||
          _gemmaTokenizerPath != tokenizerPath) {
        _gemmaEmbedder = GemmaEmbedder.create(
          modelPath: modelPath,
          tokenizerPath: tokenizerPath,
        );
        _gemmaModelPath = modelPath;
        _gemmaTokenizerPath = tokenizerPath;
      }
      final embedder = _gemmaEmbedder!;
      final query = GemmaEmbedder.formatQuery(
        query: 'Which planet is known as the Red Planet?',
      );
      const docs = [
        'Venus is often called Earth\'s twin.',
        'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
        'Jupiter has a prominent red spot.',
        'Saturn is famous for its rings.',
      ];
      final embeddings = <Float32List>[
        ...embedder.embed(
          texts: [
            query,
            ...docs.map((text) => GemmaEmbedder.formatDocument(text: text)),
          ],
        ),
      ];
      final sim = _cosine(embeddings[0], embeddings[2]);
      _appendLog('Gemma embeddings: ${embeddings.length} items');
      _appendLog('Gemma dim: ${embeddings[0].length}');
      _appendLog('Gemma cosine(query, doc1): ${sim.toStringAsFixed(4)}');
    } catch (err) {
      _appendLog('Gemma embedding failed: $err');
    } finally {
      setState(() => _loading = false);
    }
  }

  bool _requireFile(String path, String label) {
    if (path.isEmpty) {
      _appendLog('$label path is empty.');
      return false;
    }
    if (!File(path).existsSync()) {
      _appendLog('$label not found: $path');
      return false;
    }
    return true;
  }

  double _cosine(Float32List a, Float32List b) {
    final aList = a.map((v) => v.toDouble()).toList(growable: false);
    final bList = b.map((v) => v.toDouble()).toList(growable: false);
    return cosineDistance(aList, bList);
  }

  @override
  Widget build(BuildContext context) {
    final controls = <Widget>[
      Text('Docs dir: $_docsDir'),
      const SizedBox(height: 8),
      const Text('Tokenizer demo'),
      ElevatedButton(
        onPressed: _loading ? null : _runTokenizerDemo,
        child: const Text('Run tokenizer demo'),
      ),
      const SizedBox(height: 16),
      const Text('Hugging Face model download'),
      TextField(
        controller: _hfModelController,
        decoration: const InputDecoration(labelText: 'modelId'),
      ),
      const SizedBox(height: 8),
      ElevatedButton(
        onPressed: _loading ? null : _downloadHfModel,
        child: const Text('Download model'),
      ),
      const SizedBox(height: 16),
      const Text('Qwen3 paths'),
      TextField(
        controller: _qwenModelController,
        decoration: const InputDecoration(labelText: 'Qwen3 model path'),
      ),
      TextField(
        controller: _qwenTokenizerController,
        decoration: const InputDecoration(labelText: 'Qwen3 tokenizer path'),
      ),
      const SizedBox(height: 8),
      ElevatedButton(
        onPressed: _loading ? null : _runQwenDemo,
        child: const Text('Run Qwen3 embedding'),
      ),
      const SizedBox(height: 16),
      const Text('Gemma paths'),
      TextField(
        controller: _gemmaModelController,
        decoration: const InputDecoration(labelText: 'Gemma model path'),
      ),
      TextField(
        controller: _gemmaTokenizerController,
        decoration: const InputDecoration(labelText: 'Gemma tokenizer path'),
      ),
      const SizedBox(height: 8),
      ElevatedButton(
        onPressed: _loading ? null : _runGemmaDemo,
        child: const Text('Run Gemma embedding'),
      ),
    ];

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Flutter Embedding (Android)')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (!Platform.isAndroid)
                const Text('This example only supports Android.'),
              Expanded(child: ListView(children: controls)),
              const SizedBox(height: 12),
              const Text('Logs'),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: _log.length,
                  itemBuilder: (context, index) => Text(_log[index]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
