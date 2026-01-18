import 'package:integration_test/integration_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:flutter_embedder/flutter_embedder.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await initFlutterEmbedder());

  testWidgets('encode/decode via asset tokenizer', (tester) async {
    final data = await rootBundle.load('assets/tokenizer.json');
    final tokenizer = HfTokenizer.fromBytes(data.buffer.asUint8List());

    final encoded = tokenizer.encode('hello world');
    expect(encoded.ids.isNotEmpty, true);

    final decoded = tokenizer.decode(encoded.ids);
    expect(decoded.isNotEmpty, true);
  });
}
