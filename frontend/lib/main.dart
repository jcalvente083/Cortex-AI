import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'core/providers/settings_provider.dart';
import 'app.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final settings = SettingsProvider();
  await settings.load();

  runApp(
    ChangeNotifierProvider.value(
      value: settings,
      child: const CortexApp(),
    ),
  );
}
