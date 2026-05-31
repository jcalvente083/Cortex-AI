import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/api/api_client.dart';
import '../../core/providers/session_provider.dart';
import '../../core/providers/settings_provider.dart';
import '../../core/theme.dart';
import '../result/result_screen.dart';
import 'recorder_card.dart';

class RecordingScreen extends StatelessWidget {
  const RecordingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => SessionProvider(),
      child: const _RecordingView(),
    );
  }
}

class _RecordingView extends StatelessWidget {
  const _RecordingView();

  @override
  Widget build(BuildContext context) {
    final session  = context.watch<SessionProvider>();
    final settings = context.read<SettingsProvider>();
    final isAnalyzing = session.status == SessionStatus.analyzing;

    return Scaffold(
      appBar: AppBar(title: const Text('Nueva sesión')),
      body: Column(
        children: [
          LinearProgressIndicator(
            value: session.slots.where((s) => s.file != null).length / 3,
            backgroundColor: Colors.grey[200],
            color: AppTheme.primary,
          ),
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: session.slots.length,
              separatorBuilder: (_, _) => const SizedBox(height: 8),
              itemBuilder: (ctx, i) {
                final slot = session.slots[i];
                return RecorderCard(
                  index:        i,
                  label:        slot.label,
                  instruction:  slot.instruction,
                  recordedFile: slot.file,
                  onRecorded:   (file) => session.setFile(i, file),
                );
              },
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Column(
                children: [
                  if (session.status == SessionStatus.error)
                    Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppTheme.danger.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: AppTheme.danger),
                      ),
                      child: Text(session.errorMessage ?? 'Error desconocido',
                          style: const TextStyle(color: AppTheme.danger)),
                    ),
                  ElevatedButton.icon(
                    onPressed: session.allRecorded && !isAnalyzing
                        ? () async {
                            final client = ApiClient(settings.backendUrl);
                            await session.analyze(client, settings.defaultModel);
                            if (session.status == SessionStatus.done && context.mounted) {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => ResultScreen(result: session.result!),
                                ),
                              );
                            }
                          }
                        : null,
                    icon: isAnalyzing
                        ? const SizedBox(
                            width: 20, height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Icon(Icons.analytics),
                    label: Text(isAnalyzing ? 'Analizando...' : 'Analizar'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
