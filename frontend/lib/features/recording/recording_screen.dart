import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/api/api_client.dart';
import '../../core/providers/session_provider.dart';
import '../../core/providers/settings_provider.dart';
import '../../core/theme.dart';
import '../result/result_screen.dart';
import 'recorder_card.dart';
import 'scene_widget.dart';

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

class _RecordingView extends StatefulWidget {
  const _RecordingView();

  @override
  State<_RecordingView> createState() => _RecordingViewState();
}

class _RecordingViewState extends State<_RecordingView> {
  int _step = 0;

  @override
  Widget build(BuildContext context) {
    final session    = context.watch<SessionProvider>();
    final settings   = context.read<SettingsProvider>();
    final slot       = session.slots[_step];
    final isLastStep = _step == session.slots.length - 1;
    final isAnalyzing = session.status == SessionStatus.analyzing;

    return PopScope(
      canPop: _step == 0,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) setState(() => _step--);
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text('Paso ${_step + 1} de ${session.slots.length}'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () {
              if (_step > 0) {
                setState(() => _step--);
              } else {
                Navigator.pop(context);
              }
            },
          ),
        ),
        body: Column(
          children: [
            LinearProgressIndicator(
              value: session.slots.where((s) => s.file != null).length /
                  session.slots.length,
              backgroundColor: Colors.grey[200],
              color: AppTheme.primary,
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (_step == 2) ...[
                      SceneWidget(sceneIndex: session.sceneIndex),
                      const SizedBox(height: 16),
                    ],
                    RecorderCard(
                      index:        _step,
                      label:        slot.label,
                      instruction:  slot.instruction,
                      recordedFile: slot.file,
                      onRecorded:   (file) => session.setFile(_step, file),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.lock_outline, size: 13, color: Colors.grey[400]),
                        const SizedBox(width: 4),
                        Text(
                          'El audio no se almacena en el servidor',
                          style: TextStyle(fontSize: 12, color: Colors.grey[400]),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
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
                        child: Text(
                          session.errorMessage ?? 'Error desconocido',
                          style: const TextStyle(color: AppTheme.danger),
                        ),
                      ),
                    if (!isLastStep)
                      ElevatedButton.icon(
                        onPressed:
                            slot.file != null ? () => setState(() => _step++) : null,
                        icon: const Icon(Icons.arrow_forward),
                        label: const Text('Siguiente'),
                      )
                    else
                      ElevatedButton.icon(
                        onPressed: session.allRecorded && !isAnalyzing
                            ? () async {
                                final client = ApiClient(settings.backendUrl);
                                await session.analyze(client, settings.defaultModel);
                                if (session.status == SessionStatus.done &&
                                    context.mounted) {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (_) =>
                                          ResultScreen(result: session.result!),
                                    ),
                                  );
                                }
                              }
                            : null,
                        icon: isAnalyzing
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                    strokeWidth: 2, color: Colors.white))
                            : const Icon(Icons.analytics),
                        label: Text(isAnalyzing ? 'Analizando...' : 'Analizar'),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
