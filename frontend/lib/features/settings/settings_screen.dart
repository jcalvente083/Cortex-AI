import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/api/api_client.dart';
import '../../core/providers/settings_provider.dart';
import '../../core/theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _urlCtrl;
  bool _checking = false;

  static const _models = [
    ('knn',                'KNN — rápido, explicable'),
    ('xgboost',           'XGBoost — preciso, SHAP'),
    ('resnet18',          'ResNet18 — mel-espectrograma'),
    ('wav2vec_embeddings', 'Wav2Vec2 embeddings + KNN (cloud)'),
    ('wav2vec_finetune',  'Wav2Vec2 fine-tuning end-to-end (cloud)'),
  ];

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController(text: context.read<SettingsProvider>().backendUrl);
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveUrl() async {
    final url = _urlCtrl.text.trim();
    setState(() => _checking = true);
    await context.read<SettingsProvider>().setBackendUrl(url);
    try {
      await ApiClient(url).health();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✓ Conectado correctamente'),
            backgroundColor: AppTheme.success,
          ),
        );
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✗ No se pudo conectar — comprueba la URL y el puerto'),
            backgroundColor: AppTheme.danger,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _checking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<SettingsProvider>();

    return Scaffold(
      appBar: AppBar(title: const Text('Configuración')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Servidor backend', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextField(
            controller: _urlCtrl,
            decoration: const InputDecoration(
              labelText: 'URL del backend',
              hintText: 'http://raspberrypi.local:9000',
              prefixIcon: Icon(Icons.dns),
            ),
            onSubmitted: (_) => _saveUrl(),
            keyboardType: TextInputType.url,
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: FilledButton(
              onPressed: _checking ? null : _saveUrl,
              child: _checking
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Text('Guardar URL'),
            ),
          ),
          const Divider(height: 32),
          Text('Modelo por defecto', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          RadioGroup<String>(
            groupValue: settings.defaultModel,
            onChanged: (v) { if (v != null) settings.setDefaultModel(v); },
            child: Column(
              children: _models
                  .map((m) => RadioListTile<String>(
                        value: m.$1,
                        title: Text(m.$2),
                        contentPadding: EdgeInsets.zero,
                      ))
                  .toList(),
            ),
          ),
          const Divider(height: 32),
          const _InfoTile(
            icon: Icons.cloud_outlined,
            text: 'Los modelos cloud (wav2vec) requieren configurar CLOUD_API_URL en el servidor RPi5.',
          ),
          const SizedBox(height: 8),
          const _InfoTile(
            icon: Icons.lock_outline,
            text: 'Los audios se procesan localmente en el servidor y no se almacenan.',
          ),
        ],
      ),
    );
  }
}

class _InfoTile extends StatelessWidget {
  final IconData icon;
  final String text;

  const _InfoTile({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) => Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: Colors.grey, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text, style: const TextStyle(color: Colors.grey, fontSize: 13)),
          ),
        ],
      );
}
