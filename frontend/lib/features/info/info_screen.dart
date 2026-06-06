import 'package:flutter/material.dart';
import '../../core/theme.dart';

class InfoScreen extends StatelessWidget {
  const InfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Acerca de Cortex-AI')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Center(
            child: Image.asset('assets/logo.png', width: 100, height: 100),
          ),
          const SizedBox(height: 16),
          Text('Cortex-AI',
              textAlign: TextAlign.center,
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.bold, color: AppTheme.primary)),
          Text('v1.0.0 — TFG Ingeniería Informática',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
          const SizedBox(height: 24),
          _Section(
            title: 'Descripción',
            content:
                'Cortex-AI es una aplicación de análisis de voz para la detección temprana '
                'de la enfermedad de Parkinson. Utiliza modelos de aprendizaje automático '
                'entrenados sobre features acústicas y espectrogramas de mel.',
          ),
          _Section(
            title: 'Modelos disponibles',
            content: '• KNN (K-Nearest Neighbors): features acústicas clásicas.\n'
                '• XGBoost: gradient boosting con explicabilidad SHAP.\n'
                '• ResNet18: mel-espectrogramas con visión por computador.\n'
                '• Wav2Vec2 Embeddings: embeddings de audio con transformers (cloud).\n'
                '• Wav2Vec2 Fine-tune: fine-tuning end-to-end (cloud).',
          ),
          _Section(
            title: 'Protocolo de grabación',
            content: '1. Vocal sostenida /a/ durante 5 segundos.\n'
                '2. Lectura de una frase corta.\n'
                '3. Habla espontánea durante ~30 segundos.',
          ),
          _Section(
            title: 'Aviso legal',
            content:
                'Esta aplicación es un prototipo académico y NO sustituye el diagnóstico médico. '
                'Los resultados son orientativos. Consulte siempre a un neurólogo.',
          ),
          const SizedBox(height: 16),
          const Text('Desarrollado por Jesús David Calvente Zapata',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, fontSize: 12)),
          const SizedBox(height: 2),
          const Text('Trabajo de Fin de Grado — Ingeniería Informática',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, fontSize: 11)),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  final String title;
  final String content;

  const _Section({required this.title, required this.content});

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 16),
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.bold, color: AppTheme.primary)),
          const SizedBox(height: 6),
          Text(content, style: Theme.of(context).textTheme.bodyMedium),
        ],
      );
}
