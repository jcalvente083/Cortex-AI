import 'package:flutter/material.dart';
import '../../core/theme.dart';
import '../recording/recording_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cortex-AI')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Spacer(),
              Center(
                child: Image.asset('assets/logo.png', width: 120, height: 120),
              ),
              const SizedBox(height: 20),
              Text(
                'Detección de Parkinson por voz',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: AppTheme.primary,
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                'Análisis acústico de la voz mediante inteligencia artificial',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
              ),
              const Spacer(),
              _StepCard(number: '1', text: 'Graba los 3 audios pedidos'),
              const SizedBox(height: 10),
              _StepCard(number: '2', text: 'El modelo analiza tu voz'),
              const SizedBox(height: 10),
              _StepCard(number: '3', text: 'Recibe un resultado orientativo'),
              const Spacer(),
              ElevatedButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const RecordingScreen()),
                ),
                icon: const Icon(Icons.mic),
                label: const Text('Iniciar análisis'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StepCard extends StatelessWidget {
  final String number;
  final String text;

  const _StepCard({required this.number, required this.text});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          CircleAvatar(
            radius: 18,
            backgroundColor: AppTheme.primary,
            foregroundColor: Colors.white,
            child: Text(number, style: const TextStyle(fontWeight: FontWeight.bold)),
          ),
          const SizedBox(width: 12),
          Text(text, style: Theme.of(context).textTheme.bodyLarge),
        ],
      );
}
