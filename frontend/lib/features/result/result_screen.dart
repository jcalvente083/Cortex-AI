import 'package:flutter/material.dart';
import '../../core/api/models.dart';
import '../../core/theme.dart';
import 'risk_gauge.dart';
class ResultScreen extends StatelessWidget {
  final BatchPredictionResult result;

  const ResultScreen({super.key, required this.result});

  Color get _riskColor {
    switch (result.nivelRiesgo) {
      case 'Alto':   return AppTheme.danger;
      case 'Medio':  return AppTheme.warning;
      default:       return AppTheme.success;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Resultado del análisis')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    RiskGauge(
                      probability: result.probabilidadPdFinal,
                      threshold:   result.umbralPromedio,
                      label:       result.nivelRiesgo,
                    ),
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
                      decoration: BoxDecoration(
                        color: _riskColor.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(color: _riskColor),
                      ),
                      child: Text(
                        result.prediccion,
                        style: TextStyle(
                            color: _riskColor,
                            fontWeight: FontWeight.bold,
                            fontSize: 20),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Detalle por actividad',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    ...result.detallePorActividad.entries.map((e) => _DetailRow(
                          label: e.key,
                          prob:  e.value,
                          umbral: result.umbralPromedio,
                        )),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Información del análisis',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    _InfoRow('Modelo',  result.modelo),
                    _InfoRow('Umbral',  result.umbralPromedio.toStringAsFixed(4)),
                    _InfoRow('Prob. PD', '${(result.probabilidadPdFinal * 100).toStringAsFixed(1)}%'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.amber[50],
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.amber),
              ),
              child: const Text(
                'AVISO LEGAL: Este resultado es orientativo y no constituye un diagnóstico médico. '
                'Consulte siempre a un profesional de la salud.',
                style: TextStyle(fontSize: 12, color: Colors.brown),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: () => Navigator.popUntil(context, (r) => r.isFirst),
              icon: const Icon(Icons.home),
              label: const Text('Volver al inicio'),
            ),
          ],
        ),
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final double prob;
  final double umbral;

  const _DetailRow({required this.label, required this.prob, required this.umbral});

  @override
  Widget build(BuildContext context) {
    final color = prob >= umbral ? AppTheme.danger : AppTheme.success;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(child: Text(label[0].toUpperCase() + label.substring(1))),
          SizedBox(
            width: 160,
            child: LinearProgressIndicator(
              value: prob,
              backgroundColor: Colors.grey[200],
              color: color,
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
            ),
          ),
          const SizedBox(width: 8),
          Text('${(prob * 100).toStringAsFixed(1)}%',
              style: TextStyle(color: color, fontWeight: FontWeight.w600, fontSize: 12)),
        ],
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String info;
  const _InfoRow(this.label, this.info);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          children: [
            Expanded(child: Text(label, style: const TextStyle(color: Colors.grey))),
            Text(info, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
