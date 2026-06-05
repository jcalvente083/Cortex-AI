import 'dart:convert';
import 'package:flutter/material.dart';
import '../../core/theme.dart';

class GradCamWidget extends StatelessWidget {
  final Map<String, String?> gradCamPorActividad;

  const GradCamWidget({super.key, required this.gradCamPorActividad});

  static const _labels = {
    'vocal':      'Vocal',
    'frase':      'Frase',
    'espontanea': 'Espontánea',
  };

  @override
  Widget build(BuildContext context) {
    final tabs = _labels.keys
        .where((k) => gradCamPorActividad.containsKey(k))
        .toList();

    if (tabs.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(16),
        child: Text('Grad-CAM no disponible.', textAlign: TextAlign.center),
      );
    }

    return DefaultTabController(
      length: tabs.length,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TabBar(
            labelColor: AppTheme.primary,
            indicatorColor: AppTheme.primary,
            tabs: tabs.map((k) => Tab(text: _labels[k])).toList(),
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 240,
            child: TabBarView(
              children: tabs.map((k) => _GradCamTab(b64: gradCamPorActividad[k])).toList(),
            ),
          ),
          const SizedBox(height: 8),
          _ColorScaleLegend(),
          const SizedBox(height: 6),
          Text(
            'El mapa de calor indica qué regiones tiempo-frecuencia del espectrograma '
            'activaron más el modelo. Rojo = alta activación → indica Parkinson.',
            style: TextStyle(fontSize: 11, color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }
}

class _GradCamTab extends StatelessWidget {
  final String? b64;
  const _GradCamTab({required this.b64});

  @override
  Widget build(BuildContext context) {
    if (b64 == null || b64!.isEmpty) {
      return const Center(child: Text('No disponible para este audio'));
    }
    final bytes = base64Decode(b64!);
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image.memory(
        bytes,
        fit: BoxFit.fill,
        gaplessPlayback: true,
      ),
    );
  }
}

class _ColorScaleLegend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text('Baja activación', style: TextStyle(fontSize: 10, color: Colors.grey[600])),
        const SizedBox(width: 6),
        Expanded(
          child: Container(
            height: 10,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(5),
              gradient: const LinearGradient(
                colors: [Color(0xFF0000FF), Color(0xFF00FFFF), Color(0xFF00FF00),
                         Color(0xFFFFFF00), Color(0xFFFF0000)],
              ),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Text('Alta activación', style: TextStyle(fontSize: 10, color: Colors.grey[600])),
      ],
    );
  }
}
