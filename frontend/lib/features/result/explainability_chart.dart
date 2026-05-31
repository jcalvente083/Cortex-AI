import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../../core/api/models.dart';
import '../../core/theme.dart';

class ExplainabilityChart extends StatelessWidget {
  final Explicabilidad explicabilidad;

  const ExplainabilityChart({super.key, required this.explicabilidad});

  @override
  Widget build(BuildContext context) {
    if (!explicabilidad.disponible || explicabilidad.features.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(16),
        child: Text('Explicabilidad no disponible para este modelo.',
            textAlign: TextAlign.center),
      );
    }

    final features = explicabilidad.features.take(6).toList();
    final maxAbs = features
        .map((f) => f.contribucion.abs())
        .fold(0.0, (prev, v) => v > prev ? v : prev);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          explicabilidad.tipo == 'shap' ? 'Valores SHAP' : 'Desviación de features',
          style: Theme.of(context)
              .textTheme
              .titleSmall
              ?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 200,
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              barTouchData: BarTouchData(
                touchTooltipData: BarTouchTooltipData(
                  getTooltipItem: (group, groupIndex, rod, rodIndex) {
                    final f = features[group.x];
                    return BarTooltipItem(
                      '${f.nombre}\n${rod.toY.toStringAsFixed(3)}',
                      const TextStyle(color: Colors.white, fontSize: 11),
                    );
                  },
                ),
              ),
              titlesData: FlTitlesData(
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 48,
                    getTitlesWidget: (value, _) {
                      final i = value.toInt();
                      if (i < 0 || i >= features.length) return const SizedBox.shrink();
                      final name = features[i].nombre;
                      final short = name.length > 8 ? name.substring(0, 8) : name;
                      return Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: RotatedBox(
                          quarterTurns: 1,
                          child: Text(short, style: const TextStyle(fontSize: 10)),
                        ),
                      );
                    },
                  ),
                ),
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 36,
                    getTitlesWidget: (v, _) =>
                        Text(v.toStringAsFixed(2), style: const TextStyle(fontSize: 9)),
                  ),
                ),
                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                topTitles:   const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              ),
              gridData: FlGridData(
                drawHorizontalLine: true,
                getDrawingHorizontalLine: (_) =>
                    FlLine(color: Colors.grey[300]!, strokeWidth: 1),
                drawVerticalLine: false,
              ),
              borderData: FlBorderData(show: false),
              minY: -maxAbs * 1.3,
              maxY:  maxAbs * 1.3,
              barGroups: List.generate(features.length, (i) {
                final f = features[i];
                final color = f.contribucion > 0 ? AppTheme.danger : AppTheme.success;
                return BarChartGroupData(
                  x: i,
                  barRods: [
                    BarChartRodData(
                      toY: f.contribucion,
                      fromY: 0,
                      color: color,
                      width: 18,
                      borderRadius: f.contribucion >= 0
                          ? const BorderRadius.vertical(top: Radius.circular(4))
                          : const BorderRadius.vertical(bottom: Radius.circular(4)),
                    ),
                  ],
                );
              }),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _Legend(color: AppTheme.danger, label: 'Indica Parkinson'),
            const SizedBox(width: 16),
            _Legend(color: AppTheme.success, label: 'Indica Control'),
          ],
        ),
      ],
    );
  }
}

class _Legend extends StatelessWidget {
  final Color color;
  final String label;
  const _Legend({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Container(
            width: 14,
            height: 14,
            decoration: BoxDecoration(
                color: color, borderRadius: BorderRadius.circular(3)),
          ),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(fontSize: 11)),
        ],
      );
}
