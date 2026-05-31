import 'dart:math';
import 'package:flutter/material.dart';
import '../../core/theme.dart';

class RiskGauge extends StatelessWidget {
  final double probability;
  final double threshold;
  final String label;

  const RiskGauge({
    super.key,
    required this.probability,
    required this.threshold,
    required this.label,
  });

  Color get _color {
    if (probability >= threshold) return AppTheme.danger;
    if (probability >= threshold * 0.6) return AppTheme.warning;
    return AppTheme.success;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          width: 240,
          height: 130,
          child: CustomPaint(
            painter: _GaugePainter(probability: probability, threshold: threshold),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context)
              .textTheme
              .headlineSmall
              ?.copyWith(color: _color, fontWeight: FontWeight.bold),
        ),
        Text(
          '${(probability * 100).toStringAsFixed(1)}% probabilidad',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
        ),
      ],
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double probability;
  final double threshold;

  const _GaugePainter({required this.probability, required this.threshold});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height);
    final radius = size.width / 2 - 10;

    // Arc segments: low (green), medium (yellow), high (red)
    final segmentPaint = Paint()..style = PaintingStyle.stroke..strokeWidth = 18..strokeCap = StrokeCap.butt;

    final rect = Rect.fromCircle(center: center, radius: radius);

    // Green: 0–60% of threshold
    segmentPaint.color = AppTheme.success.withValues(alpha: 0.25);
    canvas.drawArc(rect, pi, pi * 0.6, false, segmentPaint);

    // Yellow: 60–100% of threshold
    segmentPaint.color = AppTheme.warning.withValues(alpha: 0.25);
    canvas.drawArc(rect, pi + pi * 0.6, pi * 0.4 * threshold, false, segmentPaint);

    // Red: above threshold
    segmentPaint.color = AppTheme.danger.withValues(alpha: 0.25);
    canvas.drawArc(rect, pi + pi * threshold, pi * (1 - threshold), false, segmentPaint);

    // Needle
    final angle = pi + pi * probability;
    final needleEnd = Offset(
      center.dx + radius * cos(angle),
      center.dy + radius * sin(angle),
    );
    final needlePaint = Paint()
      ..color = Colors.black87
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(center, needleEnd, needlePaint);

    // Center dot
    canvas.drawCircle(center, 6, Paint()..color = Colors.black87);
  }

  @override
  bool shouldRepaint(_GaugePainter old) =>
      old.probability != probability || old.threshold != threshold;
}
