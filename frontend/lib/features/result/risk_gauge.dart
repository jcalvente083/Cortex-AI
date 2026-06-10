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
    final radius = size.width / 2 - 14;
    final rect   = Rect.fromCircle(center: center, radius: radius);

    final isLow  = probability < threshold * 0.6;
    final isMid  = probability >= threshold * 0.6 && probability < threshold;
    final isHigh = probability >= threshold;

    final activeColor = isHigh ? AppTheme.danger : (isMid ? AppTheme.warning : AppTheme.success);

    final track = Paint()
      ..style      = PaintingStyle.stroke
      ..strokeWidth = 20
      ..strokeCap  = StrokeCap.butt;

    // Green: 0 → threshold*0.6
    track.color = isLow
        ? AppTheme.success
        : AppTheme.success.withValues(alpha: 0.18);
    canvas.drawArc(rect, pi, pi * threshold * 0.6, false, track);

    // Yellow: threshold*0.6 → threshold
    track.color = isMid
        ? AppTheme.warning
        : AppTheme.warning.withValues(alpha: 0.18);
    canvas.drawArc(rect, pi + pi * threshold * 0.6, pi * threshold * 0.4, false, track);

    // Red: threshold → 1
    track.color = isHigh
        ? AppTheme.danger
        : AppTheme.danger.withValues(alpha: 0.18);
    canvas.drawArc(rect, pi + pi * threshold, pi * (1 - threshold), false, track);

    // Needle pointing to exact probability position
    final angle      = pi + pi * probability;
    final needleLen  = radius - 8;
    final needleEnd  = Offset(
      center.dx + needleLen * cos(angle),
      center.dy + needleLen * sin(angle),
    );
    canvas.drawLine(
      center,
      needleEnd,
      Paint()
        ..color      = activeColor
        ..strokeWidth = 3.5
        ..strokeCap  = StrokeCap.round,
    );

    // Center dot: colored ring + white fill
    canvas.drawCircle(center, 8, Paint()..color = activeColor);
    canvas.drawCircle(center, 4, Paint()..color = Colors.white);
  }

  @override
  bool shouldRepaint(_GaugePainter old) =>
      old.probability != probability || old.threshold != threshold;
}
