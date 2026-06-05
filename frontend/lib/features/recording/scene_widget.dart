import 'package:flutter/material.dart';
import '../../core/theme.dart';

class SceneWidget extends StatelessWidget {
  final int sceneIndex;
  const SceneWidget({super.key, required this.sceneIndex});

  static const _scenes = [
    _Scene(
      emoji: '🌳  🌞  👨‍👩‍👧  🐕  🏡',
      label: 'Un parque tranquilo con árboles, sol y una familia paseando con su perro',
    ),
    _Scene(
      emoji: '🍳  👩‍🍳  🥘  🪟  🌿',
      label: 'Una cocina luminosa con alguien cocinando junto a la ventana con plantas',
    ),
    _Scene(
      emoji: '🚂  👴  📰  ☕  🧳',
      label: 'Un anciano en un tren leyendo el periódico con su café y una maleta',
    ),
    _Scene(
      emoji: '🏖️  👧  🐚  ⛱️  🌊',
      label: 'Una niña en la playa recogiendo conchas junto al mar bajo una sombrilla',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final scene = _scenes[sceneIndex % _scenes.length];
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppTheme.primary.withValues(alpha: 0.1),
            AppTheme.secondary.withValues(alpha: 0.06),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primary.withValues(alpha: 0.25)),
      ),
      child: Column(
        children: [
          Text(scene.emoji,
              style: const TextStyle(fontSize: 34),
              textAlign: TextAlign.center),
          const SizedBox(height: 10),
          Text(
            scene.label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 13,
              color: Colors.grey[700],
              fontStyle: FontStyle.italic,
            ),
          ),
        ],
      ),
    );
  }
}

class _Scene {
  final String emoji;
  final String label;
  const _Scene({required this.emoji, required this.label});
}
