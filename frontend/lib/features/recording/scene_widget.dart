import 'package:flutter/material.dart';
import '../../core/theme.dart';

class SceneWidget extends StatelessWidget {
  final int sceneIndex;
  const SceneWidget({super.key, required this.sceneIndex});

  static const _scenes = [
    _Scene(
      imagePath: 'assets/Descripcion/1.png',
      label: 'Una señora mayor junto a la ventana con su gato, mirando el pueblo andaluz',
    ),
    _Scene(
      imagePath: 'assets/Descripcion/2.png',
      label: 'Abuela preparando el almuerzo en la cocina mientras el gato la observa',
    ),
    _Scene(
      imagePath: 'assets/Descripcion/3.png',
      label: 'Abuela leyendo tranquilamente en su sillón con el gato dormido en su regazo',
    ),
    _Scene(
      imagePath: 'assets/Descripcion/4.png',
      label: 'Abuela acariciando a su gato en la mesa del desayuno con tostadas y café',
    ),
    _Scene(
      imagePath: 'assets/Descripcion/5.png',
      label: 'Mesa del desayuno: libro abierto, gafas, taza de café y tostadas con tomate',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final scene = _scenes[sceneIndex % _scenes.length];
    return Container(
      width: double.infinity,
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
      child: ClipRRect(
        borderRadius: BorderRadius.circular(15),
        child: Column(
          children: [
            Image.asset(
              scene.imagePath,
              width: double.infinity,
              height: 180,
              fit: BoxFit.cover,
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Text(
                scene.label,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.grey[700],
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Scene {
  final String imagePath;
  final String label;
  const _Scene({required this.imagePath, required this.label});
}
