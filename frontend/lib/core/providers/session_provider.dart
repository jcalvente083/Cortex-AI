import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import '../api/api_client.dart';
import '../api/models.dart';

enum SessionStatus { idle, recording, recorded, analyzing, done, error }

class RecordingSlot {
  final String label;
  final String instruction;
  File? file;
  bool isRecording = false;

  RecordingSlot({required this.label, required this.instruction});
}

class SessionProvider extends ChangeNotifier {
  SessionStatus _status = SessionStatus.idle;
  BatchPredictionResult? _result;
  String? _errorMessage;
  final int sceneIndex;

  SessionStatus get status          => _status;
  BatchPredictionResult? get result => _result;
  String? get errorMessage          => _errorMessage;

  late final List<RecordingSlot> slots;

  static const _phrases = [
    'El cielo está despejado hoy',
    'Los pájaros cantan por las mañanas',
    'María compró pan fresco en la panadería',
    'El niño juega en el parque con su perro',
    'La lluvia cae suavemente sobre el tejado',
    'Hoy hace un día muy soleado y agradable',
    'Me gusta caminar por el campo en primavera',
    'El tren llegó puntual a la estación central',
    'Ana prepara una taza de té todas las tardes',
    'El libro estaba encima de la mesa del salón',
    'Las flores del jardín huelen muy bien en abril',
    'Pedro sale a correr todos los días por la mañana',
  ];

  SessionProvider() : sceneIndex = Random().nextInt(4) {
    final phrase = _phrases[Random().nextInt(_phrases.length)];
    slots = [
      RecordingSlot(
        label:       'Vocal',
        instruction: 'Sostén la vocal /a/ durante 5 segundos',
      ),
      RecordingSlot(
        label:       'Frase',
        instruction: 'Lee en voz alta: "$phrase"',
      ),
      RecordingSlot(
        label:       'Espontánea',
        instruction: 'Describe la escena que aparece arriba durante 30 segundos',
      ),
    ];
  }

  bool get allRecorded => slots.every((s) => s.file != null);

  void setFile(int index, File file) {
    slots[index].file = file;
    if (allRecorded) _status = SessionStatus.recorded;
    notifyListeners();
  }

  void reset() {
    _status = SessionStatus.idle;
    _result = null;
    _errorMessage = null;
    for (final s in slots) {
      s.file = null;
      s.isRecording = false;
    }
    notifyListeners();
  }

  Future<void> analyze(ApiClient client, String modelo) async {
    if (!allRecorded) return;
    _status = SessionStatus.analyzing;
    _errorMessage = null;
    notifyListeners();

    try {
      _result = await client.predictBatch(
        audioVocal:      slots[0].file!,
        audioFrase:      slots[1].file!,
        audioEspontanea: slots[2].file!,
        modelo:          modelo,
      );
      _status = SessionStatus.done;
    } catch (e) {
      _errorMessage = e.toString();
      _status = SessionStatus.error;
    }
    notifyListeners();
  }
}
