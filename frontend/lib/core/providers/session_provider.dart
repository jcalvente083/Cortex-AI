import 'dart:io';
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

  SessionStatus get status       => _status;
  BatchPredictionResult? get result => _result;
  String? get errorMessage       => _errorMessage;

  final slots = [
    RecordingSlot(label: 'Vocal',      instruction: 'Sostén la vocal /a/ durante 5 segundos'),
    RecordingSlot(label: 'Frase',      instruction: 'Lee en voz alta: "El cielo está despejado hoy"'),
    RecordingSlot(label: 'Espontánea', instruction: 'Describe lo que hiciste esta mañana (30 seg)'),
  ];

  bool get allRecorded => slots.every((s) => s.file != null);

  void setFile(int index, File file) {
    slots[index].file = file;
    if (allRecorded) {
      _status = SessionStatus.recorded;
    }
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
