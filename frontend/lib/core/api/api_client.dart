import 'dart:io';
import 'package:dio/dio.dart';
import 'models.dart';

class ApiClient {
  late final Dio _dio;

  ApiClient(String baseUrl) {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 120),
    ));
  }

  Future<Map<String, dynamic>> health() async {
    final r = await _dio.get('/health');
    return r.data as Map<String, dynamic>;
  }

  Future<BatchPredictionResult> predictBatch({
    required File audioVocal,
    required File audioFrase,
    required File audioEspontanea,
    required String modelo,
  }) async {
    final form = FormData.fromMap({
      'audio_vocal': await MultipartFile.fromFile(audioVocal.path, filename: 'vocal.wav'),
      'audio_frase': await MultipartFile.fromFile(audioFrase.path, filename: 'frase.wav'),
      'audio_espontanea': await MultipartFile.fromFile(audioEspontanea.path, filename: 'espontanea.wav'),
      'modelo': modelo,
    });
    final r = await _dio.post('/predict/batch', data: form);
    return BatchPredictionResult.fromJson(r.data as Map<String, dynamic>);
  }

  Future<PredictionResult> predictSingle({
    required File audio,
    required String modelo,
    required String actividad,
  }) async {
    final form = FormData.fromMap({
      'audio': await MultipartFile.fromFile(audio.path, filename: 'audio.wav'),
      'modelo': modelo,
      'actividad': actividad,
    });
    final r = await _dio.post('/predict', data: form);
    return PredictionResult.fromJson(r.data as Map<String, dynamic>);
  }
}
