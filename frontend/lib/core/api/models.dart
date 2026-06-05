class FeatureInfo {
  final String nombre;
  final double valor;
  final double contribucion;
  final String direccion;

  const FeatureInfo({
    required this.nombre,
    required this.valor,
    required this.contribucion,
    required this.direccion,
  });

  factory FeatureInfo.fromJson(Map<String, dynamic> j) => FeatureInfo(
        nombre: j['nombre'] as String,
        valor: (j['valor'] as num).toDouble(),
        contribucion: (j['contribucion'] as num).toDouble(),
        direccion: j['direccion'] as String,
      );
}

class Explicabilidad {
  final bool disponible;
  final String? tipo;
  final List<FeatureInfo> features;
  final double? baseValue;

  const Explicabilidad({
    required this.disponible,
    this.tipo,
    this.features = const [],
    this.baseValue,
  });

  factory Explicabilidad.fromJson(Map<String, dynamic> j) => Explicabilidad(
        disponible: j['disponible'] as bool,
        tipo: j['tipo'] as String?,
        features: (j['features'] as List<dynamic>? ?? [])
            .map((e) => FeatureInfo.fromJson(e as Map<String, dynamic>))
            .toList(),
        baseValue: j['base_value'] == null ? null : (j['base_value'] as num).toDouble(),
      );
}

class PredictionResult {
  final double probabilidadPd;
  final String prediccion;
  final String nivelRiesgo;
  final double umbral;
  final String modelo;
  final String actividad;
  final Explicabilidad explicabilidad;

  const PredictionResult({
    required this.probabilidadPd,
    required this.prediccion,
    required this.nivelRiesgo,
    required this.umbral,
    required this.modelo,
    required this.actividad,
    required this.explicabilidad,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> j) => PredictionResult(
        probabilidadPd: (j['probabilidad_pd'] as num).toDouble(),
        prediccion: j['prediccion'] as String,
        nivelRiesgo: j['nivel_riesgo'] as String,
        umbral: (j['umbral'] as num).toDouble(),
        modelo: j['modelo'] as String,
        actividad: j['actividad'] as String,
        explicabilidad: Explicabilidad.fromJson(
            j['explicabilidad'] as Map<String, dynamic>? ?? {'disponible': false}),
      );
}

class BatchPredictionResult {
  final double probabilidadPdFinal;
  final String prediccion;
  final String nivelRiesgo;
  final double umbralPromedio;
  final String modelo;
  final Map<String, double> detallePorActividad;
  final Explicabilidad? explicabilidad;
  final Map<String, String?>? gradCamPorActividad;

  const BatchPredictionResult({
    required this.probabilidadPdFinal,
    required this.prediccion,
    required this.nivelRiesgo,
    required this.umbralPromedio,
    required this.modelo,
    required this.detallePorActividad,
    this.explicabilidad,
    this.gradCamPorActividad,
  });

  factory BatchPredictionResult.fromJson(Map<String, dynamic> j) => BatchPredictionResult(
        probabilidadPdFinal: (j['probabilidad_pd_final'] as num).toDouble(),
        prediccion: j['prediccion'] as String,
        nivelRiesgo: j['nivel_riesgo'] as String,
        umbralPromedio: (j['umbral_promedio'] as num).toDouble(),
        modelo: j['modelo'] as String,
        detallePorActividad: (j['detalle_por_actividad'] as Map<String, dynamic>).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
        explicabilidad: j['explicabilidad'] == null
            ? null
            : Explicabilidad.fromJson(j['explicabilidad'] as Map<String, dynamic>),
        gradCamPorActividad: j['grad_cam_por_actividad'] == null
            ? null
            : (j['grad_cam_por_actividad'] as Map<String, dynamic>).map(
                (k, v) => MapEntry(k, v as String?),
              ),
      );
}
