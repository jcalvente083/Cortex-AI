import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

void main() {
  runApp(const ParkinsonDiagnosisApp());
}

class ParkinsonDiagnosisApp extends StatelessWidget {
  const ParkinsonDiagnosisApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Diagnóstico por Voz',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      // --- CONFIGURACIÓN DE IDIOMA PARA LA FECHA (dd/mm/aaaa) ---
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: const [
        Locale('es', 'ES'), // Español
      ],
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  String _textoALeer = "Cargando texto...";
  DateTime? _fechaNacimiento;
  int? _sexo;

  final _audioRecorder = AudioRecorder();
  bool _isRecording = false;
  String? _audioPath;

  final String _apiUrl = "http://api.jcalvente083.qzz.io/diagnostico";

  @override
  void initState() {
    super.initState();
    _loadText();
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    super.dispose();
  }

  Future<void> _loadText() async {
    try {
      final String response = await rootBundle.loadString('assets/textos.txt');
      List<String> frases = response.split('\n');
      frases = frases.where((frase) => frase.trim().isNotEmpty).toList();

      if (frases.isNotEmpty) {
        final random = Random();
        setState(() {
          _textoALeer = frases[random.nextInt(frases.length)].trim();
        });
      }
    } catch (e) {
      setState(() {
        _textoALeer = "Error al cargar los textos.";
      });
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _audioRecorder.stop();
      setState(() {
        _isRecording = false;
        _audioPath = path;
      });
    } else {
      if (await _audioRecorder.hasPermission()) {
        final dir = await getTemporaryDirectory();
        final path = '${dir.path}/grabacion.wav';
        await _audioRecorder.start(const RecordConfig(encoder: AudioEncoder.wav), path: path);
        setState(() {
          _isRecording = true;
          _audioPath = null; // Reiniciamos el audio si graba de nuevo
        });
      }
    }
  }

  Future<void> _selectDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now().subtract(const Duration(days: 365 * 50)),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
      // El formato dd/mm/aaaa ya lo aplica por el Locale en MaterialApp
    );
    if (picked != null) {
      setState(() {
        _fechaNacimiento = picked;
      });
    }
  }

  int _calcularEdad(DateTime birthDate) {
    DateTime today = DateTime.now();
    int age = today.year - birthDate.year;
    if (today.month < birthDate.month || (today.month == birthDate.month && today.day < birthDate.day)) {
      age--;
    }
    return age;
  }

  // --- AVISO MÉDICO ---
  void _mostrarAvisoMedico() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orange),
            SizedBox(width: 10),
            Text("Aviso Legal y Médico"),
          ],
        ),
        content: const Text(
          "Esta aplicación es una herramienta experimental de apoyo "
              "basada en Inteligencia Artificial.\n\n"
              "Bajo ninguna circunstancia el resultado obtenido suple el diagnóstico, "
              "criterio o tratamiento de un médico o especialista colegiado.\n\n"
              "Privacidad: Los datos de voz y edad introducidos son analizados "
              "en tiempo real y NO son almacenados ni guardados en nuestros servidores.",
          textAlign: TextAlign.justify,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Entendido"),
          )
        ],
      ),
    );
  }

  Future<void> _enviarDatos() async {
    if (_audioPath == null || _fechaNacimiento == null || _sexo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Por favor completa todos los campos y graba el audio")),
      );
      return;
    }

    final edad = _calcularEdad(_fechaNacimiento!);

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const AlertDialog(
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Text("Esperando la respuesta del modelo..."),
          ],
        ),
      ),
    );

    try {
      var request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.fields['edad'] = edad.toString();
      request.fields['sexo'] = _sexo.toString();
      request.files.add(await http.MultipartFile.fromPath('audio', _audioPath!));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (mounted) Navigator.pop(context); // Cierra el loading

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);

        if (mounted) {
          // Navegamos a la pantalla de resultados y ESPERAMOS a que vuelva
          await Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultScreen(data: responseData),
            ),
          );

          // Cuando el usuario le da a "Atrás", se ejecuta esto: limpiamos el audio.
          setState(() {
            _audioPath = null;
          });
        }
      } else {
        throw Exception("Error del servidor: ${response.statusCode}");
      }
    } catch (e) {
      if (mounted) Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error de conexión: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Formulario de Análisis"),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: "Aviso Médico",
            onPressed: _mostrarAvisoMedico,
          )
        ],
      ),
      // Usamos Center para maximizar el centrado visual si la pantalla es grande
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center, // Centramos los elementos
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  _textoALeer,
                  style: const TextStyle(fontSize: 18, fontStyle: FontStyle.italic),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 20),

              ElevatedButton.icon(
                onPressed: _toggleRecording,
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(_isRecording ? "Detener Grabación" : "Grabar Texto"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.red.shade100 : null,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                ),
              ),
              if (_audioPath != null) ...[
                const SizedBox(height: 8),
                const Text("✅ Audio grabado correctamente", textAlign: TextAlign.center, style: TextStyle(color: Colors.green)),
              ],

              const Divider(height: 40),

              ListTile(
                title: const Text("Fecha de Nacimiento"),
                subtitle: Text(_fechaNacimiento == null
                    ? "No seleccionada"
                    : DateFormat('dd/MM/yyyy').format(_fechaNacimiento!)),
                trailing: const Icon(Icons.calendar_month),
                onTap: _selectDate,
                tileColor: Colors.grey[100],
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              const SizedBox(height: 15),

              DropdownButtonFormField<int>(
                decoration: InputDecoration(
                  labelText: "Sexo Biológico",
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                ),
                value: _sexo,
                items: const [
                  DropdownMenuItem(value: 0, child: Text("Femenino")),
                  DropdownMenuItem(value: 1, child: Text("Masculino")),
                ],
                onChanged: (val) => setState(() => _sexo = val),
              ),
              const SizedBox(height: 40),

              FilledButton(
                onPressed: _enviarDatos,
                style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 15)),
                child: const Text("Analizar y Diagnosticar", style: TextStyle(fontSize: 18)),
              )
            ],
          ),
        ),
      ),
    );
  }
}

// --------------------- PANTALLA DE RESULTADOS ---------------------

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultScreen({super.key, required this.data});

  String _traducirVariableSHAP(String rawFeature) {
    const Map<String, String> diccionario = {
      'Age': 'Edad del paciente',
      'Sex': 'Sexo biológico',
      'ShimmerDb': 'Inestabilidad del volumen (Shimmer dB)',
      'ATRI': 'Índice de Temblor Acústico (ATRI)',
      'Hnr': 'Calidad de voz: Armónico-Ruido (HNR)',
      'CHNR': 'Calidad de voz Cepstral (CHNR)',
      'rPPQ': 'Inestabilidad del tono (rPPQ)',
    };
    return diccionario[rawFeature] ?? rawFeature;
  }

  @override
  Widget build(BuildContext context) {
    final String diagnostico = data['diagnostico'] ?? "Desconocido";
    final double probEnfermo = data['probabilidad_parkinson_pct'] ?? 0.0;

    // --- LÓGICA DE PROBABILIDAD ---
    // Determinamos si es sano o enfermo basándonos en la palabra clave del diagnóstico
    final bool esSano = diagnostico.toLowerCase().contains("sano") || diagnostico.toLowerCase().contains("negativo");

    // Si es sano, mostramos la probabilidad de estar sano (100 - prob_enfermo)
    final double probabilidadMostrar = esSano ? (100.0 - probEnfermo) : probEnfermo;

    final shapData = data['shap_explicabilidad'] ?? {};
    final impactoVariables = shapData['impacto_variables'] as Map<String, dynamic>? ?? {};

    double maxAbsValue = 0.0;
    impactoVariables.forEach((key, value) {
      double absVal = (value is num ? value.toDouble() : 0.0).abs();
      if (absVal > maxAbsValue) maxAbsValue = absVal;
    });
    if (maxAbsValue == 0) maxAbsValue = 1.0;

    return Scaffold(
      appBar: AppBar(title: const Text("Resultados del Diagnóstico")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            Text(
              diagnostico.toUpperCase(),
              style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: esSano ? Colors.green : Colors.red
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            Text(
              "Confianza del diagnóstico: ${probabilidadMostrar.toStringAsFixed(2)}%",
              style: const TextStyle(fontSize: 18, color: Colors.grey),
            ),
            const Divider(height: 40, thickness: 2),

            const Text(
              "Variables más influyentes",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 5),
            const Text(
                "Verde: Indicador de salud | Rojo: Indicador de alerta",
                style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic)
            ),
            const SizedBox(height: 20),

            Expanded(
              child: ListView.builder(
                itemCount: impactoVariables.length,
                itemBuilder: (context, index) {
                  String rawFeature = impactoVariables.keys.elementAt(index);
                  double value = (impactoVariables[rawFeature] is num)
                      ? (impactoVariables[rawFeature] as num).toDouble()
                      : 0.0;

                  String nombreLegible = _traducirVariableSHAP(rawFeature);
                  return _buildShapBar(nombreLegible, value, maxAbsValue);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildShapBar(String feature, double value, double maxAbsValue) {
    bool isPositive = value >= 0;
    double flexPct = (value.abs() / maxAbsValue).clamp(0.0, 1.0);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(feature, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15)),
          const SizedBox(height: 4),
          Row(
            children: [
              Expanded(
                child: Align(
                  alignment: Alignment.centerRight,
                  child: !isPositive
                      ? FractionallySizedBox(
                    widthFactor: flexPct,
                    child: Container(
                      height: 12,
                      decoration: const BoxDecoration(
                          color: Colors.green,
                          borderRadius: BorderRadius.horizontal(left: Radius.circular(4))
                      ),
                    ),
                  )
                      : null,
                ),
              ),
              Container(width: 2, height: 20, color: Colors.black87),
              Expanded(
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: isPositive
                      ? FractionallySizedBox(
                    widthFactor: flexPct,
                    child: Container(
                      height: 12,
                      decoration: const BoxDecoration(
                          color: Colors.red,
                          borderRadius: BorderRadius.horizontal(right: Radius.circular(4))
                      ),
                    ),
                  )
                      : null,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
