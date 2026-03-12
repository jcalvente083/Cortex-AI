import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';
import 'dart:math';

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
  int? _sexo; // 0 para Mujer, 1 para Hombre (ajústalo según tu API)

  final _audioRecorder = AudioRecorder();
  bool _isRecording = false;
  String? _audioPath;

  // IMPORTANTE: Cambia esta IP según uses emulador o dispositivo físico
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

  // Cargar texto desde un txt
  Future<void> _loadText() async {
    try {
      // 1. Cargamos todo el contenido del archivo de texto
      final String response = await rootBundle.loadString('assets/textos.txt');

      // 2. Lo dividimos en una lista usando los saltos de línea
      List<String> frases = response.split('\n');

      // 3. Limpiamos posibles líneas vacías (por si dejas un "Enter" al final del archivo)
      frases = frases.where((frase) => frase.trim().isNotEmpty).toList();

      if (frases.isNotEmpty) {
        // 4. Elegimos una frase al azar
        final random = Random();
        final fraseAleatoria = frases[random.nextInt(frases.length)];

        setState(() {
          _textoALeer = fraseAleatoria.trim();
        });
      } else {
        setState(() {
          _textoALeer = "El archivo de textos está vacío.";
        });
      }
    } catch (e) {
      setState(() {
        _textoALeer = "Error al cargar los textos.";
      });
    }
  }

  // Lógica de grabación de audio
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
          _audioPath = null;
        });
      }
    }
  }

  // Seleccionar Fecha de Nacimiento
  Future<void> _selectDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now().subtract(const Duration(days: 365 * 50)),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
    );
    if (picked != null) {
      setState(() {
        _fechaNacimiento = picked;
      });
    }
  }

  // Calcular edad
  int _calcularEdad(DateTime birthDate) {
    DateTime today = DateTime.now();
    int age = today.year - birthDate.year;
    if (today.month < birthDate.month || (today.month == birthDate.month && today.day < birthDate.day)) {
      age--;
    }
    return age;
  }

  // Enviar a la API
  Future<void> _enviarDatos() async {
    if (_audioPath == null || _fechaNacimiento == null || _sexo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Por favor completa todos los campos y graba el audio")),
      );
      return;
    }

    final edad = _calcularEdad(_fechaNacimiento!);

    // Mostrar diálogo de carga
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

      // Cerrar el diálogo de carga
      if (mounted) Navigator.pop(context);

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);
        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultScreen(data: responseData),
            ),
          );
        }
      } else {
        throw Exception("Error del servidor: ${response.statusCode}");
      }
    } catch (e) {
      // Cerrar el diálogo de carga en caso de error
      if (mounted) Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error de conexión: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Formulario de Análisis")),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Texto a leer
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

            // Botón de grabación
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
              const Text("Audio grabado correctamente", textAlign: TextAlign.center, style: TextStyle(color: Colors.green)),
            ],

            const Divider(height: 40),

            // Selector de fecha
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

            // Selector de Sexo
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

            // Botón de Enviar
            FilledButton(
              onPressed: _enviarDatos,
              style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 15)),
              child: const Text("Analizar y Diagnosticar", style: TextStyle(fontSize: 18)),
            )
          ],
        ),
      ),
    );
  }
}

// --------------------- PANTALLA DE RESULTADOS ---------------------

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final diagnostico = data['diagnostico'] ?? "Desconocido";
    final probParkinson = data['probabilidad_parkinson_pct'] ?? 0.0;

    // Extraer datos SHAP
    final shapData = data['shap_explicabilidad'] ?? {};
    final impactoVariables = shapData['impacto_variables'] as Map<String, dynamic>? ?? {};

    // Encontrar el valor absoluto máximo para escalar la barra de -100 a 100 visualmente
    double maxAbsValue = 0.0;
    impactoVariables.forEach((key, value) {
      double absVal = (value is num ? value.toDouble() : 0.0).abs();
      if (absVal > maxAbsValue) maxAbsValue = absVal;
    });
    // Evitar división por cero
    if (maxAbsValue == 0) maxAbsValue = 1.0;

    return Scaffold(
      appBar: AppBar(title: const Text("Resultados del Diagnóstico")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            // Diagnóstico en Grande
            Text(
              diagnostico.toString().toUpperCase(),
              style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: diagnostico.toString().toLowerCase().contains('positivo') ? Colors.red : Colors.green
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            Text(
              "Probabilidad: $probParkinson%",
              style: const TextStyle(fontSize: 18, color: Colors.grey),
            ),
            const Divider(height: 40, thickness: 2),

            const Text(
              "Explicabilidad del Modelo (SHAP)",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 5),
            const Text("Hacia la izquierda (verde) reduce el riesgo. Hacia la derecha (rojo) aumenta el riesgo.", style: TextStyle(fontSize: 12)),
            const SizedBox(height: 20),

            // Lista de barras SHAP
            Expanded(
              child: ListView.builder(
                itemCount: impactoVariables.length,
                itemBuilder: (context, index) {
                  String feature = impactoVariables.keys.elementAt(index);
                  double value = (impactoVariables[feature] is num)
                      ? (impactoVariables[feature] as num).toDouble()
                      : 0.0;

                  return _buildShapBar(feature, value, maxAbsValue);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Widget personalizado para la barra de progreso bidireccional
  Widget _buildShapBar(String feature, double value, double maxAbsValue) {
    bool isPositive = value >= 0;
    // Normalizar el valor entre 0 y 1 respecto al máximo de esta ejecución
    double flexPct = (value.abs() / maxAbsValue).clamp(0.0, 1.0);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("$feature (${value.toStringAsFixed(3)})", style: const TextStyle(fontWeight: FontWeight.w500)),
          const SizedBox(height: 4),
          Row(
            children: [
              // Lado Negativo (Izquierda) - Contribuye a clase 0 (Sano)
              Expanded(
                child: Align(
                  alignment: Alignment.centerRight,
                  child: !isPositive
                      ? FractionallySizedBox(
                    widthFactor: flexPct,
                    child: Container(
                      height: 12,
                      decoration: const BoxDecoration(
                          color: Colors.green, // Verde porque resta probabilidad
                          borderRadius: BorderRadius.horizontal(left: Radius.circular(4))
                      ),
                    ),
                  )
                      : null,
                ),
              ),

              // Línea central (Valor 0)
              Container(width: 2, height: 20, color: Colors.black87),

              // Lado Positivo (Derecha) - Contribuye a clase 1 (Enfermo)
              Expanded(
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: isPositive
                      ? FractionallySizedBox(
                    widthFactor: flexPct,
                    child: Container(
                      height: 12,
                      decoration: const BoxDecoration(
                          color: Colors.red, // Rojo porque suma probabilidad
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