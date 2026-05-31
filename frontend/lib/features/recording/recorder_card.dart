import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import '../../core/theme.dart';

class RecorderCard extends StatefulWidget {
  final int index;
  final String label;
  final String instruction;
  final File? recordedFile;
  final void Function(File) onRecorded;

  const RecorderCard({
    super.key,
    required this.index,
    required this.label,
    required this.instruction,
    required this.recordedFile,
    required this.onRecorded,
  });

  @override
  State<RecorderCard> createState() => _RecorderCardState();
}

class _RecorderCardState extends State<RecorderCard> {
  final _recorder = AudioRecorder();
  final _player   = AudioPlayer();
  bool _isRecording = false;
  bool _isPlaying   = false;
  int  _elapsed     = 0;
  Timer? _timer;

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  Future<void> _toggleRecord() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) return;

    final dir  = await getTemporaryDirectory();
    final path = '${dir.path}/slot_${widget.index}_${DateTime.now().millisecondsSinceEpoch}.wav';

    await _recorder.start(
      const RecordConfig(
        encoder:    AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: path,
    );

    setState(() {
      _isRecording = true;
      _elapsed     = 0;
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() => _elapsed++);
    });
  }

  Future<void> _stopRecording() async {
    _timer?.cancel();
    final path = await _recorder.stop();
    setState(() => _isRecording = false);
    if (path != null) {
      widget.onRecorded(File(path));
    }
  }

  Future<void> _togglePlay() async {
    if (_isPlaying) {
      await _player.stop();
      setState(() => _isPlaying = false);
    } else {
      await _player.play(DeviceFileSource(widget.recordedFile!.path));
      setState(() => _isPlaying = true);
      _player.onPlayerComplete.listen((_) {
        if (mounted) setState(() => _isPlaying = false);
      });
    }
  }

  String _formatTime(int s) => '${(s ~/ 60).toString().padLeft(2, '0')}:${(s % 60).toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final isDone = widget.recordedFile != null;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  backgroundColor: isDone ? AppTheme.success : AppTheme.primary,
                  foregroundColor: Colors.white,
                  radius: 16,
                  child: isDone
                      ? const Icon(Icons.check, size: 18)
                      : Text('${widget.index + 1}', style: const TextStyle(fontWeight: FontWeight.bold)),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(widget.label,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(widget.instruction,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey[600])),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _toggleRecord,
                    icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                    label: Text(_isRecording
                        ? 'Detener  ${_formatTime(_elapsed)}'
                        : (isDone ? 'Grabar de nuevo' : 'Grabar')),
                    style: FilledButton.styleFrom(
                      backgroundColor: _isRecording ? AppTheme.danger : AppTheme.primary,
                    ),
                  ),
                ),
                if (isDone) ...[
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: _togglePlay,
                    icon: Icon(_isPlaying ? Icons.stop : Icons.play_arrow),
                    style: IconButton.styleFrom(backgroundColor: AppTheme.secondary),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}
