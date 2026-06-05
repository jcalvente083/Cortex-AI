import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsProvider extends ChangeNotifier {
  static const _keyUrl   = 'backend_url';
  static const _keyModel = 'default_model';

  String _backendUrl   = 'http://raspberrypi.local:9000';
  String _defaultModel = 'knn';

  String get backendUrl   => _backendUrl;
  String get defaultModel => _defaultModel;

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _backendUrl   = prefs.getString(_keyUrl)   ?? _backendUrl;
    _defaultModel = prefs.getString(_keyModel) ?? _defaultModel;
    notifyListeners();
  }

  Future<void> setBackendUrl(String url) async {
    _backendUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyUrl, url);
    notifyListeners();
  }

  Future<void> setDefaultModel(String model) async {
    _defaultModel = model;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyModel, model);
    notifyListeners();
  }
}
