import 'package:flutter/material.dart';
import 'core/theme.dart';
import 'features/home/home_screen.dart';
import 'features/settings/settings_screen.dart';
import 'features/info/info_screen.dart';

class CortexApp extends StatelessWidget {
  const CortexApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cortex-AI',
      theme: AppTheme.light,
      debugShowCheckedModeBanner: false,
      home: const _Shell(),
    );
  }
}

class _Shell extends StatefulWidget {
  const _Shell();

  @override
  State<_Shell> createState() => _ShellState();
}

class _ShellState extends State<_Shell> {
  int _index = 0;

  static const _screens = [
    HomeScreen(),
    SettingsScreen(),
    InfoScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Inicio'),
          NavigationDestination(icon: Icon(Icons.settings_outlined), selectedIcon: Icon(Icons.settings), label: 'Ajustes'),
          NavigationDestination(icon: Icon(Icons.info_outlined), selectedIcon: Icon(Icons.info), label: 'Info'),
        ],
      ),
    );
  }
}
