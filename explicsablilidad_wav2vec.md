Aquí el diagnóstico completo de lo que tienes actualmente y la propuesta para Wav2Vec2:

---
Explicabilidad actual

┌──────────┬───────────────────────────────────────────────────────────────────────────┬───────────────────┐
│  Modelo  │                                  Método                                   │  Nombre técnico   │
├──────────┼───────────────────────────────────────────────────────────────────────────┼───────────────────┤
│ KNN      │ Z-score de cada feature escalada, pasado por tanh para comprimir a [-1,1] │ Feature Deviation │
├──────────┼───────────────────────────────────────────────────────────────────────────┼───────────────────┤
│ XGBoost  │ SHAP TreeExplainer                                                        │ SHAP              │
├──────────┼───────────────────────────────────────────────────────────────────────────┼───────────────────┤
│ ResNet18 │ Gradientes sobre layer4[-1], ponderados por GAP                           │ Grad-CAM          │
├──────────┼───────────────────────────────────────────────────────────────────────────┼───────────────────┤
│ Wav2Vec2 │ ❌ nada                                                                   │ —                 │
└──────────┴───────────────────────────────────────────────────────────────────────────┴───────────────────┘

▎ Para la memoria: KNN no usa SHAP. Usa desviación estándar de cada feature acústica respecto a la media del entrenamiento (z-score escalado por StandardScaler, comprimido con tanh). Es interpretable porque un z-score alto en Shimmer significa "este paciente tiene mucho más Shimmer de lo normal en la población de entrenamiento".

---
Wav2Vec2 — propuesta médicamente defendible

Oclusión temporal (Temporal Saliency Map)

El único método robusto y defendible ante un tribunal médico para un modelo end-to-end sobre audio crudo:

1. Divide el audio en ventanas de ~200ms
2. Enmascara cada ventana con silencio (zeros)
3. Mide la caída en probabilidad PD: Δp = p_original - p_sin_ventana_i
4. Las ventanas con mayor Δp son las más influyentes

Por qué es médicamente defendible:
- No depende de aproximaciones de gradiente ni de la arquitectura interna
- Mide el efecto causal directo de cada segmento temporal sobre la predicción
- Puedes superponer el mapa de saliencia sobre el espectrograma y decir: "el modelo se activó principalmente en el intervalo 1.2s–1.8s, donde se aprecia temblor vocal a ~6 Hz"
- Es equivalente conceptualmente a lo que haría un médico: tapar partes del audio y ver qué importa

Las alternativas y por qué son peores:
- Integrated Gradients: más ruidoso a nivel de muestra, difícil de visualizar
- Attention weights: los transformers no garantizan que la atención = importancia (crítica bien documentada en literatura)
- SHAP KernelSHAP: funciona sobre embeddings, no sobre el audio temporal → pierde interpretación médica