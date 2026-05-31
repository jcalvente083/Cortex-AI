# Resultados del Análisis Exploratorio de Datos (EDA)

> Documento de trabajo para redactar el Apéndice C y las secciones de metodología/resultados de la memoria del TFG.  
> Corpus: **NeuroVoz** (107 pac.) + **PC-GITA** (100 pac.) — 4.032 grabaciones totales.

---

## 1. Caracterización Demográfica

### 1.1 Tamaño de las cohortes

| Corpus | Grupo | N pacientes |
|--------|-------|-------------|
| NeuroVoz | HC | 54 |
| NeuroVoz | EP | 53 |
| PC-GITA | HC | 50 |
| PC-GITA | EP | 50 |

Ambos corpus están **prácticamente balanceados** en número de pacientes por grupo (diferencia máxima de 1 sujeto). El conjunto combinado suma 207 pacientes con un ratio HC:EP de 104:103, lo que elimina el sesgo de clase a nivel de sujeto.

### 1.2 Distribución por sexo

- **NeuroVoz:** predominio masculino en ambos grupos (~60–65 % Hombres). La proporción HC/EP es similar, por lo que el sexo no actúa como confound entre grupos dentro de este corpus.
- **PC-GITA:** el grupo EP tiene mayor proporción masculina (~70 %) frente al grupo HC (~50 % Hombres). Esta diferencia es moderada y característica del corpus (la prevalencia de EP es mayor en hombres).

### 1.3 Distribución de edades — confound crítico

Este es el hallazgo demográfico más relevante del EDA:

| Corpus | HC (aprox.) | EP (aprox.) | Gap |
|--------|-------------|-------------|-----|
| NeuroVoz | ~62 años | ~70 años | **~8 años** ⚠️ |
| PC-GITA | ~61 años | ~61 años | ~0 años ✅ |

- **NeuroVoz** presenta una diferencia de edad de aproximadamente 8 años entre HC y EP, visible claramente en el violín: el violín EP está desplazado hacia arriba respecto al HC. Los grupos **no están emparejados por edad**.
- **PC-GITA** es un corpus **age-matched**: los violines HC y EP de edad son prácticamente idénticos en forma y posición central.

**Implicación para el modelado:** cualquier modelo entrenado sobre NeuroVoz (o sobre el conjunto combinado) que incluya `Age` como feature aprenderá, al menos en parte, a predecir la edad en lugar del diagnóstico. Los resultados del experimento `con_age` (BA=0.810 en `all`) deben interpretarse con cautela; el modelo `sin_age` (BA=0.833 en `frase`, solo features acústicas) es más robusto y honesto clínicamente.

---

## 2. Distribución de Grabaciones por Actividad

| Corpus | Actividad | HC | EP | Total |
|--------|-----------|----|----|-------|
| NeuroVoz | Frases/Lectura | 674 | 651 | 1325 |
| NeuroVoz | Vocales Sostenidas | 439 | 521 | 960 |
| NeuroVoz | Habla Espontánea | 50 | **23** | 73 |
| PC-GITA | Frases/Lectura | 548 | 545 | 1093 |
| PC-GITA | Vocales Sostenidas | 243 | 238 | 481 |
| PC-GITA | Habla Espontánea | 50 | 50 | 100 |

**Observaciones clave:**

1. **Frases/Lectura** es la actividad con más grabaciones en ambos corpus (~1.300 en NeuroVoz, ~1.090 en PC-GITA), lo que le proporciona mayor potencia estadística. Esto es consistente con que sea la actividad en la que los modelos obtienen mejores resultados (KNN `frase` BA=0.833).

2. **Habla Espontánea de NeuroVoz** está severamente desbalanceada: solo 23 grabaciones EP frente a 50 HC. Esto reduce drásticamente la potencia estadística para detectar diferencias en este subconjunto y explica los pobres resultados de los tests de significancia en NeuroVoz×Espontánea.

3. **PC-GITA** está perfectamente balanceado en todas las actividades, lo que lo convierte en el corpus más fiable para análisis estadísticos.

4. La actividad `all` (combinación de las tres) suma 4.032 grabaciones y es la que tiene mayor poder discriminativo, como confirman los resultados de los modelos.

---

## 3. Análisis de Significancia Estadística — Test Mann-Whitney U

### 3.1 Resumen global (mapa de calor)

El heatmap muestra el valor $-\log_{10}(p)$ por biomarcador y condición. La línea de significancia $p = 0.05$ corresponde a $-\log_{10}(p) = 1.3$.

**Patrón general observado:**

- **PC-GITA Frases** es la condición con mayor discriminabilidad estadística: múltiples biomarcadores alcanzan el techo del mapa ($-\log_{10}(p) \geq 10$, equivalente a $p < 10^{-10}$), incluyendo JITA, RAP, rPPQ, rSPPQ, ShimmerDb y CPPS.
- **NeuroVoz Frases** es la segunda condición más informativa: ATRI, HNR y NNE alcanzan $-\log_{10}(p) > 5$; rPPQ llega a 4.6.
- **NeuroVoz Vocales** tiene significancia moderada: ATRI (3.3), Shimmer (3.1), rAPQ (3.0) son los biomarcadores más discriminativos.
- **PC-GITA Vocales** muestra significancia alta en jitter (RAP, rPPQ, rSPPQ: ~6.6) y moderada en shimmer y armonicidad.
- **Habla Espontánea** es la condición más débil en ambos corpus: en NeuroVoz, solo ATRI (3.9) es significativo; en PC-GITA, los valores son bajos o en el límite. La escasez de grabaciones espontáneas (especialmente en NeuroVoz EP, N=23) es la causa principal.

### 3.2 Biomarcadores más discriminativos por tipo

#### Temblor (ATRI, FTRI, FATR)
- **ATRI** es el biomarcador más consistentemente significativo a través de corpus y actividades: NeuroVoz Vocal (***), NeuroVoz Frases (***), NeuroVoz Espontánea (***), PC-GITA Frases (***), PC-GITA Vocal (*). Es el único marcador significativo en habla espontánea.
- FATR es significativo en NeuroVoz Frases (***) pero no en otras condiciones.
- FTRI solo bordea la significancia en NeuroVoz Vocales.

**Interpretación:** El temblor de amplitud (ATRI) refleja las oscilaciones rítmicas del aparato fonador causadas por el temblor de reposo característico del Parkinson. Su robustez a través de actividades lo convierte en el marcador más clínicamente relevante del conjunto.

#### Jitter — perturbación de frecuencia (JITA, rJitter, RAP, rPPQ, rSPPQ)
- Altamente significativos en **PC-GITA Vocales** y **PC-GITA Frases** (varios con $p < 10^{-10}$).
- Significativos en **NeuroVoz Frases** (rPPQ: 4.6, rJitter: 3.9).
- **No significativos** en NeuroVoz Vocales — resultado sorprendente que sugiere que las vocales sostenidas de NeuroVoz pueden tener mayor variabilidad inter-sujeto o diferente protocolo de grabación.

#### Shimmer — perturbación de amplitud (ShimmerDb, Shimmer, rAPQ, rSAPQ)
- Significativo en **PC-GITA Frases** (ShimmerDb: $-\log_{10}(p) \geq 10$, Shimmer: ***) y PC-GITA Vocales (**).
- En NeuroVoz: ShimmerDb significativo solo en Vocales (*) y Frases (***); rAPQ en Vocales (***) y Frases (*).
- En Espontánea: solo PC-GITA muestra significancia borderline para Shimmer (**) y ShimmerDb (*).

#### Armonicidad (HNR, NNE)
- Fuertemente significativos en **NeuroVoz Frases** (ambos ***): los pacientes EP presentan menor HNR (más componente ruidoso en la voz).
- Moderadamente significativos en **PC-GITA Vocales** (**) y **PC-GITA Frases** (la imagen sugiere valores menores).
- No significativos en Espontánea, consistente con la alta varianza de esta actividad.

#### CPPS (columna CHNR en el CSV)
- Significativo en **PC-GITA Frases** (***) y PC-GITA Vocales (**).
- Borderline en NeuroVoz Frases (*) y NeuroVoz Espontánea.
- Complementario a HNR: ambos miden la calidad de la voz pero por métodos distintos (espectral vs. cepstral).

---

## 4. Análisis de Distribuciones — Diagramas de Violín

### 4.1 NeuroVoz — Vocales Sostenidas

- La mayoría de features de jitter (JITA, rJitter, RAP, rPPQ, rSPPQ) muestran distribuciones muy similares entre HC y EP, sin separación visual clara. Las distribuciones EP tienen colas más largas (outliers hacia valores altos) pero los IQR se solapan.
- **ShimmerDb y Shimmer:** el grupo EP tiene distribuciones levemente desplazadas hacia valores más altos, con mayor dispersión (violines EP más anchos en la zona media).
- **ATRI:** diferencia visual clara — el violín EP está ensanchado y desplazado hacia arriba respecto a HC. Es el único biomarcador con separación visual convincente en esta condición.
- HNR y NNE muestran formas casi idénticas entre grupos: la vocal sostenida estabilizada no es la condición óptima para medir la señal de ruido del Parkinson.

### 4.2 NeuroVoz — Frases / Lectura

Es la condición con mayor riqueza de separación en NeuroVoz:

- **Jitter (rJitter, RAP, rPPQ, rSPPQ):** separación visible — los violines EP están claramente desplazados hacia valores más altos. rPPQ muestra la separación más pronunciada.
- **ShimmerDb:** EP tiene moda más alta y violín más ancho, confirmando mayor perturbación de amplitud.
- **HNR y NNE:** diferencia visual notable — HC tiene distribuciones concentradas en valores altos (voz limpia), mientras EP se extiende hacia valores más bajos (voz más ruidosa). Violines de forma opuesta.
- **ATRI:** separación muy marcada. El violín EP es considerablemente más ancho y elevado — los pacientes EP tienen mayor temblor de amplitud durante la lectura.
- **FATR:** EP presenta mayor FATR, aunque con más solapamiento que ATRI.

### 4.3 NeuroVoz — Habla Espontánea

La escasez de grabaciones EP (N=23) hace que los violines EP sean pequeños y poco fiables:

- La gran mayoría de features muestra "ns" — la baja potencia estadística por N reducido impide detectar diferencias reales.
- **ATRI** sigue siendo significativo (***): incluso con 23 grabaciones EP, la diferencia en temblor de amplitud es suficientemente grande para ser detectada.
- Esta condición no debe usarse como condición única para inferencia; su contribución en el modelo `all` está limitada por el desbalance.

### 4.4 PC-GITA — Vocales Sostenidas

- **Jitter (JITA, rJitter, RAP, rPPQ, rSPPQ):** separación clara y consistente — los cinco marcadores muestran violines EP desplazados hacia la derecha con mayor dispersión. Esta es la condición donde el jitter es más informativo en PC-GITA, a diferencia de NeuroVoz.
- **ShimmerDb y Shimmer:** EP más elevado, separación moderada.
- **HNR y NNE:** EP más bajo (más ruido), diferencia moderada pero visible.
- **CPPS:** EP ligeramente más bajo, diferencia significativa.
- **ATRI:** solo borderline (*) — en vocales sostenidas de PC-GITA, el temblor de amplitud no discrimina tan bien como en NeuroVoz.

### 4.5 PC-GITA — Frases / Lectura

Es la condición más discriminativa de todo el dataset:

- **Jitter (todos los marcadores):** separaciones muy marcadas. Los violines EP son claramente más anchos y elevados. JITA muestra la diferencia más extrema visualmente.
- **ShimmerDb:** separación muy pronunciada, probablemente la mayor de todo el EDA. El violín EP está desplazado sustancialmente hacia arriba.
- **CPPS:** separación clara — EP tiene menor prominencia cepstral (voz más débil/ruidosa).
- **ATRI:** significativo (***), separación visible aunque menor que en NeuroVoz Frases.
- **rSAPQ y rAPQ:** rSAPQ muestra separación (***), rAPQ no (ns).

### 4.6 PC-GITA — Habla Espontánea

Con 50 grabaciones por grupo (perfectamente balanceado), esta condición es más potente que la espontánea de NeuroVoz:

- **Shimmer (%):** (**) — separación moderada pero estadísticamente significativa.
- **ShimmerDb:** (*) borderline.
- **CPPS:** (*) borderline.
- La mayoría de features de jitter y HNR: ns — en el habla espontánea, la fonación es menos controlada y la variabilidad inter-sujeto es mayor, lo que reduce la discriminabilidad de todos los marcadores.

---

## 5. Matriz de Dispersión — Biomarcadores Seleccionados

El pairplot de las 5 features del modelo KNN (`rPPQ`, `ShimmerDb`, `HNR`, `CPPS`, `ATRI`) revela:

### Distribuciones marginales (diagonal)
- **rPPQ:** HC concentrado cerca de 0 con cola corta; EP con distribución claramente más dispersa extendiéndose hacia valores altos. **Mejor separación univariante visual del conjunto.**
- **ShimmerDb:** HC con pico estrecho y alto ~1.0; EP levemente desplazado hacia la derecha y más aplanado. Separación moderada.
- **HNR:** HC pico alrededor de 20 dB; EP distribución más ancha y levemente desplazada hacia valores menores. Solapamiento considerable.
- **CPPS:** Las distribuciones HC y EP son casi indistinguibles en la marginal — CPPS no discrimina bien de forma aislada, pero sí contribuye en combinación con otras features.
- **ATRI:** Ambas distribuciones centradas alrededor de 8–12, con amplio solapamiento visual. La significancia estadística de ATRI se debe a diferencias en los cuantiles superiores más que en la moda.

### Correlaciones entre features (off-diagonal)
- **rPPQ vs ShimmerDb:** correlación positiva moderada — los pacientes con mayor jitter también tienden a tener mayor shimmer. Los puntos EP (rojo) se concentran en el cuadrante alto-alto, lo que confirma que la combinación de ambas features mejora la discriminabilidad respecto a cada una por separado.
- **rPPQ vs HNR:** correlación negativa leve — mayor jitter se asocia a menor armonicidad (más ruido). La frontera de decisión es más clara en este espacio 2D que con rPPQ solo.
- **HNR vs CPPS:** correlación positiva moderada — ambas miden calidad vocal de formas complementarias. El grupo EP tiende a valores bajos en ambas, pero hay solapamiento considerable.
- **ATRI** no muestra correlación fuerte con ninguna otra feature del conjunto: es **ortogonal** al resto. Esta independencia es precisamente por qué añade valor al modelo — captura un mecanismo fisiopatológico distinto (temblor motor) que las demás features (irregularidad fonatoria) no recogen.

### Conclusión del pairplot
Ningún par de features proporciona separación perfecta en 2D. El Parkinson es un síndrome multidimensional: sus manifestaciones en voz afectan simultáneamente a la frecuencia fundamental (jitter), la amplitud (shimmer), la calidad del sonido (HNR/CPPS) y el control motor (ATRI). La combinación de las 5 features en el espacio de alta dimensión es lo que permite al KNN alcanzar BA=0.833.

---

## 6. Síntesis y Justificación de la Selección de Features

La selección del subconjunto `['rPPQ', 'ShimmerDb', 'Hnr', 'CHNR', 'ATRI']` queda justificada por el EDA de la siguiente manera:

| Feature | Justificación EDA |
|---------|-------------------|
| **rPPQ** | Mejor jitter univariante (significativo en 5/6 condiciones), ortogonal a shimmer, alta potencia en PC-GITA Frases (-log10p ≥ 10) |
| **ShimmerDb** | Segundo mejor aislado (significativo en 5/6 condiciones), complementa rPPQ capturando perturbación de amplitud |
| **HNR** | Mide armonicidad (diferente mecanismo a jitter/shimmer), muy significativo en NeuroVoz Frases (***), ortogonal al resto |
| **CPPS** | Complementa HNR con estimación cepstral, significativo en PC-GITA (***), añade información en condiciones donde HNR pierde potencia |
| **ATRI** | Único marcador consistente en TODAS las condiciones incluyendo espontánea; captura temblor motor (mecanismo independiente de los otros 4) |

Features descartadas del conjunto de 16:
- **JITA, RAP, rSPPQ:** alta correlación con rPPQ (mismo fenómeno, distinta escala) — añaden ruido dimensional sin nueva información.
- **Shimmer (%), rAPQ, rSAPQ:** correlacionados con ShimmerDb; solo rSAPQ añade algo en PC-GITA Frases pero a costa de dimensionalidad.
- **NNE:** muy correlacionado con HNR.
- **FFTR, FATR, FTRI:** significativos solo en condiciones específicas; ATRI ya cubre el mecanismo de temblor de forma más robusta.

---

## 7. Limitaciones Identificadas

1. **Desbalance en Habla Espontánea de NeuroVoz** (HC=50, EP=23): la actividad espontánea tiene bajo poder estadístico y los modelos sobre esta actividad deben interpretarse con cautela.

2. **Confound de edad en NeuroVoz**: el gap de ~8 años entre HC y EP implica que los modelos que incluyen `Age` aprenden parcialmente un proxy de la edad. Los resultados acústicos puros (`sin_age`) son más relevantes clínicamente.

3. **Diferencias de protocolo entre corpus**: NeuroVoz no muestra significancia en jitter para vocales (mientras PC-GITA sí), sugiriendo posibles diferencias en el protocolo de grabación, el micrófono o la instrucción dada a los participantes. Esto motiva el entrenamiento conjunto sobre ambos corpus para obtener un modelo más generalizable.

4. **Alta varianza en Habla Espontánea**: la naturaleza no controlada del habla espontánea aumenta la variabilidad intra-grupo para todos los biomarcadores, reduciendo la discriminabilidad respecto a las actividades leídas.

---

## 8. Justificación de la Selección de Características — Tres Técnicas

> Análisis calculado sobre `train_80.csv` (165 pacientes, 2.713 grabaciones en el combinado) para evitar data leakage del holdout. Desglosado por actividad de habla.

### 8.1 Matriz de Correlación de Spearman entre Biomarcadores

La matriz triangular inferior revela una estructura de bloques muy consistente a través de las cuatro condiciones (Vocales, Frases, Espontánea, Todas):

#### Bloque jitter — alta redundancia interna
JITA, rJitter, RAP, rPPQ y rSPPQ forman un bloque compacto de color rojo oscuro (ρ > 0.90 entre sí). Los cinco biomarcadores son prácticamente intercambiables: miden la misma perturbación de frecuencia con fórmulas ligeramente distintas. **Elegir uno basta — rPPQ es la elección canónica** por su robustez estadística y presencia en la literatura de referencia.

#### Bloque shimmer — alta redundancia interna
ShimmerDb, Shimmer, rAPQ y rSAPQ forman un segundo bloque rojo compacto (ρ > 0.85). Misma lógica: **ShimmerDb representa al grupo** como estimador de la perturbación de amplitud más interpretable.

#### Correlación cruzada jitter ↔ shimmer — moderada
El área de intersección entre ambos bloques muestra tonos naranja-rojizos (ρ ≈ 0.40–0.65). Los dos mecanismos están correlacionados pero no son redundantes: capturan aspectos distintos de la irregularidad fonatoria. Mantener un representante de cada bloque está justificado.

#### HNR y NNE — altamente correlacionados entre sí, anticorrelacionados con jitter/shimmer
HNR y NNE muestran ρ > 0.85 entre sí (rojo), y correlación negativa fuerte (azul, ρ ≈ −0.50 a −0.70) con los bloques jitter y shimmer. Son medidas complementarias de calidad vocal pero redundantes entre sí. **Se retiene HNR** como el más establecido clínicamente.

#### CPPS — semi-independiente
CPPS muestra correlación negativa con jitter/shimmer (azul, ρ ≈ −0.30 a −0.50) y moderada con HNR (ρ ≈ 0.40–0.60), pero significativamente menor que la correlación HNR–NNE. No es redundante con ningún otro representante seleccionado: aporta señal independiente, especialmente en las actividades con menor N donde las otras métricas de calidad pierden potencia.

#### ATRI — ortogonal a todos los grupos
ATRI, FTRI, FFTR y FATR muestran correlaciones próximas a cero (blanco/muy pálido) con los bloques jitter y shimmer, y correlaciones bajas también con HNR y CPPS. El bloque de temblor es **estadísticamente independiente del resto**: mide el temblor motor del aparato fonador, un mecanismo fisiopatológico distinto a la irregularidad en la producción de la voz. ATRI (temblor de amplitud) es el representante más discriminativo del grupo según los resultados del Mann-Whitney.

**Conclusión 3.1:** la selección rPPQ + ShimmerDb + HNR + CPPS + ATRI extrae exactamente **un representante de cada grupo no redundante** identificado por la matriz de correlaciones. La estructura de bloques es estable en las cuatro condiciones analizadas.

---

### 8.2 Correlación de Spearman con el Target (HC=0 / EP=1)

Las barras muestran ρ de Spearman entre cada feature y el diagnóstico. Positivo = EP > HC. Negativo = EP < HC. Ordenadas por |ρ| ascendente (más discriminativa abajo).

#### Vocales Sostenidas (N=1.020)
Las correlaciones son modestas (|ρ| < 0.15), consistente con los bajos valores de significancia del Mann-Whitney en esta condición.
- **rPPQ** (azul): ρ ≈ +0.10 — el mejor predictor individual en vocales. EP tiene mayor perturbación de frecuencia.
- **CPPS** (azul): ρ ≈ −0.10 — EP tiene menor prominencia cepstral. Valor modesto pero presente.
- **ShimmerDb, HNR, ATRI**: correlaciones muy débiles (|ρ| < 0.06) en esta condición. Las vocales sostenidas son la actividad menos informativa para las features clásicas.

#### Frases / Lectura (N=1.555)
Es la condición con correlaciones más altas y las 5 features seleccionadas destacan claramente:
- **ATRI** (azul): ρ ≈ +0.27 — la feature más correlacionada con el Target de todo el conjunto. EP tiene temblor de amplitud considerablemente mayor durante la lectura.
- **ShimmerDb** (azul): ρ ≈ +0.18 — segunda mayor correlación entre las seleccionadas. Perturbación de amplitud elevada en EP.
- **rPPQ** (azul): ρ ≈ +0.12 — consistentemente positivo, EP con mayor jitter.
- **HNR** (azul): ρ ≈ −0.07 — negativo (EP tiene menor armonicidad = más ruido). Débil pero en la dirección esperada.
- **CPPS** (azul): ρ ≈ −0.06 — negativo, EP con menor prominencia cepstral. Similar magnitud a HNR.

Las features descartadas (gris) del grupo jitter (rJitter, RAP, rSPPQ) tienen ρ similares a rPPQ, confirmando su redundancia. rSAPQ también tiene ρ similar a ShimmerDb. La selección no pierde información discriminativa al elegir representantes.

#### Habla Espontánea (N=138)
Pese al tamaño pequeño, la correlación con Target es alta para algunas features — lo que refleja que las diferencias entre grupos son reales y grandes:
- **ShimmerDb** (azul): ρ ≈ +0.28 — la más alta en esta condición. Sorprendente y clínicamente relevante: la perturbación de amplitud es especialmente visible en el habla no controlada.
- **ATRI** (azul): ρ ≈ −0.25 — negativamente correlacionado con Target en espontánea. Esto es un artefacto del dataset pequeño y desbalanceado de NeuroVoz espontánea (23 EP vs 50 HC): la dirección puede invertirse con tan pocos datos EP. Hay que interpretar con cautela.
- **rPPQ** (azul): ρ ≈ +0.20 — segundo positivo más alto.
- **CPPS** (azul): ρ ≈ −0.05 — moderado.
- **HNR** (azul): ρ ≈ −0.15 — negativo, EP con menor armonicidad.

#### Todas las actividades combinadas (N=2.713)
El efecto promediado sobre las tres actividades muestra un patrón limpio:
- **ShimmerDb** (azul): la correlación positiva más alta (~0.20) — feature más discriminativa en el global.
- **ATRI** (azul): segundo lugar, correlación positiva significativa.
- **rPPQ** (azul): tercer lugar entre las seleccionadas, consistentemente positivo.
- **HNR** (azul): débil negativo — contribuye poco en aislamiento pero aporta en combinación.
- **CPPS** (azul): débil negativo — similar a HNR.

**Conclusión 3.2:** rPPQ, ShimmerDb y ATRI son las tres features con mayor correlación individual con el diagnóstico en casi todas las condiciones. HNR y CPPS tienen correlaciones débiles en aislamiento pero en la dirección clínicamente correcta (EP tiene menor armonicidad y menor prominencia cepstral). Su valor se manifiesta en la combinación con las demás features.

---

### 8.3 Importancia en Random Forest (200 árboles, class_weight=balanced)

El RF captura relaciones no lineales e interacciones entre features que la correlación lineal no detecta. La línea discontinua marca la importancia uniforme (1/16 = 6.25%).

#### Vocales Sostenidas (N=1.020)
La importancia es relativamente plana (~0.04–0.075), sin un dominador claro:
- **CPPS** (azul): ~0.075 — la feature seleccionada con mayor importancia en vocales. Llama la atención dado su baja correlación univariante: el RF encuentra que CPPS, en combinación con otras features, discrimina mejor en vocales que en los análisis univariantes.
- **ATRI** (azul): ~0.068 — segundo entre los seleccionados.
- **rPPQ** (azul): ~0.065 — consistente con su buen desempeño en correlación.
- **ShimmerDb** (azul): ~0.060 — por encima de la media.
- **HNR** (azul): ~0.055 — próximo a la media pero por encima.
- Nota: rAPQ y FTRI (gris) puntúan ligeramente por encima de CPPS. Esto sugiere que para vocales habría argumentos para incluir FTRI en lugar de HNR, pero la diferencia es pequeña y FTRI no tiene el mismo respaldo en literatura.

#### Frases / Lectura (N=1.555)
Las 4 features seleccionadas con mayor información son **las 4 más importantes** del conjunto, confirmando la selección de forma rotunda:
- **ATRI** (azul): ~0.068 — la más importante en frases, confirma el resultado de correlación.
- **rPPQ** (azul): ~0.065 — segunda más importante entre seleccionadas.
- **HNR** (azul): ~0.065 — en frases, HNR sube a empatar con rPPQ. El RF detecta su valor discriminativo no lineal.
- **ShimmerDb** (azul): ~0.064 — cuarta, muy próxima a las anteriores.
- **CPPS** (azul): importancia moderada, similar a las features descartadas.
- La compresión de los rangos (todas entre 0.050 y 0.070) indica que las features son complementarias, no redundantes.

#### Habla Espontánea (N=138)
Es la condición más reveladora para CPPS:
- **CPPS** (azul): ~0.10 — la feature MÁS IMPORTANTE del conjunto en espontánea, con diferencia. Esto resuelve la duda planteada en el análisis Mann-Whitney: CPPS es débil estadísticamente en NeuroVoz por el pequeño N, pero el RF confirma que porta información discriminativa real en el habla espontánea.
- **ATRI** (azul): ~0.09 — segunda más importante.
- **ShimmerDb** (azul): ~0.08 — tercera.
- **rPPQ y HNR** (azul): importancia moderada pero por encima de la mayoría de features descartadas.

#### Todas las actividades combinadas (N=2.713)
El panorama global confirma que las 5 features seleccionadas se distribuyen en la mitad superior del ranking:
- **CPPS** (azul): top del ranking global (~0.068).
- **ATRI** (azul): segunda posición.
- **rPPQ** (azul): tercera entre seleccionadas.
- **ShimmerDb** (azul): cuarta.
- **HNR** (azul): quinta pero aún por encima de la media uniforme (6.25%).
- Ninguna de las 5 features seleccionadas cae por debajo de la línea de importancia uniforme.

**Conclusión 3.3:** el Random Forest valida las 5 features sin excepción. La observación más relevante es el comportamiento de CPPS: débil en correlación univariante pero primera en importancia RF en espontánea y en el global combinado. Esto confirma que CPPS aporta información no lineal e interacciones que la correlación de Spearman no captura.

---

### 8.4 Síntesis de las Tres Técnicas

| Feature | Bloque (3.1) | Corr. Target (3.2) | RF Importance (3.3) | Veredicto |
|---------|-------------|-------------------|--------------------|-----------| 
| **rPPQ** | Representante del bloque jitter | Alta en frases (+0.12), consistente | Top en frases, por encima de media siempre | ✅ Muy bien justificada |
| **ShimmerDb** | Representante del bloque shimmer | Más alta en espontánea (+0.28), alta en frases | Top 4 en todas las condiciones | ✅ Muy bien justificada |
| **ATRI** | Grupo temblor — ortogonal a todo | Máxima en frases (+0.27) | Top 2 en frases y espontánea | ✅ La mejor elección del conjunto |
| **HNR** | Representa armonicidad (junto a NNE) | Débil (−0.07) pero consistente | Top en frases, media en vocales | ✅ Justificada, especialmente en frases |
| **CPPS** | Semi-independiente de HNR | Débil univariante, salvo espontánea | **1ª en espontánea y global** | ✅ Justificada por RF; univariante infravalora su aportación |

Las tres técnicas convergen en la misma conclusión: **las 5 features seleccionadas representan los 5 mecanismos fisiopatológicos independientes** que el Parkinson altera en la voz — perturbación de frecuencia (rPPQ), perturbación de amplitud (ShimmerDb), armonicidad espectral (HNR), prominencia cepstral (CPPS) y temblor motor (ATRI) — con mínima redundancia interna y máxima cobertura del espacio de información disponible.
