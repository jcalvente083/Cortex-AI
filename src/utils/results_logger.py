"""
results_logger.py — Persiste métricas de cada run en JSON y genera tabla LaTeX comparativa.

Uso desde scripts de entrenamiento:
    from src.utils.results_logger import save_run_json

    save_run_json(
        model_name = "KNN",
        run        = args.run,
        results    = results,
        thresholds = thresholds,
    )

Generar tabla LaTeX (formato pareado Con Edad / Sin Edad por defecto):
    uv run python -m src.utils.results_logger
    uv run python -m src.utils.results_logger --flat
    uv run python -m src.utils.results_logger --output docs/Memoria/tablas/resultados.tex
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("reports/results")

ACTIVITY_LABELS: dict[str, str] = {
    "vocal":      "Vocal",
    "frase":      "Frase",
    "espontanea": "Espontánea",
    "all":        "Todas",
}
ACTIVITY_ORDER = ["vocal", "frase", "espontanea", "all"]

METRICS: list[tuple[str, str]] = [
    ("balanced_accuracy", "BA"),
    ("roc_auc",           "AUC"),
    ("recall",            "Sensib."),
    ("specificity",       "Especif."),
    ("f1",                "F1"),
]

# Pares (nombre_display, (model, run) Con Edad, (model, run) Sin Edad)
# El orden aquí determina el orden de filas en la tabla.
MODEL_GROUPS: list[tuple[str, tuple, tuple]] = [
    ("KNN",
        ("KNN",            "con_age"),
        ("KNN",            "sin_age")),
    ("XGBoost",
        ("XGBoost",        "con_age"),
        ("XGBoost",        "sin_age")),
    ("ResNet18",
        ("ResNet18",       "specaugment_freeze"),
        ("ResNet18",       "age_matched_freeze")),
    ("ResNet10",
        ("ResNet10",       "specaugment_freeze"),
        ("ResNet10",       "age_matched_freeze")),
    ("CortexCNN",
        ("CortexCNN",      "baseline"),
        ("CortexCNN",      "age_matched")),
    ("Wav2Vec+XGBoost",
        ("Wav2Vec-XGBoost","con_age"),
        ("Wav2Vec-XGBoost","age_matched")),
    ("Wav2Vec+KNN",
        ("Wav2Vec-KNN",    "con_age"),
        ("Wav2Vec-KNN",    "age_matched")),
]


# =============================================================================
# GUARDAR JSON
# =============================================================================
def save_run_json(
    model_name: str,
    run: str,
    results: dict,
    thresholds: dict | None = None,
) -> Path:
    """Guarda métricas de un run en reports/results/{model_name}_{run}.json"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":      model_name,
        "run":        run,
        "saved":      datetime.now().isoformat(timespec="seconds"),
        "activities": {},
    }
    for act, r in results.items():
        payload["activities"][act] = {
            "threshold":        (thresholds or {}).get(act),
            "val_internal":     r.get("internal", {}),
            "holdout_external": r.get("external", {}),
        }

    out = RESULTS_DIR / f"{model_name}_{run}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [results_logger] Metricas guardadas -> {out}")
    return out


# =============================================================================
# HELPERS
# =============================================================================
def _load_all_jsons() -> dict[tuple[str, str], dict]:
    """Carga todos los JSON de RESULTS_DIR en un dict (model, run) -> payload."""
    records = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            records[(d["model"], d["run"])] = d
        except Exception:
            pass
    return records


def _fmt(val: float | None, best: float | None) -> str:
    if val is None:
        return "--"
    s = f"{val:.3f}"
    if best is not None and abs(val - best) < 1e-9:
        return f"\\textbf{{{s}}}"
    return s


def _best_vals(records: dict, set_key: str) -> dict[str, float | None]:
    """Calcula el máximo global por métrica considerando todos los runs."""
    acc: dict[str, list[float]] = {k: [] for k, _ in METRICS}
    for rec in records.values():
        for act in ACTIVITY_ORDER:
            m = rec["activities"].get(act, {}).get(set_key, {})
            for key, _ in METRICS:
                v = m.get(key)
                if v is not None:
                    acc[key].append(v)
    return {k: (max(v) if v else None) for k, v in acc.items()}


# =============================================================================
# TABLA PAREADA (Con Edad | Sin Edad) — formato principal
# =============================================================================
def generate_latex_table_paired(
    set_key: str = "holdout_external",
    output_path: Path | None = None,
) -> Path | None:
    """
    Genera tabla LaTeX con columnas dobles (Con Edad | Sin Edad) por modelo,
    siguiendo el estilo de docs/ResumenResultados/main.tex (Tabla 1).
    """
    records = _load_all_jsons()
    if not records:
        print("[results_logger] No hay JSONs en reports/results/")
        return None

    bests = _best_vals(records, set_key)
    n_m = len(METRICS)          # 5
    col_lo = 3                  # primera col de metricas Con Edad
    col_hi = col_lo + n_m - 1  # ultima col Con Edad  (3..7)
    col2_lo = col_hi + 1       # primera col Sin Edad (8)
    col2_hi = col2_lo + n_m - 1 # ultima col Sin Edad (12)

    header_metrics = " & ".join(f"\\textbf{{{lbl}}}" for _, lbl in METRICS)

    lines: list[str] = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \footnotesize",
        r"  \renewcommand{\arraystretch}{1.3}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{@{{}} l l {'c' * n_m} {'c' * n_m} @{{}}}}",
        r"    \toprule",
        f"    \\multirow{{2}}{{*}}{{\\textbf{{Modelo}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Actividad}}}}"
        f" & \\multicolumn{{{n_m}}}{{c}}{{\\textbf{{Con Edad (Baseline)}}}}"
        f" & \\multicolumn{{{n_m}}}{{c}}{{\\textbf{{Sin Edad (Age Matched)}}}} \\\\",
        f"    \\cmidrule(lr){{{col_lo}-{col_hi}}}\\cmidrule(lr){{{col2_lo}-{col2_hi}}}",
        f"    & & {header_metrics} & {header_metrics} \\\\",
        r"    \midrule",
    ]

    first = True
    for disp_name, key_con, key_sin in MODEL_GROUPS:
        rec_con = records.get(key_con)
        rec_sin = records.get(key_sin)

        # Si ninguno existe, omitir el grupo
        if rec_con is None and rec_sin is None:
            continue

        acts = ACTIVITY_ORDER
        n_acts = len(acts)

        if not first:
            lines.append(r"    \midrule")
        first = False

        for i, act in enumerate(acts):
            act_label = ACTIVITY_LABELS.get(act, act.capitalize())

            def cells(rec: dict | None) -> str:
                if rec is None:
                    return " & ".join(["--"] * n_m)
                m = rec["activities"].get(act, {}).get(set_key, {})
                return " & ".join(_fmt(m.get(k), bests.get(k)) for k, _ in METRICS)

            model_cell = f"\\multirow{{{n_acts}}}{{*}}{{\\textbf{{{disp_name}}}}}" if i == 0 else ""
            lines.append(
                f"    {model_cell} & {act_label} & {cells(rec_con)} & {cells(rec_sin)} \\\\"
            )

    set_label = "holdout externo" if set_key == "holdout_external" else "validación interna"
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{Comparativa de resultados en el {set_label} (nivel de paciente). "
        r"Los valores en \textbf{negrita} son los mejores de cada métrica. "
        r"ResNet18 Con Edad: SpecAug+Freeze. Sin Edad: Age-Matched+Freeze.}",
        r"  \label{tab:resultados_comparativa}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    out = output_path or (RESULTS_DIR / "tabla_resultados.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"[results_logger] Tabla LaTeX (pareada) guardada -> {out}")
    return out


# =============================================================================
# TABLA PLANA (un run por fila) — formato antiguo
# =============================================================================
def generate_latex_table(
    set_key: str = "holdout_external",
    output_path: Path | None = None,
) -> Path | None:
    """Genera tabla LaTeX plana (un run por fila). Usar --flat en CLI."""
    records_dict = _load_all_jsons()
    if not records_dict:
        print("[results_logger] No hay JSONs en reports/results/")
        return None

    records = list(records_dict.values())
    bests = _best_vals(records_dict, set_key)
    n_m = len(METRICS)
    col_spec = "l l " + " ".join(["c"] * n_m)

    lines: list[str] = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \renewcommand{\arraystretch}{1.3}",
        f"  \\begin{{tabularx}}{{\\textwidth}}{{@{{}} {col_spec} @{{}}}}",
        r"    \toprule",
    ]
    header_metrics = " & ".join(f"\\textbf{{{lbl}}}" for _, lbl in METRICS)
    lines.append(f"    \\textbf{{Modelo}} & \\textbf{{Actividad}} & {header_metrics} \\\\")
    lines.append(r"    \midrule")

    first_model = True
    for rec in records:
        acts_present = [a for a in ACTIVITY_ORDER if a in rec["activities"]]
        if not acts_present:
            continue
        disp = f"{rec['model']} ({rec['run']})"
        if not first_model:
            lines.append(r"    \midrule")
        first_model = False
        for i, act in enumerate(acts_present):
            m = rec["activities"][act].get(set_key, {})
            act_label = ACTIVITY_LABELS.get(act, act.capitalize())
            model_cell = f"\\multirow{{{len(acts_present)}}}{{*}}{{{disp}}}" if i == 0 else ""
            metric_cells = " & ".join(_fmt(m.get(k), bests.get(k)) for k, _ in METRICS)
            lines.append(f"    {model_cell} & {act_label} & {metric_cells} \\\\")

    set_label = "holdout externo" if set_key == "holdout_external" else "validación interna"
    lines += [
        r"    \bottomrule",
        r"  \end{tabularx}",
        f"  \\caption{{Comparativa de resultados en el {set_label} (nivel de paciente). "
        r"Los valores en \textbf{negrita} son los mejores de cada métrica.}",
        r"  \label{tab:resultados_comparativa}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    out = output_path or (RESULTS_DIR / "tabla_resultados.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"[results_logger] Tabla LaTeX (plana) guardada -> {out}")
    return out


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera tabla LaTeX comparativa a partir de los JSON de resultados."
    )
    parser.add_argument(
        "--set",
        default="holdout_external",
        choices=["holdout_external", "val_internal"],
        help="Conjunto de evaluación (default: holdout_external)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Formato plano (un run por fila) en lugar del formato pareado Con/Sin Edad",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida del .tex (default: reports/results/tabla_resultados.tex)",
    )
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None

    if args.flat:
        generate_latex_table(set_key=args.set, output_path=out_path)
    else:
        generate_latex_table_paired(set_key=args.set, output_path=out_path)
