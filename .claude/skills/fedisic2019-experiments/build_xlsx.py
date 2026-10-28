import glob
import json
import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = sys.argv[1]
OUT = sys.argv[2]
R = f"{REPO}/results_isic2019/seed42"
PAPER = f"{REPO}/flamby/results/results_benchmark_fed_isic2019"
CENTERS = ["BCN", "HAM_vidir_molemax", "HAM_vidir_modern", "HAM_rosendahl", "MSK",
           "HAM_vienna_dias"]
STRATS = ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad", "Scaffold", "Cyclic",
          "FedAvgFineTuning"]
LRS = [0.001, 0.0031622776601683, 0.01, 0.0316227766016837, 0.1, 0.316227766016838]
SLRS = LRS + [1.0, 10.0]
MUS = [0.001, 0.01, 0.1, 1.0]
# Same set as run.sh: everything in the paper's grid except FedAvgFineTuning lr 0.316.
EXPECTED = ([(s, lr, None, None) for s in ["FedAvg", "Cyclic"] for lr in LRS]
            + [("FedAvgFineTuning", lr, None, None) for lr in LRS[:5]]
            + [("FedProx", lr, None, mu) for lr in LRS for mu in MUS]
            + [(s, lr, slr, None) for s in ["FedAdam", "FedYogi", "FedAdagrad", "Scaffold"]
               for lr in LRS for slr in SLRS])

F = "Arial"
HEAD = Font(name=F, bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="305496")
BODY = Font(name=F)
BOLD = Font(name=F, bold=True)
TITLE = Font(name=F, bold=True, size=14)
GREY = Font(name=F, italic=True, color="7F7F7F")
RED_FILL = PatternFill("solid", fgColor="FCE4E4")
THIN = Border(bottom=Side(style="thin", color="BFBFBF"))
NUM = "0.000"


def wide(df):
    """FLamby long CSV -> one row per experiment: hyperparams, pooled, 6 centers."""
    df = df.copy()
    df["Strategy"] = df.Method.str.replace(r"100$", "", regex=True)
    keys = ["Strategy"] + [c for c in ["learning_rate", "server_learning_rate", "mu",
                                       "num_fine_tuning_steps"] if c in df]
    df[keys[1:]] = df[keys[1:]].astype(float)
    p = df.pivot_table(index=keys, columns="Test", values="Metric", dropna=False)
    p = p.dropna(how="all").reset_index()
    for c in ["server_learning_rate", "mu", "num_fine_tuning_steps"]:
        if c not in p:
            p[c] = float("nan")
    return p


def header(ws, cols, row=1):
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=row, column=j, value=c)
        cell.font, cell.fill = HEAD, HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 32
    ws.freeze_panes = ws.cell(row=row + 1, column=2)


def widths(ws, w):
    for j, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(j)].width = x


def val(x):
    return None if pd.isna(x) else float(x)


RES_COLS = ["Estrategia / Método", "lr cliente", "lr servidor (slr)", "mu (FedProx)",
            "pasos fine-tuning", "Pooled test"] + CENTERS + ["Media 6 centros"]


def results_sheet(ws, df, order, extra=None):
    """Writes the common results table; returns the last data row."""
    header(ws, RES_COLS + (extra or []))
    df = df.assign(_o=df.Strategy.map({s: i for i, s in enumerate(order)})) \
        .sort_values(["_o", "learning_rate", "server_learning_rate", "mu"])
    r = 1
    for _, x in df.iterrows():
        r += 1
        row = [x.Strategy, val(x.learning_rate), val(x.server_learning_rate), val(x.mu),
               val(x.num_fine_tuning_steps), val(x.get("Pooled Test"))] + \
              [val(x.get(f"client_test_{i}")) for i in range(6)]
        for j, v in enumerate(row, 1):
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.border = BODY, THIN
            if j >= 6:
                c.number_format = NUM
            elif j in (2, 3, 4):
                c.number_format = "0.00000"
        m = ws.cell(row=r, column=13,
                    value=f'=IF(COUNT(G{r}:L{r})=6,AVERAGE(G{r}:L{r}),"")')
        m.font, m.number_format, m.border = BOLD, NUM, THIN
    widths(ws, [20, 11, 13, 11, 10, 11] + [12] * 6 + [12] + [16] * len(extra or []))
    return r


BASELINES = ["Pooled Training"] + [f"Local {i}" for i in range(6)]
ORDER = BASELINES + STRATS

wb = Workbook()

# ---------------------------------------------------------------- Léeme
ws = wb.active
ws.title = "Léeme"
lines = [
    ("Fed-ISIC2019 — resultados, semilla 42", TITLE),
    ("", None),
    ("Métrica de todas las tablas: balanced accuracy (métrica oficial de FLamby para "
     "Fed-ISIC2019), en el test de cada centro y en el test agrupado (Pooled test).", BODY),
    ("Fuente: CSV de results_isic2019/seed42/ (rama fedisic2019-reports-docker, "
     "repo Ayax911/FLamby). Corridas en RTX 6000 Ada vía scripts/run_isic2019_docker.sh.",
     BODY),
    ("", None),
    ("Hojas", BOLD),
    ("Benchmark — 15 experimentos: Pooled, 6 locales y 8 estrategias con los "
     "hiperparámetros de flamby/config_isic2019.json (results_isic2019.csv).", BODY),
    ("Malla — 233 combinaciones federadas con la malla del paper (Apéndice H): lr cliente "
     "10^{-3..-0.5}, slr 10^{-3..-0.5} + 1 + 10, mu {0.001, 0.01, 0.1, 1} "
     "(results_isic2019_grid.csv). 201 completas; las 32 que divergieron (NaN: FedAdam/FedYogi "
     "con slr 1 y 10, FedAdagrad/Scaffold con slr 10, y una FedAdam con slr 0.316) aparecen "
     "en rojo, sin métricas. 155 corridas originales + 78 de la extensión del paper; "
     "se omitieron FedAvgFineTuning con lr 0.316 y repetir los 10 slr=1.0 ya divergidos.",
     BODY),
    ("Malla mejor — la mejor corrida de la malla por estrategia vs la config del paper "
     "(fórmulas sobre la hoja Malla).", BODY),
    ("Best config — las 8 estrategias de config_isic2019.json corridas de nuevo como "
     "seed42_best_* (results_isic2019_best.csv).", BODY),
    ("Benchmark rerun — 7 estrategias que se entrenaron una segunda vez por un "
     "lanzamiento duplicado (results_isic2019_benchmark_rerun.csv).", BODY),
    ("Repeticiones — misma config y semilla corrida 2–3 veces: ruido real de una "
     "corrida (fórmulas).", BODY),
    ("FLamby paper — nuestras cifras vs las 5 semillas publicadas por FLamby "
     "(flamby/results/results_benchmark_fed_isic2019/isic_seed42..46.csv).", BODY),
    ("Métricas detalladas — accuracy, AUC, F1, precisión, sensibilidad, especificidad y "
     "loss en el test agrupado, leídas de runs*/…/test/metrics.json.", BODY),
    ("", None),
    ("Advertencias", BOLD),
    ("1. val = test: elegir el mejor punto de la malla por su test es optimista.", BODY),
    ("2. Una sola semilla: corridas idénticas difieren hasta ~0.04 (hoja Repeticiones); "
     "diferencias menores entre estrategias no son concluyentes.", BODY),
    ("3. Por el lanzamiento duplicado, los reportes de runs/ para FedAvg, FedProx, FedAdam, "
     "FedYogi, FedAdagrad, Scaffold y FedAvgFineTuning (config del paper) y los de "
     "runs_best/ pueden no corresponder a la fila del CSV. Los CSV son la fuente fiable; "
     "la hoja Métricas detalladas hereda esa limitación para esas filas.", BODY),
    ("4. Ensemble no se calculó: fed_benchmark.py lo desactiva (compute_ensemble_perf=False).",
     BODY),
    ("5. Celdas en negrita de 'Media 6 centros' y las columnas de comparación son "
     "fórmulas; las demás cifras son resultados copiados de los CSV.", BODY),
]
for i, (t, f) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=t)
    if f:
        c.font = f
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 120

# ---------------------------------------------------------------- Benchmark
bench = wide(pd.read_csv(f"{R}/results_isic2019.csv"))
ws = wb.create_sheet("Benchmark")
b_last = results_sheet(ws, bench, ORDER)

# ---------------------------------------------------------------- Malla
grid = wide(pd.read_csv(f"{R}/results_isic2019_grid.csv"))
r6 = lambda v: None if v is None or pd.isna(v) else round(float(v), 6)
done = {(x.Strategy, r6(x.learning_rate), r6(x.server_learning_rate), r6(x.mu))
        for _, x in grid.iterrows()}
div = pd.DataFrame([{"Strategy": s, "learning_rate": lr,
                     "server_learning_rate": float("nan") if slr is None else slr,
                     "mu": float("nan") if mu is None else mu,
                     "num_fine_tuning_steps": float("nan")}
                    for s, lr, slr, mu in EXPECTED
                    if (s, r6(lr), r6(slr), r6(mu)) not in done])
N_GRID, N_DIV = len(EXPECTED), len(div)
grid = pd.concat([grid, div], ignore_index=True)
ws = wb.create_sheet("Malla")
g_last = results_sheet(ws, grid, STRATS, ["Estado", "Rango en su estrategia", "Clave"])
rng = f"$A$2:$A${g_last}"
prng = f"$F$2:$F${g_last}"
for r in range(2, g_last + 1):
    ok = ws.cell(row=r, column=6).value is not None
    ws.cell(row=r, column=14, value="ok" if ok else "divergió (NaN)").font = BODY
    ws.cell(row=r, column=15,
            value=f'=IF(F{r}="","",COUNTIFS({rng},A{r},{prng},">"&F{r})+1)').font = BODY
    ws.cell(row=r, column=16, value=f'=A{r}&"|"&O{r}').font = GREY
    if not ok:
        for j in range(1, 17):
            ws.cell(row=r, column=j).fill = RED_FILL
ws.cell(row=1, column=16).comment = Comment(
    "Columna auxiliar para que la hoja 'Malla mejor' encuentre la corrida de rango 1.",
    "Claude")

# ---------------------------------------------------------------- Malla mejor
ws = wb.create_sheet("Malla mejor")
header(ws, ["Estrategia", "Corridas válidas", "Mejor Pooled (malla)", "lr", "slr", "mu",
            "Pooled con config del paper", "Mejor − paper", "Pooled peor (malla)"])
G = "Malla!"
for i, s in enumerate(STRATS, 2):
    key = f'$A{i}&"|1"'
    cells = [
        s,
        f'=COUNTIFS({G}{rng},A{i},{G}$N$2:$N${g_last},"ok")',
        f"=_xlfn.MAXIFS({G}{prng},{G}{rng},A{i})",
        f"=INDEX({G}$B$2:$B${g_last},MATCH({key},{G}$P$2:$P${g_last},0))",
        f'=IF(INDEX({G}$C$2:$C${g_last},MATCH({key},{G}$P$2:$P${g_last},0))="","",'
        f"INDEX({G}$C$2:$C${g_last},MATCH({key},{G}$P$2:$P${g_last},0)))",
        f'=IF(INDEX({G}$D$2:$D${g_last},MATCH({key},{G}$P$2:$P${g_last},0))="","",'
        f"INDEX({G}$D$2:$D${g_last},MATCH({key},{G}$P$2:$P${g_last},0)))",
        f"=INDEX(Benchmark!$F$2:$F${b_last},MATCH(A{i},Benchmark!$A$2:$A${b_last},0))",
        f"=C{i}-G{i}",
        f"=_xlfn.MINIFS({G}{prng},{G}{rng},A{i})",
    ]
    for j, v in enumerate(cells, 1):
        c = ws.cell(row=i, column=j, value=v)
        c.font, c.border = BODY, THIN
        c.number_format = {4: "0.00000", 5: "0.00000", 6: "0.00000", 2: "0"}.get(j, NUM)
widths(ws, [20, 10, 14, 11, 11, 10, 16, 12, 14])
ws.cell(row=11, column=1,
        value="Si dos corridas empatan en el máximo, lr/slr/mu muestran la primera.").font = GREY

# ---------------------------------------------------------------- Best / rerun
best = wide(pd.read_csv(f"{R}/results_isic2019_best.csv"))
ws = wb.create_sheet("Best config")
best_last = results_sheet(ws, best, STRATS)
rerun = wide(pd.read_csv(f"{R}/results_isic2019_benchmark_rerun.csv"))
ws = wb.create_sheet("Benchmark rerun")
rr_last = results_sheet(ws, rerun, STRATS)

# ---------------------------------------------------------------- Repeticiones
ws = wb.create_sheet("Repeticiones")
header(ws, ["Estrategia", "Benchmark", "Rerun", "Best config", "N corridas", "Media",
            "Desv. estándar", "Rango (máx − mín)"])


def look(sheet, last, i):
    return (f'=IFERROR(INDEX(\'{sheet}\'!$F$2:$F${last},'
            f'MATCH($A{i},\'{sheet}\'!$A$2:$A${last},0)),"")')


for i, s in enumerate(STRATS, 2):
    cells = [s, look("Benchmark", b_last, i), look("Benchmark rerun", rr_last, i),
             look("Best config", best_last, i), f"=COUNT(B{i}:D{i})",
             f"=AVERAGE(B{i}:D{i})", f'=IF(E{i}>1,STDEV(B{i}:D{i}),"")',
             f"=MAX(B{i}:D{i})-MIN(B{i}:D{i})"]
    for j, v in enumerate(cells, 1):
        c = ws.cell(row=i, column=j, value=v)
        c.font, c.border = BODY, THIN
        c.number_format = "0" if j == 5 else NUM
widths(ws, [20, 12, 12, 12, 11, 11, 13, 16])
ws.cell(row=11, column=1, value="Pooled test, misma config (config_isic2019.json) y "
        "semilla 42 en la misma máquina. Cyclic no tiene rerun.").font = GREY

# ---------------------------------------------------------------- FLamby paper
ws = wb.create_sheet("FLamby paper")
header(ws, ["Método", "Seed 42", "Seed 43", "Seed 44", "Seed 45", "Seed 46",
            "Media FLamby", "Desv. FLamby", "Nuestro (seed 42)", "Nuestro − media",
            "¿Dentro de ±2 desv.?"])
paper = {}
for sd in range(42, 47):
    d = pd.read_csv(f"{PAPER}/isic_seed{sd}.csv")
    d = d[d.Test == "Pooled Test"]
    for m, v in zip(d.Method, d.Metric):
        paper.setdefault(m.replace("100", "") if m.endswith("100") else m, {})[sd] = v
i = 1
for m in ORDER:
    if m not in paper:
        continue
    i += 1
    row = [m] + [paper[m].get(sd) for sd in range(42, 47)] + [
        f"=AVERAGE(B{i}:F{i})", f"=STDEV(B{i}:F{i})",
        f'=IFERROR(INDEX(Benchmark!$F$2:$F${b_last},MATCH(A{i},Benchmark!$A$2:$A${b_last},0)),"")',
        f'=IF(I{i}="","",I{i}-G{i})',
        f'=IF(I{i}="","",IF(ABS(J{i})<=2*H{i},"sí","no"))']
    for j, v in enumerate(row, 1):
        c = ws.cell(row=i, column=j, value=v)
        c.font, c.border = BODY, THIN
        if 2 <= j <= 6:
            c.font = Font(name=F, color="0000FF")
        if j >= 2:
            c.number_format = NUM
widths(ws, [20] + [10] * 5 + [13, 13, 15, 14, 14])
ws.cell(row=i + 2, column=1, value=(
    "Seeds 42–46 (azul): pooled test publicado por FLamby en "
    "flamby/results/results_benchmark_fed_isic2019/isic_seed<S>.csv. FedAvgFineTuning no "
    "está en sus resultados.")).font = GREY

# ---------------------------------------------------------------- Métricas detalladas
ws = wb.create_sheet("Métricas detalladas")
MET = ["balanced_accuracy", "accuracy", "auc", "f1", "f1_macro", "precision",
       "sensitivity", "specificity", "loss"]
header(ws, ["Origen", "Carpeta de la corrida"] + MET)
rows = []
for src in ["runs", "runs_best"]:
    for d in sorted(glob.glob(f"{R}/{src}/fed_isic2019/seed42/*")):
        for sub in ["test", "pooled_eval/test"]:
            p = f"{d}/{sub}/metrics.json"
            if os.path.exists(p):
                rows.append([src, os.path.basename(d)] + [json.load(open(p)).get(k)
                                                          for k in MET])
key = {n: i for i, n in enumerate(ORDER)}
rows.sort(key=lambda x: (x[0], key.get(x[1].split("-")[0].replace("pooled",
          "Pooled Training"), 99), x[1]))
for i, row in enumerate(rows, 2):
    for j, v in enumerate(row, 1):
        c = ws.cell(row=i, column=j, value=v)
        c.font, c.border = BODY, THIN
        if j > 2:
            c.number_format = NUM
widths(ws, [10, 62] + [12] * len(MET))
ws.auto_filter.ref = f"A1:{get_column_letter(2 + len(MET))}{len(rows) + 1}"

for name in ["Benchmark", "Malla", "Best config", "Benchmark rerun"]:
    w = wb[name]
    w.auto_filter.ref = f"A1:{get_column_letter(w.max_column)}{w.max_row}"

assert N_GRID == g_last - 1, (N_GRID, g_last)
wb.save(OUT)
print("benchmark", b_last - 1, "grid", g_last - 1, "best", best_last - 1,
      "rerun", rr_last - 1, "diverged", N_DIV, "detailed", len(rows))
