---
name: fedisic2019-experiments
description: Resume, lanza, monitorea y sube los experimentos Fed-ISIC2019 (seed 42) del fork Ayax911/FLamby que corren en la workstation con Docker. Úsalo cuando el usuario pregunte qué experimentos hay, cuáles faltan, cómo lanzarlos/pararlos/ver estado, qué hacer cuando terminan (rsync, commit, push, revisar duplicados, actualizar el Excel), o por resultados de la malla/benchmark/best-config. Palabras clave: isic2019, malla, grid, benchmark, best config, run.sh, workstation, PROGRESS, terminaron los experimentos.
---

# Experimentos Fed-ISIC2019 (seed 42)

Todo es **seed 42** (decisión del usuario: no correr seeds 43–46). Rama `fedisic2019-reports-docker`
de `Ayax911/FLamby`. Se entrena en la workstation (RTX 6000 Ada 48 GB), no en el portátil.

## Grupos de experimentos

| Grupo | Runs | Nombre de run | CSV (en `OUT_DIR`) | W&B group |
|---|---|---|---|---|
| Benchmark | 15 | Pooled, Local 0–5, 8 estrategias de `flamby/config_isic2019.json` | `results_isic2019.csv` (105 filas = 15×7 tests) | `benchmark` |
| Malla original | 155 (145 OK, 10 NaN) | `seed42_grid_<S>_lr<LR>[_mu<MU>][_slr<SLR>]` | `grid/<run>.csv` → `results_isic2019_grid.csv` | `grid` |
| Malla paper (extensión) | 78 | igual que la malla | igual | `grid` |
| Best config | 8 | `seed42_best_<Strategy>` | `best/` → `results_isic2019_best.csv` | `best_config` |
| Benchmark rerun (accidental) | 7 | — | `results_isic2019_benchmark_rerun.csv` (49 filas) | — |

Valores de malla (paper, Apéndice H): client lr 10^{-3,-2.5,-2,-1.5,-1,-0.5};
server lr los mismos + 1.0 + 10.0; mu {0.001, 0.01, 0.1, 1}. Por estrategia:
FedAvg/Cyclic/FedAvgFineTuning solo lr; FedProx lr×mu; FedAdam/FedYogi/FedAdagrad/Scaffold lr×slr.
`GRID_SKIP` omite a propósito los 10 FedAdam/FedYogi slr=1.0 (ya divergieron) y FAFT lr0.316.
Total malla: 234 combinaciones, 11 omitidas → 223 esperadas.
Pendiente opcional: si el slr "garbled" del paper era 10^0.5, añadir `3.1622776601683795` a `GRID_SLRS` (+24 runs).

Hiperparámetros del JSON (= paper = best config), 9 rondas × 100 pasos locales, batch 64:
FedAvg lr 0.01 · FedProx lr 0.01 mu 0.001 · FedAdam/FedYogi lr 0.01 slr 10^-2.5 · FedAdagrad lr 0.01 slr 10^-1.5 ·
Scaffold lr 0.01 slr 1 · Cyclic lr 10^-2.5 · FAFT lr 0.01 nft 3. Pooled: Adam 5e-4, 20 épocas.
No hay "Ensemble" (FLamby fija `compute_ensemble_perf=False`).

## Rutas

- Workstation: repo `/media/sosagro4c/DATA3/03_pregradojulian/FLamby`, salida
  `/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019`, logs `.../outputs/isic2019/logs/`.
- Repo: copia de la salida en `results_isic2019/seed42/`; Excel `results_isic2019/resultados_isic2019.xlsx`.
- Local: `/home/akira/snap/steam/federallearning/FLamby`, python con pandas/openpyxl:
  `~/miniconda3/envs/flamby/bin/python`.

## Lanzar / monitorear (en la workstation)

`run.sh` fija qué se corre (BENCH/GRID/BEST, listas de malla, GRID_SKIP); edítalo para cada tanda.

```bash
cd /media/sosagro4c/DATA3/03_pregradojulian/FLamby && git pull
export OUT_DIR=/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019
DRY_RUN=1 bash run.sh | grep -c "docker run"      # cuántos runs nuevos
read -s WANDB_API_KEY && export WANDB_API_KEY      # opcional; nunca pegar la key en el chat
nohup bash run.sh > ../outputs/run_<tanda>.log 2>&1 &
bash run.sh status        # PROGRESS.md
bash run.sh stop          # pausar (no Ctrl+C); relanzar el mismo comando reanuda
nvidia-smi                # JOBS procesos python ~5.8 GB c/u, util ~100%
```

Un solo lanzamiento por `OUT_DIR` (flock en `$OUT_DIR/.lock`); un segundo lanzamiento se rechaza.

## Cuando terminan

1. Estado y fallos (en la workstation):
   ```bash
   bash run.sh status | head -40
   grep -l Traceback $OUT_DIR/logs/*.log
   grep -l "Input contains NaN" $OUT_DIR/logs/*.log   # divergencias: esperables con slr≥1 / lr 0.316
   ```
   Fallo por NaN = resultado (divergió), no se relanza. Otro Traceback = bug → investigar.
2. Copiar al repo y subir:
   ```bash
   cd /media/sosagro4c/DATA3/03_pregradojulian/FLamby
   rsync -a --exclude='*.pt' --exclude='*.pth' --exclude='wandb/' $OUT_DIR/ results_isic2019/seed42/
   du -sh results_isic2019/seed42; git status --short | wc -l
   git add results_isic2019/seed42 && git commit -m "Fed-ISIC2019 seed 42: <tanda>" && git push
   ```
3. En local: `git pull`, luego verificar:
   - `results_isic2019.csv` sigue con 105 filas y sin duplicados (method×test).
   - `results_isic2019_grid.csv`: filas = 7 × CSVs en `grid/`; sin duplicados por (run, test).
   - Ningún run aparece dos veces en `grid/` ni en `best/`.
4. Regenerar el Excel con el script de esta skill (hojas Léeme, Benchmark, Malla, Malla mejor,
   Best config, Benchmark rerun, Repeticiones, FLamby paper, Métricas detalladas; los runs esperados
   sin CSV salen en rojo como divergidos) y recalcular las fórmulas con el `recalc.py` de la skill xlsx:
   ```bash
   cd /home/akira/snap/steam/federallearning/FLamby
   PY=~/miniconda3/envs/flamby/bin/python; OUT=results_isic2019/resultados_isic2019.xlsx
   $PY .claude/skills/fedisic2019-experiments/build_xlsx.py $PWD $OUT
   $PY ~/.claude/skills/synced/*/xlsx/scripts/recalc.py $OUT 90   # total_errors debe ser 0
   ```
   Si cambia la malla (p. ej. slr 10^0.5), actualizar `LRS`/`SLRS`/`EXPECTED` y el texto de Léeme en el script.
   Estado al 2026-10-28: malla 233 combinaciones = 201 completas + 32 divergidas.
5. Commit/push solo si el usuario lo pide. Nunca incluir CLAUDE.md, ARCHITECTURE.md, EXPERIMENTS.md.

## Advertencias al interpretar

- Una sola seed: el ruido de repetición (misma config/seed/máquina) llega a ~0.04 de balanced accuracy,
  así que el top-4 federado está empatado estadísticamente.
- Val = test: elegir el mejor punto de la malla es optimista.
- Los nombres de carpeta en `runs/` dependen solo de los hiperparámetros: reruns sobrescriben reportes.
  Los CSV son la fuente fiable.
- `best/` se sobrescribió en el doble lanzamiento (Scaffold log 0.646 vs CSV 0.624).
- La imagen Docker se reconstruyó con pandas 3.1 / scikit-learn 1.9 / wandb 0.31 antes de la malla de 78:
  posible efecto menor en comparabilidad.
- Referencia FLamby publicada: `flamby/results/results_benchmark_fed_isic2019/isic_seed4{2..6}.csv`
  (sin FAFT); nuestros 14 métodos caen dentro de ±2σ.
