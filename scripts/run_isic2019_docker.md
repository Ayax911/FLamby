# Fed-ISIC2019 en Docker — guía de ejecución

Corre todos los experimentos de Fed-ISIC2019 con `scripts/run_isic2019_docker.sh`:
benchmark (pooled, 6 locales, ensemble y 8 estrategias federadas) y la malla de
hiperparámetros (155 corridas federadas), todo con la semilla 42. Cada corrida deja
reportes estilo FedMammoBench y se registra en Weights & Biases.

Rutas de la workstation (Precision 5860, RTX 6000 Ada 48 GB):

| Qué | Ruta |
|---|---|
| Repo | `/media/sosagro4c/DATA3/03_pregradojulian/FLamby` (rama `fedisic2019-reports-docker`) |
| Datos | `/media/sosagro4c/DATA3/03_pregradojulian/fed_isic2019` |
| Resultados | `/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019` |
| Caché de pesos | `/media/sosagro4c/DATA3/03_pregradojulian/cache_torch` |
| W&B | proyecto `flamby` (grupos `benchmark` y `grid`) en forge.coreweave.com |

## 1. Requisitos (una sola vez)

- Driver NVIDIA ≥ 550 (la imagen usa CUDA 12.4).
- Docker con el runtime `nvidia`: `docker info | grep -i runtime` debe mostrar `nvidia`.
  Si no aparece, alguien con sudo instala `nvidia-container-toolkit` y ejecuta
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- Tu usuario en el grupo `docker` (si no, sale `permission denied ... docker.sock`):
  ```bash
  sudo usermod -aG docker $USER
  ```
  Luego cierra sesión y vuelve a entrar. Mientras no lo hagas, cada terminal o ventana de
  tmux nueva necesita `newgrp docker`.
- ~11 GB libres en `/var/lib/docker` para la imagen y ~20 GB en `DATA3` para preparar los datos.
- API key de W&B: <https://wandb.ai/authorize>.

## 2. Código

```bash
cd /media/sosagro4c/DATA3/03_pregradojulian
git clone -b fedisic2019-reports-docker https://github.com/Ayax911/FLamby.git
cd FLamby
```

Si ya está clonado: `git pull`. Después del pull, `git log --oneline -1` debe mostrar el
último commit de la rama.

## 3. Variables (en cada terminal nueva)

```bash
newgrp docker                      # solo si no cerraste sesión tras el usermod
cd /media/sosagro4c/DATA3/03_pregradojulian/FLamby
export DATA_DIR=/media/sosagro4c/DATA3/03_pregradojulian/fed_isic2019
export OUT_DIR=/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019
export CACHE_DIR=/media/sosagro4c/DATA3/03_pregradojulian/cache_torch
export WANDB_API_KEY=<tu_key>
echo ${WANDB_API_KEY:+key-ok}      # debe imprimir key-ok
unset WANDB_PROJECT                # proyecto por defecto: flamby
```

`newgrp docker` abre una shell nueva: haz los `export` **después**.

## 4. Datos: descarga y preprocesamiento (una sola vez)

```bash
tmux new -s isic
bash scripts/run_isic2019_docker.sh prepare     # escribe "y" para aceptar la licencia ISIC
```

Construye la imagen, descarga el zip de 9,1 GB (~15 min), lo descomprime, arma los 6
centros y preprocesa las imágenes (lado corto 224 px + color constancy). Al final imprime
`23247`. Si se corta durante el preprocesamiento, relanzarlo retoma solo esa parte.

Quedan en `DATA_DIR`:

| Entrada | Origen | ¿La usa el entrenamiento? |
|---|---|---|
| `ISIC_2019_Training_Input/` | zip descargado (imágenes originales) | No; se puede borrar después (~9 GB) |
| `ISIC_2019_Training_Metadata.csv` | descargado | No directamente |
| `ISIC_2019_Training_GroundTruth.csv` | descargado | No directamente |
| `ISIC_2019_Training_Metadata_FL.csv` | generado (centros + split train/test) | **Sí** |
| `ISIC_2019_Training_Input_preprocessed/` | generado (~190 MB) | **Sí** |

El modelo ve recortes de **200 × 200** (RandomCrop en train, CenterCrop en test), como
el código original de FLamby.

## 5. Prueba rápida (1 ronda / 1 época)

```bash
DEBUG=1 OUT_DIR=$OUT_DIR-debug WANDB_PROJECT=flamby-debug bash scripts/run_isic2019_docker.sh
```

Revisa al terminar:

```bash
ls $OUT_DIR-debug/runs/fed_isic2019/seed42/     # 15 carpetas
grep -l Traceback $OUT_DIR-debug/logs/*.log     # no debe imprimir nada
```

## 6. Corrida completa

```bash
tmux new -s isic                   # o tmux attach -t isic
GRID=1 JOBS=4 bash scripts/run_isic2019_docker.sh
```

Al arrancar, comprueba:

1. La línea `docker run …` incluye `--init` y `--label flamby-isic2019=seed42_benchmark`
   (si no, el código está viejo: Ctrl+C, `git pull` y relanza).
2. W&B dice `Syncing run pooled-seed42` con un enlace, no `offline`
   (si dice offline: Ctrl+C, revisa `WANDB_API_KEY` y relanza).

Luego sal de tmux con **Ctrl+B, D**; sigue corriendo aunque cierres RustDesk.
Para volver: `tmux attach -t isic`.

Qué hace, en orden:

1. Benchmark de la semilla 42 en un contenedor: pooled (Adam, lr 5e-4, 20 épocas),
   6 locales, ensemble y 8 estrategias con los hiperparámetros del paper
   (`flamby/config_isic2019.json`): 9 rondas × 100 pasos locales. ~2 h.
2. Malla: 155 corridas federadas, `JOBS` a la vez. ~8–11 h con `JOBS=4`.
   - lr de cliente: 0,001 · 0,00316 · 0,01 · 0,0316 · 0,1
   - FedProx mu: 0,001 · 0,01 · 0,1 · 1
   - FedAdam / FedYogi / FedAdagrad / Scaffold server lr: 0,001 · 0,00316 · 0,01 · 0,0316 · 0,1 · 1,0

Mientras corre, `nvidia-smi` y `htop` en otra ventana (Ctrl+B, C). Si la GPU y la CPU van
holgadas, la próxima vez sube a `JOBS=6`.

## 7. Pausar y retomar

**Pausar** (por ejemplo antes de las 8:00), desde otra ventana de tmux con permiso de Docker:

```bash
export OUT_DIR=/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019
bash scripts/run_isic2019_docker.sh stop
docker ps                          # no debe quedar ningún contenedor flamby-isic2019
```

- Usa `stop`, no Ctrl+C: Ctrl+C puede dejar contenedores de la malla usando la GPU.
- `stop` detiene **todos** los contenedores de FLamby, sin importar el `OUT_DIR`.
- Lo terminado se conserva. Lo que estaba entrenando se repite al retomar
  (si se corta en medio de los locales, se repiten los 6 por el ensemble).
- En W&B, las corridas cortadas quedan como *crashed*; filtra por estado *finished*.

**Retomar**, con las variables de la sección 3:

```bash
GRID=1 JOBS=4 bash scripts/run_isic2019_docker.sh
```

Solo corre lo pendiente.

**Ver dónde quedó**:

```bash
bash scripts/run_isic2019_docker.sh status
```

Muestra `$OUT_DIR/PROGRESS.md`, que se reescribe tras cada corrida: estado
(en curso / pausado / detenido), último comando lanzado, corridas en curso, checklist del
benchmark y malla hecha/pendiente.

## 8. Resultados (`OUT_DIR`)

| Ruta | Contenido |
|---|---|
| `runs/fed_isic2019/seed42/pooled/`, `local_<i>/` | `config.yaml`, `metrics.csv`, `plots/` (8 curvas), `test/` (métricas, matriz de confusión, ROC, predicciones, desglose por centro) |
| `runs/fed_isic2019/seed42/<Estrategia>-…/` | `server/` (`metrics.csv`, `best.json`, curvas), `nodes/<centro>/` (`metrics.csv`, `rounds.csv`, curvas, `test/`), `pooled_eval/test/` |
| `results_isic2019.csv` | tabla de FLamby del benchmark |
| `grid/<corrida>.csv` y `results_isic2019_grid.csv` | tabla por corrida de la malla y la unión al final |
| `logs/<corrida>.log` | salida de cada contenedor |
| `PROGRESS.md` | reporte de avance |

Val = test (FLamby no tiene validación): las columnas `val_*` se calculan sobre test,
`test/` evalúa siempre el modelo final y `best.json` es solo informativo. La métrica
oficial de FLamby para ISIC es la balanced accuracy.

## 9. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `permission denied ... docker.sock` o `Cannot talk to the Docker daemon` | Esta terminal no tiene el grupo `docker` | `newgrp docker` (o cerrar sesión y entrar) y repetir los `export` |
| `Docker has no NVIDIA runtime` | Falta `nvidia-container-toolkit` | Sección 1 |
| W&B dice `offline` | `WANDB_API_KEY` no está en esta shell | `export WANDB_API_KEY=…`; lo offline se sube después con `wandb sync <carpeta wandb>` |
| El `docker run` no trae `--init`/`--label` | Código viejo | `git pull` y relanzar |
| `stop` no detiene nada | Contenedores lanzados con código viejo (sin label) | `docker ps` y `docker stop <ID>` |
| Una corrida dice `FAILED` | Error en esa corrida | Ver `logs/<corrida>.log`; relanzar el mismo comando repite solo las fallidas |
| Relanzar no entrena nada | Ya está todo en los CSV de resultados | Correcto; para repetir, borra sus filas o usa otro `OUT_DIR` |
| `DEPRECATED: legacy builder` / `Running pip as the 'root' user` | Avisos de Docker y pip | Inofensivos (opcional: `sudo apt-get install docker-buildx`) |

## Variables del script

| Variable | Por defecto | Para qué |
|---|---|---|
| `DATA_DIR` | `~/datasets/fed_isic2019` | Datos preprocesados |
| `OUT_DIR` | `outputs/isic2019` (`_debug` con `DEBUG=1`) | Resultados |
| `CACHE_DIR` | `~/.cache/torch` | Pesos de EfficientNet |
| `ZIP_DIR` | `DATA_DIR` | Dónde queda el zip en `prepare` |
| `GPU` | `0` | Índice de la GPU en el host |
| `SEEDS` | `42` | Semillas del benchmark |
| `GRID` / `GRID_SEED` | `0` / `42` | Activar la malla / su semilla |
| `GRID_LRS`, `GRID_MUS`, `GRID_SLRS` | ver sección 6 | Valores de la malla |
| `JOBS` | `1` | Corridas de la malla en paralelo |
| `WORKERS` | `4` | Workers del dataloader por contenedor |
| `WANDB_PROJECT` | `flamby` | Proyecto W&B (`""` lo desactiva) |
| `DEBUG` | `0` | 1 ronda / 1 época |
| `DRY_RUN` | `0` | Solo imprime los comandos |
