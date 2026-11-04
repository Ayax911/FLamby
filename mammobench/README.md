# Fed-MammoBench en FLamby (benigno vs maligno)

FedMammoBench adaptado a FLamby: mismo `fed_benchmark.py`, mismas estrategias (FedAvg, FedProx,
FedAdam, FedAdagrad, FedYogi, Scaffold, Cyclic, FedAvgFineTuning), y un dataset nuevo con el mismo
contrato que `fed_isic2019_binary`. Es **binario** (0 = Benign, 1 = Malignant), así que no hizo
falta una variante multiclase.

| pieza | dónde |
|---|---|
| paquete del dataset (`FedMammobench`, `Baseline`, `BaselineLoss`, `metric`) | `flamby/datasets/fed_mammobench/` |
| manifests y generador | `flamby/datasets/fed_mammobench/dataset_creation_scripts/` |
| config de estrategias | `flamby/config_mammobench.json` |
| nombres de centros/clases para los reportes | `flamby/reporting.py` (`fed_mammobench`) |
| lanzador conda y prueba rápida | `mammobench/run.sh`, `mammobench/smoke_test.py` |
| Docker (workstation) | `mammobench/run_docker_*.sh`, `scripts/run_mammobench_docker.sh`, `dockers/Dockerfile.mammobench` |
| variante EfficientNetV2 | `flamby/datasets/fed_mammobench_efficientnetv2/`, `flamby/config_mammobench_efficientnetv2.json` |
| variante ResNet18 | `flamby/datasets/fed_mammobench_resnet18/`, `flamby/config_mammobench_resnet18.json` |

## Clientes y partición

4 clientes, uno por base de datos (`center=i`): **0 cmmd, 1 kau-bcmd, 2 cdd-cesm, 3 inbreast**
(mismo orden que los nodos federados de FedMammoBench). La partición train/val/test es la de
`FedMammoBench/manifests/fedmammobench_norm_0_1.csv`, **sin tocarla** (disjunta por paciente;
`make_manifests.py` lo verifica).

| | train | val | test |
|---|---|---|---|
| cmmd | 3776 | 474 | 472 |
| kau-bcmd | 1764 | 220 | 222 |
| cdd-cesm | 803 | 100 | 100 |
| inbreast | 328 | 42 | 40 |
| **total** | **6671** | **836** | **834** |

Manifests (`dataset_creation_scripts/`):
- `mammobench_split.csv`: equivalente a `train_test_split` de ISIC. Columnas `image` (ruta relativa
  a la raíz de imágenes), `binary_target`, `center`, `split`, `fold` (= `split`, selección pooled) y
  `fold2` (`<split>_<center>`, selección por cliente).
- `manifests/<base>_<split>.csv`: lo mismo, un archivo por cliente y partición, para inspeccionar.

Regenerar (también para los TIFF en [-1, 1] con `--normalization neg1_1`):
```bash
python flamby/datasets/fed_mammobench/dataset_creation_scripts/make_manifests.py \
  --source ../FedMammoBench/manifests/fedmammobench_norm_0_1.csv --image-root ../data/preproccesed_julian --check
```
Si cambian los números, copiar `NUM_TRAIN_SAMPLES` (common.py) y `alpha` (loss.py) que imprime.

**Validación.** FLamby solo tiene train/test. `FedMammobench(center=i, train=...)` mapea
`train=True` → train y `train=False` → test (lo que evalúa `fed_benchmark.py`). Para val:
`FedMammobench(center=i, split="val")` (o `pooled=True`); sin augmentation.

## Imágenes y modelo (como Fed-ISIC2019 binario, con ImageNet y `norm_0_1`)

- TIFF float32 de 1 canal en [0, 1], 224×224 → réplica a 3 canales → `Normalize` con media 0.449 /
  std 0.226 (promedio de las de ImageNet, como exp17 de FedMammoBench). Augmentation de train: hflip
  0.5, vflip 0.2, rotación 15°, blur 0.3 (torchvision sobre tensores, sin albumentations).
- `Baseline` de `fed_mammobench`: **EfficientNet-B0** (`efficientnet-pytorch`, pesos ImageNet), igual
  que ISIC binario: se entrena **entera** y la última capa es `Linear(1280, 2)`.
- Pérdida: focal ponderada de FLamby con `alpha = N/n_clase = [1.5186, 2.9284]`; métrica: balanced
  accuracy.

## Ejecutar

```bash
conda activate flamby
bash mammobench/run.sh                         # pooled + 4 locales + 8 estrategias, semilla 42
bash mammobench/run.sh -s FedAvg -lr 0.01      # una estrategia
python mammobench/smoke_test.py --scratch      # prueba rápida sin descargar pesos
```
Las imágenes se buscan en: `data_path=`, `$MAMMOBENCH_DATA` (`DATA_DIR` en run.sh),
`dataset_creation_scripts/dataset_location.yaml` (`dataset_path:`), y por defecto `../data/preproccesed_julian`.
La primera vez torchvision descarga los pesos ImageNet (~100 MB).

## EfficientNetV2 (`fed_mammobench_efficientnetv2`)

Mismos datos, partición, clientes, pérdida, métrica y presupuesto; solo cambia el backbone:
torchvision `efficientnet_v2_s` (`IMAGENET1K_V1`), entrenado entero, última capa `Linear(1280, 2)`.
Config: `flamby/config_mammobench_efficientnetv2.json`. `MAMMOBENCH_EFFNET=m|l` cambia el tamaño.

## ResNet18 (`fed_mammobench_resnet18`)

Tercera variante, con la misma receta: torchvision `resnet18` (`IMAGENET1K_V1`), entrenado entero,
última capa `Linear(512, 2)`. Config: `flamby/config_mammobench_resnet18.json`; lanzador:
`mammobench/run_docker_resnet18.sh`. Salidas en `outputs/mammobench_resnet18`.

## Docker en la workstation (GPU)

Mismo mecanismo que `run_binary.sh` de ISIC (`scripts/run_mammobench_docker.sh`, imagen
`dockers/Dockerfile.mammobench`): benchmark (pooled + 4 locales + 8 estrategias) + `best` + malla de
233 corridas, semilla 42, `JOBS` contenedores a la vez, pausa/reanuda, `PROGRESS.md`.

```bash
cd /media/sosagro4c/DATA3/03_pregradojulian/FLamby && git pull
export WANDB_API_KEY=...            # https://wandb.ai/authorize (o `wandb login` una vez)
export WANDB_ENTITY=<usuario-o-equipo>   # opcional
bash mammobench/run_docker_efficientnetb0.sh weights     # una vez: cachea pesos ImageNet (ver abajo)
DEBUG=1 OUT_DIR=outputs/mm_debug bash mammobench/run_docker_efficientnetb0.sh   # humo: 1 ronda
bash mammobench/run_docker_efficientnetb0.sh             # EfficientNet-B0 completo
bash mammobench/run_docker_efficientnetv2.sh       # EfficientNetV2 completo (después, o con otra GPU)
bash mammobench/run_docker_resnet18.sh             # ResNet18 completo
bash mammobench/run_docker_<modelo>.sh status | stop
```
Variables (todas opcionales salvo la de W&B si quieres subir en línea):

| variable | para qué | default |
|---|---|---|
| `WANDB_API_KEY` | subir a W&B en línea; sin ella (ni `~/.netrc`) queda offline en `outputs/.../runs/` → `wandb sync` | — |
| `WANDB_ENTITY` | usuario/equipo de W&B | el de tu cuenta |
| `WANDB_PROJECT` | proyecto (`""` desactiva W&B) | `flamby-mammobench` |
| `WANDB_MODE` | `offline` / `online` / `disabled` | automático |
| `DATA_DIR` | carpeta que contiene `norm_0_1/` | `<repo>/../data/preproccesed_julian` |
| `CACHE_DIR` | caché de pesos ImageNet (`TORCH_HOME`) | `<repo>/../cache_torch` |
| `OUT_DIR` | salidas | `outputs/mammobench[_efficientnetv2]` |
| `GPU` / `JOBS` | índice de GPU / contenedores simultáneos | `0` / `4` |
| `GRID` `BEST` `BENCH` | `0` omite esa parte | `1` |
| `MAMMOBENCH_EFFNET` | tamaño de EfficientNetV2 (`s`, `m`, `l`) | `s` |

Con el layout actual de la workstation (`03_pregradojulian/{FLamby,data,cache_torch}`) los defaults
de `DATA_DIR` y `CACHE_DIR` ya apuntan bien; no hace falta exportarlos. W&B agrupa por
`<DATASET>_benchmark|grid|best_config`, así que ambos backbones conviven en el mismo proyecto.
Ejecutar los dos a la vez en la misma GPU es posible pero cada uno ya usa `JOBS` contenedores.

## Lo que hay que saber

- `NUM_EPOCHS_POOLED = 60` (presupuesto de los federados de FedMammoBench, 30 rondas × 2 épocas);
  con `num_updates=100` son 15 rondas. FLamby/ISIC usa 20.
- Los hiperparámetros de `config_mammobench.json` son copia de `config_isic2019_binary.json`, sin
  ajustar a este dataset.
- kau-bcmd tiene 6 malignos en test (82 en train): su balanced accuracy local es muy ruidosa.
- `scripts/run_isic2019_docker.sh` no se tocó; MammoBench usa su propio `scripts/run_mammobench_docker.sh`.
- Verificado: scripts Docker en `DRY_RUN` (233 corridas de malla + 8 best; la clave de W&B no se imprime), manifests (8341 TIFF existen), conteos por cliente, rango de los tensores, freeze,
  pérdida/métrica y una ronda de FedAvg con subconjuntos y pesos aleatorios. **No** se corrió el
  benchmark completo ni con pesos ImageNet (la descarga no terminó en esta máquina).
