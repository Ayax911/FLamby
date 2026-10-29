# Fed-ISIC2019 binario: modificaciones y justificación

`flamby/datasets/fed_isic2019_binary/` (antes `binario/`) es una copia de `fed_isic2019`
adaptada para clasificar **benigno (0) vs maligno (1)** en lugar de 8 clases. Usa las mismas
imágenes, los mismos 6 centros y el mismo split train/test por centro, de modo que sus resultados
son comparables con los del multiclase.

Contenido:
1. [Definición de la tarea](#1-definición-de-la-tarea)
2. [Cambios en el paquete, archivo por archivo](#2-cambios-en-el-paquete-archivo-por-archivo)
3. [Cambios fuera del paquete](#3-cambios-fuera-del-paquete)
4. [Qué no se tocó](#4-qué-no-se-tocó)
5. [Verificación realizada](#5-verificación-realizada)
6. [Advertencias y pendientes](#6-advertencias-y-pendientes)
7. [Cómo usarlo](#7-cómo-usarlo)

---

## 1. Definición de la tarea

| `target` original | Clase | Etiqueta binaria |
|---|---|---|
| 0 | MEL (melanoma) | **1 · maligno** |
| 2 | BCC (carcinoma basocelular) | **1 · maligno** |
| 7 | SCC (carcinoma espinocelular) | **1 · maligno** |
| 1 | NV (nevus) | 0 · benigno |
| 4 | BKL (queratosis benigna) | 0 · benigno |
| 5 | DF (dermatofibroma) | 0 · benigno |
| 6 | VASC (lesión vascular) | 0 · benigno |
| 3 | AK (queratosis actínica, precancerosa) | **excluida** |

**Por qué se excluye AK:** es una lesión precancerosa. Asignarla a cualquiera de las dos clases
sería una decisión clínica discutible; quitarla deja una frontera benigno/maligno limpia.

**Tamaños resultantes**

| | Multiclase | Binario |
|---|---|---|
| Imágenes totales | 23 247 | 22 380 (−867 AK) |
| Train | 18 413 | 17 885 (11 419 benignas / 6 466 malignas) |
| Test | 4 834 | 4 495 (2 825 / 1 670) |

Por centro (`center=i`), imágenes de train y fracción de malignos:

| Centro | Nombre | Train | % maligno (train) |
|---|---|---|---|
| 0 | BCN | 9 324 | 52,1 % |
| 1 | HAM_vidir_molemax | 3 163 | **0,7 %** (22 imágenes) |
| 2 | HAM_vidir_modern | 2 671 | 26,9 % |
| 3 | HAM_rosendahl | 1 721 | 37,4 % |
| 4 | MSK | 655 | 25,3 % |
| 5 | HAM_vienna_dias | 351 | 17,9 % |

---

## 2. Cambios en el paquete, archivo por archivo

### Carpeta: `binario/` → `fed_isic2019_binary/`
- **Qué:** renombrada.
- **Por qué:** `conf.get_dataset_args` deduce el nombre de la clase a partir del nombre del
  paquete (`fed_isic2019_binary` → `FedIsic2019Binary`). Con `binario` buscaba una clase
  `Binario` que no existía, así que el dataset no se podía cargar desde `fed_benchmark.py`. El
  nombre nuevo se entiende sin contexto y funciona con ese mecanismo sin modificarlo.

### `dataset_creation_scripts/train_test_split`
- **Qué:** nueva columna final **`binary_target`** con 0, 1 o vacío (AK). Ejemplo de cabecera:
  `image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK,target,center,fold,fold2,binary_target`.
- **Por qué una columna nueva y no editar `target`:** no se modifica el resto del CSV (`fold`,
  `fold2`, `center`, one-hot), el split queda idéntico al del multiclase y se puede volver al
  problema de 8 clases leyendo `target`. Es una copia propia de este paquete; el CSV de
  `fed_isic2019` no se tocó.
- **Mapeo aplicado:** `{1:0, 4:0, 5:0, 6:0, 0:1, 2:1, 7:1}`; `target == 3` queda vacío.

### `dataset.py`
| Antes | Ahora |
|---|---|
| `Isic2019Raw`, `FedIsic2019` | `Isic2019BinaryRaw`, `FedIsic2019Binary` |
| `self.targets = df2.target` | `self.targets = df2.binary_target.astype(int)` |
| `pd.read_csv(split)` | `_read_split(split)`: lee el CSV y descarta las filas sin `binary_target` |

- **Por qué:** el filtrado ocurre al leer el CSV, antes de `query("fold == ...")`, así el dataset
  por centro, el pooled y el `__len__` excluyen AK de forma consistente. `reset_index(drop=True)`
  mantiene válido `self.targets[idx]`, que indexa por posición.
- Las augmentations, el `Normalize` y el tamaño de recorte (200) son los mismos que en el
  multiclase.
- Las imágenes se siguen leyendo con `check_dataset_from_config("fed_isic2019")`. No hay una
  segunda copia de los datos, y no hizo falta añadir el nombre nuevo a las listas fijas de
  `flamby/utils.py` (`get_config_file_path` y la de licencias).

### `model.py`
- **Qué:** `nn.Linear(nftrs, 8)` → `nn.Linear(nftrs, 2)`. Se eliminaron el import de
  `FedIsic2019` y el bloque `__main__`.
- **Por qué 2 salidas y no 1:** así la loss (`log_softmax` + `gather`), la métrica (`argmax`) y
  `reporting.predict` (softmax) funcionan sin cambios. Con 1 salida habría que reescribir la loss
  con BCE y la métrica.
- **Por qué se quitó `__main__`:** llamaba a `FedIsic2019(0, True, "train", augmentations=...)`, una
  firma que no existe, y obligaba a importar el dataset desde el modelo (import circular).

### `loss.py`
- **Qué:** `alpha` pasa de 8 pesos a 2: `[1.5662, 2.7660]` (benigno, maligno).
- **Cómo se calcularon:** `N_train / n_clase` sobre el train binario: `17885/11419` y `17885/6466`.
  Es la misma fórmula que usa `benchmark.py` para el multiclase.
- **Por qué:** los 8 pesos originales eran de las 8 clases. Con etiquetas 0/1 el `gather` no
  fallaría, pero tomaría los pesos de MEL (5,58) y NV (2,05), que no corresponden a benigno/maligno.
  Los pesos nuevos compensan el desbalance real (~64 % benigno / 36 % maligno).
- Se mantienen la focal loss y `gamma = 2.0`. Se eliminaron el bloque `__main__` y los imports
  circulares, igual que en `model.py`.

### `metric.py`
- **Sin cambios** (copia idéntica). La *balanced accuracy* con `argmax` sobre logits `(N, 2)` es
  válida en binario y equivale al promedio de sensibilidad y especificidad.

### `common.py`
- **Qué:** `FedClass = FedIsic2019Binary`; nueva constante `NUM_TRAIN_SAMPLES = 17885`;
  `get_nb_max_rounds` la usa en lugar del 18413 fijo.
- **Por qué:** el 18413 contaba las AK. Con `num_updates = 100` el resultado sigue siendo 9 rondas
  (`17885 // 6 // 64 = 46`, `46 * 20 // 100 = 9`; antes `47 * 20 // 100 = 9`), pero el valor queda
  correcto si se cambia `num_updates`.
- `NUM_CLIENTS=6`, `BATCH_SIZE=64`, `NUM_EPOCHS_POOLED=20`, `LR=5e-4` y `Adam` no cambian.

### `__init__.py`
- **Qué:** reexporta desde este paquete. Antes importaba todo de `fed_isic2019`.
- **Por qué:** `fed_benchmark.py` toma `Baseline`, `BaselineLoss`, `metric` y las constantes de
  este `__init__`. Sin el cambio habría seguido entrenando el multiclase aunque el JSON apuntara
  al paquete nuevo.

### `benchmark.py` (entrenador pooled oficial, no lo usa `fed_benchmark.py`)
- Importa de este paquete y usa `FedIsic2019Binary`; `weights = [0] * 2` y `range(2)` en el cálculo
  de pesos; las carpetas de reporte pasan a `fed_isic2019_binary`; se quitó el `import dataset`
  local (dependía del directorio de trabajo).
- Sigue usando `check_dataset_from_config("fed_isic2019")` para encontrar las imágenes y el
  archivo donde guarda el modelo.

### `heterogeneity_pic.py`
- Solo cambian el import y la clase (`FedIsic2019Binary`).

### `README.md`
- Cabecera que remite a este documento. El resto es el README original del dataset.

---

## 3. Cambios fuera del paquete

### `flamby/reporting.py`
- **Qué:** nueva entrada `"fed_isic2019_binary"` en `DATASET_LABELS`: los mismos 6 centros y
  `classes = ["Benigno", "Maligno"]`.
- **Por qué:** `fed_benchmark.make_reporter` devuelve `None` si el dataset no está en
  `DATASET_LABELS`, y entonces la corrida **no genera reportes** (curvas, matriz de confusión,
  ROC, desglose por centro) **sin ningún aviso**. Con 2 clases `compute_metrics` usa las métricas
  binarias (AUC, sensibilidad, especificidad, F1, precisión).

### `flamby/config_isic2019_binary.json` (nuevo)
- Copia de `config_isic2019.json` con `"dataset": "fed_isic2019_binary"` y
  `"results_file": "results_isic2019_binary_42.csv"`.
- **Por qué otro CSV:** `find_xps_in_df` salta los experimentos que ya aparecen en el CSV de
  resultados; reutilizar el del multiclase haría que no se ejecutara nada.
- Los hiperparámetros son los del multiclase, solo como punto de partida (ver §6).

### `CLAUDE.md`
- Se documentó la variante binaria y se quitó la advertencia obsoleta sobre `binario`.

---

## 4. Qué no se tocó

`fed_benchmark.py`, `benchmark_utils.py`, `conf.py`, todas las estrategias (`flamby/strategies/`),
`_Model`, `DataLoaderWithMemory`, `evaluate_model_on_tests`, `flamby/utils.py`.
Reciben el modelo, la loss y los dataloaders por argumento y no
dependen del número de clases.

---

## 5. Verificación realizada

Entorno: conda `flamby`, datos en `~/datasets/fed_isic2019`, sin entrenar.

| Comprobación | Resultado |
|---|---|
| `get_dataset_args("fed_isic2019_binary")` | resuelve `FedIsic2019Binary`, `NUM_CLIENTS=6`, `get_nb_max_rounds(100)=9` |
| Tamaños por centro y pooled (train/test) | coinciden con las cifras de §1 (17 885 / 4 495) |
| Etiquetas del dataset | solo 0 y 1; el centro 1 tiene 22 malignos en train |
| Forward del modelo + loss | logits `(2, 2)`; loss finita (0,42) con etiquetas `[0, 1]` |
| `fed_benchmark.py -s FedAvg --hard-debug --seed 42` con `--runs-dir` | terminó sin error; escribió el CSV de resultados y los reportes (matrices de confusión, ROC, métricas por centro) en `fed_isic2019_binary/seed42/...` |

La corrida de humo usa 0 rondas, así que la balanced accuracy (~0,41 pooled) es la de un modelo
sin entrenar y **no es un resultado**.

**No verificado:** ninguna corrida con entrenamiento real (Pooled, Local o estrategias con
rondas), `benchmark.py` y `heterogeneity_pic.py` (solo se cambiaron imports y constantes, no se
ejecutaron), y el script de Docker.

---

## 6. Advertencias y pendientes

1. **Centro 1 casi sin malignos** (22 de 3 163 en train, unos 4 en test). Sus métricas por centro
   (sensibilidad, AUC) son muy ruidosas; conviene reportarlas con esa salvedad o excluirlo de las
   comparaciones por centro.
2. **Hiperparámetros sin re-tunear.** Los de `config_isic2019.json` se eligieron para 8 clases con
   la focal loss multiclase. Para resultados publicables hay que repetir la malla (lr, `mu`,
   `server_learning_rate`).
3. **Docker.** Resuelto: ver §8. El script acepta `DATASET=fed_isic2019_binary`.
4. **`alpha` fijo en el código.** Si cambia la definición de maligno o el split, hay que
   recalcularlo.
5. **Métrica principal.** Se conserva la balanced accuracy por compatibilidad con el multiclase;
   en binario puede interesar priorizar AUC o sensibilidad. `reporting.py` ya las calcula.
6. **Definición de maligno.** Si se decide incluir AK (como maligna o benigna), hay que cambiar el
   mapeo en el CSV, `NUM_TRAIN_SAMPLES` y `alpha`.

---

## 7. Cómo usarlo

```bash
conda activate flamby
cd flamby/benchmarks

# una estrategia con hiperparámetros por CLI
python fed_benchmark.py -cfp ../config_isic2019_binary.json --seed 42 -s FedAvg -lr 0.01

# baseline pooled
python fed_benchmark.py -cfp ../config_isic2019_binary.json --seed 42 -scb Pooled

# comprobación rápida sin entrenar (0 rondas, 0 épocas)
python fed_benchmark.py -cfp ../config_isic2019_binary.json --seed 42 --hard-debug -s FedAvg -lr 0.01
```

Los reportes quedan en `runs/fed_isic2019_binary/seed<N>/<corrida>/`.

---

## 8. Ejecución en Docker y en la workstation

### Cambios en `scripts/run_isic2019_docker.sh`
- Nueva variable **`DATASET`** (`fed_isic2019` por defecto, o `fed_isic2019_binary`). Fija el
  config (`config_isic2019[_binary].json`), el `OUT_DIR` por defecto (`outputs/isic2019[_binary]`),
  la carpeta de reportes (`runs/$DATASET`) y el `BEST_CONFIG`. Un valor desconocido aborta el script.
- Cada contenedor recibe `-cfp ../$CONFIG_FILE`. `argparse` se queda con el último `-cfp`, así que
  pisa el del `ENTRYPOINT` del Dockerfile y no hizo falta cambiar la imagen.
- `.launch_env` guarda también `DATASET`, `CONFIG_FILE` y `BEST_CONFIG`, para que `stop` y `status`
  vean la misma tarea que el último lanzamiento.
- Sin `DATASET` el comportamiento es idéntico al anterior. Se comprobó con `DRY_RUN=1` para ambas tareas.
- El nombre de la imagen y la etiqueta Docker (`flamby-isic2019`) no cambian, por lo que `stop`
  detiene los contenedores de las dos tareas.

### Rutas de la workstation
```bash
cd /media/sosagro4c/DATA3/03_pregradojulian/FLamby
export DATA_DIR=/media/sosagro4c/DATA3/03_pregradojulian/fed_isic2019
export CACHE_DIR=/media/sosagro4c/DATA3/03_pregradojulian/cache_torch
export OUT_DIR=/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019_binary
DATASET=fed_isic2019_binary DEBUG=1 OUT_DIR=$OUT_DIR-debug bash scripts/run_isic2019_docker.sh
DATASET=fed_isic2019_binary bash scripts/run_isic2019_docker.sh
```
`DATA_DIR` debe contener `ISIC_2019_Training_Input_preprocessed/` y
`ISIC_2019_Training_Metadata_FL.csv`; el binario no necesita datos adicionales.

### Modificar el CSV
El CSV de etiquetas es `dataset_creation_scripts/train_test_split` (dentro del repo, no en
`DATA_DIR`). No se edita a mano: se regenera con

```bash
cd flamby/datasets/fed_isic2019_binary/dataset_creation_scripts
python make_binary_target.py --dry-run                                   # ver números sin escribir
python make_binary_target.py --malignant MEL BCC SCC AK --benign NV BKL DF VASC   # ejemplo
```

Requisitos: pandas (entorno conda `flamby`, o dentro de la imagen:
`docker run --rm -v "$PWD":/w -w /w --entrypoint python flamby-isic2019 make_binary_target.py ...`),
permiso de escritura en el repo, y que cada clase esté en un solo grupo (las no listadas se
excluyen). El script valida que `target` coincida con las columnas one-hot y, con los grupos por
defecto, reproduce el CSV actual byte a byte.

Tras cambiar el mapeo hay que copiar a mano los números que imprime:
`NUM_TRAIN_SAMPLES` (`common.py`) y `alpha` (`loss.py`); si no, `get_nb_max_rounds` y los pesos de
la loss quedan desfasados. Después, relanzar el script de Docker (reconstruye la imagen) y usar un
`OUT_DIR` nuevo: los CSV de resultados anteriores ya no son comparables.
