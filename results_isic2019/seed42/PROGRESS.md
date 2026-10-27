# Progreso Fed-ISIC2019

- Actualizado: 2026-10-27 15:42:19
- Estado: **detenido (terminado, o cortado sin `stop`)**
- Último lanzamiento: `SEEDS="42" GRID=1 BEST=1 JOBS=4 DEBUG=0 bash /media/sosagro4c/DATA3/03_pregradojulian/FLamby/scripts/run_isic2019_docker.sh`
- Para retomar: relanza ese mismo comando; solo corre lo pendiente.

## Benchmark, semilla 42: 15/15

- [x] Pooled Training
- [x] Local 0
- [x] Local 1
- [x] Local 2
- [x] Local 3
- [x] Local 4
- [x] Local 5
- [x] FedAvg100
- [x] FedProx100
- [x] FedAdam100
- [x] FedYogi100
- [x] FedAdagrad100
- [x] Scaffold100
- [x] Cyclic100
- [x] FedAvgFineTuning100

## Malla, semilla 42: 145/155

Pendientes (incluye las que estén en curso):

- seed42_grid_FedAdam_lr0.001_slr1.0
- seed42_grid_FedYogi_lr0.001_slr1.0
- seed42_grid_FedAdam_lr0.0031622776601683_slr1.0
- seed42_grid_FedYogi_lr0.0031622776601683_slr1.0
- seed42_grid_FedAdam_lr0.01_slr1.0
- seed42_grid_FedYogi_lr0.01_slr1.0
- seed42_grid_FedAdam_lr0.0316227766016837_slr1.0
- seed42_grid_FedYogi_lr0.0316227766016837_slr1.0
- seed42_grid_FedAdam_lr0.1_slr1.0
- seed42_grid_FedYogi_lr0.1_slr1.0

## Mejor config (config_isic2019.json), semilla 42: 8/8

- [x] seed42_best_Cyclic
- [x] seed42_best_FedAdagrad
- [x] seed42_best_FedAdam
- [x] seed42_best_FedAvg
- [x] seed42_best_FedProx
- [x] seed42_best_FedYogi
- [x] seed42_best_Scaffold
- [x] seed42_best_FedAvgFineTuning
