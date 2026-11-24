#!/usr/bin/env bash
# Encola los despliegues de Fed-MammoBench: espera a que termine el run_mammobench_docker.sh
# que esté activo y luego lanza, uno tras otro, EfficientNet-B0 (retoma) y ResNet18.
# Cada paso es idempotente (omite lo ya hecho). Un fallo no detiene los siguientes.
# Uso: nohup bash scripts/queue_mammobench.sh > outputs/queue_mammobench.log 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[$(date '+%F %T')] esperando a que termine el run actual..."
while pgrep -f "scripts/run_mammobench_docker.sh" >/dev/null; do sleep 30; done
for w in run_docker_efficientnetb0.sh run_docker_resnet18.sh; do
    echo "[$(date '+%F %T')] lanzando mammobench/$w"
    bash "$REPO_DIR/mammobench/$w"; echo "[$(date '+%F %T')] $w terminó con código $?"
done
echo "[$(date '+%F %T')] cola completa"
