#!/usr/bin/env bash
# Everything runs inside Docker. Nothing is installed on the host.
#
# Usage:
#   ./run.sh build              build the CPU image
#   ./run.sh train [args...]    train the GraphSAGE model on the showcase split
#   ./run.sh cv [args...]       5-fold strict-inductive cross-validation
#   ./run.sh ablation [args...] GNN vs edges-removed MLP, plus k sensitivity sweep
#   ./run.sh report             regenerate report.md from existing metrics.json
#   ./run.sh all                train -> cv -> ablation -> report
#   ./run.sh check              leakage check: shuffled labels must give R2 ~ 0
#
# Hybrid ML-DEA study (4 learners x 3 normalisations, independent of the GNN):
#   ./run.sh ml-dea [args...]   run all 12 combinations, figures and report
#   ./run.sh ml-dea-selftest    gradient check + leakage checks
#   ./run.sh ml-dea-report      regenerate outputs/ml_dea/report.md
#   ./run.sh shell              interactive shell in the container
#   ./run.sh clean              remove outputs/
set -euo pipefail

cd "$(dirname "$0")"
IMAGE=gnn-bcc:cpu
PROJ_DIR="$(pwd)"

docker_run() {
  mkdir -p outputs/figures
  docker run --rm -t \
    --user "$(id -u):$(id -g)" \
    -v "${PROJ_DIR}/data:/app/data:ro" \
    -v "${PROJ_DIR}/outputs:/app/outputs" \
    -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp \
    "$IMAGE" "$@"
}

cmd="${1:-train}"; shift || true

case "$cmd" in
  build)
    docker build -t "$IMAGE" .
    echo
    echo "Verifying the image..."
    docker run --rm "$IMAGE" python -c \
      "import torch, torch_geometric, sklearn, matplotlib, pandas; \
       print('torch', torch.__version__); \
       print('torch_geometric', torch_geometric.__version__); \
       print('sklearn', sklearn.__version__); \
       print('OK')"
    ;;
  train)    docker_run python -m src.train "$@" ;;
  cv)       docker_run python -m src.cv "$@" ;;
  ablation) docker_run python -m src.ablation "$@" ;;
  report)   docker_run python -m src.report "$@" ;;
  ml-dea)          docker_run python -m src.mldea.run "$@" ;;
  ml-dea-selftest) docker_run python -m src.mldea.selftest "$@" ;;
  ml-dea-report)   docker_run python -m src.mldea.report "$@" ;;
  check)    docker_run python -m src.train --shuffle-labels --tag shuffled "$@" ;;
  selftest) docker_run python -m src.selftest "$@" ;;
  all)
    docker_run python -m src.train "$@"
    docker_run python -m src.cv "$@"
    docker_run python -m src.ablation "$@"
    docker_run python -m src.report
    ;;
  shell)    docker_run /bin/bash ;;
  clean)    rm -rf outputs && mkdir -p outputs/figures && echo "outputs/ cleared" ;;
  *)        sed -n '2,26p' "$0"; exit 1 ;;
esac
