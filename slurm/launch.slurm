#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=<ADD-YOUR-PARTITION>
#SBATCH --account=<ADD-YOUR-ACCOUNT>

# Check if dump argument is provided
if [ -z "$1" ]; then
  echo "Error: Dump argument is required. See https://commoncrawl.org/overview"
  echo "Usage: $0 <dump>"
  exit 1
fi

# Set dump argument
DUMP=$1
BASE_PATH=<ADD-CommonCrawl-CreativeCommons-PATH>
OUTPUT_PATH="$BASE_PATH/output/"
PARTITION=<ADD-YOUR-PARTITION>
ACCOUNT=<ADD-YOUR-ACCOUNT>

source ~/.bashrc
free -h | awk '/^Mem:/ {print $7 " available"}'

set -e
cd "$BASE_PATH"
source ".venv/bin/activate"

# Only pip install if new changes pulled
if ! git pull | grep -q "Already up to date."; then
    python -m pip install -e . --no-cache-dir
fi

echo "Launching!"

python scripts/run_slurm.py \
  --dump "$DUMP" \
  --output_path "$OUTPUT_PATH" \
  --pipelines_config "$BASE_PATH/configs/config-slurm.yaml" \
  --venv_path "$BASE_PATH/.venv/bin/activate" \
  --account "$ACCOUNT" \
  --partition "$PARTITION"
  