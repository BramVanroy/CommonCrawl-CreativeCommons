#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=<ADD-YOUR-PARTITION>
#SBATCH --account=<ADD-YOUR-ACCOUNT>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

source ~/.bashrc

set -e

BASE_PATH=<ADD-CommonCrawl-CreativeCommons-PATH>
PARTITION=<ADD-YOUR-PARTITION>
ACCOUNT=<ADD-YOUR-ACCOUNT>
REPO_NAME=<ADD-REPO-NAME>

source "$BASE_PATH/.venv/bin/activate"

cd "$BASE_PATH"

# Only pip install if new changes pulled
if ! git pull | grep -q "Already up to date."; then
    python -m pip install -e . --no-cache-dir
fi

python scripts/upload_slurm.py \
  -j output/CC-MAIN-2020-05 \
  -o parquet_output/CC-MAIN-2020-05 \
  -r "$REPO_NAME" \
  --pipelines_config "$BASE_PATH/configs/upload-config-slurm.yaml" \
  --venv_path "$BASE_PATH/.venv/bin/activate" \
  --account "$ACCOUNT" \
  --partition "$PARTITION"
