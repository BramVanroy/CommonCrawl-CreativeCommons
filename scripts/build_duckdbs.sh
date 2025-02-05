#!/bin/bash
# Simple while loop for robustly running the DuckDB script
# The Python script may fail from time to time due to network/hub issues
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
    python "$SCRIPT_DIR/build_duckdbs.py"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script completed!"
        break
    else
        echo "Script crashed with exit code $EXIT_CODE. Restarting after sleeping 15m..."
        sleep -m 15
    fi
done
