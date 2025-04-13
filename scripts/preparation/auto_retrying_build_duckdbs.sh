#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to run and monitor a script with specific parameters
run_and_monitor() {
    local command="$1"
    local process_name="$2"
    
    echo "Starting $process_name..."
    while true; do
        echo "[$process_name] Running: $command"
        eval "$command"
        
        # Check the exit status of the script
        if [ $? -eq 0 ]; then
            echo "[$process_name] Script completed successfully!"
            break
        else
            # If the script failed (non-zero exit code), retry
            echo "[$process_name] Script failed. Retrying in 30 seconds..."
            sleep 30
        fi
    done
}

# Run both processes in background
run_and_monitor "python build_duckdbs.py --fw-version fineweb-2" "fineweb-2" &
FW2_PID=$!

run_and_monitor "python build_duckdbs.py --fw-version fineweb" "fineweb" &
FW_PID=$!

# Wait for both processes to complete
echo "Monitoring both processes. Press Ctrl+C to stop all processes."
wait $FW2_PID $FW_PID

echo "All processes completed."
cd - > /dev/null