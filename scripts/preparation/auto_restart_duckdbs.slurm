#!/bin/bash

# ---- SLURM job setup ----
#SBATCH --job-name=auto_restart_duckdbs
#SBATCH --time=3-00:00:00  # 3 days walltime
#SBATCH --output=auto_restart_duckdbs_%j.out
#SBATCH --partition=cpu_rome_all
#SBATCH --account=2024_107
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

source ~/.bashrc

BASE_PATH="/dodrio/scratch/projects/2024_107/CommonCrawl-CreativeCommons"
cd "$BASE_PATH" || { echo "Failed to cd into $BASE_PATH"; exit 1; }
source ".venv/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }


# Only pip install if new changes pulled
if ! git pull | grep -q "Already up to date."; then
    python -m pip install -e . --no-cache-dir
fi

trap 'echo "Caught signal, cleaning up..."; kill "$TIME_CHECK_PID" 2>/dev/null; exit 1' SIGINT SIGTERM SIGHUP

RESUBMIT_WHEN_MIN_LEFT=10
CHECK_RESUBMIT_EVERY_SECONDS=300

# Require one positional argument, either `fineweb` or `fineweb-2`
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <fineweb|fineweb-2>"
    exit 1
fi

FW_VERSION="$1"

# Check if the argument is valid
if [ "$FW_VERSION" != "fineweb" ] && [ "$FW_VERSION" != "fineweb-2" ]; then
    echo "Invalid argument: $FW_VERSION. Use 'fineweb' or 'fineweb-2'."
    exit 1
fi


SCRIPT_DIR="$BASE_PATH/scripts/preparation"
cd "$SCRIPT_DIR"

# Variable to track if we've already resubmitted a job
JOB_RESUBMITTED=false

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

# Convert slurm string to minutes
# Format can be either D-HH:MM:SS or HH:MM:SS
get_minutes() {
    local time_str="$1"
    echo "$time_str" | awk -F'[-:]' '{
        if (NF == 4) {
            # Format: D-HH:MM:SS
            print $1 * 24 * 60 + $2 * 60 + $3
        } else if (NF == 3) {
            # Format: HH:MM:SS
            print $1 * 60 + $2
        } else {
            print "Error: Unexpected time format" > "/dev/stderr"
            exit 1
        }
    }'
}


# Function to check remaining time and resubmit job if needed
check_and_resubmit() {
    # Only proceed if running under SLURM and if we haven't already resubmitted
    if [ -n "$SLURM_JOB_ID" ] && [ "$JOB_RESUBMITTED" = false ]; then
        # Get information about current job
        JOB_INFO=$(scontrol show job $SLURM_JOB_ID)

        # Extract the time limit and elapsed time
        TIME_LIMIT=$(echo "$JOB_INFO" | grep -oP 'TimeLimit=\K[0-9:-]+')
        TIME_ELAPSED=$(echo "$JOB_INFO" | grep -oP 'RunTime=\K[0-9:-]+')
        # Convert to minutes for easy comparison
        TIME_LIMIT_MIN=$(get_minutes "$TIME_LIMIT")
        TIME_ELAPSED_MIN=$(get_minutes "$TIME_ELAPSED")

        # Resubmit if less than 10 minutes remaining (adjust as needed)
        if [ $((TIME_LIMIT_MIN - TIME_ELAPSED_MIN)) -lt $RESUBMIT_WHEN_MIN_LEFT ]; then
            echo "Approaching time limit. Resubmitting job with dependency..."

            # Get the current script path
            SCRIPT_PATH="$(realpath "$0")"

            # Resubmit the job with a dependency on the current job
            # This ensures the new job starts only after the current job completes
            NEW_JOB_ID=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$FW_VERSION" | awk '{print $NF}')

            # Mark that we've resubmitted to avoid multiple submissions
            JOB_RESUBMITTED=true

            echo "Job resubmitted as SLURM job $NEW_JOB_ID with dependency on current job $SLURM_JOB_ID"

            # Exit the current job gracefully - the new job will start after this one ends
            echo "Current job will continue until completion or time limit."
        fi
    fi
}

# Set up a periodic check for time limit
(
    while true; do
        check_and_resubmit
        sleep $CHECK_RESUBMIT_EVERY_SECONDS
    done
) &

TIME_CHECK_PID=$!

CMD="HF_HUB_DOWNLOAD_TIMEOUT=120 python build_duckdbs.py --fw-version $FW_VERSION"

run_and_monitor "$CMD" "$FW_VERSION"  &
RUN_PID=$!

# Wait for active processes to complete
echo "Monitoring processes. Press Ctrl+C to stop all processes."
wait $RUN_PID

trap - SIGINT SIGTERM SIGHUP
if [ -n "${TIME_CHECK_PID:-}" ]; then
    kill "$TIME_CHECK_PID" 2>/dev/null
fi
exit 0

