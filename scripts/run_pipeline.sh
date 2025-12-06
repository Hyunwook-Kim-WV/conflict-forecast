#!/bin/bash
# Run complete GDELT conflict prediction pipeline

echo "Starting GDELT Conflict Prediction Pipeline..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run pipeline
python main.py "$@"

echo "Pipeline completed!"
