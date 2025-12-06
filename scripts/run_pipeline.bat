@echo off
REM Run complete GDELT conflict prediction pipeline (Windows)

echo Starting GDELT Conflict Prediction Pipeline...

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run pipeline
python main.py %*

echo Pipeline completed!
pause
