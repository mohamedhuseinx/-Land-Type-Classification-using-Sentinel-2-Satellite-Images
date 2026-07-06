@echo off
title Sentinel-2 Land Classifier API
cd /d "C:\DEPI project"
echo Activating environment...
call .venv\Scripts\activate.bat
echo Starting API server...
start "" http://localhost:8000
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
pause
