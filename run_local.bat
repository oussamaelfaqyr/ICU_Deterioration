@echo off
echo ========================================================
echo Starting ICU Project Locally (Without Docker)
echo ========================================================
echo.

echo [1/4] Installing Python requirements...
pip install -r requirements-api.txt
echo.

echo [2/4] Starting MLflow Tracking Server (Port 5000)...
:: We use 'start' to open this process in a new terminal window
start "MLflow Server" cmd /k "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"

echo Waiting a few seconds for MLflow to initialize...
timeout /t 5 /nobreak >nul
echo.

echo [3/4] Starting FastAPI Backend (Port 8000)...
:: Set environment variable for API to find MLflow
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
start "FastAPI Server" cmd /k "uvicorn api:app --host 127.0.0.1 --port 8000"

echo Waiting a few seconds for API to initialize...
timeout /t 5 /nobreak >nul
echo.

echo [4/4] Starting Streamlit Dashboard (Port 8501)...
:: Set environment variable for Dashboard to find the API
set API_URL=http://127.0.0.1:8000
start "Streamlit Dashboard" cmd /k "streamlit run dashboard/app.py --server.port=8501"

echo.
echo ========================================================
echo All services are starting! 
echo 3 new command prompt windows should have opened.
echo To stop the services later, just close those new windows.
echo ========================================================
pause
