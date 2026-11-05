@echo off
chcp 65001 >nul
cd /d C:\Users\O.G\football_analytics
call .venv\Scripts\activate
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
if not exist predictions mkdir predictions
echo ============================================ >> logs\predictions.log
echo Run started: %date% %time% >> logs\predictions.log
echo ============================================ >> logs\predictions.log
python src\predict_live.py >> logs\predictions.log 2>&1
python src\bet_slip_builder.py >> logs\predictions.log 2>&1
echo Run finished: %date% %time% >> logs\predictions.log
echo. >> logs\predictions.log