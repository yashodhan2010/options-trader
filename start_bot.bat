@echo off
REM ============================================
REM Options Trading Bot - Auto Launcher
REM Scheduled via Windows Task Scheduler
REM ============================================

REM Activate conda environment
call C:\Users\Yashodhan\miniconda3\Scripts\activate.bat
call conda activate options-trader

REM Change to project directory
cd /d C:\Users\Yashodhan\OneDrive\Documents\Algo\OptionsTrader\options-trader

REM Log startup
echo [%date% %time%] Bot starting... >> logs\scheduler.log

REM Run the bot (paper trading mode)
python run.py --bot --paper >> logs\scheduler.log 2>&1

REM Log exit
echo [%date% %time%] Bot exited with code %ERRORLEVEL% >> logs\scheduler.log
