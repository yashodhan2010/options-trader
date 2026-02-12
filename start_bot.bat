@echo off
REM ============================================
REM Options Trading Bot - Auto Launcher
REM Scheduled via Windows Task Scheduler
REM ============================================

REM Log startup
echo [%date% %time%] Task scheduler triggered >> "C:\Users\Yashodhan\OneDrive\Documents\Algo\OptionsTrader\options-trader\logs\scheduler.log"

REM Wait for network connectivity (up to 60 seconds)
set RETRIES=0
:CheckNetwork
ping -n 1 api.kite.trade >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /a RETRIES+=1
    if %RETRIES% geq 12 (
        echo [%date% %time%] ERROR: No network after 60s, proceeding anyway >> "C:\Users\Yashodhan\OneDrive\Documents\Algo\OptionsTrader\options-trader\logs\scheduler.log"
        goto StartBot
    )
    timeout /t 5 /nobreak >nul
    goto CheckNetwork
)
echo [%date% %time%] Network OK >> "C:\Users\Yashodhan\OneDrive\Documents\Algo\OptionsTrader\options-trader\logs\scheduler.log"

:StartBot
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
