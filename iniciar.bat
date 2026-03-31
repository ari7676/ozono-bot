@echo off
title Ozono Bot
echo.
echo  ================================
echo   OZONO BOT - Iniciando...
echo  ================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no encontrado.
    pause & exit
)
echo  Instalando dependencias (primera vez puede tardar 2 min)...
pip install flask flask-cors yfinance pandas numpy requests -q
echo.
echo  Cargando datos... (primera vez tarda ~30 seg)
echo  Abrir: http://localhost:5000
echo.
start "" cmd /c "timeout /t 5 >nul && start http://localhost:5000"
python app.py
pause
