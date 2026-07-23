@echo off
chcp 65001 >nul
cd /d "%~dp0"
call D:\Software\anaconda\Scripts\activate.bat songruixuan
start http://127.0.0.1:7860
python web_app.py
pause
