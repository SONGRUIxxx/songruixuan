@echo off
chcp 65001 >nul
cd /d "%~dp0"
call D:\Software\anaconda\Scripts\activate.bat songruixuan
python web_app.py
pause
