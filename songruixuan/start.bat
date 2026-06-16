@echo off
set HF_ENDPOINT=https://hf-mirror.com
start powershell -NoExit -Command "conda activate songruixuan; cd '%~dp0'; python songruixuan.py"
