@echo off
rem Telegram buy-signal alert (PC version). Needs TELEGRAM_TOKEN / TELEGRAM_CHAT_ID user env vars.
chcp 65001 > nul
cd /d "%~dp0"
echo ===== %date% %time% ===== >> alert_log.txt
python telegram_alert.py --app ..\app.py >> alert_log.txt 2>&1
