@echo off
set REMOTE=root@68.183.227.9
echo.
echo === Starting Deployment to %REMOTE% ===
echo.

:: Uploading files individually to home (no leading slash)
echo [1/3] Uploading core files...
scp main.py %REMOTE%:main.py
scp .env %REMOTE%:.env
scp -r schemas %REMOTE%:schemas
scp -r services %REMOTE%:services
scp -r routers %REMOTE%:routers

:: Update files on server using cp -rf to correctly merge/overwrite directories
echo [2/4] Updating files in /opt/chatbotv2/...
ssh %REMOTE% "cp -rf main.py .env schemas services routers /var/www/chatbot_backend/"

:: Clean up home directory
echo [3/4] Cleaning up root home...
ssh %REMOTE% "rm -rf main.py .env schemas services routers"

:: Restart service
echo [4/4] Restarting backend service...
ssh %REMOTE% "systemctl restart chatbot_backend && systemctl status chatbot_backend --no-pager"

echo.
echo === Deployment Complete! ===
pause
