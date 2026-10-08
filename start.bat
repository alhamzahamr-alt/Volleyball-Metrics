@echo off
REM Launches the backend (FastAPI/uvicorn), frontend (Vite), and Caddy
REM reverse proxy - as three tabs in one Windows Terminal window if it's
REM installed (it is by default on Win11, and commonly on Win10), otherwise
REM three separate windows. Backend/frontend bind to 0.0.0.0 (--host 0.0.0.0
REM / vite.config.ts's server.host) so another device on the same LAN can
REM reach them - but talk to Caddy's one HTTPS port (443) instead of
REM hitting 5173/8000 directly, see Caddyfile. Find this machine's LAN IP
REM via `ipconfig`, then open https://<that IP> on another device (Caddy
REM installs its own root CA into this machine's trust store automatically
REM the first time it runs - no separate step - so Chrome/Edge here show a
REM normal padlock; other devices still see a one-time warning from their
REM own browser unless they install that CA too).
REM
REM Each service's actual startup logic lives in its own small script
REM (Backend/_run_backend.bat, Frontend/_run_frontend.bat, _run_proxy.bat)
REM rather than inline here - wt's multi-tab command-line syntax turned out
REM to choke on a long &&-chained/quoted commandline in ways that were hard
REM to predict from its docs (failed two different ways before this), so
REM each tab's wt commandline here is just a bare filename instead - no
REM nested quotes, no &&, nothing for it to misparse.

set ROOT=%~dp0
REM %~dp0 always ends in a trailing backslash - fine folded into "...Backend"/
REM "...Frontend" below (no longer trailing), but fatal used bare: a
REM backslash immediately before a closing quote is read as an ESCAPED quote
REM by Windows' argument parsing, not "end of string" - so `-d "%ROOT%"`
REM never actually closes its quote, and swallows everything after it
REM (including the next tab's own arguments) into the directory argument.
REM Strip the trailing backslash for that one bare-root case.
set "ROOT_NOSLASH=%ROOT:~0,-1%"

REM Setup check before anything launches - see Backend/doctor.py. Its exit
REM code 2 means the backend can't start at all (missing packages, a broken
REM PyTorch install), so stop here rather than open three tabs of errors; 1
REM means the app runs but processing a video will fail (missing model
REM files), so pause long enough to read its fix-up list - this window is
REM replaced by the launched ones the moment they open - then carry on.
REM No echo below contains parentheses or ">": inside an if (...) block
REM either one would end the block or redirect the output.
if not exist "%ROOT%Backend\.venv\Scripts\python.exe" (
    echo Backend\.venv not found - see the Getting started section of README.md.
    pause
    exit /b 1
)
if not exist "%ROOT%Frontend\node_modules" (
    echo Frontend\node_modules not found - run npm install in the Frontend folder first.
    pause
    exit /b 1
)
"%ROOT%Backend\.venv\Scripts\python.exe" "%ROOT%Backend\doctor.py"
set DOCTOR_EXIT=%errorlevel%
if %DOCTOR_EXIT% geq 2 (
    echo.
    echo The backend can't start until the problems above are fixed.
    pause
    exit /b 1
)
if %DOCTOR_EXIT% equ 1 (
    echo.
    echo Starting anyway - existing games still open, but processing a new video will fail until the above is fixed.
    pause
)

where wt >nul 2>&1
if not errorlevel 1 (
    wt -w -1 new-tab --title Backend -d "%ROOT%Backend" cmd /k _run_backend.bat ; new-tab --title Frontend -d "%ROOT%Frontend" cmd /k _run_frontend.bat ; new-tab --title Proxy -d "%ROOT_NOSLASH%" cmd /k _run_proxy.bat
) else (
    start "Backend" cmd /k "%ROOT%Backend\_run_backend.bat"
    start "Frontend" cmd /k "%ROOT%Frontend\_run_frontend.bat"
    start "Proxy" cmd /k "%ROOT%_run_proxy.bat"
)
