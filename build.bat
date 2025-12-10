@ECHO OFF
title Build QATCH nanovisQ software


REM Sanity: check if venv exists (note: `%~dp0` will *always* end with a slash)
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Virtual environment not found!
    pause
    exit /b 1
)

REM Load and sync the virtual environment
REM call .venv\Scripts\activate & REM sadly this hangs, do not use
set _OLD_VIRTUAL_PATH=%PATH%
set VIRTUAL_ENV=%~dp0.venv
set PATH=%VIRTUAL_ENV%\Scripts;%PATH%
REM echo VENV: %VIRTUAL_ENV%
REM echo %PATH% & pause & REM testing only
REM NOTE: The above changes are "undone" by calling `deactivate.bat`

REM clean any existing build folders to force a clean build process
echo Clean
rmdir /s dist

REM abort process if user says "no"
if exist "dist\" (
    echo Cleaning "dist" folder was canceled or failed. Cannot proceed.
    pause
    exit /b 1
)

REM if user said yes, delete "build" folder quietly (no prompt)
rmdir /s /q build >nul
if exist "build\" (
    echo Cleaning "build" folder was canceled or failed. Cannot proceed.
    pause
    exit /b 1
)

echo Build folders cleaned successfully.
echo.

py -m pip install --upgrade pip-tools pip setuptools wheel & REM pyinstaller --no-warn-script-location
pip-sync requirements.txt requirements-dev.txt

REM get location to '.venv' folder for current Python installation (if configured correctly)
py -c "import sys; import os; print(os.path.dirname(os.path.dirname(sys.executable)))" > pypath.txt
set /p PY_SCRIPTS=<pypath.txt
del pypath.txt
echo Path to PY scripts: %PY_SCRIPTS%

if not "%VIRTUAL_ENV%" == "%PY_SCRIPTS%" (
    echo Wrong python loaded. Please check your environment paths and try again!
    echo VENV = %VIRTUAL_ENV%
    echo PATH = %PY_SCRIPTS%
    exit /b 1
)

REM set "PATH=%PATH%;%PY_SCRIPTS%"
REM echo %PATH%

REM set seed to a known repeatable integer value for a reproducible build
set "PYTHONHASHSEED=1"

REM set SOURCE_DATE_EPOCH to guarantee a reproducible build checksum
set SOURCE_DATE_EPOCH=
py -c "from datetime import datetime; epoch = int((datetime.now() - datetime(1970, 1, 1, 4, 0, 0)).total_seconds()); epoch -= epoch %% (60*60*24); epoch += int(60*60*24/2); print(epoch)" > "source_date.txt"
set /p SOURCE_DATE_EPOCH= <"source_date.txt"
echo SOURCE_DATE_EPOCH = %SOURCE_DATE_EPOCH%

set "TF_CPP_MIN_LOG_LEVEL=3" & REM HIDE TENSORFLOW MSGS
make_data_db.py & REM create data\app.db base database for VisQ.AI
make_version.py & REM modify version.rc to reflect current version
make_clean.py & REM clean the working directory of build artifacts after making DB and version info
build_pyinstaller.py & REM pyinstaller --log-level WARN "QATCH nanovisQ.spec"
REM PyInstaller --onedir --name "QATCH nanovisQ" --clean ^
REM	--splash "QATCH\icons\qatch-splash.png" --noupx ^
REM	--icon "QATCH\ui\favicon.ico" ^
REM	--version-file "version.rc" --console app.py
REM modify .spec SPLASH:   text_pos=(10,470), text_size=10

REM capture escape character to replace prior echo line
for /f %%A in ('echo prompt $E^| cmd') DO SET "ESC=%%A"
<nul set /p "=Calculating MD5..." & <nul set /p "=%ESC%[G"

cd dist\QATCH nanovisQ
set checksum=
certutil -hashfile "QATCH nanovisQ.exe" MD5 | find /i /v "md5" | find /i /v "certutil" > "app.checksum"
set /p checksum= <"app.checksum"
echo Calculated MD5: %checksum%

REM mkdir "QATCH nanovisQ"
REM move "QATCH nanovisQ.exe" "QATCH nanovisQ" >NUL & REM move to subdir
REM move "app.checksum" "QATCH nanovisQ" >NUL & REM move to subdir
cd ..\.. & REM back to "dev" folder

xcopy /y "docs" "dist\QATCH nanovisQ\"
echo F|xcopy /y "launch_exe.bat" "dist\QATCH nanovisQ\launch.bat" >NUL
REM attrib +s +h  "dist\QATCH nanovisQ\launch.bat"

REM All these (except for "docs" in the root) can be removed
REM xcopy /y /w "QATCH\icons" "dist\QATCH nanovisQ\QATCH\icons\"
REM xcopy /y /s "QATCH\models" "dist\QATCH nanovisQ\QATCH\models\"
REM xcopy /y /s "QATCH\resources" "dist\QATCH nanovisQ\QATCH\resources\"
REM xcopy /y "QATCH\ui\*.ico" "dist\QATCH nanovisQ\QATCH\ui\"
REM xcopy /y "QATCH\*.pdf" "dist\QATCH nanovisQ\QATCH\"
REM xcopy /y "docs" "dist\QATCH nanovisQ\docs\"

REM cd QATCH_Q-1_FW_*
REM for %%I in (.) do set folder=%%~nxI
REM xcopy /y "*.hex" "..\dist\QATCH nanovisQ\%folder%\"
REM xcopy /y "*.pdf" "..\dist\QATCH nanovisQ\%folder%\"

pause

REM Close the virtual environment
deactivate
