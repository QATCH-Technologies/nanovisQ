@echo off
TITLE Build Requirements
REM Requires Python modules "pipreqs" and "pip-tools"

REM get location to 'Scripts' folder for current Python installation
py -c "import sys; import os; print(os.path.join(os.path.split(sys.executable)[0], 'Scripts'))" > pypath.txt
set /p PY_SCRIPTS=<pypath.txt
del pypath.txt

set "PATH=%PATH%;%PY_SCRIPTS%"
echo %PATH%

ECHO.
ECHO Step 1 / 4: Updating 'pip', 'wheel', 'pipreqs' and 'pip-tools' modules...
ECHO.
py -m pip install --upgrade pip wheel pipreqs pip-tools
pause
ECHO.
ECHO Step 2 / 4: Running 'pipreqs' to generate a short list of dependencies...
ECHO.
if exist "requirements.in" del requirements.in
"%PY_SCRIPTS%\pipreqs" --encoding=utf8 --force --debug --savepath=requirements.out "./QATCH/"
REM remove "QATCH", "qmodel" and any found modules that contain "~" as they are invalid distributions that should not be traced:
type requirements.out | findstr /v ~ | findstr /V QATCH | findstr /V qmodel | findstr /V PyAutoGUI > requirements.io
for /F "tokens=1 delims==" %%a in (requirements.io) do (echo %%a >> requirements.in)
pause
ECHO.
ECHO Step 3 / 4: Running 'pip-compile' to generate a full list of dependencies...
ECHO.
if exist "requirements.txt" del requirements.txt
"%PY_SCRIPTS%\pip-compile" --strip-extras --resolver=backtracking
if not exist "requirements.txt" goto compile_error
del requirements.in
del requirements.io
del requirements.out
ECHO.
ECHO INFO: Successfully saved requirements file in requirements.txt
ECHO.
CHOICE /M "DO YOU WANT TO SYNC YOUR PYTHON ENVIRONMENT WITH THESE REQUIREMENTS NOW"
IF NOT %ERRORLEVEL%==1 GOTO FINISHED
ECHO.
ECHO Step 4 / 4: Running 'pip-sync' to update the active python environment...
ECHO.
REM "%PY_SCRIPTS%\pip-sync" # This command uninstalls other modules which is undesired for dev
py -m pip install --upgrade -r requirements.txt --no-warn-script-location
GOTO FINISHED
:compile_error
ECHO.
ECHO ERROR: pip-compile encountered a fatal error!
:FINISHED
ECHO.
ECHO INFO: Requirements script has finished.
PAUSE

