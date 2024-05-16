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
ECHO Step 1 / 3: Updating 'pip', 'wheel', 'pipreqs' and 'pip-tools' modules...
ECHO.
py -m pip install --upgrade pip wheel pipreqs pip-tools
pause
ECHO.
ECHO Step 2 / 3: Running 'pipreqs' to generate a short list of dependencies...
ECHO.
"%PY_SCRIPTS%\pipreqs" --encoding=utf8 --force --debug --savepath=requirements.out "./QATCH/"
REM remove "QATCH" and any found modules that contain "~" as they are invalid distributions that should not be traced:
type requirements.out | findstr /v ~ | findstr /V QATCH > requirements.in
pause
ECHO.
ECHO Step 3 / 3: Running 'pip-compile' to generate a full list of dependencies...
ECHO.
if exist "requirements.txt" del requirements.txt
"%PY_SCRIPTS%\pip-compile" --resolver=backtracking
del requirements.in
del requirements.out
ECHO.
ECHO INFO: Successfully saved requirements file in requirements.txt
ECHO.
PAUSE
