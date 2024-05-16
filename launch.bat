@ECHO OFF
COLOR F9
REM This script will install Python (if missing) and launch QATCH program
set "python_ver=3.11.7"
set "python_use=3.11"

REM QUERY REGISTRY TO FIND THE PERSONAL DIRECTORY
for /f "usebackq tokens=3*" %%D IN (`reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Personal`) do set "PERSONAL=%%D"
for /f "tokens=* USEBACKQ" %%E IN (`call echo %personal%`) do set "PERSONAL=%%E"

REM QUERY REGISTRY TO FIND THE DESKTOP DIRECTORY
for /f "usebackq tokens=3*" %%D IN (`reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Desktop`) do set "DESKTOP=%%D"
for /f "tokens=* USEBACKQ" %%E IN (`call echo %desktop%`) do set "DESKTOP=%%E"

REM RUN THE NEW APPLICATION INSTANCE
set "WORKING=%PERSONAL%\QATCH nanovisQ" 
mkdir "%WORKING%" & REM Create working dir if none exists

echo Install directory is: "%CD%"
echo Working directory is: "%WORKING%"
echo Desktop directory is: "%DESKTOP%"
echo.

timeout 3

cls
cd /d %~dp0
if exist "requirements.txt" goto :CheckPyVer
goto :Launch

:CheckPyVer
title Checking installed Python version...
echo Checking internet connection...
ping www.google.com -n 1 -w 1000 >nul 2>&1
if %errorlevel% equ 1 (
color fc
echo Not connected to internet. Skipping Python environment update checks.
echo.
echo Please connect to internet or delete "requirements.txt" to remove this check.
echo.
echo Application will launch in 10 seconds, or press any key to continue...
timeout 10 >nul 2>&1
goto :Launch
)
echo Checking installed Python version...
py -%python_use% -c "from platform import python_version; print(python_version())" > python-version.txt
set "current_ver=" & REM unset first to be sure
set /p current_ver=<python-version.txt
del python-version.txt

echo Current Version: %current_ver%
echo Desired Version: %python_ver%
echo.

if "%current_ver%" == "%python_ver%" (
echo Your Python version is up-to-date.
goto :InstallReqsFirst
)

echo Your Python version is out-of-date!
choice /m "Install Python %python_ver% (default:'Y' in 3s)" /T 3 /D Y

if %errorlevel% neq 1 (
echo Skipped Python install.
goto :InstallReqsFirst
)

title Install Python %python_ver% for QATCH nanovisQ...
echo Downloading Python %python_ver% from https://www.python.org/...
set "python_url=https://www.python.org/ftp/python/%python_ver%/python-%python_ver%-amd64.exe"
powershell -Command "(New-Object Net.WebClient).DownloadFile('%python_url%', 'python-install.exe')"

echo Installing Python %python_ver%... (see setup window to install)
REM See https://docs.python.org/3/using/windows.html for options
python-install.exe ^
  InstallAllUsers=0 ^
  PrependPath=0 ^
  Include_doc=0 ^
  Include_launcher=1 ^
  InstallLauncherAllUsers=0 ^
  Include_tcltk=0 ^
  Include_test=0 ^
  SimpleInstall=1 ^
  SimpleInstallDescription="Python for QATCH nanovisQ"
del python-install.exe

echo.
echo Python %python_ver% installed! Environment variables have changed.
echo Re-launching application to finish the setup process...
explorer launch.bat & REM must go thru 'explorer' to refresh ENV VAR %PATH%
goto :NoSelfDestruct

:InstallReqsFirst
echo.
title Checking installed Python module dependencies...
echo Checking internet connection...
ping www.google.com -n 1 -w 1000 >nul 2>&1
if %errorlevel% equ 1 (
color fc
echo Not connected to internet. Skipping Python environment update checks.
echo.
echo Please connect to internet or delete "requirements.txt" to remove this check.
echo.
echo Application will launch in 10 seconds, or press any key to continue...
timeout 10 >nul 2>&1
goto :Launch
)
echo Checking installed Python module dependencies...
py -%python_use% -m pip install --upgrade pip --no-warn-script-location                 | findstr -v /c:"Requirement already satisfied"
py -%python_use% -m pip install --upgrade setuptools wheel --no-warn-script-location    | findstr -v /c:"Requirement already satisfied"
py -%python_use% -m pip install --upgrade -r requirements.txt --no-warn-script-location | findstr -v /c:"Requirement already satisfied"
echo.
echo All required Python modules are up-to-date!
echo.
REM FIRST, EXTRACT VERSION FOLDER FROM CURRENT DIRECTORY
set INSTALL_DIR=%CD%
if "%INSTALL_DIR:~-1%" == "\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
for %%f in ("%INSTALL_DIR%") do set "VERSION=%%~nxf"
echo Version: %VERSION%
set SCRIPT="createDesktopLink.vbs"
py -%python_use% -c "import sys; print(sys.executable)" > where.python
set /p PYTHONPATH=<where.python
set PYTHONPATH=%PYTHONPATH:python.exe=pythonw.exe%
del where.python
ECHO Absolute launch command: "%PYTHONPATH%" -m QATCH
echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%CD%\QATCH nanovisQ.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%PYTHONPATH%" >> %SCRIPT%
echo oLink.Arguments = "-m QATCH" >> %SCRIPT%
echo oLink.Description = "%VERSION%" >> %SCRIPT%
echo oLink.IconLocation = "%CD%\QATCH\ui\favicon.ico" >> %SCRIPT%
echo oLink.WorkingDirectory = "%CD%" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%
cscript /nologo %SCRIPT%
del %SCRIPT%
choice /m "Create desktop shortcut (default:'Y' in 3s)" /T 3 /D Y
if %errorlevel% neq 1 (
echo Desktop shortcut not created.
goto :FinishReqsInstall
)

:CreateDesktopLink
REM for /f "usebackq tokens=3*" %%D IN (`reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Desktop`) do set "DESKTOP=%%D"
REM for /f "tokens=* USEBACKQ" %%E IN (`call echo %desktop%`) do set "DESKTOP=%%E"
echo Creating desktop shortcut in "%DESKTOP%"...
copy /y *.lnk "%DESKTOP%"

:FinishReqsInstall
echo.
echo Application will launch in 3 seconds, or press any key to continue...
timeout 3 >nul 2>&1
del requirements.txt
goto :Launch

:Launch
mkdir "%WORKING%\logged_data" & REM Create working logged_data dir if none exists
cls
echo Requesting administrative privileges to create a symbolic link for logged_data...
timeout 3 >nul 2>&1
powershell -Command "Start-Process cmd -ArgumentList '/c echo working directory is %cd% & mklink /d \"%cd%\logged_data\" \"%WORKING%\logged_data\" & timeout 3' -Verb RunAs -WorkingDirectory '%cd%'"
cls
color & REM f9
set title=QATCH nanovisQ
title %title%
echo Launching application...
START "%title%" /B "QATCH nanovisQ.lnk"
echo This window will close in 3 seconds, or press any key to continue...
timeout 3 >nul 2>&1
color & REM this will restore normal console colors
if exist "requirements.txt" goto :NoSelfDestruct
(goto) 2>nul & del "%~f0"

:NoSelfDestruct

