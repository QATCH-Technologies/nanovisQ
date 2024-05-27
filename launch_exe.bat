@ECHO OFF
TITLE QATCH nanovisQ

REM DO NOT MODIFY THIS LAUNCH SCRIPT FOR QATCH nanovisQ
REM CREATED BY QATCH TECHNOLOGIES LLC, ALL RIGHTS RESERVED

REM QUERY REGISTRY TO FIND THE PERSONAL DIRECTORY
for /f "usebackq tokens=3*" %%D IN (`reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Personal`) do set "PERSONAL=%%D"
for /f "tokens=* USEBACKQ" %%E IN (`call echo %personal%`) do set "PERSONAL=%%E"

set "WORKING=%PERSONAL%\QATCH nanovisQ"

REM QUERY REGISTRY TO FIND THE DESKTOP DIRECTORY
for /f "usebackq tokens=3*" %%D IN (`reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Desktop`) do set "DESKTOP=%%D"
for /f "tokens=* USEBACKQ" %%E IN (`call echo %desktop%`) do set "DESKTOP=%%E"

echo Install directory is: "%CD%"
echo Working directory is: "%WORKING%"
echo Desktop directory is: "%DESKTOP%"
echo.
echo Verifying application checksum...

REM REPORT EXPECTED APPLICATION CHECKSUM WHILE CALCULATING ACTUAL
set "expect=" & REM unset first to be sure
set /p expect=<app.checksum
echo Expect: %expect%

REM CALCULATE THE ACTUAL APPLICATION CHECKSUM FROM BINARY SOURCE
certutil -hashfile "QATCH nanovisQ.exe" MD5 | find /i /v "md5" | find /i /v "certutil" > calc.checksum
set "actual=" & REM unset first to be sure
set /p actual=<calc.checksum
echo Actual: %actual%

REM VERIFY THE ACTUAL CHECKSUM MATCHES THE EXPECTED CHECKSUM
if NOT "%actual%" == "%expect%" (goto error)
echo PASS: Verified!
del calc.checksum
del app.checksum

echo.
echo Launching verified application...

REM RUN THE NEW APPLICATION INSTANCE
REM Create working dir if none exists (hide 'exists' error)
mkdir "%WORKING%" >nul 2>&1 
START "QATCH nanovisQ" /D "%WORKING%" /B "QATCH nanovisQ.exe"

echo.
echo Creating desktop shortcut...

REM FIRST, EXTRACT VERSION FOLDER FROM CURRENT DIRECTORY
set INSTALL_DIR=%CD%
if "%INSTALL_DIR:~-1%" == "\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
for %%f in ("%INSTALL_DIR%") do set "VERSION=%%~nxf"
echo Version: %VERSION%

REM CREATE AND EXECUTE SHELL SCRIPT TO CREATE DESKTOP LINK
set SCRIPT="createDesktopLink.vbs"
echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%CD%\QATCH nanovisQ.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%CD%\QATCH nanovisQ.exe" >> %SCRIPT%
echo oLink.Arguments = "" >> %SCRIPT% & REM explicity remove '-m QATCH' if upgrading PY install to EXE
echo oLink.Description = "%VERSION%" >> %SCRIPT%
echo oLink.IconLocation = "%CD%\QATCH nanovisQ.exe,0" >> %SCRIPT%
echo oLink.WorkingDirectory = "%WORKING%" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%
cscript /nologo %SCRIPT%
del %SCRIPT%

REM COPY THE SHORTCUT TO THE DESKTOP (retain link in project folder)
copy /y *.lnk "%DESKTOP%" & echo.

REM Wait 3 seconds before closing and user a chance to pause
echo The application will automatically open soon...
REM echo Press 'Ctrl+C' once to abort application launch.
REM echo Press any other key to open the application now.
REM timeout 10 >nul 2>&1

REM SELF-DESTRUCT THIS SCRIPT
(goto) 2>nul & del "%~f0"

:error
REM INDICATE CHECKSUM ERROR TO USER
echo ERROR: Application checksum verification failed!
echo Please re-download and try again, or contact support.
echo.
pause