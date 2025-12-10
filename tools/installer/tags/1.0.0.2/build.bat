@ECHO OFF
title Build QATCH nanovisQ SW Installer

REM get location to 'Scripts' folder for current Python installation
py -3.10 -c "import sys; import os; print(os.path.join(os.path.split(sys.executable)[0], 'Scripts'))" > pypath.txt
set /p PY_SCRIPTS=<pypath.txt
del pypath.txt

set "PATH=%PATH%;%PY_SCRIPTS%"
REM echo %PATH%

echo Clean
rmdir /s dist

REM set seed to a known repeatable integer value for a reproducible build
set "PYTHONHASHSEED=1"

pyinstaller --log-level INFO "installer.spec"

cd dist
set checksum=
certutil -hashfile "QATCH installer.exe" MD5 | find /i /v "md5" | find /i /v "certutil" > "installer.checksum"
set /p checksum= <"installer.checksum"
echo Calculated MD5: %checksum%

pause
