@ECHO OFF
title Build QATCH nanovisQ software

REM Check for updates to PyInstaller
py -3.11 -m pip install --upgrade pyinstaller --no-warn-script-location

REM get location to 'Scripts' folder for current Python installation
py -3.11 -c "import sys; import os; print(os.path.join(os.path.split(sys.executable)[0], 'Scripts'))" > pypath.txt
set /p PY_SCRIPTS=<pypath.txt
del pypath.txt

set "PATH=%PATH%;%PY_SCRIPTS%"
REM echo %PATH%

echo Clean
rmdir /s dist

REM set seed to a known repeatable integer value for a reproducible build
set "PYTHONHASHSEED=1"

REM set SOURCE_DATE_EPOCH to guarantee a reproducible build checksum
set SOURCE_DATE_EPOCH=
py -c "from datetime import datetime; epoch = int((datetime.now() - datetime(1970, 1, 1, 4, 0, 0)).total_seconds()); epoch -= epoch %% (60*60*24); epoch += int(60*60*24/2); print(epoch)" > "source_date.txt"
set /p SOURCE_DATE_EPOCH= <"source_date.txt"
echo SOURCE_DATE_EPOCH = %SOURCE_DATE_EPOCH%

set "TF_CPP_MIN_LOG_LEVEL=3" & REM HIDE TENSORFLOW MSGS
make_clean.py & REM clean the working directory of build artifacts
make_data_db.py & REM create data\app.db base database for VisQ.AI
make_version.py & REM modify version.rc to reflect current version
pyinstaller --log-level WARN "QATCH nanovisQ.spec"
REM PyInstaller --onedir --name "QATCH nanovisQ" --clean ^
REM	--splash "QATCH\icons\qatch-splash.png" --noupx ^
REM	--icon "QATCH\ui\favicon.ico" ^
REM	--version-file "version.rc" --console app.py
REM modify .spec SPLASH:   text_pos=(10,470), text_size=10

cd dist
set checksum=
certutil -hashfile "QATCH nanovisQ.exe" MD5 | find /i /v "md5" | find /i /v "certutil" > "app.checksum"
set /p checksum= <"app.checksum"
echo Calculated MD5: %checksum%

mkdir "QATCH nanovisQ"
move "QATCH nanovisQ.exe" "QATCH nanovisQ" >NUL & REM move to subdir
move "app.checksum" "QATCH nanovisQ" >NUL & REM move to subdir
cd .. & REM back to "dev" folder

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
