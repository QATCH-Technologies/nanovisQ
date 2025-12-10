@ECHO OFF
cd %LOCALAPPDATA%\QATCH\nanovisQ\database
if %errorlevel% == 1 goto dirnotfound
ECHO Searching for existing DB files in QATCH application data...

set cnt=0
for %%A in (*.db) do set /a cnt+=1
echo DB file count = %cnt%
if %cnt% == 0 goto notokens

echo Would you like to clear all DB files from the cache?
CHOICE /C YNC /M "Press Y for Yes, N for No or C for Cancel."
if %errorlevel% == 1 goto cleartokens
goto end

:cleartokens
del *.db
echo DB file(s) cleared!
goto end

:notokens
echo No DB file(s) found.
goto end

:dirnotfound
echo ERROR: Directory not found! No DB files to clear.
goto end

:end
echo Finished.
pause