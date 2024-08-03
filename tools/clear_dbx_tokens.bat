@ECHO OFF
cd %LOCALAPPDATA%\QATCH\nanovisQ\tokens
if %errorlevel% == 1 goto dirnotfound
ECHO Searching for dbx access tokens in QATCH application data...

set cnt=0
for %%A in (*.pem) do set /a cnt+=1
echo Token count = %cnt%
if %cnt% == 0 goto notokens

echo Would you like to clear all tokens from the cache?
CHOICE /C YNC /M "Press Y for Yes, N for No or C for Cancel."
if %errorlevel% == 1 goto cleartokens
goto end

:cleartokens
del *.pem
echo Token(s) cleared!
goto end

:notokens
echo No token(s) found.
goto end

:dirnotfound
echo ERROR: Directory not found! No tokens to clear.
goto end

:end
echo Finished.
pause