@echo off

REM Merge ".visq" file extension keys to registry
REGEDIT.EXE "%CD%\visq_file_extension_keys.reg"

CHOICE /C YN /M "Do you want to restart file explorer"

IF %ERRORLEVEL% EQU 1 GOTO YES_ACTION
IF %ERRORLEVEL% EQU 2 GOTO NO_ACTION

:YES_ACTION
ECHO You chose Yes. Restarting...

REM Restart file explorer to use new registry keys
tskill explorer

GOTO END

:NO_ACTION
ECHO You chose No. Exiting...
GOTO END

:END
PAUSE