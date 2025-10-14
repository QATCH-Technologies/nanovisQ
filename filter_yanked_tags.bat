@echo off
setlocal

set "mainfile=tags.txt"
set "removefile=yanked.txt"
set "tempfile=tags_temp.txt"

:: Create temp file with filtered results
type nul > "%tempfile%"

for /f "usebackq delims=" %%a in ("%mainfile%") do (
    findstr /x /c:"%%a" "%removefile%" >nul
    if errorlevel 1 echo %%a>> "%tempfile%"
)

:: Replace original
move /y "%tempfile%" "%mainfile%"

echo Done!