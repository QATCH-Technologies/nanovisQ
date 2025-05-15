@echo off
setlocal enabledelayedexpansion

:: Get the full current working directory
set "curDir=%cd%"

:: Remove the last folder name and store as parent path
for %%A in ("%curDir%") do set "parentDir=%%~dpA"
set "parentDir=%parentDir:~0,-1%"

:: Get the names of the last and second-to-last folder
for %%A in ("%curDir%") do set "lastFolder=%%~nxA"
for %%A in ("%parentDir%") do set "parentFolder=%%~nxA"

:: Extract text after last underscore
set "reversedLast="
for /l %%I in (0,1,255) do (
    set "char=!lastFolder:~%%I,1!"
    if "!char!"=="" goto :afterLastReversed
    set "reversedLast=!char!!reversedLast!"
)
:afterLastReversed
for /f "tokens=1* delims=_" %%A in ("!reversedLast!") do set "revSuffixLast=%%A"
set "suffixLast="
for /l %%I in (0,1,255) do (
    set "char=!revSuffixLast:~%%I,1!"
    if "!char!"=="" goto :afterRevLast
    set "suffixLast=!char!!suffixLast!"
)
:afterRevLast

set "reversedParent="
for /l %%I in (0,1,255) do (
    set "char=!parentFolder:~%%I,1!"
    if "!char!"=="" goto :afterParentReversed
    set "reversedParent=!char!!reversedParent!"
)
:afterParentReversed
for /f "tokens=1* delims=_" %%A in ("!reversedParent!") do set "revSuffixParent=%%A"
set "suffixParent="
for /l %%I in (0,1,255) do (
    set "char=!revSuffixParent:~%%I,1!"
    if "!char!"=="" goto :afterRevParent
    set "suffixParent=!char!!suffixParent!"
)
:afterRevParent

:: Check if both folder suffixes are the same
if /I "!lastSuffix!"=="!parentSuffix!" (
    echo Detected matching suffixes for folders: %parentFolder%\%lastFolder%
    echo Copying contents of "%curDir%" to "%parentDir%"...
    xcopy "%curDir%\*" "%parentDir%\" /E /Y /C >nul

    echo Copy complete.
    cd "%parentDir%"

    :: Wait for caller script to exit
    timeout 1

    echo Deleting "%curDir%"...
    rmdir /S /Q "%curDir%"

    :: Launch application
    start launch

    echo Done.

    timeout 10
    exit
) else (
    echo Folder suffixes do not match. No action taken.
    echo Parent suffix: !parentSuffix!
    echo Last suffix: !lastSuffix!
)

endlocal
