git tag -a v2.6r46 -m "nanovisQ_SW_v2.6r46 (2024-08-18)"
git push origin v2.6r46
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause