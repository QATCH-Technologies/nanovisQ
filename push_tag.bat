git tag -a v2.6r60 -m "nanovisQ_SW_v2.6r60 (2025-05-19)"
git push origin v2.6r60
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause