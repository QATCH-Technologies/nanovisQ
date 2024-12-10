git tag -a v2.6b50 -m "nanovisQ_SW_v2.6b50 (2024-12-06)"
git push origin v2.6b50
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause