git tag -a v2.6b49 -m "nanovisQ_SW_v2.6b49 (2024-11-08)"
git push origin v2.6b49
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause