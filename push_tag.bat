git tag -a v2.6b52 -m "nanovisQ_SW_v2.6b52 (2025-02-03)"
git push origin v2.6b52
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause