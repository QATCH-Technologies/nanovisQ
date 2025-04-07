git tag -a v2.6b56 -m "nanovisQ_SW_v2.6b56 (2025-04-07)"
git push origin v2.6b56
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause