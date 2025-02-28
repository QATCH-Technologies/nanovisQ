git tag -a v2.6b54 -m "nanovisQ_SW_v2.6b54 (2025-02-17)"
git push origin v2.6b54
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause