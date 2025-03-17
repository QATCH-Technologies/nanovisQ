git tag -a v2.6b55 -m "nanovisQ_SW_v2.6b55 (2025-03-17)"
git push origin v2.6b55
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause