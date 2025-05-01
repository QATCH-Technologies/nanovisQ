git tag -a v2.6b59 -m "nanovisQ_SW_v2.6b59 (2025-05-01)"
git push origin v2.6b59
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause