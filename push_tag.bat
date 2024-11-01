git tag -a v2.6b48 -m "nanovisQ_SW_v2.6b48 (2024-11-01)"
git push origin v2.6b48
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause