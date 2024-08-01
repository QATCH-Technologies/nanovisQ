git tag -a v2.6b45 -m "nanovisQ_SW_v2.6b45 (2024-08-01)"
git push origin v2.6b45
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause