git tag -a v2.6b47 -m "nanovisQ_SW_v2.6b47 (2024-10-03)"
git push origin v2.6b47
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause