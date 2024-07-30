git tag -a v2.6b44 -m "nanovisQ_SW_v2.6b44 (2024-07-30)"
git push origin v2.6b44
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause