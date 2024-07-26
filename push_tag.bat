git tag -a v2.6b43 -m "nanovisQ_SW_v2.6b43 (2024-07-26)"
git push origin v2.6b43
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause