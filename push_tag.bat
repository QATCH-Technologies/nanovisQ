git tag -a v2.6b58 -m "nanovisQ_SW_v2.6b58 (2025-04-30)"
git push origin v2.6b58
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause