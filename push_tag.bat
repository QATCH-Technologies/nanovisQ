git tag -a v2.6b57 -m "nanovisQ_SW_v2.6b57 (2025-04-23)"
git push origin v2.6b57
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause