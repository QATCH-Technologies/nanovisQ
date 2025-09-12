git tag -a v2.6b62 -m "nanovisQ_SW_v2.6b62 (2025-09-12)"
git push origin v2.6b62
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause