git tag -a v2.6r64 -m "nanovisQ_SW_v2.6r64 (2025-10-13)"
git push origin v2.6r64
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause