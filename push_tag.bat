git tag -a v2.6r63 -m "nanovisQ_SW_v2.6r63 (2025-10-09)"
git push origin v2.6r63
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause