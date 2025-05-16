git tag -a v2.6b60 -m "nanovisQ_SW_v2.6b60 (2025-05-16)"
git push origin v2.6b60
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause