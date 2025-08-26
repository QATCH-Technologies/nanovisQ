git tag -a v2.6b61 -m "nanovisQ_SW_v2.6b61 (2025-07-02)"
git push origin v2.6b61
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause