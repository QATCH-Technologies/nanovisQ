git tag -a v2.6b46 -m "nanovisQ_SW_v2.6b46 (2024-08-18)"
git push origin v2.6b46
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause