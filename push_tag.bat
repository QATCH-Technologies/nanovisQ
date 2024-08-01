git tag -a v2.6r45 -m "nanovisQ_SW_v2.6r45 (2024-08-01)"
git push origin v2.6r45
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause