git tag -a v2.6r53 -m "nanovisQ_SW_v2.6r53 (2025-02-10)"
git push origin v2.6r53
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause