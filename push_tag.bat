git tag -a v2.6b51 -m "nanovisQ_SW_v2.6b51 (2024-12-23)"
git push origin v2.6b51
git tag -l --sort=taggerdate > tags.txt
REM move 'tags.txt' to 'dist' folder
pause