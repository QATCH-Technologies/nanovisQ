git tag -a v2.7b12 -m "nanovisQ_SW_v2.7b12 (2026-10-07)"
git push origin v2.7b12
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause