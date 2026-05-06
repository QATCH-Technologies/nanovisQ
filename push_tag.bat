git tag -a v2.7r1 -m "nanovisQ_SW_v2.7r1 (2026-05-06)"
git push origin v2.7r1
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause