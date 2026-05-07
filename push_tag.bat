git tag -a v2.6r70 -m "nanovisQ_SW_v2.6r70 (2026-05-06)"
git push origin v2.6r70
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause