git tag -a v2.7b8 -m "nanovisQ_SW_v2.7b8 (2026-09-08)"
git push origin v2.7b8
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause