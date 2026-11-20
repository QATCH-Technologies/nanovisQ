git tag -a v2.7b13 -m "nanovisQ_SW_v2.7b13 (2026-11-20)"
git push origin v2.7b13
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause