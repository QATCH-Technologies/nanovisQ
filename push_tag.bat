git tag -a v2.7r6 -m "nanovisQ_SW_v2.7r6 (2026-07-16)"
git push origin v2.7r6
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause