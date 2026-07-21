git tag -a v2.7r7 -m "nanovisQ_SW_v2.7r7 (2026-07-21)"
git push origin v2.7r7
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause