git tag -a v2.7r9 -m "nanovisQ_SW_v2.7r9 (2026-09-09)"
git push origin v2.7r9
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause