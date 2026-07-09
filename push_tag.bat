git tag -a v2.7b5 -m "nanovisQ_SW_v2.7b5 (2026-07-09)"
git push origin v2.7b5
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause