git tag -a v2.7b3 -m "nanovisQ_SW_v2.7b3 (2026-05-18)"
git push origin v2.7b3
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause