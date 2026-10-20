git tag -a v2.7r12 -m "nanovisQ_SW_v2.7r12 (2026-10-08)"
git push origin v2.7r12
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause