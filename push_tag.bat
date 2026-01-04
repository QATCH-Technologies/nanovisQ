git tag -a v2.6b67 -m "nanovisQ_SW_v2.6b67 (2025-12-22)"
git push origin v2.6b67
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause