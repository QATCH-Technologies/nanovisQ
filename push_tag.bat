git tag -a v2.6r66 -m "nanovisQ_SW_v2.6r66 (2025-11-20)"
git push origin v2.6r66
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause