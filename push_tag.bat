git tag -a v2.6r69 -m "nanovisQ_SW_v2.6r69 (2026-02-24)"
git push origin v2.6r69
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause