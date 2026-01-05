git tag -a v2.6b68 -m "nanovisQ_SW_v2.6b68 (2026-01-05)"
git push origin v2.6b68
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause