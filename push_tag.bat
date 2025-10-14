git tag -a v2.6r65 -m "nanovisQ_SW_v2.6r65 (2025-10-14)"
git push origin v2.6r65
git tag -l --sort=taggerdate > tags.txt
call filter_yanked_tags
REM move 'tags.txt' to 'dist' folder
pause