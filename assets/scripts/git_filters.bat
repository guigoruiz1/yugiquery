@echo off
setlocal

:: Define the custom filter in the local Git configuration
git config filter.strip_secrets.clean "sed 's/=[^=]*/=/'"
git config filter.strip_secrets.smudge "cat"

:: Check if the .gitattributes file exists and if it contains the filter definition
if not exist .gitattributes (
    echo "**/*secret* filter=strip_secrets" >> .gitattributes
) else (
    findstr /x "**/*secret* filter=strip_secrets" .gitattributes > nul
    if %ERRORLEVEL% neq 0 (
        echo "**/*secret* filter=strip_secrets" >> .gitattributes
    )
)

:: Install nbstripout and configure it with the specified attributes file in the notebooks directory
pip install nbstripout
nbstripout --install --attributes "notebooks/.gitattributes"

endlocal