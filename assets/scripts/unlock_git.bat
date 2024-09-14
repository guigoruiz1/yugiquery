@echo off
setlocal

:: Get the credential store using Git config
for /f "delims=" %%i in ('git config --get credential.credentialstore') do set credential_store=%%i

:: Handle GPG credential store
if "%credential_store%"=="gpg" (
    set passphrase=%1
    if "%passphrase%"=="" (
        echo Passphrase needed.
        exit /b 1
    ) else (
        echo %passphrase% | gpg --batch --yes --pinentry-mode loopback --sign > nul
    )
) else (
    echo GPG not used.
    exit /b 0
)

endlocal