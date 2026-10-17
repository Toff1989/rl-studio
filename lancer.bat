@echo off
rem ---------------------------------------------------------------------------------------------
rem  RL Studio - lanceur : mode developpement ou mode compile.
rem
rem    lancer.bat              menu interactif (taper 1, 2, 3 ou Q puis Entree)
rem    lancer.bat dev          lance directement le mode developpement
rem    lancer.bat compile      lance directement la version compilee
rem    lancer.bat build        reconstruit la version compilee puis la lance
rem    lancer.bat menu dry     le deuxieme argument "dry" affiche la commande sans l'executer
rem
rem  La fenetre se ferme toute seule apres le lancement ; elle ne reste ouverte (pause) que
rem  si quelque chose ne va pas, pour que vous puissiez lire le message.
rem ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
title RL Studio - lancement

set "PYTHON=.venv\Scripts\python.exe"
set "EXE=dist\rlstudio\rlstudio.exe"
set "MODE=%~1"
set "DRY=%~2"
set "TENTATIVES=0"
set "MSG="

if /i "%MODE%"=="dev" goto dev
if /i "%MODE%"=="compile" goto compile
if /i "%MODE%"=="build" goto build

:menu
cls
echo ============================================================
echo    RL Studio - choix du mode de lancement
echo ============================================================
echo.
echo    [1]  Mode DEVELOPPEMENT   code source + environnement .venv
echo                              (console visible : erreurs et logs)
echo    [2]  Mode COMPILE         %EXE%
echo                              (comme chez un utilisateur final)
echo    [3]  RECONSTRUIRE         recompile puis lance le mode compile
echo    [Q]  Quitter
echo.
if defined MSG echo    ^>^> %MSG%
if defined MSG echo.
set "CHOIX="
set /p "CHOIX=Votre choix (1, 2, 3 ou Q) puis Entree : "
if /i "%CHOIX%"=="1" goto dev
if /i "%CHOIX%"=="2" goto compile
if /i "%CHOIX%"=="3" goto build
if /i "%CHOIX%"=="Q" goto fin
set /a TENTATIVES+=1
if %TENTATIVES% GEQ 5 goto fin
set "MSG=Choix invalide : tapez 1, 2, 3 ou Q."
goto menu

:dev
if exist "%PYTHON%" goto dev_ok
echo.
echo [ERREUR] Environnement de developpement introuvable : %PYTHON%
echo          Creez-le avec :
echo            python -m venv .venv
echo            .venv\Scripts\pip install -e ".[gui,runtime,dev]"
goto erreur
:dev_ok
echo.
echo Lancement en mode DEVELOPPEMENT (cette fenetre affiche les logs) ...
if /i "%DRY%"=="dry" goto dev_dry
"%PYTHON%" -m rlstudio.launcher
if errorlevel 1 goto dev_echec
goto fin
:dev_dry
echo [dry] "%PYTHON%" -m rlstudio.launcher
goto fin
:dev_echec
echo.
echo [ERREUR] L'application s'est arretee avec une erreur : voir les messages ci-dessus.
goto erreur

:compile
if exist "%EXE%" goto compile_ok
echo.
echo [ATTENTION] La version compilee n'existe pas encore : %EXE%
set "REP="
set /p "REP=La construire maintenant (environ 2 minutes) ? (O/N) : "
if /i "%REP%"=="O" goto build
goto fin
:compile_ok
echo.
echo Lancement de la version COMPILEE ...
if /i "%DRY%"=="dry" goto compile_dry
start "" "%EXE%"
goto fin
:compile_dry
echo [dry] start "" "%EXE%"
goto fin

:build
echo.
echo Reconstruction de la version compilee (PyInstaller) ...
if /i "%DRY%"=="dry" goto build_dry
powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\build.ps1" -SkipInstaller
if errorlevel 1 goto build_echec
if not exist "%EXE%" goto build_echec
echo.
echo Construction terminee.
goto compile_ok
:build_dry
echo [dry] powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\build.ps1" -SkipInstaller
goto fin
:build_echec
echo.
echo [ERREUR] La construction a echoue : voir les messages ci-dessus.
goto erreur

:erreur
echo.
pause
endlocal
exit /b 1

:fin
endlocal
exit /b 0
