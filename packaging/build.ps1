<#
.SYNOPSIS
  Construit RL Studio : PyInstaller (interface + outils), assemblage du dossier d'installation, Inno Setup.

.EXAMPLE
  .\packaging\build.ps1                      # build complet + installateur
  .\packaging\build.ps1 -SkipInstaller       # seulement dist\rlstudio (test rapide)
  .\packaging\build.ps1 -SkipVendor          # sans telecharger Python embarque (installateur incomplet !)

  Prerequis : Python 3.12 (py -3.12), Inno Setup 6. Telechargements (sauf -SkipVendor) :
  python.org (Python embarque) et bootstrap.pypa.io (get-pip.py).
#>
param(
  [string]$Version = "0.3.3",
  [string]$PythonEmbedVersion = "3.12.10",   # derniere 3.12 avec binaires Windows
  [switch]$SkipVendor,
  [switch]$SkipInstaller
)
# PowerShell 5.1 traite tout ecrit sur stderr (avertissements pip) comme une erreur : on verifie les codes de sortie.
$ErrorActionPreference = "Continue"
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what a echoue (code $LASTEXITCODE)" } }
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 1. environnement de build (interface seulement : ni PyTorch ni Stable-Baselines3)
function Find-Python312 {
  # lanceur py, sinon le python du PATH s'il est en 3.12
  foreach ($cand in @(@{exe="py"; args=@("-3.12")}, @{exe="python"; args=@()})) {
    try {
      $v = & $cand.exe @($cand.args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
      if ($LASTEXITCODE -eq 0 -and $v -eq "3.12") { return $cand }
    } catch { }
  }
  throw "Python 3.12 introuvable (necessaire : l'interface et le runtime visent 3.12)."
}
if (-not (Test-Path ".buildvenv\Scripts\python.exe")) {
  $p = Find-Python312
  & $p.exe @($p.args) -m venv .buildvenv
}
$py = ".buildvenv\Scripts\python.exe"
# Python conda (Anaconda) : ses DLL (libffi, openssl...) sont dans Library\bin et PyInstaller ne les trouve
# que si ce dossier est dans le PATH. Un Python python.org n'en a pas besoin (recommande).
$base = (& $py -c "import sys; print(sys.base_prefix)")
if (Test-Path "$base\Library\bin") { $env:PATH = "$base\Library\bin;$env:PATH"; Write-Host "Python conda detecte : PATH complete." }
# pip du venv de build : on ecarte toute configuration pip de la machine (PIP_USER, PIP_REQUIRE_VIRTUALENV,
# pip.ini utilisateur...) et on affiche la sortie (sinon l'erreur reelle est masquee).
Get-ChildItem Env: | Where-Object { $_.Name -like "PIP_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:PIP_USER = "0"
$env:PIP_CONFIG_FILE = "NUL"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
function Pip-Install($what, [string[]]$pipArgs) {
  $out = & $py -m pip install @pipArgs 2>&1 | ForEach-Object { "$_" }
  $code = $LASTEXITCODE
  if ($code -ne 0) { $out | ForEach-Object { Write-Host $_ } }
  $global:LASTEXITCODE = $code
  Check $what
}
& $py -m pip install --quiet --upgrade pip 2>&1 | Out-Null   # non bloquant
Pip-Install "pip (dependances de build)" @("-r", "packaging\requirements-gui.txt")
Pip-Install "pip install -e ." @("--no-deps", "-e", ".")

# 2. icone et PyInstaller
if (-not (Test-Path "packaging\icon.ico")) { & $py packaging\make_icon.py }
# Nettoyage robuste : une instance de RL Studio lancee depuis dist\, OneDrive (synchronisation) ou l'antivirus
# verrouillent des fichiers -> « Acces refuse ». On arrete les instances, on reessaie, puis on echoue clairement.
Get-Process rlstudio, rlstudio-tools, rlstudio-scan -ErrorAction SilentlyContinue | Stop-Process -Force
foreach ($d in "dist", "build") {
  for ($i = 1; (Test-Path $d) -and $i -le 5; $i++) {
    Get-ChildItem $d -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object { try { $_.Attributes = "Normal" } catch { } }
    Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
    if (Test-Path $d) { Start-Sleep -Seconds 2 }
  }
  if (Test-Path $d) {
    throw "Impossible de supprimer '$d' (verrouille). Ferme RL Studio / l'Explorateur sur ce dossier, mets le depot hors OneDrive (ou clic droit > Toujours conserver sur cet appareil, ou suspends la synchro), puis relance."
  }
}
& $py -m PyInstaller packaging\rlstudio.spec --noconfirm --distpath dist --workpath build\spec
Check "PyInstaller (application)"
& $py -m PyInstaller --onefile --console --name rlstudio-scan --icon "$root\packaging\icon.ico" `
  --paths "$root\src" --distpath "$root\dist" --workpath "$root\build\scan" --specpath "$root\build\scan" `
  --exclude-module PySide6 --exclude-module numpy --exclude-module yaml "$root\packaging\scan_entry.py"
Check "PyInstaller (scan)"

# 3. disposition de l'application installee (voir src/rlstudio/paths.py)
$app = "dist\rlstudio"
New-Item -ItemType Directory -Force "$app\resources", "$app\lib", "$app\vendor" | Out-Null
Copy-Item -Recurse -Force templates "$app\resources\templates"
Copy-Item -Recurse -Force matlab "$app\resources\matlab"
Copy-Item packaging\icon.ico "$app\resources\icon.ico"
# code Python pur vu par le runtime de calcul (sans l'interface)
Copy-Item -Recurse -Force src\rlstudio "$app\lib\rlstudio"
Remove-Item -Recurse -Force "$app\lib\rlstudio\gui" -ErrorAction SilentlyContinue
Get-ChildItem "$app" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Copy-Item packaging\requirements-runtime.txt "$app\vendor\"
# mentions legales (LGPL v3 de Qt : texte de licence, notices, offre de sources)
Copy-Item THIRD_PARTY_NOTICES.md "$app\"
Copy-Item -Recurse -Force licenses "$app\licenses"
if (Test-Path LICENSE) { Copy-Item LICENSE "$app\LICENSE" } else { Write-Warning "Pas de fichier LICENSE a la racine : a ajouter avant toute diffusion." }

# 4. Python embarque + get-pip (le runtime est cree a l'installation)
if (-not $SkipVendor) {
  New-Item -ItemType Directory -Force packaging\vendor | Out-Null
  $zip = "packaging\vendor\python-$PythonEmbedVersion-embed-amd64.zip"
  if (-not (Test-Path $zip)) {
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonEmbedVersion/python-$PythonEmbedVersion-embed-amd64.zip" -OutFile $zip
  }
  if (-not (Test-Path "packaging\vendor\get-pip.py")) {
    Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile "packaging\vendor\get-pip.py"
  }
  Copy-Item $zip, "packaging\vendor\get-pip.py" "$app\vendor\"
} else {
  Write-Warning "Python embarque ignore : l'installateur ne pourra pas creer le runtime de calcul."
}

# 5. verification du build gele
$env:QT_QPA_PLATFORM = "offscreen"
& $py packaging\check_bundle.py $app; Check "Controle des licences (modules Qt LGPL uniquement)"
& "$app\rlstudio-tools.exe" --selftest; Check "Autotest (console)"
# l'exe sans console n'est pas attendu par `&` : Start-Process -Wait recupere son code de sortie
$gui = Start-Process "$app\rlstudio.exe" -ArgumentList "--selftest" -Wait -PassThru
if ($gui.ExitCode -ne 0) { throw "Autotest de rlstudio.exe a echoue (code $($gui.ExitCode))" }
$scan = & "$app\rlstudio-tools.exe" --scan
$scan | Select-Object -First 3
$global:LASTEXITCODE = 0

# 6. installateur
if (-not $SkipInstaller) {
  $iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $iscc) { throw "Inno Setup 6 introuvable" }
  & $iscc "/DAppVersion=$Version" packaging\installer.iss
  Check "Inno Setup"
  Write-Host "Installateur : dist-installer\RLStudio-Setup-$Version.exe"
}
