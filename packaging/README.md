# Empaquetage de RL Studio (PyInstaller + Inno Setup)

## Architecture

L'application installée est **découpée en deux** parce que PyTorch (CUDA) et MATLAB Engine dépendent
du matériel de chaque utilisateur et ne peuvent pas être figés dans un exécutable :

```
Programme (PyInstaller, ~160 Mo)             Runtime de calcul (créé à l'installation)
  rlstudio.exe        interface (sans console)   %LOCALAPPDATA%\RLStudio\runtime\
  rlstudio-tools.exe  scan / provision (console)   python\            Python 3.12 embarqué
  resources\          templates, scripts MATLAB    Lib\site-packages  numpy, gymnasium, stable-baselines3,
  lib\rlstudio\       code Python pur du worker                       PyTorch (cu130 | cu128 | cu126 | CPU),
  vendor\             Python embarqué, get-pip                        matlabengine (version de TON MATLAB)
```

L'interface lance le runtime en sous-processus (`python -m rlstudio.worker`), qui répond en JSON ligne par
ligne : entraînement, explorateur Simulink, test MATLAB, versions installées.

Données utilisateur (jamais dans `Program Files`) : `%LOCALAPPDATA%\RLStudio\` — `settings.json`, `runtime\`,
`logs\`. Les projets sont dans `Documents\RLStudio\projects` par défaut (réglable dans Paramètres).

## Ce que fait l'installateur

1. **Page « Matériel détecté »** : `rlstudio-scan.exe` (stdlib, 8 Mo, extrait dans `{tmp}`) détecte
   - CPU, RAM ;
   - GPU NVIDIA : nom, pilote, *compute capability*, CUDA maximal du pilote (`nvidia-smi`) ;
   - installations MATLAB (registre + dossiers usuels) : release, Simulink, Engine Python présent,
     versions de Python acceptées par l'Engine (lues dans `extern\engines\python\setup.py`).
2. Copie des fichiers (installation par utilisateur par défaut, sans droits administrateur).
3. `rlstudio-tools.exe --provision` crée le runtime avec les versions **adaptées** (`components.py`) :

| Matériel détecté | Composant installé |
|---|---|
| NVIDIA Blackwell (RTX 50xx, cc ≥ 12.0), CUDA pilote ≥ 13.0 / 12.8 | PyTorch `cu130` puis `cu128` (repli automatique) |
| NVIDIA Ampere/Ada, CUDA pilote 12.6 | PyTorch `cu126` |
| Pilote trop ancien pour la carte, CUDA inconnu, pas de NVIDIA, GPU désactivé | PyTorch CPU (+ message explicatif) |
| MATLAB R2026b complet | `matlabengine==26.2.*` |
| MATLAB dont l'Engine n'accepte pas Python 3.12 (ex. R2023a) | pas de MATLAB Engine (raison affichée) |
| MATLAB incomplet / absent | MATLAB désactivé |

   Si un index PyTorch échoue (réseau, version retirée), la suivante est essayée, puis PyTorch CPU.
4. Tout peut être refait plus tard depuis **Paramètres > Composants** (« Installer / réparer »).

Installation silencieuse : `RLStudio-Setup-x.y.z.exe /VERYSILENT` (composants installés par défaut ; `/NOPROVISION=1` pour
les ignorer, `/TASKS="!associate"` pour ne pas associer les fichiers `.rls`).
La désinstallation supprime le runtime et les journaux, **pas** les réglages ni les projets.

## Construire

Prérequis : Python 3.12 (python.org recommandé ; Anaconda fonctionne, le PATH est complété), Inno Setup 6.

```powershell
.\packaging\build.ps1                 # build complet -> dist-installer\RLStudio-Setup-<version>.exe
.\packaging\build.ps1 -SkipInstaller  # seulement dist\rlstudio (vérifie le build gelé avec --selftest)
```

Le script télécharge (une fois, mis en cache dans `packaging\vendor\`) le Python embarqué
(python.org, 3.12.10 = dernière 3.12 avec binaires Windows) et `get-pip.py` (bootstrap.pypa.io).
Il construit l'environnement `.buildvenv` (interface seulement, sans PyTorch), lance PyInstaller, assemble
`dist\rlstudio`, exécute les autotests (`rlstudio-tools.exe --selftest`, scan) puis `ISCC`.

## Points d'attention

- **Signature** : l'installateur et les exe ne sont pas signés (SmartScreen avertira) — à signer pour une diffusion publique.
- **PySide6** est limité à `<6.9` : la 6.12 échouait au chargement des DLL sur la machine de développement.
- Les index PyTorch (`cu130`, `cu128`, `cu126`) évoluent : la liste est dans `components.TORCH_INDEXES`.
- Le Python embarqué fixe la version de Python du runtime ; MATLAB doit l'accepter (voir tableau).
