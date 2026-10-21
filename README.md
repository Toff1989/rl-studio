<p align="center">
  <img src="docs/logo.png" width="120" alt="Logo de RL Studio">
</p>

<h1 align="center">RL Studio</h1>

<p align="center">
  <b>Créer, partager et entraîner des projets d'apprentissage par renforcement — en Python ou sur un modèle Simulink.</b>
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-0.4.1-4f7cf0">
  <img alt="licence MIT" src="https://img.shields.io/badge/licence-MIT-3ecf8e">
  <img alt="Python 3.10 à 3.12" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-3776AB?logo=python&logoColor=white">
  <img alt="Windows" src="https://img.shields.io/badge/plateforme-Windows-0078D6?logo=windows&logoColor=white">
  <br>
  <img alt="Qt PySide6" src="https://img.shields.io/badge/Qt%20(PySide6)-6.8%20%C2%B7%20LGPLv3-41CD52?logo=qt&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU%20auto-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Stable-Baselines3" src="https://img.shields.io/badge/Stable--Baselines3-SAC%20%C2%B7%20PPO%20%C2%B7%20TD3-4f7cf0">
  <img alt="MATLAB Simulink" src="https://img.shields.io/badge/MATLAB-R2026b%20%C2%B7%20Simulink-0076A8">
  <img alt="format .rls" src="https://img.shields.io/badge/projets-.rls-8b93a7">
</p>

## Fonctionnalités

- **Gestion de projets** : créer, dupliquer, renommer, supprimer (corbeille récupérable), un dossier par projet.
- **Deux types d'environnements** : Python (`gymnasium.Env`) ou modèle Simulink décrit sans code (signaux, paramètres, récompense).
- **Conversion d'un modèle Simulink existant** : le bouton *Convertir un modèle Simulink…* (barre latérale) analyse un `.slx`, propose entrées → actions et sorties → observations, puis crée un projet avec une **copie** adaptée du modèle (l'original n'est jamais modifié) et la teste à vide.
- **Partage** : export/import de projets en un seul fichier **`.rls`** (double-clic ou glisser-déposer).
- **Entraînement** en direct (récompense, observation, actions) avec comparaison à une politique de référence ; algorithmes SAC, PPO et TD3.
- **Évaluation** : rejeu d'un entraînement, d'un modèle mathématique extrait ou de paliers d'actions aléatoires, avec courbes en direct, plusieurs épisodes, graine et vitesse au choix ; un **modèle identifié** peut être rejoué à côté du vrai système (courbes en pointillé, écart par signal).
- **Modèle mathématique** : extraction de la politique apprise sous forme de PID, table de gains, arbre de décision ou équation, validée en boucle fermée, exportable en Python / MATLAB / Simulink ([détails](docs/distillation.md)).
- **Identification sans politique** (`rlstudio identify`) : balayage d'un modèle Simulink ou Python → carte entrées → sorties **en régime permanent** (défaut) ou dynamique `x(k+1) = x + f(x, u)` ; modèle réduit (linéaire, polynôme, équations parcimonieuses SINDy, réseau de neurones) validé en simulation libre, exportable en Python / MATLAB ([détails](docs/identification.md)).
- **Aide intégrée pour débuter** : une infobulle sur chaque bouton, champ, colonne de tableau et graphique, et un **guide d'utilisation** (touche **F1**, bouton *Guide d'utilisation* de la barre latérale, ou *Paramètres → Guide d'utilisation*) avec sommaire et recherche, ouvert à la section de l'onglet affiché ([docs/guide.md](docs/guide.md)).
- **Erreurs toujours visibles** : toute erreur (fin de calcul en échec, ligne d'erreur du journal, exception inattendue) s'affiche aussi dans un bandeau en haut de la fenêtre, avec un bouton *Détails…* (texte complet, copie).
- **Installateur Windows** qui détecte le matériel (GPU NVIDIA, MATLAB) et installe les composants adaptés.
- **Thèmes** Sombre, Clair, Matrix et personnalisables ; tous les réglages généraux dans la fenêtre *Paramètres*.

```
projects/<projet>/        un dossier par projet : project.yaml, env.py / model.slx, runs/
templates/                huit modèles de départ (voir plus bas)
matlab/                   rlstudio_step.m (simulation pas à pas), rlstudio_inspect.m (explorateur), conversion de modèles
docs/                     guide d'utilisation, distillation, identification, notes de version
src/rlstudio/             project.py, archive.py (.rls), envs/, training.py, worker.py, distill/, identify/, components.py, hardware.py
src/rlstudio/gui/         app.py (fenêtre), tab_*.py (un module par onglet), project_actions.py, distill_tab.py, identify_tab.py,
                          eval_tab.py, guide.py, notify.py (bandeau d'erreurs), settings_dialog.py, convert_dialog.py
packaging/                PyInstaller + Inno Setup : voir packaging/README.md
tests/                    pytest : voir « Développement »
```

## Application installée (Windows)

Installateur `RLStudio-Setup-x.y.z.exe` : il détecte le matériel (GPU NVIDIA, MATLAB) et installe les
composants de calcul adaptés (PyTorch CUDA ou CPU, MATLAB Engine de la bonne version). Les réglages
généraux sont dans la fenêtre **Paramètres** (apparence, dossier des projets, MATLAB, GPU, composants).
Il associe aussi les fichiers `.rls` à l'application.
Détails d'architecture et de build : [packaging/README.md](packaging/README.md).

## Modèles de projet fournis

| Modèle | Type | Contenu |
|---|---|---|
| `python_blank` | Python | Environnement Gymnasium vide à compléter dans `env.py` |
| `simulink_blank` | Simulink | Projet Simulink vide : on déclare signaux, paramètres et récompense |
| `cartpole` | Python | Pendule inverse sur chariot (exemple classique) |
| `cartpole_simulink` | Simulink | Pendule inverse modélisé dans Simulink (modèle MIT de Jitendra Singh, adapté) |
| `pac_zone` | Python | Pompe à chaleur air/eau : régulation d'une zone (modèle RC, COP selon la météo) |
| `ventilation_co2` | Python | Ventilation à la demande : tenir le CO₂ sous 1000 ppm au moindre coût |
| `ballon_ecs` | Python | Ballon d'eau chaude : tarif heures pleines/creuses et photovoltaïque |
| `usine_identification` | Python | Banc d'essai de l'onglet Identification (2 entrées, 2 sorties non linéaires) |

## Développement

Python **3.10 à 3.12** (3.12 recommandé : c'est le Python embarqué de l'installateur). Une autre version affiche un avertissement
(`rlstudio/pyversion.py` est la source unique de cette plage, alignée sur `requires-python`).

```powershell
python -m venv .venv                      # Python 3.12
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # ou cu130 selon pilote ; voir Paramètres > Calcul
pip install -e ".[gui,runtime,dev]"
pip install -e ".[matlab]"                # seulement pour les projets Simulink
```

```powershell
rlstudio-gui                                  # interface graphique
rlstudio list | templates | new | train | distill | identify | delete | export | import
python -m rlstudio.launcher --scan            # détection du matériel
python -m rlstudio.launcher --provision --dry-run   # plan d'installation des composants
pytest
```

`rlstudio distill <projet>` extrait un modèle mathématique d'un entraînement ; `rlstudio identify <projet>` identifie un modèle réduit
sans politique (`--mode steady|dynamic`). `rlstudio <commande> --help` détaille les options.

**Tests.** Le noyau (numpy, pyyaml) suffit à la plupart des tests. Les tests qui ont besoin de plus portent un marqueur et sont
**ignorés avec la raison affichée** quand les paquets manquent, au lieu d'échouer : `runtime` (gymnasium, stable-baselines3, torch :
entraînement, extraction, évaluation) et `gui` (PySide6, pyqtgraph : fenêtres et graphiques). `pip install -e ".[gui,runtime,dev]"`
(avec PyTorch) exécute tout ; `pytest -m "not gui and not runtime"` lance le seul noyau. Sous Linux sans écran :
`QT_QPA_PLATFORM=offscreen pytest`.

En développement, l'interpréteur de calcul est celui du `.venv` ; dans l'application installée, c'est le runtime
de `%LOCALAPPDATA%\RLStudio\runtime` (ou un Python choisi dans Paramètres > Composants).

## Deux types de projets

**Python** — `env.py` expose `make_env(**params)` (un `gymnasium.Env`) ; `params` vient de
`env.params` dans `project.yaml`. Optionnel : `baseline_policy(env)` pour tracer une référence
pendant l'entraînement (sinon : politique aléatoire).

**Simulink** — aucun code : on déclare dans la GUI (onglet *Environnement*) ou dans `project.yaml`
- **observations** : n'importe quelle sortie de bloc (`bloc` + `port`), journalisée en mémoire
  (le `.slx` n'est jamais modifié) ;
- **actions** : paramètre de bloc (`Constant/Value`, `Gain/Gain`…) ou variable de workspace,
  avec bornes min/max ;
- **état initial aléatoire** : variables du workspace tirées dans un intervalle à chaque épisode ;
- **récompense** et **arrêt anticipé** : expressions utilisant les noms des observations/actions
  et `t`, `dt`, `step` (ex. `-(temp - 20)**2 - 0.01*u**2`).

Le bouton **Explorer le modèle** ouvre un arbre des blocs (via MATLAB) pour ajouter d'un clic
un signal en observation ou un paramètre en action. L'état du modèle est conservé d'un pas à
l'autre côté MATLAB (état final → état initial), un seul appel `sim` par pas de décision.

**Convertir un modèle existant** — le bouton *Convertir un modèle Simulink…* ouvre une fenêtre en quatre étapes : analyse du modèle (entrées, sorties, constantes, solveur, variables manquantes, blocs d'affichage), choix des actions et des observations, variables initiales aléatoires, puis pas de décision et récompense. La conversion enregistre une copie dans un nouveau projet : les entrées racine deviennent des constantes pilotables (`Rl_<nom>`), les sorties sont résolues vers le bloc qui les alimente, les scopes et affichages sont retirés, les variables du workspace sont embarquées dans le modèle (modèle autonome) et le solveur peut être passé à pas fixe. Un script `.m` ou un `.mat` d'initialisation peut être indiqué pour les modèles qui dépendent de variables définies hors du `.slx` (il est exécuté tel quel dans MATLAB). Un épisode court sur la copie vérifie que tout simule.

Limites : signaux de bus non supportés ; un moteur MATLAB par environnement (pour
paralléliser, un process par env). Simulink crée son dossier de cache `slprj` dans
`%LOCALAPPDATA%\RLStudio\cache\simulink`, jamais dans le dossier du projet.

## Partager un projet : le format `.rls`

Un fichier `.rls` est une archive ZIP (manifeste `rlstudio.json` + dossier `project/`). Dans l'application :
*Actions du projet → Exporter (.rls)…* (avec ou sans résultats d'entraînement), et *Importer un projet (.rls)…*,
glisser-déposer du fichier dans la fenêtre, ou double-clic dans l'Explorateur (association faite par l'installateur).
En ligne de commande : `rlstudio export <projet> fichier.rls [--runs none|last|all]` et `rlstudio import fichier.rls`.

L'export exclut les caches (`__pycache__`, `slprj`, `*.slxc`…) et, par défaut, les entraînements. L'import crée toujours
un **nouveau** projet (jamais d'écrasement).

> **Sécurité** — un projet contient du code (`env.py`, modèle Simulink, scripts) qui s'exécute sur votre machine.
> N'importez que des projets de personnes de confiance : l'application affiche la liste des fichiers de code et demande
> une confirmation. L'import refuse les chemins dangereux (`..`, absolus, liens), les archives piégées (taille,
> compression) et tout exécutable ou script système (`.exe`, `.dll`, `.bat`, `.ps1`…).

## Extraire un modèle mathématique de l'agent

L'onglet **Modèle mathématique** (ou `rlstudio distill <projet>`) transforme la politique apprise en loi de commande
lisible : régulateur linéaire/PID, table de gains, arbre de décision, équation parcimonieuse, ou copie exacte du réseau.
Les modèles sont ajustés sur les décisions de l'agent, corrigés par DAgger, comparés en simulation (score normalisé,
complexité, fidélité) ; le plus simple qui reste dans la tolérance est recommandé. Export Python, MATLAB et contrôleur
Simulink. Voir [docs/distillation.md](docs/distillation.md).

## Remarques

- GPU : les petits MLP de SAC/PPO sont souvent plus rapides sur CPU ; un projet en `device: auto` utilise
  le GPU s'il est disponible et autorisé dans Paramètres.

## Licence et composants tiers

RL Studio est distribué sous **licence MIT** — © 2026 Christophe Barras, voir [LICENSE](LICENSE).

- L'interface utilise **Qt 6 via PySide6 sous LGPL v3**, en liaison dynamique (build PyInstaller en mode « dossier »,
  jamais `--onefile`) : l'utilisateur peut remplacer les bibliothèques Qt. Textes dans [`licenses/`](licenses),
  inventaire complet dans [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Seuls des modules Qt **LGPL** sont embarqués. Certains modules Qt (Virtual Keyboard, Charts, Data Visualization…)
  ne sont disponibles qu'en GPL v3 / commercial : `packaging/check_bundle.py` (liste blanche) fait échouer le build
  si l'un d'eux apparaît.
- MATLAB, Simulink, PyTorch CUDA et `matlabengine` ne sont **pas redistribués** : ils sont installés à la demande,
  avec la licence de l'utilisateur ou de leur éditeur.
