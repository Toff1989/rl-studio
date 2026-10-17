# Changelog

Toutes les modifications notables de RL Studio sont consignées ici.
Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet suit le
[versionnement sémantique](https://semver.org/lang/fr/).

## [Non publié]

## [0.3.0] - 2026-10-17

Extraire l'intelligence d'un agent entraîné sous forme de modèle mathématique lisible.

### Ajouté
- **Extraction d'un modèle mathématique** (onglet *Modèle mathématique*, commande `rlstudio distill`, commande worker
  `distill`) : la politique apprise (SAC, PPO, TD3) est rejouée avec un léger bruit d'action, puis des modèles simples
  sont ajustés sur ses décisions déterministes :
  régulateur **linéaire / PID** (avec action intégrale optionnelle), **table de gains** 1D/2D, **arbre de décision**
  (profondeurs 3, 5, 8), **équation symbolique** parcimonieuse (3, 6, 12 termes) et **copie exacte du réseau**
  (vérifiée numériquement contre `model.predict`).
- **DAgger** pour éviter la dérive de distribution : le modèle pilote, la politique réétiquette les états visités.
- **Validation en boucle fermée** sur des épisodes appariés : score normalisé (0 = aléatoire, 1 = politique), fidélité (R²),
  complexité, part d'épisodes menés à terme ; **recommandation** du modèle le plus simple dans la tolérance.
  Rapport `report.json` / `report.md` avec points d'attention (saturation, arrêts anticipés, extrapolation, état).
- **Exports** par modèle : `policy.py`, `policy.m` (compatible génération de code), `equations.txt`, `surrogate.json`, et
  **contrôleur Simulink** (bloc *MATLAB Function*, une entrée par observation, une sortie par action) via
  `matlab/rlstudio_build_controller.m`. Le code MATLAB et le modèle Simulink générés sont vérifiés contre Python.
- Documentation [`docs/distillation.md`](docs/distillation.md).
- **Template `cartpole_simulink`** : pendule inverse sur chariot modélisé dans Simulink, exemple complet de projet
  Simulink (action = force du chariot, 4 observations, état initial aléatoire, récompense et arrêt anticipé en
  expressions). Basé sur un modèle MIT de Jitendra Singh, adapté par le script reproductible
  `tools/build_cartpole_simulink.m` ; licence d'origine et provenance fournies avec le template. Validé avec MATLAB R2026b :
  le pendule tombe en boucle ouverte, un LQR dérivé du modèle le stabilise (force maximale < 1 N sur une borne de 5 N).
- **Lanceur `lancer.bat`** : choix du mode développement ou application compilée.

### Modifié
- **Nouvelle icône** : boucle agent/environnement et point de récompense, dessinée taille par taille
  (`icon.ico` contient désormais 16, 20, 24, 32, 40, 48, 64, 128 et 256 px au lieu d'une seule taille) ; elle apparaît
  aussi dans la fenêtre, la barre des tâches et l'en-tête de l'application. Générée par `packaging/make_icon.py`.

### Corrigé
- Les fichiers de réglages et de projet enregistrés avec un BOM UTF-8 (par exemple par PowerShell 5.1) sont désormais lus
  correctement au lieu d'être ignorés silencieusement.

## [0.2.0] - 2026-10-17

Première version publique : RL Studio devient une application Windows installable de gestion de projets
d'apprentissage par renforcement, pilotant des environnements Python ou des modèles Simulink.

### Ajouté
- **Gestion de projets** : créer, dupliquer, renommer, supprimer (corbeille récupérable `projects/.trash/`), un dossier
  par projet (`project.yaml`, `env.py` ou modèle Simulink, `runs/`).
- **Environnements Python** (`make_env(**params)` + `baseline_policy` optionnel) et **Simulink sans code** : observations
  (sortie de n'importe quel bloc), actions (paramètre de bloc ou variable de workspace), état initial aléatoire,
  récompense et arrêt anticipé en expressions ; **explorateur de blocs** alimenté par MATLAB.
- **Entraînement** SAC / PPO / TD3 (Stable-Baselines3) avec suivi en direct : récompense par épisode, observation,
  actions, comparaison à une politique de référence.
- **Format de partage `.rls`** : export (avec ou sans résultats d'entraînement), import sécurisé, glisser-déposer,
  double-clic dans l'Explorateur, commandes `rlstudio export` / `rlstudio import`.
- **Fenêtre Paramètres** : apparence, dossier des projets, MATLAB / Simulink, GPU, composants de calcul, à propos.
- **Thèmes** Sombre, Clair, Matrix et thèmes personnalisés avec aperçu en direct.
- **Installateur Windows** (PyInstaller + Inno Setup) qui détecte le matériel — GPU NVIDIA (pilote, CUDA), installations
  MATLAB — et installe un runtime de calcul adapté : PyTorch `cu130` / `cu128` / `cu126` / CPU et `matlabengine` de la
  version de MATLAB détectée. Options `/VERYSILENT`, `/NOPROVISION=1`, tâche d'association `.rls`.
- **Exemples génériques** : `python_blank`, `cartpole` (SAC atteint 499/500 en 30 000 pas), `simulink_blank`.
- Mentions légales : licence **MIT**, `THIRD_PARTY_NOTICES.md`, textes LGPL / GPL, contrôle automatique du contenu Qt
  du build (`packaging/check_bundle.py`).

### Sécurité
- L'import d'un `.rls` valide chaque chemin (zip-slip, chemins absolus, liens symboliques), limite les tailles,
  **refuse les exécutables et scripts système**, confine `env.entry` et `env.model` au dossier du projet et demande une
  confirmation qui liste les fichiers de code.
- Le runtime de calcul ignore le site-packages utilisateur : l'installation ne peut ni voir ni désinstaller les paquets
  Python de l'utilisateur.

### Notes de conformité
- Qt 6 est utilisé via PySide6 sous **LGPL v3**, en liaison dynamique, avec uniquement des modules Qt LGPL (Core, Gui,
  Widgets, Network, OpenGL, OpenGLWidgets, Svg, Test). Qt Virtual Keyboard, Charts… (GPL v3 / commercial) sont exclus.

### Limites connues
- Installateur **non signé** : Windows SmartScreen affichera un avertissement.
- Windows uniquement ; Simulink : signaux de bus non supportés, un moteur MATLAB par environnement.
- Testé avec MATLAB R2026b, Python 3.12, NVIDIA RTX 5080 (CUDA 13) ; les autres configurations sont couvertes par des
  tests unitaires de planification mais n'ont pas été installées pour de vrai.

[Non publié]: https://github.com/Toff1989/rl-studio/compare/0.3.0...HEAD
[0.3.0]: https://github.com/Toff1989/rl-studio/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Toff1989/rl-studio/releases/tag/0.2.0
