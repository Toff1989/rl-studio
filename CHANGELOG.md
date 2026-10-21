# Changelog

Toutes les modifications notables de RL Studio sont consignées ici.
Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet suit le
[versionnement sémantique](https://semver.org/lang/fr/).

## [Non publié]

### Corrigé
- **Identification en régime permanent : « Trop peu de points stabilisés »** sur un projet dont les observations contiennent des signaux indépendants de l'entrée (horloge, occupation, météo : modèle *Ventilation CO2*). Aucun point ne se stabilisait, car ces signaux bougent à chaque pas. Ils sont désormais repérés automatiquement (observation strictement identique sous trois actions différentes, à départ égal), écartés des sorties à modéliser avec un message dans le journal et une remarque dans le rapport. Si rien ne se stabilise malgré tout, l'erreur nomme les sorties fautives et indique de les décocher dans « Sorties à modéliser ». Le test de stabilisation ne porte plus que sur les sorties choisies.
- Plus d'avertissement numpy « Mean of empty slice » quand aucun point n'est retenu.

## [0.4.1] - 2026-10-21

### Ajouté
- **Infobulles partout** : chaque bouton, champ, case, liste, tableau (en-têtes et cellules), graphique et onglet explique son rôle au survol — fenêtre principale, Paramètres, éditeur de thème, assistant de conversion Simulink et explorateur de blocs. Un test vérifie qu'aucun contrôle n'en est dépourvu.
- **Guide d'utilisation** (`docs/guide.md`, embarqué dans l'installateur) : vocabulaire, démarrage en 5 minutes, description de chaque onglet et de chaque champ, conversion Simulink, partage, paramètres, dépannage. Accessible par **F1** (ouvert à la section de l'onglet affiché), le bouton *Guide d'utilisation* de la barre latérale ou *Paramètres → Guide d'utilisation* ; sommaire et recherche. Il reste consultable pendant un calcul.
- **Bandeau d'erreurs** en haut de la fenêtre, visible quel que soit l'onglet, avec *Détails…* (texte complet, copie) : fin de calcul en échec (entraînement, extraction, identification, évaluation), ligne d'erreur écrite dans un journal, échec d'un export Simulink ou de la lecture des signaux, exception Python inattendue. La fenêtre d'installation des composants signale aussi les erreurs de sa sortie.
- Avertissement (bandeau, ou stderr en ligne de commande) quand Python sort de la plage validée 3.10 à 3.12 (`rlstudio/pyversion.py`).

### Modifié
- **Textes accentués** : interface, messages du calcul, validations, erreurs d'import, modèles de projet fournis. Le nom d'un projet accentué donne un dossier lisible (`Régulation` → `regulation`). Le code MATLAB généré reste en ASCII (indépendant de l'encodage lu par MATLAB).
- **`gui/app.py` découpé** (1 244 → environ 230 lignes) : un module par onglet (`tab_project`, `tab_env`, `tab_algo`, `tab_train`, `tab_runs`), les actions de projet (`project_actions`) et la table de spécification (`spec_table`). Aucun changement de comportement.
- Tests : deux marqueurs, `runtime` (gymnasium, stable-baselines3, torch) et `gui` (PySide6, pyqtgraph). Sans ces paquets les tests concernés sont **ignorés avec la raison affichée** au lieu d'échouer.
- README à jour : les huit modèles de projet, toutes les commandes (`distill`, `identify`), plage de Python, organisation des tests.

### Corrigé
- Une erreur écrite dans le journal d'un onglet n'apparaissait nulle part ailleurs : elle est maintenant remontée dans le bandeau.
- Le contrôle du bundle (`check_bundle.py`) exige aussi le guide d'utilisation (ressource distincte des licences).

## [0.4.0] - 2026-10-20

### Ajouté
- **Identification d'un modèle réduit, sans politique ni entraînement** (module `rlstudio.identify`, commande `rlstudio identify`, onglet **Identification**). Le modèle (Simulink ou Python) est balayé puis remplacé par des équations ou un réseau de neurones, validé sur des points ou des épisodes jamais vus.
  - **Régime permanent** (par défaut) : carte statique entrées → sorties `y = f(u)`. Hypercube latin sur les entrées, chaque point simulé jusqu'à stabilisation, points non convergés écartés. Option `--scenario` : les variables `initial` (perturbations) deviennent des entrées.
  - **Dynamique** (`--mode dynamic`) : `x(k+1) = x(k) + f(x, u)`, paliers d'action aléatoires, validation en **simulation libre** (fidélité, erreur par horizon, courbe d'apprentissage).
  - **Familles** : linéaire, polynôme (degrés 2 et 3), équations parcimonieuses (SINDy : monomes, `sin`, `cos`, `tanh`, `abs`, produits fonction × variable), réseau de neurones. NumPy pur.
  - **Réglages complets** dans l'onglet : entrées balayées (plage Min/Max) ou fixées, sorties à modéliser, critères de stabilisation, nombre de points / d'épisodes, hyperparamètres des modèles, tolérance de recommandation.
  - **Exports** : Python et MATLAB (`steady_state` / `dynamics`, compatibles génération de code), et **bloc Simulink** (MATLAB Function ; modèle dynamique avec état `persistent`).
  - Nouvelles commandes du worker : `identify`, `env-signals`, `export-identify-simulink`.
- **Identification dans les projets et les `.rls`** : réglages dans `identify/options.json` (toujours inclus à l'export), résultats `none|last|all` (`rlstudio export --identify`, choix dans le dialogue d'export), code généré non embarqué et régénéré depuis `model.json`. Le résumé d'import indique les identifications incluses.
- **Projet fourni « Usine non linéaire (identification) »** : Python, 2 entrées / 2 sorties, formules du régime permanent connues, pour essayer la fonction.

### Corrigé
- L'export `.rls` n'embarquait pas de choix pour le dossier `identify/` ; il est désormais filtré (données seulement).

## [0.3.5] - 2026-10-19

### Corrigé
- **Interface fluide pendant les calculs** (entraînement, évaluation, extraction de modèle) : les graphiques ajoutent un point en temps constant et se redessinent par paquets (≈ 25 fois par seconde au plus, moins souvent sur les longs épisodes) avec au plus ~1500 points tracés par courbe (pics conservés) ; les journaux sont bornés à 5000 lignes ; le bouton *Arrêter* ne bloque plus l'interface (`taskkill` en arrière-plan) ; le processus de calcul, MATLAB compris, tourne en priorité inférieure à la normale (`RLSTUDIO_NORMAL_PRIORITY=1` pour l'annuler).

## [0.3.4] - 2026-10-19

### Ajouté
- **Modèle mathématique** : nouvelle famille **Polynôme** (degrés 2 et 3 par défaut, termes croisés inclus, régression régularisée sur les observations normalisées). Équations lisibles, export Python / MATLAB / Simulink, sauvegarde JSON, option `poly_degrees` et CLI `--methods polynomial`.
- **Trois modèles de projet CVC / bâtiment** (Python autonome, avec régulateur de référence) : *PAC air/eau – régulation d'une zone* (modèle RC air + masse, COP et capacité selon la météo, consigne jour/nuit), *Ventilation à la demande (CO₂)* (occupation variable, coût ventilateur + chauffage de l'air neuf) et *Ballon d'ECS – tarif et photovoltaïque* (heures pleines/creuses, PV, soutirages aléatoires).

## [0.3.3] - 2026-10-19

### Corrigé
- **Installation des composants de calcul** : pip ne bascule plus en installation « --user » (erreur « User site-packages are disabled for this Python ») sur les postes où le site utilisateur est désactivé. `Lib\site-packages` est créé d'avance et l'environnement de pip est isolé (variables `PIP_*` héritées supprimées, `PIP_USER=0`, configuration pip ignorée).
- **Construction de l'installateur** (`packaging/build.ps1`) : la configuration pip de la machine est écartée et l'erreur réelle de pip s'affiche ; nettoyage robuste de `dist` et `build` (arrêt des instances en cours, réessais, message clair si un dossier est verrouillé par OneDrive ou l'antivirus).

## [0.3.2] - 2026-10-17

### Ajouté
- **Conversion d'un modèle Simulink existant** : bouton *Convertir un modèle Simulink…* sous « Importer un projet (.rls) ». Fenêtre en quatre étapes (analyse du modèle, actions, observations et variables initiales, pas et récompense) qui crée un nouveau projet avec une **copie convertie** du modèle (entrées → constantes pilotables, sorties → bloc source, affichages retirés, variables du workspace embarquées, solveur à pas fixe au choix) puis la teste à vide. L'original n'est jamais modifié. Commandes worker `analyze-model` et `convert-model`, scripts `matlab/rlstudio_analyze.m` et `rlstudio_convert.m`, explorateur de blocs partagé (`gui/explorer.py`).
- **Entraînement** : journal détaillé et horodaté dans une colonne à droite des graphiques (démarrage de MATLAB/Simulink, évaluation de référence épisode par épisode avec temps restant, début de l'apprentissage, une ligne par épisode avec vitesse en pas/s et temps restant) ; la barre de progression suit l'évaluation de référence avant l'apprentissage.

### Corrigé
- **Bouton Arrêter** : il répond immédiatement. Le process de calcul est tué avec ses process fils (moteur MATLAB compris) au lieu d'attendre la fin de l'épisode ou 20 s ; seul l'apprentissage en cours garde un arrêt propre (3 s maximum) pour enregistrer le modèle partiel. Un run interrompu avant l'enregistrement du modèle est supprimé, et l'arrêt ne s'affiche plus comme une erreur.
- Graphiques temporels (Entraînement, Évaluation) : la légende des signaux n'est plus dessinée sur la courbe mais sous le graphique, en couleurs lisibles.
- Graphiques temporels avec beaucoup de signaux : une couleur distincte par signal (plus de couleurs répétées en pointillé), légende compacte à hauteur plafonnée, dont les entrées sont cliquables pour masquer / afficher une courbe (et un raccourci « tout masquer / afficher »).

## [0.3.1] - 2026-10-17

### Ajouté
- **Onglet Évaluation** : rejoue un run entraîné (ou un modèle mathématique extrait) avec la politique déterministe et
  trace en direct les observations, les actions, la récompense cumulée et le retour par épisode ; choix du run, de la
  politique, du nombre d'épisodes, de la graine et de la vitesse. Commande worker `evaluate`.
- **Extraction du modèle mathématique** : progression fine (un pas par épisode simulé), étape en cours détaillée, temps écoulé et estimation du temps restant, journal visible pendant le calcul.
- **Entraînement** : les observations et les actions sont désormais tracées en **courbes temporelles** (une courbe par
  signal, légende avec les noms), avec option de normalisation pour comparer des signaux d'échelles différentes.

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

[Non publié]: https://github.com/Toff1989/rl-studio/compare/0.3.1...HEAD
[0.3.1]: https://github.com/Toff1989/rl-studio/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/Toff1989/rl-studio/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Toff1989/rl-studio/releases/tag/0.2.0
