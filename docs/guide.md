# Guide d'utilisation de RL Studio

Ce guide s'adresse à quelqu'un qui découvre l'apprentissage par renforcement. Il est accessible à tout moment : touche **F1**, bouton **? Guide d'utilisation** de la barre latérale, ou *Paramètres → Guide d'utilisation*. Passez la souris sur n'importe quel bouton, champ ou titre : une infobulle explique à quoi il sert.

## 1. À quoi sert RL Studio ?

RL Studio permet d'**entraîner un programme (l'« agent ») à piloter un système** par essais et erreurs, sans écrire l'algorithme d'apprentissage : régler un chauffage, équilibrer un pendule, doser une ventilation…

Le principe, en quatre mots :

- **Environnement** : le système à piloter, simulé (en Python, ou dans un modèle Simulink).
- **Observations** : ce que l'agent voit à chaque instant (une température, un angle…).
- **Actions** : ce que l'agent peut commander (une puissance, une force…).
- **Récompense** : un nombre, calculé à chaque pas, qui dit si la situation est bonne (grand) ou mauvaise (petit). L'agent apprend à maximiser la somme des récompenses.

Après l'entraînement, RL Studio sait aussi :

- **rejouer** l'agent pour vérifier son comportement (onglet *Évaluation*) ;
- en tirer une **loi de commande lisible** : régulateur PID, table de gains, arbre de décision, équation (onglet *Modèle mathématique*) ;
- **remplacer un modèle lourd par un modèle réduit** (équations ou petit réseau de neurones), sans entraînement (onglet *Identification*).

## 2. Démarrage rapide (5 minutes)

1. Cliquez sur **+ Nouveau projet** (barre latérale), choisissez **Pendule inverse (CartPole)** et donnez-lui un nom.
2. Ouvrez l'onglet **Entraînement** et cliquez sur **▶ Démarrer**. Les courbes se tracent en direct ; le journal à droite indique l'avancement et le temps restant.
3. À la fin, la courbe « Récompense par épisode » monte : l'agent a appris. Ouvrez l'onglet **Résultats** pour retrouver l'entraînement.
4. Ouvrez **Évaluation**, choisissez le dernier entraînement et cliquez sur **▶ Lancer l'évaluation** pour voir l'agent à l'œuvre.
5. Pour obtenir une loi de commande lisible, passez à **Modèle mathématique** et cliquez sur **Extraire le modèle**.

> Les composants de calcul (Python, PyTorch, MATLAB Engine) doivent être installés. Si un bouton rouge « Composants de calcul manquants » apparaît dans la barre latérale, cliquez dessus : il ouvre *Paramètres → Composants*.

## 3. Vocabulaire

| Terme | Signification |
|---|---|
| **Agent** | Le programme qui apprend à piloter le système. |
| **Politique** | La règle qui donne l'action à partir des observations. L'entraînement fabrique la politique. |
| **Épisode** | Un essai complet : du départ du système jusqu'à la fin (durée écoulée ou arrêt anticipé). |
| **Pas** | Un instant de décision : l'agent observe, agit, reçoit une récompense. |
| **Durée d'entraînement** | Nombre total de pas de l'apprentissage (pas le nombre d'épisodes). |
| **Retour** | Somme des récompenses d'un épisode. C'est ce que l'agent cherche à maximiser. |
| **Référence** | Politique de comparaison (régulateur classique du modèle de projet, sinon actions aléatoires). |
| **Run** | Un entraînement enregistré dans le projet (dossier `runs/`). |
| **Graine** | Nombre qui fixe le hasard : même graine = mêmes tirages, donc résultats reproductibles. |
| **SAC, PPO, TD3** | Trois algorithmes d'apprentissage (voir §5.3). |
| **Régime permanent** | État stabilisé d'un système : les sorties ne changent plus pour des entrées constantes. |

## 4. La barre latérale

- **+ Nouveau projet** : crée un projet à partir d'un modèle de départ (Python vide, Simulink vide, pendule, pompe à chaleur, ventilation CO₂, ballon d'eau chaude, usine non linéaire…).
- **Importer un projet (.rls)…** : ouvre un projet partagé par quelqu'un d'autre. Vous pouvez aussi **glisser-déposer** un fichier `.rls` dans la fenêtre. Un projet contient du code qui s'exécute sur votre ordinateur : n'importez que des fichiers de confiance.
- **Convertir un modèle Simulink…** : assistant qui prépare un modèle Simulink existant pour l'apprentissage (voir §7).
- **Liste des projets** : cliquez sur un projet pour l'ouvrir. Impossible d'en changer pendant un calcul.
- **Actions du projet** : dupliquer, renommer, ouvrir le dossier, exporter en `.rls`, supprimer (le projet part dans une corbeille récupérable `projects/.trash/`).
- **? Guide d'utilisation** : ce guide.
- **⚙ Paramètres** : apparence, dossier des projets, MATLAB, GPU, composants (voir §9).

La barre du haut affiche le nom du projet et un bouton **Enregistrer**. Un point orange « Modifications non enregistrées » apparaît dès qu'un champ change ; l'entraînement enregistre automatiquement avant de démarrer.

## 5. Les onglets d'un projet

### 5.1 Projet

Fiche du projet : **nom** (sert aussi de nom de dossier), **description** libre, et indicateurs (type, algorithme, durée, nombre d'entraînements, date du dernier). Les boutons mènent directement à l'entraînement, à l'édition de l'environnement ou au dossier du projet.

### 5.2 Environnement

Décrit le système à piloter. Il y a deux types de projets.

**Projet Python.** Le système est un fichier Python (`env.py`).

- **Point d'entrée** : `fichier:fonction`, par défaut `env.py:make_env`. La fonction renvoie un environnement Gymnasium.
- **Ouvrir dans l'éditeur** : ouvre le fichier avec l'éditeur associé de Windows.
- **Paramètres** : texte YAML transmis à `make_env(**paramètres)`. Pratique pour changer la durée d'un épisode ou un poids de la récompense sans toucher au code.

**Projet Simulink.** Aucun code : on décrit le modèle.

- **Modèle Simulink** : fichier `.slx` ou `.mdl`. *Parcourir…* le copie dans le projet ; *Explorer le modèle* ouvre l'arbre des blocs (MATLAB requis) pour ajouter d'un clic un signal ou un paramètre.
- **Pas de décision** : temps simulé entre deux actions de l'agent. **Durée d'un épisode** : temps simulé d'un essai complet.
- **Actions** : ce que l'agent commande, soit un *paramètre de bloc* (colonnes Bloc + Paramètre, ex. `Constant` / `Value`), soit une *variable du workspace* (colonne ou Variable). **Min** et **Max** bornent la commande.
- **Observations** : signaux lus par l'agent à chaque pas (colonnes Bloc, Port, Taille). Min et Max servent à normaliser.
- **État initial aléatoire** : variables du workspace tirées dans [Min, Max] au début de chaque épisode (optionnel).
- **Récompense** : expression calculée à chaque pas avec les noms des actions et des observations, plus `t` (temps), `dt` (pas) et `step` (numéro de pas). Exemple : `-(temp - 20)**2 - 0.01*u**2` pénalise l'écart à 20 °C et l'effort.
- **Arrêt anticipé** (optionnel) : condition qui termine l'épisode, par exemple `temp > 100`.
- **Vérifier la configuration** : contrôle les noms, les bornes et les expressions avant de lancer un entraînement.

### 5.3 Algorithme

- **Algorithme** : **SAC** (par défaut, efficace et économe en essais, actions continues), **TD3** (proche de SAC, plus déterministe), **PPO** (plus stable mais demande beaucoup plus de pas).
- **Durée** : nombre total de pas d'apprentissage. Plus c'est long, meilleur est souvent l'agent, mais plus l'attente est grande. Commencez par une valeur modeste pour vérifier que tout fonctionne.
- **Graine aléatoire** : change les tirages. Relancer avec une autre graine permet de vérifier que le résultat n'est pas dû au hasard.
- **Périphérique** : `cpu` (processeur), `cuda` (carte NVIDIA) ou `auto` (carte graphique si disponible et autorisée). Pour de petits réseaux, le processeur est souvent aussi rapide.
- **Évaluation de référence** : nombre d'épisodes joués *avant* l'apprentissage avec la politique de référence ; sa moyenne est tracée en rouge pour juger les progrès.
- **Hyperparamètres** : réglages avancés de Stable-Baselines3 en YAML (`learning_rate: 0.0003`). Laissez vide pour les valeurs par défaut.

### 5.4 Entraînement

- **▶ Démarrer** enregistre le projet, vérifie la configuration (projets Simulink) et lance le calcul. **■ Arrêter** interrompt immédiatement (le modèle partiel est conservé s'il l'a pu).
- La **barre de progression** suit d'abord l'évaluation de référence puis l'apprentissage ; le texte à droite donne l'état et le temps restant.
- **Indicateurs** : nombre d'épisodes et de pas, dernière récompense, meilleure, référence.
- **Graphiques** : récompense par épisode (la courbe jaune est une moyenne glissante, la ligne rouge pointillée la référence) ; observations, actions et récompense de l'épisode en cours. Cliquez sur un nom de la légende pour masquer ou afficher une courbe. **Normaliser les courbes** ramène les signaux à la même échelle.
- **Journal** (colonne de droite) : tout ce que fait le calcul, avec l'heure. Les erreurs y apparaissent aussi **en haut de la fenêtre** (voir §10).

Une bonne courbe de récompense **monte puis se stabilise**. Si elle reste plate, essayez une durée plus longue, une autre graine ou une récompense plus informative.

### 5.5 Évaluation

Rejoue un entraînement terminé avec la politique déterministe (sans bruit d'exploration).

- **Run** : l'entraînement à rejouer. **Politique** : l'agent entraîné ou un modèle mathématique extrait de ce run.
- **Épisodes** et **Graine** : combien d'essais, et quelle suite de départs (la même graine donne les mêmes départs).
- **Vitesse** : *Maximale* ou ralentie, pour suivre les courbes en direct.
- Les graphiques montrent observations, actions, récompense cumulée et le retour de chaque épisode.

### 5.6 Résultats

La liste des entraînements du projet (date, référence, modèle enregistré ou non) et, pour celui qui est sélectionné, la configuration exacte utilisée. **Ouvrir le dossier du run** affiche les fichiers.

### 5.7 Modèle mathématique

Transforme la politique apprise (un réseau de neurones opaque) en **loi de commande lisible**.

1. Choisissez le **run** et cochez les **familles de modèles** à essayer : *Linéaire / PID*, *Polynôme*, *Table de gains*, *Arbre de décision*, *Équation symbolique*, *Réseau exact*.
2. Réglez si besoin : **Épisodes de collecte** (données tirées de l'agent), **Bruit d'action** (variété des situations rencontrées), **Tours de DAgger** et **Épisodes / tour** (corrige la dérive du modèle en boucle fermée), **Épisodes d'évaluation**, **Tolérance** (écart de score accepté pour préférer un modèle plus simple), **Dimensions de la table** (1 ou 2), **Observations intégrées (PID)** (observations dont on ajoute l'intégrale, ex. `angle, position`).
3. Cliquez sur **Extraire le modèle**. Le tableau compare les modèles : *Complexité* (nombre de paramètres), *R2* (fidélité aux décisions de l'agent), *Retour*, *Score* (0 = aléatoire, 1 = agent) et *Menés à terme*. Le modèle **recommandé** est le plus simple dont le score reste dans la tolérance.
4. Sélectionnez une ligne pour lire ses équations. Exportez avec **Copier Python / MATLAB**, **Enregistrer…** ou **Exporter vers Simulink…** (contrôleur prêt à intégrer, MATLAB requis).

### 5.8 Identification

Remplace un modèle (Simulink ou Python) par un **modèle réduit**, **sans agent ni entraînement**.

- **Mode** :
  - *Régime permanent* : carte statique « entrées → sorties » (`y = f(u)`). Chaque point est simulé jusqu'à stabilisation. Le plus rapide et le plus lisible.
  - *Dynamique* : `x(k+1) = x(k) + f(x, u)`, pour les systèmes qui n'ont pas de régime permanent (instables, oscillants).
- **Familles de modèles** : *Linéaire*, *Polynôme*, *Équation parcimonieuse (SINDy)* qui retrouve les formules avec peu de termes, *Réseau de neurones*.
- **Entrées** : cochez *Balayer* pour faire varier une entrée entre Min et Max ; décochée, elle reste à sa **valeur fixée**. **Lire les signaux du projet** recharge la liste (projets Python).
- **Sorties à modéliser** : les observations à reproduire.
- **Balayage** : nombre de points (ou d'épisodes), points de validation, part des points aux bornes, critères de stabilisation (tolérances, nombre de pas stables, durée maximale), perturbations « initiales » comme entrées, etc.
- **Modèles** : degrés du polynôme, seuils SINDy (plus grand = moins de termes), couches et époques du réseau, tolérance de recommandation, graine.
- **Résultats** : historique des identifications, tableau comparatif (complexité, R2, NRMSE, erreur…), graphiques complexité / fidélité et modèle contre système, équations, export Python, MATLAB ou bloc Simulink.

Un **NRMSE** proche de 0 signifie un modèle fidèle. Les points qui ne convergent pas sont écartés et signalés.

## 6. Exemple de bonne démarche

1. Partir d'un modèle de départ proche de votre besoin.
2. Vérifier l'**Environnement** (récompense, bornes des actions).
3. Lancer un **premier entraînement court** pour contrôler que rien n'échoue.
4. Lancer l'entraînement long, comparer à la **référence**.
5. **Évaluer** l'agent sur d'autres graines.
6. **Extraire un modèle mathématique** pour comprendre et valider ce que fait l'agent avant tout usage réel.

## 7. Convertir un modèle Simulink existant

Le bouton **Convertir un modèle Simulink…** guide en quatre étapes sans jamais modifier votre fichier d'origine (une copie est faite).

1. **Modèle** : choisissez le `.slx`/`.mdl` et, si le modèle dépend de variables définies ailleurs, un script `.m` ou un `.mat` d'initialisation. Cliquez sur *Analyser le modèle*.
2. **Actions** : les entrées du modèle deviennent des constantes que l'agent pilote. Vous pouvez ajouter un paramètre de bloc.
3. **Observations** et variables initiales aléatoires : les sorties du modèle sont proposées ; ajoutez d'autres signaux si besoin.
4. **Réglages** : pas de décision, durée d'un épisode, solveur, récompense, arrêt anticipé.

La conversion crée un nouveau projet et teste la copie « à vide » (un épisode court). Limites : pas de signaux de bus ; un moteur MATLAB par environnement.

## 8. Partager un projet

*Actions du projet → Exporter (.rls)…* produit un fichier unique. Choisissez d'y inclure ou non les résultats d'entraînement (le plus léger : sans) et d'identification. *Importer un projet (.rls)…* (ou glisser-déposer) crée toujours un **nouveau** projet et ne remplace jamais un projet existant. Avant l'import, RL Studio affiche la liste des fichiers de code contenus dans le projet et demande confirmation.

## 9. Les paramètres

- **Apparence** : thème *Sombre*, *Clair*, *Matrix* ou personnalisé (*Personnaliser…* règle les couleurs et la police ; l'aperçu est en direct, *Annuler* revient au thème précédent).
- **Général** : dossier des projets et dossier des données de l'application.
- **MATLAB / Simulink** : activer l'intégration, choisir l'installation, *Détecter à nouveau*, *Tester MATLAB*.
- **Calcul (CPU / GPU)** : autoriser le GPU NVIDIA (CUDA), *Scanner le matériel*.
- **Composants** : état de Python, PyTorch et MATLAB Engine ; *Installer / réparer les composants…* ; Python de calcul personnalisé (avancé) ; accès aux journaux.
- **Guide d'utilisation** : ce document.
- **À propos** : version, emplacements des fichiers, licences.

Les paramètres sont verrouillés pendant un entraînement.

## 10. Erreurs et dépannage

**Où voir les erreurs ?** Toute erreur apparaît dans le journal de l'onglet concerné **et** dans un bandeau rouge en haut de la fenêtre, visible quel que soit l'onglet. Le bouton **Détails…** ouvre le texte complet (avec **Copier** pour le transmettre), **✕** masque le bandeau. Un bandeau jaune signale un avertissement.

| Symptôme | Cause probable et solution |
|---|---|
| « Composants de calcul non installés » | Ouvrez *Paramètres → Composants* puis *Installer / réparer les composants…*. |
| MATLAB introuvable ou erreur au démarrage | *Paramètres → MATLAB / Simulink* : *Détecter à nouveau* puis *Tester MATLAB*. Vérifiez qu'il s'agit d'une installation complète avec l'Engine Python. |
| « Configuration Simulink » invalide | Onglet *Environnement* → *Vérifier la configuration* ; corrigez les noms ou les bornes signalés. |
| L'entraînement n'améliore rien | Allongez la durée, changez de graine, vérifiez que la récompense augmente quand la situation s'améliore. |
| L'application semble figée au lancement d'un calcul | Le démarrage de PyTorch ou de MATLAB prend quelques secondes à une minute. |
| Import refusé | L'archive contient un chemin dangereux ou un exécutable : elle n'est pas importée par sécurité. |

Les journaux détaillés sont dans *Paramètres → Composants → Ouvrir le dossier des journaux*.

## 11. Raccourcis et astuces

- **F1** : ouvre ce guide à la section de l'onglet affiché.
- **Glisser-déposer** un `.rls` dans la fenêtre pour l'importer.
- **Clic sur la légende** d'un graphique : masque ou affiche une courbe.
- **Survol** de n'importe quel élément : infobulle explicative.
- Un double-clic sur un fichier `.rls` ouvre RL Studio et propose l'import (si l'installateur a associé les fichiers).

## 12. Glossaire rapide

- **DAgger** : méthode qui rejoue le modèle extrait, demande à l'agent la bonne action dans les situations réellement rencontrées, et corrige ainsi la dérive.
- **NRMSE** : erreur normalisée entre le modèle et le système (0 = parfait).
- **PID** : régulateur classique à trois termes (proportionnel, intégral, dérivé).
- **R2** : part de la variation reproduite par le modèle (1 = parfait).
- **SINDy** : régression qui sélectionne peu de termes pour retrouver une équation lisible.
- **Solveur à pas fixe** : Simulink avance par pas constants ; recommandé pour conserver l'état d'un pas de décision à l'autre.
