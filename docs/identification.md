# Identifier un modele reduit sans politique : regime permanent (defaut) ou dynamique

## Interface : onglet **Identification**

Tout se parametre dans l'onglet (ou en ligne de commande, plus bas) :

- **Mode** : regime permanent ou dynamique ; **familles** de modeles a ajuster.
- **Entrees** (tableau lu dans le projet ; bouton *Lire les signaux* pour un projet Python) : decocher une entree pour la maintenir a une
  *valeur fixee* (elle n'est alors plus une entree du modele), ou restreindre sa plage *Min / Max* (dans les bornes de l'action).
- **Sorties** (regime permanent) : cocher les observations a modeliser.
- **Balayage** : nombre de points / d'episodes, validation, part aux bornes, criteres de stabilisation (tolerances relative et absolue, pas
  consecutifs, duree max), perturbations `initial` comme entrees, tours de collecte, duree des paliers (dynamique).
- **Modeles** : degres du polynome, seuils SINDy, couches cachees et epoques du reseau, tolerance de recommandation, graine.
- **Resultats** : historique des identifications du projet, tableau comparatif, graphique complexite / fidelite, nuage modele vs systeme (par
  sortie) ou erreur par horizon, equations, copie / enregistrement en Python ou MATLAB.

Les resultats sont enregistres dans `projects/<projet>/identify/<date>/`, et les reglages dans `identify/options.json` : ils sont
retrouves a la reouverture du projet.

### Export vers Simulink

*Exporter vers Simulink...* (modele selectionne) genere un `.slx` (MATLAB requis) : Inport (une par entree) -> bloc **MATLAB Function** ->
Outport (une par sortie), code compatible generation de code.
- **Regime permanent** : `y = fcn(u)`, sans etat.
- **Dynamique** : les entrees sont les actions, les sorties sont l'etat ; la fonction garde l'etat (`persistent`) et avance d'un pas a chaque pas
  d'echantillonnage `dt`. Etat initial = moyenne des etats initiaux du balayage.
En ligne de commande du worker : `export-identify-simulink --project ... --report <date> --candidate <id> --out modele.slx`.

### Projets et fichiers `.rls`

L'identification fait partie du projet : `identify/` voyage avec lui. Dans l'export `.rls` (menu *Exporter*, ou `rlstudio export --identify none|last|all`) :
les **reglages** sont toujours inclus ; les **resultats** (rapport, equations, `model.json`) selon le choix (dernier par defaut) ; le **code genere**
(`.py`, `.m`) n'est jamais embarque - il est regenere depuis `model.json` a la demande, ce qui evite d'afficher un avertissement de securite pour du code
que l'on peut reconstruire. A l'import, tout est retrouve dans l'onglet Identification.

### Modeles pour essayer

- **Usine non lineaire (identification)** : projet fourni (*Nouveau projet > Usine non lineaire*), Python, 2 entrees / 2 sorties, formules
  du regime permanent connues : l'identification doit les retrouver (`y1 = 0,5 u2^2 + 3 tanh u1`, `y2 = |u1| + sin u1 . u2`).
- **Pendule inverse (Simulink)** : projet fourni, utile en mode *dynamique* (instable : pas de regime permanent).
- **Demos livrees avec Simulink** (aucun telechargement) : `sldemo_engine` (moteur : regime permanent regime = f(papillon, couple resistant),
  tables non lineaires), `sldemo_househeat` (maison chauffee : thermostat a hysteresis, donc cycle sans regime permanent unique). Pour un
  modele a vous, utiliser *Convertir un modele Simulink* puis l'onglet Identification.


## Regime permanent : entrees -> sorties (`rlstudio identify <projet>`)

On ne cherche pas la dynamique, seulement l'**etat stabilise** en fonction des entrees : `y = f(u)`. Le modele (Simulink ou Python) est une
boite noire que l'on balaie ; aucune politique, aucun entrainement.

1. **Plan d'experience** : hypercube latin sur les entrees (actions de l'environnement, entre leurs bornes `low` / `high`), dont 15 % des
   coordonnees aux bornes (saturations).
2. **Etablissement** : chaque point est simule, action constante, jusqu'a stabilisation des sorties
   (`|dy| <= atol + rtol.|y|` pendant 3 pas) ; la sortie stabilisee est relevee. Un point qui ne converge pas est ecarte et compte dans le rapport.
3. **Ajustement** (memes familles que ci-dessous) : lineaire, polynome (degres 2 et 3), equation parcimonieuse (SINDy : monomes
   jusqu'au degre 3, `sin`, `cos`, `tanh`, `abs`, produits fonction x variable), reseau de neurones.
4. **Validation** sur des points neufs (R2 global et par sortie) ; recommandation du modele le plus simple avec R2 >= 98 %.

```bash
rlstudio identify mon-projet-simulink                                     # 200 points + 50 de validation
rlstudio identify mon-projet --samples 1000 --max-settle-steps 400 --methods sparse,neural
rlstudio identify mon-projet --scenario                                   # variables `initial` (perturbations) = entrees du modele
```

Fichiers : `projects/<projet>/identify/<date>/` : `report.json`, `report.md`, et par modele `steady_state.py`, `steady_state.m`
(`y = steady_state(u)`, compatible generation de code), `equations.txt`, `model.json`.

Points d'attention : domaine valable = plages balayees ; une entree `u` hors plage est extrapolee ; en cas de regimes permanents multiples
(hysteresis), la sortie depend de l'histoire et le modele statique n'en donne qu'une moyenne ; `--scenario` n'a de sens que si les
variables `initial` sont de vraies perturbations (pas des conditions initiales d'etat, qui n'influencent pas le regime permanent).

---

## Dynamique : `x(k+1) = x(k) + f(x, u)` (`rlstudio identify <projet> --mode dynamic`)

Le mode dynamique traite un modele (Simulink ou Python) comme une boite noire a **balayer** : aucune politique,
aucun entrainement. On en tire un modele reduit `x(k+1) = x(k) + f(x(k), u(k))`, en equations ou en reseau de neurones,
exportable en Python et MATLAB. (A ne pas confondre avec [l'extraction d'une politique](distillation.md).)

## Principe

1. **Balayage.** Chaque episode part de conditions initiales tirees par l'environnement (cle `initial` d'un projet Simulink)
   et applique des *paliers d'action aleatoires* : duree de 1 a `hold-max` pas, amplitude variable d'un episode a l'autre
   (faible excitation proche d'un equilibre, forte excitation), 20 % des paliers aux bornes pour couvrir les saturations.
2. **Ajustement** de l'increment `x(k+1) - x(k)` (normalise) par plusieurs familles :

| Famille | Forme | Quand |
|---|---|---|
| Lineaire | `A.x + B.u + c` | reference, systemes proches d'un point de fonctionnement |
| Polynome | monomes degre 2 et 3, termes croises | non-linearites lisses |
| Equation parcimonieuse | SINDy : monomes jusqu'au degre 3, `sin`, `cos`, `tanh`, `abs`, regression a seuillage sequentiel (3 seuils) | physique reconnaissable : donne des equations lisibles |
| Reseau de neurones | perceptron `tanh` 32-32, Adam, bruit sur les entrees | fortes non-linearites, precision maximale, mais opaque |

3. **Validation en simulation libre** sur des episodes *jamais vus* : le modele est simule seul, son etat reinjecte, avec les
   memes actions que le vrai systeme. La **fidelite** vaut 1 si la trajectoire est reproduite, 0 si le modele ne fait pas mieux
   qu'une constante. L'erreur par horizon (1 a 100 pas) montre ou le modele derive.
4. **Recommandation** : le modele le plus simple dont la fidelite atteint `1 - margin` (5 % par defaut).

`--rounds N` ajoute des tours de collecte : chaque tour mesure d'abord un polynome sur des donnees neuves (courbe
d'apprentissage : faut-il plus de donnees ?), puis les agrege.

## Utilisation

```bash
rlstudio identify mon-projet-simulink --mode dynamic
rlstudio identify mon-projet --mode dynamic --samples 200 --val-samples 20 --rounds 3 --hold-max 15 --methods polynomial,sparse,neural
```

Resultats dans `projects/<projet>/identify/<date>/` : `report.json`, `report.md`, et un dossier par modele
(`dynamics.py`, `dynamics.m` - compatible generation de code -, `equations.txt`, `model.json`).
Les fonctions exportees ont la signature `xn = dynamics(x, u)` ; `DT` donne le pas de temps.

## Limites a connaitre

- **Observations = etat.** Si le Simulink contient un etat cache (integrateur, retard, inertie thermique non observee), le modele est bon
  a un pas mais derive en simulation libre : le rapport le signale. Ajouter les signaux manquants aux observations.
- **Couverture.** Le modele n'est valable que dans les plages balayees (actions et conditions initiales). Hors de ces plages,
  les polynomes divergent vite et le reseau extrapole mal.
- **Cout.** Chaque pas est une simulation Simulink : prevoir de quelques centaines a quelques milliers d'episodes pour des
  systemes fortement non lineaires ; `--rounds` indique quand la fidelite sature.
- **Pas fixe.** Le modele reproduit le pas de decision de l'environnement (`step_s`) ; les actions sont maintenues sur ce pas.
