# Identifier un modele reduit de la dynamique (sans politique)

`rlstudio identify <projet>` traite un modele (Simulink ou Python) comme une boite noire a **balayer** : aucune politique,
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
rlstudio identify mon-projet-simulink
rlstudio identify mon-projet --episodes 200 --val-episodes 20 --rounds 3 --hold-max 15 --methods polynomial,sparse,neural
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
