# Origine du modèle `pendulum.slx`

Ce modèle est **dérivé** d'un travail tiers sous licence MIT :

- **Auteur** : Jitendra Singh — © 2021
- **Dépôt** : <https://github.com/jitendra825/Inverted-Pendulum-Simulink>
- **Fichier d'origine** : `IP_Nonlinear_model.slx` (commit `b34c2f7579d93626e06abe5a337bf8e9f913a04b`)
- **Licence** : MIT — texte d'origine conservé dans [`LICENSE-modele.txt`](LICENSE-modele.txt)

Les paramètres physiques (`m`, `M`, `I`, `l`, `g`, `b`, `c`) proviennent du script `Init_Setup_LQRArd.m` du même dépôt.

## Modifications apportées

Le script reproductible [`tools/build_cartpole_simulink.m`](../../tools/build_cartpole_simulink.m) du dépôt RL Studio
applique ces changements :

1. suppression du sous-système « 3D Animation » (les blocs *Virtual Reality* ont été retirés de MATLAB R2026b, le modèle
   d'origine ne s'y compile plus) et des blocs d'affichage (Scope, Mux, Demux, conversions en degrés) ;
2. l'échelon de force devient la constante pilotable `Force_cmd` (variable `u_force`) : c'est **l'action** de l'agent ;
3. les quatre sorties `x`, `theta`, `x_dot`, `theta_dot` du sous-système dynamique sont les **observations** ; le
   modèle d'origine mesure `theta` depuis la position **pendante** (stable), un bloc `Bias` (−π) fournit donc l'angle par
   rapport à la **verticale** ;
4. position et angle initiaux deviennent les variables `x0` et `theta0` (écart à la verticale), tirées au hasard à chaque
   épisode : le pendule démarre près de la position haute ;
5. les paramètres sont placés dans le workspace du modèle (il est autonome, sans script d'initialisation) ;
6. diagnostic « boucle algébrique » désactivé : elle est intrinsèque au couplage chariot/pendule et résolue par Simulink ;
7. solveur `ode4`, pas fixe 0,01 s ; modèle exporté pour **MATLAB R2024b** et plus récent.

## Conventions

- `force` : force appliquée au chariot, en newtons ; `theta` = 0 correspond au pendule **vertical** (position haute,
  instable) ; `x` en mètres.
- Seuls les produits **MATLAB et Simulink** sont nécessaires (ni Simscape, ni 3D Animation, ni Control System Toolbox).
- Chaque pas de décision lance une simulation Simulink : comptez environ 0,1 s par pas, soit plusieurs dizaines de
  minutes pour 30 000 pas.
