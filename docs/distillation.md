# Extraire un modèle mathématique d'un run

Un agent entraîné par RL est un réseau de neurones : performant, mais opaque. L'onglet **Modèle mathématique** (et la
commande `rlstudio distill`) en extrait une **loi de commande lisible** — gains d'un PID, table de gains, arbre de
décision, équation — que l'on peut relire, valider, et embarquer ailleurs (Python, MATLAB, Simulink).

## Principe

1. **Collecte.** La politique apprise rejoue des épisodes avec un léger bruit d'action (pour visiter des états voisins) ;
   chaque état est étiqueté par l'*action déterministe* de la politique (moyenne pour SAC/PPO, sans bruit
   d'exploration).
2. **Ajustement.** Plusieurs familles de modèles sont ajustées sur ces paires état → action.
3. **DAgger.** Un modèle simple qui imite la politique finit par visiter d'autres états qu'elle (*dérive de
   distribution*). À chaque tour, le modèle pilote (en partie) la simulation, la politique **réétiquette** les états
   réellement visités, on agrège et on réajuste.
4. **Validation en boucle fermée.** Chaque modèle pilote la simulation sur les mêmes graines que la politique.
   Le **score** est normalisé : 0 = politique aléatoire, 1 = politique apprise.
5. **Recommandation.** Le modèle **le plus simple** dont le score atteint `1 − tolérance` (5 % par défaut).

## Familles de modèles

| Famille | Forme | Quand |
|---|---|---|
| Linéaire / PID | `u = K·x + b`, avec action intégrale optionnelle (observations à intégrer) | systèmes proches d'un régulateur classique |
| Table de gains | interpolation 1D/2D sur les observations les plus influentes | non-linéarités douces, gain programmé |
| Arbre de décision | règles `si x < seuil alors …` (profondeurs 3, 5, 8) | actions par paliers, saturation, commutations |
| Équation symbolique | somme parcimonieuse de termes (`x`, `x²`, `sin`, `tanh`, produits…), 3 / 6 / 12 termes | loi compacte et lisible |
| Réseau exact | copie *exacte* du réseau SB3 (vérifiée numériquement), équations matricielles | référence : même comportement, aucune interprétation |

## Utilisation

Interface : onglet **Modèle mathématique** → choisir le run, les familles, **Extraire le modèle**. Le tableau compare
complexité, fidélité (R²), retour en boucle fermée et score ; un graphe montre le compromis simplicité / performance.
Chaque modèle peut être copié, enregistré (`.py`, `.m`) ou exporté vers **Simulink** (bloc *MATLAB Function* avec une
entrée par observation et une sortie par action ; nécessite MATLAB).

Ligne de commande :

```bash
rlstudio distill mon-projet                      # dernier run, toutes les familles
rlstudio distill mon-projet --methods linear,tree --dagger 5 --episodes 40
rlstudio distill mon-projet --methods linear --integrators angle   # PID avec action intégrale sur « angle »
```

Résultats dans `projects/<projet>/runs/<run>/distill/` : `report.json`, `report.md`, et un dossier par modèle
(`policy.py`, `policy.m`, `equations.txt`, `surrogate.json`).

## Lire les résultats

Sur le CartPole fourni (SAC, 30 000 pas, retour moyen 332 ± 43), l'extraction recommande une **équation à 3 termes**
(complexité 4, score 1,10) ; un PID linéaire (5 paramètres) et un arbre de profondeur 3 (8 paramètres) font aussi bien.
Un score **supérieur à 1** est normal quand la politique apprise est imparfaite : un modèle simple peut la
*lisser* et mieux se comporter qu'elle. Les scores fluctuent d'un jeu d'évaluation à l'autre (10 épisodes par défaut) :
augmenter « Épisodes d'évaluation » avant de trancher entre deux modèles proches.

## Pièges pris en compte

- **Action déterministe** : les étiquettes sont la moyenne de la politique (SAC/PPO), pas un tirage bruité.
- **Bornes de l'action** : les modèles sont écrêtés aux bornes ; une note signale si la politique sature souvent
  (un modèle lisse reproduit mal les paliers : préférer l'arbre).
- **Dérive de distribution** : DAgger (désactivable) ; sans lui, un avertissement est ajouté au rapport.
- **Extrapolation** : hors des états visités un modèle simple extrapole ; toujours valider en simulation.
- **Fidélité ≠ performance** : un R² élevé sur les actions ne garantit pas un bon comportement ; le score en boucle
  fermée est le critère.
- **Modèles à état** (intégrale) : à réinitialiser au début de chaque épisode (`reset()` / `clear` en MATLAB).
- **Stabilité** : aucune garantie formelle n'est apportée ; le rapport mesure la proportion d'épisodes menés à terme.

## Limites

Actions continues (`Box`) uniquement ; les observations sont celles de l'environnement (un modèle dépend du
pas de temps de la simulation pour l'action intégrale).
