# Licences des composants tiers

RL Studio est construit sur des logiciels libres. Ce fichier liste ce qui est **distribué avec l'application**,
ce qui est **installé à la demande sur la machine de l'utilisateur** (non redistribué par nous), et les
obligations correspondantes. RL Studio lui-même est distribué sous **licence MIT** (© 2026 Christophe Barras) : voir le fichier `LICENSE`.

## 1. Interface graphique : Qt 6 via PySide6 (LGPL v3)

| | |
|---|---|
| Composants | Qt 6.8.3 (Core, Gui, Widgets, Network, OpenGL, OpenGLWidgets, Svg, Test) et PySide6 / shiboken6 6.8.3 |
| Licence | **GNU LGPL v3** (PySide6 et Qt sont aussi disponibles sous GPL v2/v3 et licence commerciale ; RL Studio utilise l'option LGPL v3) — texte : [`licenses/LGPL-3.0.txt`](licenses/LGPL-3.0.txt), qui renvoie à [`licenses/GPL-3.0.txt`](licenses/GPL-3.0.txt) |
| Copyright | © The Qt Company Ltd. et contributeurs |
| Liaison | **Dynamique** : les bibliothèques Qt et PySide6 sont des fichiers séparés dans `_internal\PySide6\` (aucune liaison statique, application PyInstaller en mode « dossier », pas en `--onefile`) |

**Vous avez le droit de remplacer Qt / PySide6** par une autre version (modifiée ou non) : remplacez les fichiers
`Qt6*.dll`, les modules `Qt*.pyd` et les dossiers `plugins` de `_internal\PySide6\` (et `_internal\shiboken6\`),
puis relancez `rlstudio.exe`. Aucune restriction technique ne l'empêche.

**Code source de Qt et PySide6** (versions distribuées, non modifiées par RL Studio) :

- Qt 6.8.3 : <https://download.qt.io/archive/qt/6.8/6.8.3/single/qt-everywhere-src-6.8.3.tar.xz>
- PySide6 6.8.3 : <https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3>

Les binaires distribués sont ceux des paquets `PySide6-Essentials` 6.8.3 publiés par The Qt Company sur PyPI.
Si ces adresses devenaient indisponibles, vous pouvez demander les sources correspondantes aux mainteneurs de
RL Studio (offre écrite valable trois ans à compter de la distribution, conformément à la LGPL v3).

**Modules Qt exclus volontairement** : Qt Virtual Keyboard (distribué uniquement sous GPL v3 / licence commerciale,
non LGPL), ainsi que Qt Qml, Qt Quick et Qt PDF (inutiles). Le script `packaging/check_bundle.py` échoue si un
module Qt hors liste blanche apparaît dans une version construite.

> « Qt » est une marque déposée de The Qt Company Ltd.

## 2. Distribué avec l'application (interface gelée par PyInstaller)

| Composant | Licence | Remarque |
|---|---|---|
| [pyqtgraph](https://www.pyqtgraph.org) | MIT | graphiques |
| [NumPy](https://numpy.org) | BSD-3-Clause (et licences des bibliothèques embarquées) | |
| [PyYAML](https://pyyaml.org) | MIT | |
| [PyInstaller](https://pyinstaller.org) | GPL v2+ **avec exception** autorisant les applications gelées sous toute licence | bootloader des exécutables |
| Python (CPython 3.12) | PSF License | interpréteur de l'interface |
| Bibliothèques d'exécution Microsoft Visual C++ | Conditions de redistribution Microsoft | `vcruntime140*.dll`, `msvcp140*.dll` |

## 3. Distribué avec l'installateur (runtime de calcul)

| Composant | Licence | Remarque |
|---|---|---|
| Python 3.12 « embeddable » (python.org) | PSF License | copié dans le runtime ; son `LICENSE.txt` est conservé |
| [get-pip.py](https://github.com/pypa/get-pip) | MIT (inclut pip et ses dépendances, licences MIT/BSD/Apache/PSF) | |

## 4. Téléchargé à l'installation sur la machine de l'utilisateur (non redistribué)

L'installateur télécharge ces paquets directement depuis leurs dépôts officiels ; RL Studio ne les redistribue
pas et chacun reste soumis à sa propre licence, que l'utilisateur accepte auprès de l'éditeur.

| Composant | Licence | Source |
|---|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause | download.pytorch.org |
| Bibliothèques NVIDIA CUDA / cuDNN (incluses dans les builds CUDA de PyTorch) | **Licence NVIDIA** (EULA propriétaire) | download.pytorch.org |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | MIT | PyPI |
| [Gymnasium](https://gymnasium.farama.org) | MIT | PyPI |
| NumPy, PyYAML | voir ci-dessus | PyPI |
| `matlabengine` (MATLAB Engine API for Python) | **Licence MathWorks** | PyPI |

## 5. MATLAB et Simulink

RL Studio **ne contient ni ne redistribue** MATLAB ni Simulink. L'intégration (explorateur de blocs, simulation pas
à pas) nécessite une installation de MATLAB/Simulink **avec licence valide de l'utilisateur**. Les scripts
`matlab/rlstudio_*.m` sont des fichiers de RL Studio.

> MATLAB et Simulink sont des marques déposées de The MathWorks, Inc. NVIDIA et CUDA sont des marques déposées
> de NVIDIA Corporation. Python est une marque de la Python Software Foundation. Ces noms ne sont utilisés que
> pour désigner les logiciels compatibles ; RL Studio n'est ni affilié à ces sociétés ni approuvé par elles.

## 6. Outil de construction

L'installateur est produit avec [Inno Setup](https://jrsoftware.org/isinfo.php), dont la licence autorise la
distribution libre des installateurs générés.
