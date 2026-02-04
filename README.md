# JUNO Neutrino Oscillation Simulation

Ce projet propose une chaîne d'analyse modulaire pour simuler le flux d'antineutrinos électroniques provenant de réacteurs nucléaires et leur détection par l'expérience **JUNO** (Jiangmen Underground Neutrino Observatory).

L'objectif principal est d'évaluer la sensibilité de JUNO à la **Hiérarchie de Masse des Neutrinos (NMO)** en comparant les spectres attendus pour l'ordonnancement Normal (NO) et l'ordonnancement Inversé (IO).

Ce code simule :
1. Le flux de issus de la fission des isotopes .
2. Le calcul de la probabilité de survie en prenant en compte les paramètres d'oscillation à 3 saveurs.
3. La détection (en cours...)


## Structure du Projet

Le projet est divisé en modules Python pour faciliter l'évolution vers des modèles plus complexes:

* **`main.py`** : Script principal qui orchestre la simulation, boucle sur les différents réacteurs (Yangjiang, Taishan, etc.) et génère les graphiques de comparaison.
* **`JunoPhysics.py`** : Contient les constantes physiques, les paramètres du détecteur (masse, nombre de protons) et les fonctions de calcul des événements journaliers.
* **`NeutrinoFlux.py`** : Gère la partie "Source" : spectres de fission (modèle de Vogel-Engel), taux de fission par réacteur et calcul du flux géométrique.
* **`Oscillation.py`** : Implémente la probabilité de survie des neutrinos en tenant compte de la hiérarchie de masse (NO ou IO).


## Installation et Utilisation

* Python 3.x
* NumPy
* Matplotlib


## Résultats

La simulation produit un graphique comparant :
* Le spectre théorique **sans oscillation**.
* Le spectre avec **Normal Ordering (NO)**.
* Le spectre avec **Inverted Ordering (IO)**.
* Les spectres avec les **effets de résolution en énergie** du détecteur.


Le code calcule également le nombre total d'événements détectés par jour, un indicateur clé pour l'analyse statistique de JUNO.

## Évolutions prévues

Conformément aux objectifs du TIPP:


## Réferences

[1] F. An et al. [JUNO Collaboration], J. Phys. G 43 (2016) no.3, 030401 [arXiv:1507.05613].


