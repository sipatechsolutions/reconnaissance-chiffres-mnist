# Reconnaissance de chiffres manuscrits — MNIST face aux photos

Entraînement d'un modèle de classification de chiffres manuscrits sur **MNIST**,
puis évaluation sur un second jeu de test représentant des **chiffres écrits sur
papier et photographiés**. Le projet **compare** les deux performances et en
fait une **analyse** détaillée (rassemblée dans le document PDF).

## Résultat principal

| Jeu de données | Précision |
|---|---|
| MNIST (test, laboratoire) | ~98 % |
| Jeu « papier » (conditions réelles) | ~93 % |

La baisse est surtout concentrée sur certains chiffres : l'analyse par classe et
la matrice de confusion le montrent en détail dans le PDF.

## Contenu

| Fichier | Rôle |
|---|---|
| `donnees.py` | Chargement de MNIST, génération du jeu « papier », prétraitement des captures. |
| `modele.py` | Le réseau de neurones (MLP) et son entraînement. |
| `analyse.py` | Pipeline complet : entraînement, évaluation, figures, `resultats.json`. |
| `generer_pdf.py` | Construit le document explicatif. |
| `reconnaissance_chiffres_analyse.pdf` | Document pédagogique complet avec l'analyse. |
| `polices/` | Polices d'écriture manuscrite (licence Open Font License) utilisées pour le jeu papier. |
| `figures/` | Illustrations et graphiques produits par l'analyse. |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python analyse.py        # entraîne, évalue, produit les figures et resultats.json
python generer_pdf.py    # construit le PDF à partir des résultats
```

MNIST est téléchargé automatiquement au premier lancement. Le jeu « papier » et
le modèle entraîné sont mis en cache pour accélérer les exécutions suivantes.

## Le jeu « papier »

Ces images sont **reconstituées** (à partir de vraies polices manuscrites, d'une
texture de papier et d'effets de capture : éclairage inégal, ombres, flou, bruit,
rotation, perspective) afin d'imiter des photos réelles de façon reproductible.
Le prétraitement (`pretraiter_capture` dans `donnees.py`) accepte également de
**vraies photos** : il suffit de les fournir en niveaux de gris, chiffre sombre
sur papier clair.

## Le prétraitement (photo → format MNIST)

1. Niveaux de gris + léger flou.
2. Inversion (chiffre clair sur fond sombre).
3. Séparation encre / papier par seuil d'Otsu.
4. Recadrage sur le chiffre, mise à l'échelle 20×20.
5. Centrage dans une image 28×28 d'après le centre de masse.
