# Fiche de présentation — Reconnaissance de chiffres manuscrits

## Objectif
Entraîner un modèle à reconnaître des chiffres manuscrits (0 à 9), puis répondre
à une vraie question de terrain : **un modèle brillant en laboratoire tient-il
face à des conditions réelles ?**

## Les données
- **MNIST** : le jeu de référence (70 000 images 28×28, propres et centrées).
- **Jeu « papier »** : un second jeu **reconstitué** représentant des chiffres
  écrits sur papier et photographiés (vraies écritures manuscrites + texture de
  papier + flou, ombres, bruit, rotation, perspective).

## Méthode
1. Un **réseau de neurones (MLP)** est entraîné sur MNIST.
2. Un **prétraitement** ramène chaque photo au format de MNIST (mise en niveaux
   de gris, inversion, seuil d'Otsu, recadrage, centrage).
3. On **compare** les performances sur les deux jeux et on **analyse** les erreurs.

## Résultats clés
| Jeu de données | Précision |
|---|---|
| MNIST (laboratoire) | **~98 %** |
| Jeu papier (réel) | **~93 %** |

Le plus intéressant : la baisse n'est **pas uniforme**. Le chiffre **« 1 »
s'effondre (~66 %)**, confondu avec le « 7 » — parce que les écritures manuscrites
lui donnent un grand crochet en haut, absent des « 1 » de MNIST. C'est un cas
concret d'**écart de domaine** (*domain shift*). Détails : matrices de confusion
et exemples d'erreurs dans `figures/`.

## Comment lancer
```bash
pip install -r requirements.txt
python analyse.py        # entraînement, évaluation, figures, resultats.json
python generer_pdf.py    # (optionnel) régénère le document d'analyse
```
> MNIST est téléchargé automatiquement au premier lancement.

## Pour aller plus loin
Le document **`reconnaissance_chiffres_analyse.pdf`** contient toute l'analyse
détaillée, expliquée depuis zéro.

## À dire à l'oral (points clés)
- Un bon score **moyen** peut cacher une **faiblesse ciblée** (ici, le « 1 »).
- L'analyse **par classe** et la **matrice de confusion** sont indispensables.
- Le **prétraitement** est décisif pour limiter l'écart entre labo et réel.
- Pistes d'amélioration : data augmentation, ajout de vraies photos, réseau
  convolutif (CNN).
