# -*- coding: utf-8 -*-
"""
Le modèle de classification.

On utilise un réseau de neurones (Perceptron Multicouche, MLPClassifier de
scikit-learn). Il prend en entrée les 784 pixels d'une image 28x28 et renvoie
le chiffre reconnu (0 à 9).
"""

import os
import joblib
from sklearn.neural_network import MLPClassifier

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_MODELE = os.path.join(DOSSIER, "modele_entraine")
CHEMIN_MODELE = os.path.join(DOSSIER_MODELE, "mlp_mnist.joblib")


def entrainer_modele(X_train, y_train, graine=0):
    """Entraîne le réseau de neurones sur les données fournies."""
    modele = MLPClassifier(
        hidden_layer_sizes=(256, 128),   # deux couches cachées
        activation="relu",
        solver="adam",
        alpha=1e-4,                       # régularisation (évite le surapprentissage)
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,             # arrêt quand la validation ne progresse plus
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=graine,
        verbose=False,
    )
    modele.fit(X_train, y_train)
    return modele


def charger_ou_entrainer(X_train, y_train, forcer=False):
    """Charge le modèle déjà entraîné, ou l'entraîne puis le sauvegarde."""
    os.makedirs(DOSSIER_MODELE, exist_ok=True)
    if os.path.exists(CHEMIN_MODELE) and not forcer:
        return joblib.load(CHEMIN_MODELE)
    modele = entrainer_modele(X_train, y_train)
    joblib.dump(modele, CHEMIN_MODELE)
    return modele
