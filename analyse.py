# -*- coding: utf-8 -*-
"""
Pipeline complet :
  1. Charge MNIST.
  2. Fabrique (ou recharge) le jeu de test « papier + photo ».
  3. Entraîne le réseau de neurones sur MNIST.
  4. Évalue le modèle sur MNIST ET sur le jeu papier.
  5. Produit toutes les figures et un fichier de résultats (resultats.json)
     réutilisés par le document PDF.

Lancement :  python analyse.py
"""

import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, accuracy_score

from donnees import charger_mnist, generer_jeu_papier
from modele import charger_ou_entrainer

DOSSIER = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(DOSSIER, "figures")
DATA = os.path.join(DOSSIER, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

BLEU = "#1d4ed8"
ORANGE = "#ea580c"
GRIS = "#334155"
N_PAR_CHIFFRE = 250


# ---------------------------------------------------------------------------
def preparer_donnees():
    print("1) Chargement de MNIST...")
    X_train, y_train, X_test, y_test = charger_mnist()

    chemin = os.path.join(DATA, "jeu_papier.npz")
    if os.path.exists(chemin):
        print("2) Rechargement du jeu papier depuis le cache...")
        d = np.load(chemin, allow_pickle=True)
        X_pap, y_pap = d["X"], d["y"]
        exemples = list(d["exemples"])
    else:
        print("2) Génération du jeu papier (peut prendre ~1 min)...")
        t = time.time()
        X_pap, y_pap, exemples = generer_jeu_papier(n_par_chiffre=N_PAR_CHIFFRE)
        np.savez_compressed(chemin, X=X_pap, y=y_pap,
                            exemples=np.array(exemples, dtype=object))
        print("   fait en %.0f s" % (time.time() - t))
    return X_train, y_train, X_test, y_test, X_pap, y_pap, exemples


# ---------------------------------------------------------------------------
def precision_par_classe(y_vrai, y_pred):
    acc = []
    for c in range(10):
        masque = y_vrai == c
        acc.append(100.0 * np.mean(y_pred[masque] == c))
    return acc


def paires_confondues(cm, k=5):
    """Renvoie les k confusions les plus fréquentes (vrai -> prédit)."""
    paires = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                paires.append((int(cm[i, j]), i, j))
    paires.sort(reverse=True)
    return paires[:k]


# ---------------------------------------------------------------------------
def figure_exemples(exemples):
    """Photos papier brutes (une ligne) + version prétraitée (ligne du dessous)."""
    par_chiffre = {}
    for chiffre, photo, img28 in exemples:
        par_chiffre.setdefault(chiffre, (photo, img28))
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.1))
    for c in range(10):
        photo, img28 = par_chiffre[c]
        axes[0, c].imshow(photo, cmap="gray"); axes[0, c].axis("off")
        axes[0, c].set_title(str(c), fontsize=11, color=GRIS)
        axes[1, c].imshow(img28, cmap="gray"); axes[1, c].axis("off")
    axes[0, 0].set_ylabel("photo", fontsize=9)
    fig.text(0.01, 0.72, "Photo\npapier", fontsize=9, color=GRIS, va="center")
    fig.text(0.01, 0.28, "Prétraité\n28x28", fontsize=9, color=GRIS, va="center")
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    plt.savefig(os.path.join(FIG, "exemples_papier.png"), dpi=150)
    plt.close()


def figure_exemples_mnist(X_test, y_test):
    fig, axes = plt.subplots(2, 10, figsize=(14, 2.9))
    for c in range(10):
        idx = np.where(y_test == c)[0][:2]
        for r in range(2):
            axes[r, c].imshow(X_test[idx[r]].reshape(28, 28), cmap="gray")
            axes[r, c].axis("off")
        axes[0, c].set_title(str(c), fontsize=11, color=GRIS)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "exemples_mnist.png"), dpi=150)
    plt.close()


def figure_matrice(cm, titre, nom):
    cmap = LinearSegmentedColormap.from_list("bl", ["#ffffff", BLEU])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Chiffre prédit par le modèle", color=GRIS)
    ax.set_ylabel("Vrai chiffre", color=GRIS)
    ax.set_title(titre, color=GRIS, fontsize=12)
    seuil = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > seuil else GRIS)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, nom), dpi=150)
    plt.close()


def figure_precision_par_classe(acc_mnist, acc_pap):
    x = np.arange(10)
    largeur = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - largeur / 2, acc_mnist, largeur, label="MNIST (labo)", color=BLEU)
    ax.bar(x + largeur / 2, acc_pap, largeur, label="Papier (réel)", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xlabel("Chiffre", color=GRIS)
    ax.set_ylabel("Précision (%)", color=GRIS)
    ax.set_ylim(0, 105)
    ax.set_title("Précision par chiffre : MNIST vs jeu papier", color=GRIS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "precision_par_classe.png"), dpi=150)
    plt.close()


def figure_resume(acc_mnist, acc_pap):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    barres = ax.bar(["MNIST\n(labo)", "Papier\n(réel)"], [acc_mnist, acc_pap],
                    color=[BLEU, ORANGE], width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Précision globale (%)", color=GRIS)
    ax.set_title("Précision globale", color=GRIS)
    for b, v in zip(barres, [acc_mnist, acc_pap]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, "%.1f %%" % v,
                ha="center", fontsize=12, fontweight="bold", color=GRIS)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "resume.png"), dpi=150)
    plt.close()


def figure_erreurs(X_pap, y_pap, y_pred):
    """Grille d'exemples MAL classés par le modèle sur le jeu papier."""
    mauvais = np.where(y_pred != y_pap)[0]
    rng = np.random.default_rng(0)
    if len(mauvais) > 15:
        mauvais = rng.choice(mauvais, 15, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(9, 5.6))
    for ax, idx in zip(axes.ravel(), mauvais):
        ax.imshow(X_pap[idx].reshape(28, 28), cmap="gray")
        ax.set_title("vu %d, dit %d" % (y_pap[idx], y_pred[idx]),
                     fontsize=9, color=ORANGE)
        ax.axis("off")
    for ax in axes.ravel()[len(mauvais):]:
        ax.axis("off")
    fig.suptitle("Exemples d'erreurs du modèle sur le jeu papier",
                 color=GRIS, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG, "erreurs_papier.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
def main():
    (X_train, y_train, X_test, y_test,
     X_pap, y_pap, exemples) = preparer_donnees()

    print("3) Entraînement du modèle sur MNIST...")
    t = time.time()
    modele = charger_ou_entrainer(X_train, y_train)
    print("   fait en %.0f s" % (time.time() - t))

    print("4) Évaluation...")
    pred_train = modele.predict(X_train[:10000])
    pred_test = modele.predict(X_test)
    pred_pap = modele.predict(X_pap)

    acc_train = 100.0 * accuracy_score(y_train[:10000], pred_train)
    acc_test = 100.0 * accuracy_score(y_test, pred_test)
    acc_pap = 100.0 * accuracy_score(y_pap, pred_pap)

    cm_test = confusion_matrix(y_test, pred_test)
    cm_pap = confusion_matrix(y_pap, pred_pap)
    acc_cls_mnist = precision_par_classe(y_test, pred_test)
    acc_cls_pap = precision_par_classe(y_pap, pred_pap)

    print("   MNIST (train)  : %.2f %%" % acc_train)
    print("   MNIST (test)   : %.2f %%" % acc_test)
    print("   Jeu papier     : %.2f %%" % acc_pap)

    print("5) Figures...")
    figure_exemples_mnist(X_test, y_test)
    figure_exemples(exemples)
    figure_matrice(cm_test, "Matrice de confusion — MNIST", "matrice_mnist.png")
    figure_matrice(cm_pap, "Matrice de confusion — jeu papier", "matrice_papier.png")
    figure_precision_par_classe(acc_cls_mnist, acc_cls_pap)
    figure_resume(acc_test, acc_pap)
    figure_erreurs(X_pap, y_pap, pred_pap)

    resultats = {
        "n_train": int(len(y_train)),
        "n_test_mnist": int(len(y_test)),
        "n_papier": int(len(y_pap)),
        "n_par_chiffre": N_PAR_CHIFFRE,
        "acc_mnist_train": round(acc_train, 2),
        "acc_mnist_test": round(acc_test, 2),
        "acc_papier": round(acc_pap, 2),
        "chute": round(acc_test - acc_pap, 2),
        "acc_cls_mnist": [round(a, 1) for a in acc_cls_mnist],
        "acc_cls_papier": [round(a, 1) for a in acc_cls_pap],
        "confusions_papier": paires_confondues(cm_pap, 6),
        "architecture": "MLP (784 -> 256 -> 128 -> 10), activation ReLU",
    }
    with open(os.path.join(DOSSIER, "resultats.json"), "w") as f:
        json.dump(resultats, f, indent=2)
    print("Terminé. Résultats écrits dans resultats.json")
    return resultats


if __name__ == "__main__":
    main()
