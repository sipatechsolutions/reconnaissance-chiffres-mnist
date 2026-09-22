# -*- coding: utf-8 -*-
"""
Gestion des données pour la reconnaissance de chiffres manuscrits.

Ce module fournit trois choses :

  1. charger_mnist()          -> le jeu de référence MNIST (chiffres propres,
                                 déjà centrés, 28x28).
  2. pretraiter_capture()     -> transforme une PHOTO (chiffre écrit sur papier)
                                 au format exact de MNIST. C'est ce même
                                 traitement qui serait appliqué à de vraies
                                 photos prises au téléphone.
  3. generer_jeu_papier()     -> fabrique un jeu de test « papier + photo »
                                 réaliste, à partir de vraies polices
                                 d'écriture manuscrite déposées sur une texture
                                 de papier, avec des artefacts de capture
                                 (flou, ombres, bruit, rotation, perspective).

Aucune donnée n'est inventée dans les résultats : MNIST est le vrai jeu de
LeCun et al., et le jeu « papier » est clairement un jeu SIMULÉ dont le but
est de mesurer l'écart de performance entre des données de laboratoire et des
données du monde réel.
"""

import os
import glob
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DATA = os.path.join(DOSSIER, "data")
DOSSIER_POLICES = os.path.join(DOSSIER, "polices")
URL_MNIST = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


# ===========================================================================
# 1. MNIST
# ===========================================================================
def charger_mnist():
    """Charge MNIST (le télécharge une seule fois si nécessaire).

    Renvoie X_train, y_train, X_test, y_test.
    Les images X sont aplaties en vecteurs de 784 valeurs dans [0, 1].
    """
    os.makedirs(DOSSIER_DATA, exist_ok=True)
    chemin = os.path.join(DOSSIER_DATA, "mnist.npz")
    if not os.path.exists(chemin):
        print("Téléchargement de MNIST (une seule fois)...")
        urllib.request.urlretrieve(URL_MNIST, chemin)

    with np.load(chemin) as d:
        x_train, y_train = d["x_train"], d["y_train"]
        x_test, y_test = d["x_test"], d["y_test"]

    X_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


# ===========================================================================
# 2. PRÉTRAITEMENT D'UNE CAPTURE (photo -> format MNIST)
# ===========================================================================
def _seuil_otsu(gris):
    """Calcule automatiquement un seuil séparant l'encre du papier (Otsu)."""
    hist, _ = np.histogram(gris, bins=256, range=(0, 256))
    total = gris.size
    somme_tot = np.dot(np.arange(256), hist)
    somme_b, poids_b, var_max, seuil = 0.0, 0.0, 0.0, 0
    for t in range(256):
        poids_b += hist[t]
        if poids_b == 0:
            continue
        poids_f = total - poids_b
        if poids_f == 0:
            break
        somme_b += t * hist[t]
        moy_b = somme_b / poids_b
        moy_f = (somme_tot - somme_b) / poids_f
        var_inter = poids_b * poids_f * (moy_b - moy_f) ** 2
        if var_inter > var_max:
            var_max, seuil = var_inter, t
    return seuil


def pretraiter_capture(image_gris):
    """Transforme une photo en niveaux de gris au format MNIST (28x28, [0,1]).

    Étapes (identiques à celles qu'on appliquerait à une vraie photo) :
      1. Léger flou pour réduire le bruit.
      2. Inversion : MNIST attend un chiffre CLAIR sur fond SOMBRE, alors qu'une
         photo montre un chiffre sombre sur papier clair.
      3. Seuil d'Otsu pour effacer le papier (le fond devient noir).
      4. Recadrage sur le chiffre, redimensionnement à 20x20 en gardant les
         proportions.
      5. Centrage dans une image 28x28 d'après le centre de masse, exactement
         comme dans la construction originale de MNIST.

    'image_gris' est un tableau numpy 2D (valeurs 0-255).
    Renvoie un vecteur de 784 valeurs dans [0, 1].
    """
    img = Image.fromarray(image_gris.astype("uint8")).filter(
        ImageFilter.GaussianBlur(1))
    gris = np.asarray(img, dtype=np.float32)

    inv = 255.0 - gris                      # chiffre clair sur fond sombre
    seuil = _seuil_otsu(255.0 - inv)        # seuil calculé sur l'image d'origine
    inv[inv < (255.0 - seuil)] = 0.0        # on efface le papier

    coords = np.argwhere(inv > 0)
    if coords.size == 0:                     # image vide : on renvoie du noir
        return np.zeros(784, dtype="float32")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    decoupe = inv[y0:y1, x0:x1]

    # Redimensionnement à 20x20 en conservant les proportions
    h, w = decoupe.shape
    if h > w:
        nh, nw = 20, max(1, int(round(w * 20.0 / h)))
    else:
        nh, nw = max(1, int(round(h * 20.0 / w))), 20
    petit = np.asarray(
        Image.fromarray(decoupe.astype("uint8")).resize((nw, nh), Image.LANCZOS),
        dtype=np.float32)

    # Collage centré dans 28x28
    canevas = np.zeros((28, 28), dtype=np.float32)
    dy, dx = (28 - nh) // 2, (28 - nw) // 2
    canevas[dy:dy + nh, dx:dx + nw] = petit

    # Recentrage fin par le centre de masse (comme MNIST)
    if canevas.sum() > 0:
        cy, cx = ndimage.center_of_mass(canevas)
        canevas = ndimage.shift(canevas, (13.5 - cy, 13.5 - cx), order=1)

    canevas = np.clip(canevas, 0, 255) / 255.0
    return canevas.reshape(784).astype("float32")


# ===========================================================================
# 3. GÉNÉRATION DU JEU « PAPIER + PHOTO »
# ===========================================================================
def _charger_polices():
    fichiers = sorted(glob.glob(os.path.join(DOSSIER_POLICES, "*.ttf")))
    if not fichiers:
        raise RuntimeError(
            "Aucune police manuscrite trouvée dans le dossier 'polices/'.")
    return fichiers


def _fond_papier(taille, rng):
    """Fabrique un fond de papier : blanc cassé, grain, éclairage inégal."""
    base = rng.integers(232, 250)
    fond = np.full((taille, taille), base, dtype=np.float32)

    # Éclairage inégal (dégradé linéaire, comme une lumière de côté)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    yy, xx = np.mgrid[0:taille, 0:taille] / taille - 0.5
    fond += (gx * xx + gy * yy) * rng.uniform(10, 30)

    # Grain fin du papier
    fond += rng.normal(0, rng.uniform(2, 6), size=(taille, taille))
    return fond


def generer_photo_chiffre(chiffre, polices, rng, taille=280):
    """Crée UNE photo réaliste d'un chiffre écrit à la main sur du papier.

    Renvoie une image PIL en niveaux de gris (la « photo »).
    """
    fond = _fond_papier(taille, rng)
    img = Image.fromarray(np.clip(fond, 0, 255).astype("uint8"), mode="L")

    # Le chiffre, dessiné sur un calque transparent puis tourné
    calque = Image.new("L", (taille, taille), 0)
    dessin = ImageDraw.Draw(calque)
    police = ImageFont.truetype(rng.choice(polices), rng.integers(150, 210))
    encre = int(rng.integers(20, 70))            # gris foncé de l'encre

    texte = str(chiffre)
    bbox = dessin.textbbox((0, 0), texte, font=police)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px = (taille - tw) / 2 - bbox[0] + rng.integers(-18, 18)
    py = (taille - th) / 2 - bbox[1] + rng.integers(-18, 18)
    dessin.text((px, py), texte, fill=255, font=police)

    # Rotation (inclinaison de la feuille / de l'écriture)
    calque = calque.rotate(rng.uniform(-18, 18), resample=Image.BICUBIC,
                           expand=False)

    # Perspective légère (photo prise de biais), une fois sur deux
    if rng.random() < 0.5:
        m = taille
        d = rng.uniform(0.02, 0.10) * m
        coeffs = _coeffs_perspective(
            [(0, 0), (m, 0), (m, m), (0, m)],
            [(rng.uniform(0, d), rng.uniform(0, d)),
             (m - rng.uniform(0, d), rng.uniform(0, d)),
             (m - rng.uniform(0, d), m - rng.uniform(0, d)),
             (rng.uniform(0, d), m - rng.uniform(0, d))])
        calque = calque.transform((m, m), Image.PERSPECTIVE, coeffs,
                                  Image.BICUBIC)

    # Composition : l'encre assombrit le papier
    fond_arr = np.asarray(img, dtype=np.float32)
    encre_arr = np.asarray(calque, dtype=np.float32) / 255.0
    resultat = fond_arr * (1 - encre_arr) + encre * encre_arr
    photo = Image.fromarray(np.clip(resultat, 0, 255).astype("uint8"), mode="L")

    # Artefacts de capture : flou d'objectif puis bruit de capteur
    photo = photo.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6)))
    arr = np.asarray(photo, dtype=np.float32)
    arr += rng.normal(0, rng.uniform(3, 9), size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L")


def _coeffs_perspective(source, cible):
    """Calcule les 8 coefficients d'une transformation perspective PIL."""
    matrice = []
    for (xs, ys), (xc, yc) in zip(source, cible):
        matrice.append([xc, yc, 1, 0, 0, 0, -xs * xc, -xs * yc])
        matrice.append([0, 0, 0, xc, yc, 1, -ys * xc, -ys * yc])
    A = np.array(matrice, dtype=np.float64)
    B = np.array(source, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


def generer_jeu_papier(n_par_chiffre=250, graine=7, n_exemples_sauves=60):
    """Fabrique le jeu de test « papier », prétraité au format MNIST.

    Renvoie :
      X_papier : (N, 784) float dans [0, 1]
      y_papier : (N,) int
      exemples : liste de (chiffre, photo_np, image28_np) pour les figures
    """
    rng = np.random.default_rng(graine)
    polices = _charger_polices()

    X, y, exemples = [], [], []
    for chiffre in range(10):
        for i in range(n_par_chiffre):
            photo = generer_photo_chiffre(chiffre, polices, rng)
            vecteur = pretraiter_capture(np.asarray(photo))
            X.append(vecteur)
            y.append(chiffre)
            if len(exemples) < n_exemples_sauves and i < 6:
                exemples.append((chiffre, np.asarray(photo),
                                 vecteur.reshape(28, 28)))

    return np.array(X, dtype="float32"), np.array(y, dtype=int), exemples


if __name__ == "__main__":
    # Petit test rapide
    Xtr, ytr, Xte, yte = charger_mnist()
    print("MNIST :", Xtr.shape, Xte.shape)
    Xp, yp, ex = generer_jeu_papier(n_par_chiffre=5)
    print("Jeu papier :", Xp.shape, "exemples sauvés :", len(ex))
