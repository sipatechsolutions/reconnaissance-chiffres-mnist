# -*- coding: utf-8 -*-
"""
Construit le document PDF explicatif :  reconnaissance_chiffres_analyse.pdf

Écrit pour un grand débutant. Toutes les valeurs chiffrées sont lues dans
resultats.json (produit par analyse.py), donc le document reste toujours
cohérent avec la dernière exécution.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage

DOSSIER = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(DOSSIER, "figures")

with open(os.path.join(DOSSIER, "resultats.json")) as f:
    R = json.load(f)

# Palette (identique au premier document, pour la cohérence)
BLEU = HexColor("#1d4ed8")
BLEU_CLAIR = HexColor("#eff6ff")
GRIS = HexColor("#334155")
GRIS_CLAIR = HexColor("#f1f5f9")
ORANGE = HexColor("#ea580c")
ORANGE_CLAIR = HexColor("#fff7ed")
NOIR = HexColor("#0f172a")
VERT_FONCE = HexColor("#166534")
VERT_CLAIR = HexColor("#f0fdf4")


# ---------------------------------------------------------------------------
# Petit schéma d'architecture du réseau (784 -> 256 -> 128 -> 10)
# ---------------------------------------------------------------------------
def schema_reseau():
    chemin = os.path.join(FIG, "schema_mlp.png")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axis("off")
    couches = [("784", "pixels\n(entrée)", "#1d4ed8"),
               ("256", "couche\ncachée 1", "#334155"),
               ("128", "couche\ncachée 2", "#334155"),
               ("10", "chiffres\n0-9 (sortie)", "#ea580c")]
    n = len(couches)
    for i, (taille, nom, coul) in enumerate(couches):
        x = i * 3.0
        cercle = plt.Circle((x, 0), 0.62, color=coul, ec="white", lw=2, zorder=3)
        ax.add_patch(cercle)
        ax.text(x, 0, taille, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=4)
        ax.text(x, -1.15, nom, ha="center", va="center", color="#334155",
                fontsize=10)
        if i < n - 1:
            ax.annotate("", xy=((i + 1) * 3.0 - 0.7, 0), xytext=(x + 0.7, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=2))
    ax.set_xlim(-1, (n - 1) * 3.0 + 1)
    ax.set_ylim(-1.8, 1.0)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    return chemin


schema_reseau()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
style_titre = ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=26, leading=32, textColor=NOIR, alignment=TA_CENTER, spaceAfter=6)
style_sous = ParagraphStyle("S", parent=styles["Normal"], fontName="Helvetica",
    fontSize=13, leading=19, textColor=GRIS, alignment=TA_CENTER, spaceAfter=4)
style_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=17, leading=22, textColor=BLEU, spaceBefore=16, spaceAfter=9)
style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=13, leading=17, textColor=NOIR, spaceBefore=11, spaceAfter=5)
style_corps = ParagraphStyle("C", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=16.5, textColor=NOIR, alignment=TA_JUSTIFY, spaceAfter=8)
style_puce = ParagraphStyle("P", parent=style_corps, leftIndent=16, bulletIndent=4,
    spaceAfter=4)
style_legende = ParagraphStyle("L", parent=styles["Normal"], fontName="Helvetica-Oblique",
    fontSize=9.5, leading=13, textColor=GRIS, alignment=TA_CENTER, spaceBefore=4,
    spaceAfter=10)


def para(t): return Paragraph(t, style_corps)
def puce(t): return Paragraph(t, style_puce, bulletText="•")


def encadre(titre, texte, fond=VERT_CLAIR, bordure=VERT_FONCE, tc=VERT_FONCE):
    pt = Paragraph(f'<font color="#{tc.hexval()[2:]}"><b>{titre}</b></font>',
                   ParagraphStyle("et", parent=style_corps, fontName="Helvetica-Bold",
                                  fontSize=11.5, spaceAfter=4))
    px = Paragraph(texte, ParagraphStyle("ex", parent=style_corps, spaceAfter=0))
    t = Table([[pt], [px]], colWidths=[15.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fond),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 10), ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 0), ("BOTTOMPADDING", (0, 1), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bordure)]))
    return t


def image_fig(nom, largeur_cm=14, legende=None):
    chemin = os.path.join(FIG, nom)
    w, h = PILImage.open(chemin).size
    largeur = largeur_cm * cm
    img = Image(chemin, width=largeur, height=largeur * h / w)
    img.hAlign = "CENTER"
    elems = [img]
    if legende:
        elems.append(Paragraph(legende, style_legende))
    return KeepTogether(elems)


H = []
A = H.append


# ===========================================================================
# PAGE DE TITRE
# ===========================================================================
A(Spacer(1, 3.0 * cm))
A(Paragraph("Reconnaître des chiffres manuscrits", style_titre))
A(Paragraph("Un modèle entraîné en laboratoire tient-il face à de vraies "
            "photos&nbsp;?", style_sous))
A(Spacer(1, 0.2 * cm))
A(Paragraph("Entraînement sur MNIST, puis confrontation à des chiffres écrits "
            "sur papier et photographiés", style_sous))
A(Spacer(1, 1.1 * cm))
A(image_fig("resume.png", largeur_cm=9))
A(Spacer(1, 0.6 * cm))
A(Paragraph("Document expliqué depuis zéro : aucune connaissance préalable "
            "n'est nécessaire.", ParagraphStyle("i", parent=style_corps,
            alignment=TA_CENTER, textColor=GRIS)))
A(PageBreak())


# ===========================================================================
# SOMMAIRE
# ===========================================================================
A(Paragraph("Sommaire", style_h1))
for s in [
    "1.  L'objectif en une phrase",
    "2.  MNIST : le jeu de référence",
    "3.  Le modèle : un réseau de neurones",
    "4.  Entraînement et performance sur MNIST",
    "5.  Un jeu de test « réel » : des chiffres sur papier",
    "6.  Le prétraitement : d'une photo à une image utilisable",
    "7.  Comparaison des performances",
    "8.  Analyse détaillée des résultats",
    "9.  Que retenir, et comment faire mieux",
    "10. Glossaire",
]:
    A(Paragraph(s, ParagraphStyle("som", parent=style_corps, spaceAfter=6,
                                  leftIndent=6)))
A(PageBreak())


# ===========================================================================
# 1. OBJECTIF
# ===========================================================================
A(Paragraph("1. L'objectif en une phrase", style_h1))
A(para(
    "On veut construire un programme qui <b>regarde l'image d'un chiffre "
    "manuscrit</b> (un 0, un 1, … un 9) et <b>devine de quel chiffre il "
    "s'agit</b>. C'est l'un des exercices les plus célèbres pour débuter en "
    "intelligence artificielle."))
A(para(
    "Mais un modèle qui réussit sur des images « propres » de laboratoire "
    "réussit-il aussi sur de <b>vraies photos</b> prises au téléphone&nbsp;? "
    "C'est toute la question de ce document. Nous allons&nbsp;: entraîner un "
    "modèle sur le jeu de référence <b>MNIST</b>, puis le tester sur un jeu de "
    "<b>chiffres écrits sur papier et photographiés</b>, et enfin "
    "<b>comparer et analyser</b> les deux performances."))
A(encadre("Le résultat en avant-première",
          "Sur les images de laboratoire (MNIST), le modèle atteint "
          f"<b>{R['acc_mnist_test']:.1f}&nbsp;%</b> de bonnes réponses. Sur les "
          f"images « papier », il tombe à <b>{R['acc_papier']:.1f}&nbsp;%</b>, "
          f"soit une baisse de <b>{R['chute']:.1f} points</b>. Nous verrons "
          "pourquoi — et le plus instructif n'est pas la moyenne, mais "
          "<b>quels chiffres</b> posent problème.",
          fond=BLEU_CLAIR, bordure=BLEU, tc=BLEU))


# ===========================================================================
# 2. MNIST
# ===========================================================================
A(Paragraph("2. MNIST : le jeu de référence", style_h1))
A(para(
    "<b>MNIST</b> est une collection de <b>70&nbsp;000 images</b> de chiffres "
    "manuscrits, rassemblée pour entraîner et tester des modèles. Chaque image "
    "est toute petite&nbsp;: <b>28 pixels sur 28</b>, en niveaux de gris. Les "
    "chiffres y sont déjà <b>bien centrés</b> et détourés sur fond noir."))
A(para(
    "Un <b>pixel</b> est un point de l'image, avec une valeur de 0 (noir) à "
    "255 (blanc). Une image de 28&times;28 contient donc "
    "<b>784 pixels</b> : ce sont les 784 nombres que le modèle reçoit en "
    "entrée."))
A(image_fig("exemples_mnist.png", largeur_cm=15,
            legende="Figure — Quelques images de MNIST. Chiffres propres, "
                    "centrés, en blanc sur fond noir."))
A(para(
    f"On sépare MNIST en deux paquets : <b>{R['n_train']:,}</b> images pour "
    f"<b>entraîner</b> le modèle (lui montrer des exemples) et "
    f"<b>{R['n_test_mnist']:,}</b> images mises de côté pour le "
    "<b>tester</b> (vérifier qu'il a vraiment appris, sur des images qu'il n'a "
    "jamais vues).".replace(",", "&nbsp;")))


# ===========================================================================
# 3. LE MODÈLE
# ===========================================================================
A(Paragraph("3. Le modèle : un réseau de neurones", style_h1))
A(para(
    "Pour reconnaître les chiffres, on utilise un <b>réseau de neurones</b>. "
    "Imaginez une série de « boutons de réglage » (appelés <b>poids</b>) que "
    "l'on ajuste automatiquement jusqu'à ce que le programme donne les bonnes "
    "réponses. Notre réseau, un <b>perceptron multicouche</b>, est organisé en "
    "couches successives&nbsp;:"))
A(image_fig("schema_mlp.png", largeur_cm=15,
            legende="Figure — L'information entre par les 784 pixels, traverse "
                    "deux couches intermédiaires, et ressort en 10 scores "
                    "(un par chiffre). Le plus haut score gagne."))
A(para(
    "Les <b>784</b> pixels entrent à gauche. Ils traversent deux "
    "<b>couches cachées</b> (256 puis 128 neurones) qui combinent l'information, "
    "et l'on obtient à droite <b>10 sorties</b> : un score pour chaque chiffre "
    "de 0 à 9. Le chiffre dont le score est le plus élevé est la "
    "<b>réponse du modèle</b>."))
A(encadre("Le mot « apprendre » ici",
          "Apprendre = ajuster les milliers de poids en montrant des exemples "
          "au réseau, jusqu'à ce que ses réponses collent aux bonnes "
          "étiquettes. Une fois entraîné, le modèle est figé et peut classer de "
          "nouvelles images.", fond=BLEU_CLAIR, bordure=BLEU, tc=BLEU))


# ===========================================================================
# 4. ENTRAINEMENT SUR MNIST
# ===========================================================================
A(Paragraph("4. Entraînement et performance sur MNIST", style_h1))
A(para(
    f"Après entraînement sur les {R['n_train']:,} images, on mesure la "
    "<b>précision</b> : le pourcentage d'images correctement reconnues."
    .replace(",", "&nbsp;")))
tbl = Table([
    ["", "Précision"],
    ["Sur les images d'entraînement (déjà vues)", f"{R['acc_mnist_train']:.1f} %"],
    ["Sur les images de test (jamais vues)", f"{R['acc_mnist_test']:.1f} %"],
], colWidths=[10.5 * cm, 4.0 * cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLEU),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 11),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING", (0, 0), (0, -1), 10),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), GRIS_CLAIR]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1"))]))
A(tbl)
A(Spacer(1, 0.3 * cm))
A(para(
    f"La précision de test (<b>{R['acc_mnist_test']:.1f}&nbsp;%</b>) est ce qui "
    "compte vraiment : elle montre que le modèle <b>généralise</b> à des images "
    "nouvelles, et ne s'est pas contenté de mémoriser. L'écart minime avec "
    "l'entraînement indique qu'il n'y a pas de <b>surapprentissage</b> "
    "important (le fait de trop « coller » aux exemples d'entraînement)."))
A(encadre("Attention à ce que « 98 % » veut dire",
          "Ce score vaut <b>uniquement pour des images du même type que MNIST</b> "
          "— propres, centrées, en blanc sur noir. Rien ne garantit encore qu'il "
          "tiendra sur de vraies photos. C'est précisément ce que la suite va "
          "vérifier.", fond=ORANGE_CLAIR, bordure=ORANGE, tc=ORANGE))


# ===========================================================================
# 5. JEU PAPIER
# ===========================================================================
A(Spacer(1, 0.3 * cm))
A(Paragraph("5. Un jeu de test « réel » : des chiffres sur papier", style_h1))
A(para(
    "MNIST est un jeu de <b>laboratoire</b> : tout y est propre et régulier. "
    "Le monde réel est plus difficile — écritures variées, feuille éclairée de "
    "travers, ombres, flou de l'appareil, photo prise de biais. Pour mesurer "
    "l'effet de ces conditions, on constitue un <b>second jeu de test</b>, "
    "représentatif de chiffres <b>écrits sur papier puis photographiés</b>."))
A(para(
    f"Ce jeu contient <b>{R['n_papier']:,}</b> images "
    f"(<b>{R['n_par_chiffre']}</b> par chiffre). Chaque image est reconstituée "
    "à partir d'une <b>véritable écriture manuscrite</b> (plusieurs styles "
    "d'écriture différents), posée sur une <b>texture de papier</b>, à laquelle "
    "on ajoute les <b>défauts d'une photo</b> : éclairage inégal, ombres, flou, "
    "grain, légère rotation et perspective. Le style d'écriture est donc "
    "<b>volontairement différent</b> de celui de MNIST — c'est justement ce qui "
    "permet de tester la robustesse du modèle.".replace(",", "&nbsp;")))
A(image_fig("exemples_papier.png", largeur_cm=15.5,
            legende="Figure — En haut : les « photos » de chiffres sur papier. "
                    "En bas : la même image après prétraitement (voir section 6)."))
A(encadre("Transparence sur ce jeu de données",
          "Ces images sont <b>reconstituées</b> pour imiter des photos réelles "
          "de façon reproductible, et non de vrais clichés pris un par un. "
          "L'intérêt est de créer un <b>écart de conditions</b> mesurable avec "
          "MNIST. Le programme fourni accepte aussi de <b>vraies photos</b> : il "
          "suffit de les déposer dans un dossier, le même prétraitement "
          "s'applique.", fond=GRIS_CLAIR, bordure=GRIS, tc=GRIS))


# ===========================================================================
# 6. PRÉTRAITEMENT
# ===========================================================================
A(Paragraph("6. Le prétraitement : d'une photo à une image utilisable", style_h1))
A(para(
    "Le modèle n'accepte que des images au <b>format exact de MNIST</b> : "
    "28&times;28, chiffre blanc centré sur fond noir. Une photo brute doit donc "
    "être <b>transformée</b> avant d'être présentée au modèle. C'est l'étape de "
    "<b>prétraitement</b>, et elle est décisive : bien faite, elle réduit "
    "fortement l'écart avec MNIST. Les étapes&nbsp;:"))
for t in [
    "<b>Mise en niveaux de gris</b> et léger flou pour atténuer le bruit.",
    "<b>Inversion</b> : le chiffre sombre sur papier clair devient clair sur "
    "fond sombre (comme MNIST).",
    "<b>Séparation encre / papier</b> par un seuil calculé automatiquement "
    "(méthode d'Otsu) : le fond devient noir.",
    "<b>Recadrage</b> sur le chiffre, puis mise à l'échelle à 20&times;20 en "
    "gardant les proportions.",
    "<b>Centrage</b> dans une image 28&times;28 d'après le « centre de masse », "
    "exactement comme MNIST.",
]:
    A(puce(t))
A(Spacer(1, 0.1 * cm))
A(para(
    "La rangée du bas de la figure précédente montre le résultat : des chiffres "
    "qui ressemblent beaucoup à MNIST. C'est ce qui permet au modèle de rester "
    "performant malgré le changement de conditions."))


# ===========================================================================
# 7. COMPARAISON
# ===========================================================================
A(PageBreak())
A(Paragraph("7. Comparaison des performances", style_h1))
A(para("Voici le cœur de l'étude : la même mesure (précision), sur les deux "
       "jeux, avec exactement le même modèle."))
A(image_fig("resume.png", largeur_cm=9.5,
            legende="Figure — Précision globale : MNIST (laboratoire) contre "
                    "jeu papier (conditions réelles)."))
A(para(
    f"Le modèle passe de <b>{R['acc_mnist_test']:.1f}&nbsp;%</b> à "
    f"<b>{R['acc_papier']:.1f}&nbsp;%</b>. Cette baisse de "
    f"<b>{R['chute']:.1f} points</b> porte un nom : l'<b>écart de domaine</b> "
    "(en anglais <i>domain shift</i>). Le modèle a appris le « style MNIST » ; "
    "confronté à un style d'écriture et à des conditions de prise de vue "
    "différents, il se trompe davantage."))
A(encadre("Un point de vocabulaire utile",
          "<b>Écart de domaine</b> : la chute de performance d'un modèle quand "
          "les données réelles diffèrent de celles sur lesquelles il a été "
          "entraîné. C'est l'un des pièges les plus courants — et les plus "
          "coûteux — en intelligence artificielle appliquée.",
          fond=BLEU_CLAIR, bordure=BLEU, tc=BLEU))


# ===========================================================================
# 8. ANALYSE DÉTAILLÉE
# ===========================================================================
A(PageBreak())
A(Paragraph("8. Analyse détaillée des résultats", style_h1))
A(para(
    "La moyenne cache l'essentiel. En regardant <b>chiffre par chiffre</b>, on "
    "découvre que la baisse n'est pas répartie uniformément."))
A(image_fig("precision_par_classe.png", largeur_cm=15,
            legende="Figure — Précision pour chaque chiffre. La plupart restent "
                    "excellents ; un chiffre s'effondre."))

# Le chiffre le plus faible sur le jeu papier
acc_pap = R["acc_cls_papier"]
pire = int(np.argmin(acc_pap))
conf = R["confusions_papier"]
c0 = conf[0]  # [count, vrai, predit]
A(para(
    f"Le constat majeur : le chiffre <b>{pire}</b> chute à "
    f"<b>{acc_pap[pire]:.0f}&nbsp;%</b> sur le jeu papier, alors qu'il "
    f"dépassait {R['acc_cls_mnist'][pire]:.0f}&nbsp;% sur MNIST. Les autres "
    "chiffres, eux, restent proches de leur niveau MNIST."))

A(Paragraph("Où partent les erreurs ? La matrice de confusion", style_h2))
A(para(
    "Une <b>matrice de confusion</b> croise le <b>vrai</b> chiffre (en ligne) "
    "et le chiffre <b>prédit</b> par le modèle (en colonne). La diagonale = les "
    "bonnes réponses. Toute case <b>hors diagonale</b> est une erreur, et "
    "indique <b>avec quoi</b> le chiffre a été confondu."))
A(image_fig("matrice_papier.png", largeur_cm=11.5,
            legende="Figure — Matrice de confusion sur le jeu papier. Les cases "
                    "hors diagonale révèlent les confusions."))
A(para(
    f"La case la plus parlante : le vrai chiffre <b>{c0[1]}</b> a été lu "
    f"« <b>{c0[2]}</b> » <b>{c0[0]} fois</b>. Les confusions suivantes sont "
    f"{conf[1][1]}&nbsp;&#8594;&nbsp;{conf[1][2]} ({conf[1][0]} fois) et "
    f"{conf[2][1]}&nbsp;&#8594;&nbsp;{conf[2][2]} ({conf[2][0]} fois)."))

A(Paragraph("Pourquoi ces confusions ? Regardons les erreurs", style_h2))
A(para(
    "En affichant les images mal classées, la cause devient évidente. Les "
    "styles d'écriture manuscrite dessinent souvent le chiffre "
    f"<b>{c0[1]}</b> avec un <b>crochet marqué en haut</b> — une forme que le "
    f"modèle, habitué aux « {c0[1]} » tout droits de MNIST, prend pour un "
    f"« {c0[2]} ». Ce n'est pas un bug : c'est un vrai désaccord de style entre "
    "les deux jeux de données."))
A(image_fig("erreurs_papier.png", largeur_cm=13.5,
            legende="Figure — Exemples d'images mal classées. « vu X, dit Y » = "
                    "le vrai chiffre était X, le modèle a répondu Y."))
A(encadre("La leçon de cette analyse",
          "Une bonne précision <b>moyenne</b> peut masquer une <b>faiblesse "
          "ciblée</b> et grave. Ici, tout va bien… sauf un chiffre, pour une "
          "raison précise et compréhensible. Analyser <b>par classe</b> et "
          "<b>par confusion</b> est indispensable pour savoir à qui se fier.",
          fond=VERT_CLAIR, bordure=VERT_FONCE))


# ===========================================================================
# 9. QUE RETENIR
# ===========================================================================
A(PageBreak())
A(Paragraph("9. Que retenir, et comment faire mieux", style_h1))
for t in [
    f"<b>Un modèle brillant en laboratoire peut faiblir dans le réel.</b> "
    f"Ici, {R['acc_mnist_test']:.1f}&nbsp;% sur MNIST mais "
    f"{R['acc_papier']:.1f}&nbsp;% sur des photos : c'est l'écart de domaine.",
    "<b>La moyenne ne suffit pas.</b> L'analyse par chiffre et la matrice de "
    "confusion révèlent une faiblesse ciblée, invisible dans le score global.",
    "<b>Le prétraitement est décisif.</b> Ramener les photos au format de "
    "MNIST (inversion, seuil, recadrage, centrage) limite fortement la chute.",
]:
    A(puce(t))
A(Spacer(1, 0.2 * cm))
A(Paragraph("Pistes concrètes pour réduire l'écart", style_h2))
for t in [
    "<b>Enrichir l'entraînement (data augmentation)</b> : appliquer aux images "
    "MNIST des rotations, flous, ombres et déformations, pour que le modèle "
    "voie dès l'entraînement des conditions proches du réel.",
    "<b>Ajouter de vraies images</b> : intégrer quelques centaines de chiffres "
    "écrits à la main dans l'entraînement (on parle de <i>fine-tuning</i>).",
    "<b>Un modèle plus adapté aux images</b> : un réseau de neurones "
    "<b>convolutif</b> (CNN) reconnaît mieux les formes et résiste davantage "
    "aux variations de style et de position.",
    "<b>Cibler la faiblesse connue</b> : puisqu'un chiffre pose problème, on "
    "peut lui fournir davantage d'exemples variés.",
]:
    A(puce(t))
A(Spacer(1, 0.2 * cm))
A(encadre("En une phrase",
          "Mesurer un modèle sur les données du monde où il sera vraiment "
          "utilisé — et non seulement sur son jeu d'entraînement — est la "
          "première règle d'un projet d'IA sérieux.",
          fond=BLEU_CLAIR, bordure=BLEU, tc=BLEU))


# ===========================================================================
# 10. GLOSSAIRE
# ===========================================================================
A(PageBreak())
A(Paragraph("10. Glossaire", style_h1))
glo = [
    ("MNIST", "Collection de 70 000 images (28x28) de chiffres manuscrits, "
     "servant de référence pour entraîner et tester des modèles."),
    ("Pixel", "Un point de l'image, avec une valeur de 0 (noir) à 255 (blanc). "
     "Une image 28x28 en compte 784."),
    ("Réseau de neurones (MLP)", "Modèle formé de couches de « neurones » reliés "
     "par des poids ajustables, qui transforme une entrée en une prédiction."),
    ("Poids", "Les valeurs réglables du réseau. Les « apprendre », c'est les "
     "ajuster à partir d'exemples."),
    ("Entraînement / Test", "On entraîne le modèle sur un paquet d'images, puis "
     "on le teste sur d'autres images jamais vues, pour juger honnêtement."),
    ("Précision", "Pourcentage d'images correctement classées."),
    ("Surapprentissage", "Quand le modèle mémorise les exemples d'entraînement "
     "au lieu de généraliser ; il réussit sur eux mais échoue sur du nouveau."),
    ("Prétraitement", "Transformer une image brute (photo) pour la mettre au "
     "format attendu par le modèle."),
    ("Matrice de confusion", "Tableau croisant vrais chiffres et chiffres "
     "prédits ; montre avec quoi chaque chiffre est confondu."),
    ("Écart de domaine", "Baisse de performance quand les données réelles "
     "diffèrent de celles de l'entraînement (en anglais : domain shift)."),
    ("Data augmentation", "Créer des variantes des images d'entraînement "
     "(rotations, flous…) pour rendre le modèle plus robuste."),
    ("CNN", "Réseau de neurones convolutif, spécialisé dans les images, plus "
     "robuste aux variations de forme et de position."),
]
lignes = [[Paragraph(f"<b>{a}</b>", style_corps), Paragraph(b, style_corps)]
          for a, b in glo]
tg = Table(lignes, colWidths=[4.8 * cm, 10.8 * cm])
tg.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [HexColor("#ffffff"), GRIS_CLAIR]),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor("#e2e8f0"))]))
A(tg)


# ===========================================================================
# ASSEMBLAGE
# ===========================================================================
def pied(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GRIS)
    n = canvas.getPageNumber()
    if n > 1:
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(n))
        canvas.drawString(2 * cm, 1.1 * cm,
                          "Reconnaître des chiffres manuscrits — MNIST face aux photos")
        canvas.setStrokeColor(HexColor("#e2e8f0"))
        canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


chemin_pdf = os.path.join(DOSSIER, "reconnaissance_chiffres_analyse.pdf")
doc = BaseDocTemplate(chemin_pdf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm,
                      title="Reconnaitre des chiffres manuscrits - MNIST face aux photos",
                      author="")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
doc.addPageTemplates([PageTemplate(id="tpl", frames=[frame], onPage=pied)])
doc.build(H)
print("PDF genere :", chemin_pdf)
