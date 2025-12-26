# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

En utilisant ces informations, un modèle HMM peut
être construit pour calculer la probabilité qu'un mot appartienne à une langue donnée
"""

import numpy as np
import pandas as pd
import unidecode
import unicodedata
import re

"Lecture et nettoyage des corpus"

def lire_corpus(nom_fichier):
    texte_nettoye = ""
    try:
        with open(nom_fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                # Supprimer les accents
                ligne = unicodedata.normalize('NFD', ligne)
                ligne = ligne.encode('ascii', 'ignore').decode('utf-8')
                ligne = ligne.lower()
                # Supprimer la ponctuation et les caractères non a-z
                ligne = re.sub(r'[^a-z\s]', '', ligne)
                # Ajouter la ligne nettoyée au texte global
                texte_nettoye += ligne.strip() + " "
    except FileNotFoundError:
        print(f"Fichier '{nom_fichier}' introuvable.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
    
    # Supprimer les espaces superflus et retourner sous forme de liste contenant une seule chaîne
    return texte_nettoye.strip()

corpus_fr = lire_corpus("C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/french.txt")
corpus_en = lire_corpus("C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/english.txt")
corpus_it = lire_corpus("C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/italian.txt")


def matrice_emission(nom_fichier_emission):
    # Lecture du fichier Excel
    B = pd.read_excel(nom_fichier_emission,index_col=0)
    return B.to_numpy()

# B = matrice_emission("C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/matrice_emission.xls")
B = np.eye(26)


def matrice_transition(nom_fichier):    
    # Extraire toutes les lettres du corpus
    X = lire_corpus(nom_fichier)
    n = 26
    M = np.zeros((n, n))  # matrice 26x26

    # Remplir la matrice de transition
    for i in range(len(X) - 1):
        # Convertir les caracteres en indices numeriques
        c1 = ord(X[i]) - ord('a')
        c2 = ord(X[i+1]) - ord('a')
        # Eviter les espaces entre les caracteres
        if 0<= c1 <=25 and  0<= c2 <=25:
            M[c1,c2] += 1
    
    #matrice stochastique
    ligne = M.sum(axis=1)
    
    # eviter les division par 0
    ligne[ligne==0] = 1 
    
    A = M/ligne[:,np.newaxis]
    

    return A

nom_fichier_fr = "C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/french.txt"
nom_fichier_en = "C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/english.txt"
nom_fichier_it = "C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/italian.txt"

A_fr = matrice_transition(nom_fichier_fr)
A_en = matrice_transition(nom_fichier_en)
A_it = matrice_transition(nom_fichier_it)

def forward(O, A, B, PI):
    N = len(A) #Nombre d'etat
    T = len(O) #longueur de la sequence d'observation
    alpha = np.zeros((T,N))
    
    for i in range (N):
        alpha[0,i] = PI[i,0] * B[i,O[0]] 
        
    for t in range(T-1):
        for j in range(N):
            somme = 0 
            for i in range(N):
                somme += alpha[t,i]*A[i,j]
            alpha[t+1,j] = B[j,O[t+1]]*somme           
        
    P = 0       
    for i in range(N):
            P += alpha[T-1,i]
            
    return P, alpha

def backward(O, A, B, PI):
    N = len(A)
    T = len(O)
    beta = np.zeros((T, N))
    # Initialisation
    beta[T-1, :] = 1 
    # Boucle d'induction
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :]) 
    # Terminaison
    P = np.sum(PI * B[:, O[0]] * beta[0, :])
    return P, beta



def mot_en_indices(mot):
    return [ord(c)-ord('a') for c in mot.lower() if c.isalpha()]


def tester_et_confusion(mots, A_fr, A_en, A_it, B, PI, mot2idx, forward, backward):
    langues = ["FR", "EN", "IT"]
    matrices = [A_fr, A_en, A_it]
    priors = np.array([1/3, 1/3, 1/3])

    # matrice de confusion 3x3 initialisée à 0
    confusion = np.zeros((3, 3), dtype=int)

    for i, mot in enumerate(mots):
        # convertir mot → indices
        O = mot2idx(mot)

        # forward
        P_fw = np.array([forward(O, A, B, PI)[0] for A in matrices])
        posterior_fw = P_fw * priors
        posterior_fw /= posterior_fw.sum()

        # backward
        P_bw = np.array([backward(O, A, B, PI)[0] for A in matrices])
        posterior_bw = P_bw * priors
        posterior_bw /= posterior_bw.sum()

        # prédiction = langue avec la plus grande probabilité (forward ici)
        predicted = np.argmax(posterior_fw)

        # vérité terrain (FR=0, EN=1, IT=2)
        true_class = i  

        # mise à jour de la matrice de confusion
        confusion[true_class, predicted] += 1

        print("\n------ Résultats pour le mot :", mot, "------")
        print("Posterior FORWARD  :", posterior_fw)
        print("Posterior BACKWARD :", posterior_bw)
        print("Langue prédite     :", langues[predicted])

    return confusion

langues_reelles = ["FR", "EN", "IT"]
langues = ["FR", "EN", "IT"]
A_dict = {"FR": A_fr, "EN": A_en, "IT": A_it}
PI = np.ones((26,1))/26

mots = ["probablement", "probably", "probabilmente"]

confusion_matrix = tester_et_confusion(
    mots=mots,
    A_fr=A_fr,
    A_en=A_en,
    A_it=A_it,
    B=B,
    PI=PI,
    mot2idx=mot_en_indices,
    forward=forward,
    backward=backward
)

print("\n===== MATRICE DE CONFUSION =====")
print(confusion_matrix)

"Partie2 du HMM"


def lire_fichier(nom_fichier):
    """
    Lecture d'un fichier texte et retour d'une liste de mots nettoyés :
    - gestion des encodages (UTF-8 ou fallback Latin-1)
    - suppression des accents
    - conversion en minuscules
    - suppression des caractères non alphabétiques
    - découpage en mots
    """
    try:
        # Essai avec UTF-8
        with open(nom_fichier, "r", encoding="utf-8") as f:
            texte = f.read()
    except UnicodeDecodeError:
        # Si erreur, fallback en Latin-1
        with open(nom_fichier, "r", encoding="latin-1") as f:
            texte = f.read()

    # Nettoyage du texte
    texte = unidecode.unidecode(texte)          # supprime les accents
    texte = texte.lower()                       # met en minuscules
    texte = re.sub(r'[^a-z\s]', ' ', texte)     # garde seulement lettres et espaces

    # Découpage en mots
    mots = texte.split()
    return mots

test_1 = r"C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/texte_1.txt"
test_2 = r"C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/texte_2.txt"
test_3 = r"C:/Users/GLC/Desktop/3IL/Semestre7/I2_2026/sem7/Modelisation Sys/TP/TP/texte_3.txt"

fichiers = [test_1, test_2, test_3]

test_11 = lire_fichier(fichiers[0])
test_11 = lire_fichier(fichiers[1])
test_11 = lire_fichier(fichiers[2])


def proba_texte(nom_fichier):
    mots = lire_fichier(nom_fichier)
    # on stocke des log-probabilités
    log_scores = {lang: 0.0 for lang in langues}

    for mot in mots:
        O = mot_en_indices(mot)
        if not O:
            continue
        for lang in langues:
            # utiliser une version stable du forward (avec scaling ou log)
            logP, _ = forward(O, A_dict[lang],B, PI)
            log_scores[lang] += logP

    # conversion en probabilités avec softmax
    m = max(log_scores.values())  # stabilité numérique
    probs = {lang: np.exp(log_scores[lang] - m) for lang in langues}
    Z = sum(probs.values())
    probs = {lang: probs[lang] / Z for lang in langues}
    return probs

# matrice confusion (3x3) normalisée (0–1)
confusion = np.zeros((3, 3), dtype=float)

for idx, f in enumerate(fichiers):
    # calcul des probabilités pour chaque texte
    probs = proba_texte(f)

    # conversion en tableau numpy
    probs_array = np.array([probs[lang] for lang in langues])

    # normalisation (somme = 1)
    total = np.sum(probs_array)
    if total > 0:
        probs_array = probs_array / total

    # indices réel et prédiction
    pred_idx = np.argmax(probs_array)
    real_idx = langues.index(langues_reelles[idx])

    # ajout des valeurs dans la matrice
    confusion[real_idx, :] += probs_array

    # affichage détaillé
    print(f"Texte_{idx+1} : Réel: {langues_reelles[idx]}, Prédit: {langues[pred_idx]}")
    for l, p in zip(langues, probs_array):
        print(f"  {l}: {p:.2f}")

# Moyenne si plusieurs textes par langue
for i in range(len(langues)):
    nb = langues_reelles.count(langues[i])
    if nb > 0:
        confusion[i, :] /= nb

# affichage final
df_conf = pd.DataFrame(confusion, index=langues, columns=langues)
print("\nMatrice de confusion (valeurs normalisées 0–1) :")
print(df_conf.round(2))
