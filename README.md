🌍🔤 Détection automatique de la langue avec les Modèles de Markov Cachés (HMM)
📌 Présentation du projet
Ce projet met en œuvre un système de reconnaissance automatique de la langue basé sur les Modèles de Markov Cachés (Hidden Markov Models – HMM).
🎯 L’objectif : identifier la langue d’un mot ou d’un texte en exploitant les régularités statistiques des séquences de lettres.

Ce dépôt a été conçu pour mettre en valeur des compétences en modélisation probabiliste, algorithmique et Python, dans un contexte proche des problématiques réelles du Traitement Automatique du Langage (TAL/NLP).

🎯 Objectifs techniques
🧠 Implémenter un modèle probabiliste HMM from scratch

🔡 Analyser des séquences de caractères pour la classification linguistique

⚖️ Comparer différentes stratégies de modélisation et mesurer leurs performances

📝 Produire une analyse critique des résultats obtenus

🧩 Compétences mises en avant
📊 Modélisation statistique (HMM)

🔁 Algorithmes probabilistes : Forward / Backward

🧬 Analyse de séquences

🧮 Calcul matriciel & algèbre linéaire

📉 Évaluation de modèles (matrices de confusion)

🐍 Python scientifique

🛠️ Technologies utilisées
🐍 Python

🔢 NumPy – calcul matriciel

🗂️ Pandas – manipulation de données

📈 Matplotlib – visualisation

⚙️ SciPy – outils numériques

🧪 Démarche et méthodologie
1️⃣ Prétraitement des données
🧹 Nettoyage des corpus textuels

🔤 Normalisation (minuscules, suppression des accents, caractères spéciaux)

🔁 Conversion des mots en séquences de lettres (a–z)

2️⃣ Construction du modèle HMM
Un modèle HMM est construit pour chaque langue :

🔀 Matrice de transition : probabilité de passage entre lettres

🎯 Matrice d’émission : probabilité d’émission des symboles

🚀 Vecteur de probabilité initiale

Chaque langue est représentée par un modèle statistique distinct.

3️⃣ Inférence probabiliste
⚙️ Implémentation des algorithmes Forward et Backward

📊 Calcul de la probabilité qu’un mot/texte appartienne à une langue

🏆 Sélection de la langue la plus probable

4️⃣ Évaluation et analyse
🧪 Classification mot par mot et texte par texte

🧩 Construction de matrices de confusion

🔍 Analyse de l’impact :

de la longueur des mots

de la structure interne des séquences

de la matrice d’émission

⭐ Résultats clés
📏 Les mots longs sont beaucoup mieux classés

❓ Les mots courts sont plus ambigus

🎯 La matrice d’émission influence fortement les performances

⚠️ Une matrice d’émission identité → forte baisse de précision

💼 Valeur pour un recruteur
Ce projet démontre :

🧠 Une capacité à implémenter des modèles mathématiques complexes

📚 Une maîtrise solide des fondements probabilistes

🧪 Une approche rigoureuse de l’évaluation de modèles

🧐 Une aptitude à analyser et expliquer les limites d’un système

🚀 Des compétences transférables vers le Machine Learning, le NLP et la Data Science

🚀 Pistes d’amélioration
📚 Enrichissement des corpus d’apprentissage

🌍 Ajout de nouvelles langues

⚙️ Optimisation des paramètres du modèle

🤖 Introduction d’algorithmes d’apprentissage (Baum-Welch)

✍️ Auteur
TSAGUA YEMEWA Beyoncé