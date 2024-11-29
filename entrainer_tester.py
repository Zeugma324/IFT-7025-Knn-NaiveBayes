import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
#import Knn # importer la classe du Knn
from Knn import Knn
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres
train_ratio = 0.7
# Initialisez/instanciez vos classifieurs avec leurs paramètres
classifier = NaiveBayes.BayesNaif()
knn = Knn()
# Charger/lire les datasets
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
# Entrainez votre classifieur
classifier.train(train, train_labels)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
train_results = classifier.evaluate(train, train_labels)

print("################ DATASET IRIS ################")
print("Résultats d'Entraînement du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {train_results[0]}""")
print(f"""precision : {train_results[1]}""")
print(f"""recall : {train_results[2]}""")
print(f"""F1_score : {train_results[3]}""")
print(f"confusion_matrix : \n{train_results[4]}")

print("")
print("Résultats d'Entraînement du KNN:")
print("--------------------------------")
knn.train(train, train_labels)

# Tester votre classifieur
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

test_results = classifier.evaluate(test, test_labels)

print("")
print("################ DATASET IRIS ################")
print("Résultats de Test du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {test_results[0]}""")
print(f"""precision : {test_results[1]}""")
print(f"""recall : {test_results[2]}""")
print(f"""F1_score : {test_results[3]}""")
print(f"confusion_matrix : \n{test_results[4]}")

knn_test_results = knn.evaluate(test, test_labels)

print("")
print("Résultats de Test du KNN:")
print("--------------------------------")
print(f"confusion_matrix : \n{knn_test_results[1]}")
print(f"""accuracy : {knn_test_results[2]}""")
print(f"""precision : {knn_test_results[3]}""")
print(f"""recall : {knn_test_results[4]}""")
print(f"""F1_score : {knn_test_results[5]}""")

#Comparaison
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

gnb = GaussianNB()
gnb.fit(train, train_labels)
y_pred = gnb.predict(test)

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
f1 = f1_score(test_labels, y_pred, average='macro')
conf_matrix = confusion_matrix(test_labels, y_pred)

print("")
print("################ DATASET IRIS ################")
print("=== Métriques pour le modèle Naive Bayes de sklearn ===")
print(f"Accuracy : {accuracy}")
print(f"Macro-Précision : {precision}")
print(f"Macro-Rappel : {recall}")
print(f"Macro-F1-score : {f1}")
print("Matrice de confusion :")
print(conf_matrix)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(train, train_labels)
knn_y_pred = sklearn_knn.predict(test)

knn_accuracy = accuracy_score(test_labels, knn_y_pred)
knn_precision = precision_score(test_labels, knn_y_pred, average='macro')
knn_recall = recall_score(test_labels, knn_y_pred, average='macro')
knn_f1 = f1_score(test_labels, knn_y_pred, average='macro')
knn_conf_matrix = confusion_matrix(test_labels, knn_y_pred)

print("")
print("=== Métriques pour le modèle KNN de sklearn ===")
print(f"Accuracy : {knn_accuracy}")
print(f"Macro-Précision : {knn_precision}")
print(f"Macro-Rappel : {knn_recall}")
print(f"Macro-F1-score : {knn_f1}")
print("Matrice de confusion :")
print(knn_conf_matrix)

print("")
print("##################################### Test sur le Dataset Wine #####################################")
print("")
# Initialisez/instanciez vos classifieurs avec leurs paramètres
classifier = NaiveBayes.BayesNaif()
knn = Knn()
# Charger/lire les datasets
train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
# Entrainez votre classifieur
classifier.train(train, train_labels)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
train_results = classifier.evaluate(train, train_labels)

print("################ DATASET WINE ################")
print("Résultats d'Entraînement du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {train_results[0]}""")
print(f"""precision : {train_results[1]}""")
print(f"""recall : {train_results[2]}""")
print(f"""F1_score : {train_results[3]}""")
print(f"confusion_matrix : \n{train_results[4]}")

print("")
print("Résultats d'Entraînement du KNN:")
print("--------------------------------")
knn.train(train, train_labels)

# Tester votre classifieur
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

test_results = classifier.evaluate(test, test_labels)

print("")
print("################ DATASET WINE ################")
print("Résultats de Test du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {test_results[0]}""")
print(f"""precision : {test_results[1]}""")
print(f"""recall : {test_results[2]}""")
print(f"""F1_score : {test_results[3]}""")
print(f"confusion_matrix : \n{test_results[4]}")

knn_test_results = knn.evaluate(test, test_labels)

print("")
print("Résultats de Test du KNN:")
print("--------------------------------")
print(f"confusion_matrix : \n{knn_test_results[1]}")
print(f"""accuracy : {knn_test_results[2]}""")
print(f"""precision : {knn_test_results[3]}""")
print(f"""recall : {knn_test_results[4]}""")
print(f"""F1_score : {knn_test_results[5]}""")

#Comparaison
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

gnb = GaussianNB()
gnb.fit(train, train_labels)
y_pred = gnb.predict(test)

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
f1 = f1_score(test_labels, y_pred, average='macro')
conf_matrix = confusion_matrix(test_labels, y_pred)

print("")
print("################ DATASET WINE ################")
print("=== Métriques pour le modèle Naive Bayes de sklearn ===")
print(f"Accuracy : {accuracy}")
print(f"Macro-Précision : {precision}")
print(f"Macro-Rappel : {recall}")
print(f"Macro-F1-score : {f1}")
print("Matrice de confusion :")
print(conf_matrix)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(train, train_labels)
knn_y_pred = sklearn_knn.predict(test)

knn_accuracy = accuracy_score(test_labels, knn_y_pred)
knn_precision = precision_score(test_labels, knn_y_pred, average='macro')
knn_recall = recall_score(test_labels, knn_y_pred, average='macro')
knn_f1 = f1_score(test_labels, knn_y_pred, average='macro')
knn_conf_matrix = confusion_matrix(test_labels, knn_y_pred)

print("")
print("=== Métriques pour le modèle KNN de sklearn ===")
print(f"Accuracy : {knn_accuracy}")
print(f"Macro-Précision : {knn_precision}")
print(f"Macro-Rappel : {knn_recall}")
print(f"Macro-F1-score : {knn_f1}")
print("Matrice de confusion :")
print(knn_conf_matrix)

print("")
print("################################### Test sur le Dataset Ormeaux #####################################")
print("")
# Initialisez/instanciez vos classifieurs avec leurs paramètres
classifier = NaiveBayes.BayesNaif()
knn = Knn()
# Charger/lire les datasets
train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)
# Entrainez votre classifieur
classifier.train(train, train_labels)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
train_results = classifier.evaluate(train, train_labels)

print("################ DATASET Ormeaux ################")
print("Résultats d'Entraînement du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {train_results[0]}""")
print(f"""precision : {train_results[1]}""")
print(f"""recall : {train_results[2]}""")
print(f"""F1_score : {train_results[3]}""")
print(f"confusion_matrix : \n{train_results[4]}")

print("")
print("Résultats d'Entraînement du KNN:")
print("--------------------------------")
knn.train(train, train_labels)

# Tester votre classifieur
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

test_results = classifier.evaluate(test, test_labels)

print("")
print("################ DATASET Ormeaux ################")
print("Résultats de Test du Classificateur Naïf Bayésien:")
print("---------------------------------------------------------")
print(f"""accuracy : {test_results[0]}""")
print(f"""precision : {test_results[1]}""")
print(f"""recall : {test_results[2]}""")
print(f"""F1_score : {test_results[3]}""")
print(f"confusion_matrix : \n{test_results[4]}")

knn_test_results = knn.evaluate(test, test_labels)

print("")
print("Résultats de Test du KNN:")
print("--------------------------------")
print(f"confusion_matrix : \n{knn_test_results[1]}")
print(f"""accuracy : {knn_test_results[2]}""")
print(f"""precision : {knn_test_results[3]}""")
print(f"""recall : {knn_test_results[4]}""")
print(f"""F1_score : {knn_test_results[5]}""")

#Comparaison
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

gnb = GaussianNB()
gnb.fit(train, train_labels)
y_pred = gnb.predict(test)

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
f1 = f1_score(test_labels, y_pred, average='macro')
conf_matrix = confusion_matrix(test_labels, y_pred)

print("")
print("################ DATASET Ormeaux ################")
print("=== Métriques pour le modèle Naive Bayes de sklearn ===")
print(f"Accuracy : {accuracy}")
print(f"Macro-Précision : {precision}")
print(f"Macro-Rappel : {recall}")
print(f"Macro-F1-score : {f1}")
print("Matrice de confusion :")
print(conf_matrix)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(train, train_labels)
knn_y_pred = sklearn_knn.predict(test)

knn_accuracy = accuracy_score(test_labels, knn_y_pred)
knn_precision = precision_score(test_labels, knn_y_pred, average='macro')
knn_recall = recall_score(test_labels, knn_y_pred, average='macro')
knn_f1 = f1_score(test_labels, knn_y_pred, average='macro')
knn_conf_matrix = confusion_matrix(test_labels, knn_y_pred)

print("")
print("=== Métriques pour le modèle KNN de sklearn ===")
print(f"Accuracy : {knn_accuracy}")
print(f"Macro-Précision : {knn_precision}")
print(f"Macro-Rappel : {knn_recall}")
print(f"Macro-F1-score : {knn_f1}")
print("Matrice de confusion :")
print(knn_conf_matrix)