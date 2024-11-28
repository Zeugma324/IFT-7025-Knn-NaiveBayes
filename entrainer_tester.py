import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
#import Knn # importer la classe du Knn
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
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)



# Initialisez/instanciez vos classifieurs avec leurs paramètres

classifier = NaiveBayes.BayesNaif()
classifier.train(train, train_labels)



# Charger/lire les datasets




# Entrainez votre classifieur


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

predictions = np.array([classifier.predict(x) for x in train])
classes = np.unique(train_labels)
train_results = classifier.evaluate(train, train_labels)

print("Résultats :")
print(f"""accuracy : {train_results[0]}""")
print(f"""precision : {train_results[1]}""")
print(f"""recall : {train_results[2]}""")
print(f"""F1_score : {train_results[3]}""")
print(f"confusion_matrix : \n{train_results[4]}")

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
print("Résultats :")
print(f"""accuracy : {test_results[0]}""")
print(f"""precision : {test_results[1]}""")
print(f"""recall : {test_results[2]}""")
print(f"""F1_score : {test_results[3]}""")
print(f"confusion_matrix : \n{test_results[4]}")

#Comparaison

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


train_ratio = 0.7
X_train, y_train, X_test, y_test = load_datasets.load_iris_dataset(train_ratio)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("")
print("=== Métriques pour le modèle Naive Bayes de sklearn ===")
print(f"Accuracy : {accuracy}")
print(f"Macro-Précision : {precision}")
print(f"Macro-Rappel : {recall}")
print(f"Macro-F1-score : {f1}")
print("Matrice de confusion :")
print(conf_matrix)