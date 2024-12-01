"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import load_datasets as ld


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class BayesNaif: #nom de la class à changer

	def __init__(self):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.classes = None 
		self.prior= {}  #probabilité à priori
		self.mean = {}  #moyenne
		self.var = {}   #variance
    
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		self.classes=np.unique(train_labels)
		for classe in self.classes :
			#nous commençons par l'extraction des exemples de la classe
			data_classe = train[train_labels == classe]
			self.prior [classe]= len (data_classe) / len (train)
			self.mean [classe]= np.mean(data_classe,axis=0)
			self.var [classe]= np.var(data_classe,axis=0) 
			# ASTUCE : on peut ajouter + 1e-9 au calcul de la variance si jamais on rencontre des problemes, le com est à supprimer avant de soumettre le tp
			# #(Ajout d'un petit epsilon pour éviter la division par zéro au cas où on tombe sur ce cas afin d'être sur de l'éviter)



	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		posteriors= []
		for classe in self.classes:
			prior = self.prior [classe]
			#on initialise la prob conditionnelle 
			likelihood = 1
			for i in range(len(x)):
				mean=self.mean[classe][i]
				var=self.var[classe][i]
				# calcul de la probabilité gaussienne
				exponent = np.exp (-((x[i] - mean) ** 2) / (2 * var))
				prob = (1 / np.sqrt(2 * np.pi * var)) * exponent
				likelihood *= prob
			#calculer la probabilité postérieure
			posterior = prior * likelihood
			posteriors.append(posterior)
		return self.classes[np.argmax(posteriors)]
	
	def precision_and_recall_and_F1score(self, y_true, y_pred, class_label):

		#calcul des vrais positifs
		TP = np.sum((y_true == class_label) & (y_pred == class_label))

		#calcul des faux positifs
		FP = np.sum((y_true != class_label) & (y_pred == class_label))

		#calcul des faux negatifs
		FN = np.sum((y_true == class_label) & (y_pred != class_label))

		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		F1_score = 2 * precision * recall / (precision + recall)

		return precision, recall, F1_score
	
	def confusion_matrix(self, y_true, y_pred, classes):
		n = len(classes)

		#création d'une matrice n*n rempli de 0
		matrix = np.zeros((n,n), dtype = int)

		#parcourt simultanement chaque couple
		for true, pred in zip(y_true, y_pred):

			#trouve l'indice de la classe réelle
			true_i = np.where(classes == true)[0][0]

			#trouve l'indice de la classe prédit
			pred_i = np.where(classes == pred)[0][0]

			#ajoute 1 au coordonnée dans la matrice
			matrix[true_i, pred_i] += 1
		
		return matrix
	
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		exp_pred= np.array([self.predict(x) for x in X])
		#calcluer l'exactitude (accuracy)
		accuracy = np.mean (exp_pred ==y)

		#initialisation des listes de metriques par classes
		precision_list, recall_list, F1_score_list = [], [], []

		#pour chaque classe on calcul les metriques et les stockage dans leur liste
		for classe in self.classes:
			precision, recall, F1_score = self.precision_and_recall_and_F1score(y, exp_pred, classe)
			precision_list.append(precision)
			recall_list.append(recall)
			F1_score_list.append(F1_score)
		
		#on calcule la moyenne pour avoir un resultat unique
		mean_precision = np.mean(precision_list)
		mean_recall = np.mean(recall_list)
		mean_F1_score = np.mean(F1_score_list)
		confusion_matrix = self.confusion_matrix(y, exp_pred, self.classes)

		return accuracy, mean_precision, mean_recall, mean_F1_score, confusion_matrix