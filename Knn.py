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
import random


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.num_train_data = np.array([])
		self.cat_train_data = np.array([])
		self.train_labels = np.array([])
		self.k = [1, 10]
	
	def is_float(self, string):
		"""
		Permet de vérifier si une chaîne de caractères
		est un nombre à virgule

		Paramètres
		----------
		string : str: La chaîne de caractères à vérifier

		Retours
		-------
		bool : True si la chaîne est un nombre à virgule, False
		sinon

		"""
		try:
			float(string)
			return True
		except ValueError:
			return False
	
	def detect_cat(self, data):
		"""
		Prend un ensemble de données sous forme de matrice numpy en entrée
		et sépare l'ensemble des données catégorielles de l'ensemble des données
		numériques. Renvoie deux matrices numpy. S'il n'y a pas de variable
		catégorielle, la matrice cat_data sera une matrice vide.

		Paramètres
		----------
		data : np.array : L'ensemble de données à traiter

		Retours
		-------
		cat_data : np.array : La matrice des données catégorielles
		num_data : np.array : La matrice des données numériques

		"""
		# Vérifie si des variables catégorielles sont présentes
		cat_idx = []
		num_idx = []
		for i in range(len(data[0])):
			if self.is_float(data[0][i]):
				num_idx.append(i)
			else:
				cat_idx.append(i)
		if len(cat_idx) == 0:
			cat_data = np.array([])
			num_data = data
		else:
			cat_data = []
			for i in cat_idx:
				# Convertir les variables catégorielles en variables numériques
				## Retrieve the values of categorical col
				cat_col = [element[i] for element in data]
				unique_vals = list(set(cat_col))
				## Create a mapping from feature to id
				feature_to_id = {val: i for i, val in enumerate(unique_vals)}
				encoded_col = [feature_to_id[val] for val in cat_col]
				cat_data.append(encoded_col)
			# Transpose the cat_data matrix to get the same order as the original data
			cat_data = np.array(cat_data).T
			# Save the numerical data of the object
			num_data = []
			for i in num_idx:
				num_col = [element[i] for element in data]
				num_data.append(num_col)
			num_data = np.array(num_data).T
		
		return cat_data, num_data

	def predict(self, x, train, train_labels, k):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm

		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		"""
		# Passer sur ttes les instances
		# Calculer la diff avec ttes les autres instances
		# Trouver les k plus proches
		# Sélectionner la classe la plus présente parmi les K plus proches voisins
		
		# Pour les données d'entraîneùent: séparer les données
		# catégorielles des données numériques et les sauver
		# dans les attributs de l'objet
		self.cat_train_data, self.num_train_data = self.detect_cat(train)
		self.train_labels = train_labels
		
		# Séparer les données catégorielles des données numériques
		x = np.array([list(x)])
		cat_x, num_x = self.detect_cat(x)
		nb_cat_features = len(cat_x)
		nb_num_features = len(num_x)
		total_nb_features = len(cat_x) + len(num_x)
		# Calcul de la distance euclidienne
		unsorted_eucl_dist = np.array(
			[np.linalg.norm(num_x - data) for data in self.num_train_data]
			)
		# On calcule maintenant la distance pour les variables catégorielles
		# On ajoute 1 à la distance si les valeurs sont équivalentes
		# S'il n'y a pas de variable catégorielles, matching coefficient
		# sera un tableau de zéros
		if nb_cat_features == 0:
			matching_coef = np.zeros((len(self.num_train_data)))
		else:
			matching_coef = np.array(
				[
					[
						1 if cat_x[i] == data[i] else 0
						for i in range(nb_cat_features)
						]
					for data in self.cat_train_data
					]
				)
			matching_coef = np.sum(
				matching_coef, axis= 1, keepdims = True
				)/nb_cat_features
		# On fait la somme pondérée des distances numériques et catégorielles
		# pour obtenir une distance globale entre les instances
		weighted_eucl_dist = unsorted_eucl_dist *\
			(nb_num_features / total_nb_features)
		weighted_matching_coef = matching_coef *\
			(nb_cat_features / total_nb_features)
		unsorted_dist = np.add(weighted_eucl_dist, weighted_matching_coef)
		# On efefctue un tri des distances pour pouvoir récupérer les k
		# plus proches voisins. On fait d'abord une copie du tableau des 
		# distances pour être capable de retrouver les indices des k plus
		# proches voisins après le tri afin de récupérer leur label
		sorted_dist = np.copy(unsorted_dist)
		sorted_dist.sort()
		# On ne garde que les labels des k plus proches voisins
		k_nearest_labels = [
			self.train_labels[np.where(unsorted_dist == dist)[0][0]]
			for dist in sorted_dist[:k]
			]
		# Trouver la classe la plus présente parmi les k plus proches voisins
		# On utilise la méthode des votes majoritaires
		unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
		most_frequent_label = unique_labels[np.argmax(counts)]

		return most_frequent_label
	
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
        # Pour le KNN il n'y a pas de phase d'entraînement
		# le modèle garde juste les instances et les labels d'entrainement
		# pour les utiliser lors de la phase de prédiction

		# Utiliser la validation croisée pour déterminer le bon nombre
		# de voisins
		val_errors = []
		for k in range (self.k[0], self.k[1] + 1):
			# On divise les données en 10 parties
			nb_folds = 10
			fold_size = len(train)//nb_folds
			# On divise les données et les labels en 10 parties
			folded_data = [
				train[i*fold_size:(i+1)*fold_size]
				for i in range(nb_folds)
				]
			folded_labels = [
				train_labels[i*fold_size:(i+1)*fold_size]
				for i in range(nb_folds)
				]
			accuracies = []
			for fold in range(nb_folds):
				# On utilise 9 parties pour l'entrainement et 1 pour la validation
				cv_train_data = np.concatenate(
					folded_data[:fold] + folded_data[fold + 1:]
					)
				cv_train_labels = np.concatenate(
					folded_labels[:fold] + folded_labels[fold + 1:]
					)
				val_data = folded_data[fold]
				val_labels = folded_labels[fold]
				# On prédit les classes pour les données de validation
				# et on calcule l'accuracy
				predicted_labels = [
					self.predict(x, cv_train_data, cv_train_labels, k)
					for x in val_data
					]
				accuracy = np.sum(predicted_labels == val_labels)/len(val_labels)
				accuracies.append(accuracy)
			# On calcule l'erreur de validation pour le nombre de voisins k
			val_error = 1 - np.mean(accuracies)
			val_errors.append(val_error)
		# On choisit le nombre de voisins qui minimise l'erreur de validation
		self.k = np.argmin(val_errors) + 1

		# Afficher les erreurs de validation pour chaque nombre de voisins
		# et le nombre de voisins choisi
		print("Erreurs de validation pour chaque nombre de voisins:")
		for i in range(len(val_errors)):
			print(f"Pour {i + 1} voisins: {val_errors[i]}")
		print(f"Nombre de voisins choisi: {self.k}")

        
	def evaluate(self, X, y, train, train_labels):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		predicted_labels = [
			self.predict(x, train, train_labels, self.k)
			for x in X
			]
		# Calcul de la matrice de confusion
		nb_test_labels = len(set(y))
		nb_train_labels = len(set(self.train_labels))
		if nb_test_labels > nb_train_labels:
			nb_labels = nb_test_labels
		else:
			nb_labels = nb_train_labels
		confusion_matrix = np.zeros((nb_labels, nb_labels))
		for i in range(len(y)):
			confusion_matrix[y[i]][predicted_labels[i]] += 1
		# Calcul de l'accuracy
		accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
		# Calcul de la précision
		precision = np.zeros(nb_labels)
		for i in range(nb_labels):
			precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
		mean_precision = np.mean(precision)
		# Calcul du rappel
		recall = np.zeros(nb_labels)
		for i in range(nb_labels):
			recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:, i])
		mean_recall = np.mean(recall)
		# Calcul du F1-score
		f1_score = np.zeros(nb_labels)
		for i in range(nb_labels):
			f1_score[i] = 2*precision[i]*recall[i]/(precision[i] + recall[i])
		mean_f1_score = np.mean(f1_score)
		
		return (predicted_labels, confusion_matrix,
		  		accuracy, mean_precision, mean_recall,
				mean_f1_score)