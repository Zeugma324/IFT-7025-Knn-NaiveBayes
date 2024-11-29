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
		self.train_data = np.array([])
		self.train_labels = np.array([])
		self.k = [1, 5]
	
	def predict(self, x, k):
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
		
		# Séparer les données catégorielles des données numériques
		# Calcul de la distance euclidienne
		unsorted_eucl_dist = np.array(
			[np.linalg.norm(x - data) for data in self.train_data]
			)
		# On efefctue un tri des distances pour pouvoir récupérer les k
		# plus proches voisins. On fait d'abord une copie du tableau des 
		# distances pour être capable de retrouver les indices des k plus
		# proches voisins après le tri afin de récupérer leur label
		sorted_dist = np.copy(unsorted_eucl_dist)
		sorted_dist.sort()
		# On ne garde que les labels des k plus proches voisins
		k_nearest_labels = [
			self.train_labels[np.where(unsorted_eucl_dist == dist)[0][0]]
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
					)# Pour les données d'entraîneùent: séparer les données
				# catégorielles des données numériques et les sauver
				# dans les attributs de l'objet
				self.train_data = cv_train_data
				self.train_labels = cv_train_labels
				# Récupérer les données de validation
				val_data = folded_data[fold]
				val_labels = folded_labels[fold]
				# On prédit les classes pour les données de validation
				# et on calcule l'accuracy
				predicted_labels = [
					self.predict(x, k)
					for x in val_data
					]
				accuracy = np.sum(predicted_labels == val_labels)/len(val_labels)
				accuracies.append(accuracy)
			# On calcule l'erreur de validation pour le nombre de voisins k
			val_error = 1 - np.mean(accuracies)
			val_errors.append(val_error)
		# On choisit le nombre de voisins qui minimise l'erreur de validation
		self.k = np.argmin(val_errors) + 1
		# On remet à jour les données d'entraînement avec toutes les données
		# d'entraînement
		# Pour les données d'entraîneùent: séparer les données
		# catégorielles des données numériques et les sauver
		# dans les attributs de l'objet
		self.train_data = train
		self.train_labels = train_labels

		# Afficher les erreurs de validation pour chaque nombre de voisins
		# et le nombre de voisins choisi
		print("Erreurs de validation pour chaque nombre de voisins:")
		for i in range(len(val_errors)):
			print(f"Pour {i + 1} voisins: {val_errors[i]}")
		print(f"Nombre de voisins choisi: {self.k}")
        
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
		predicted_labels = [
			self.predict(x, self.k)
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
		# Convertir le tableau y en float pour éviter les problèmes
		# d'indices
		y = y.astype(float).astype(int)
		predicted_labels = np.array(predicted_labels).astype(float).astype(int)
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