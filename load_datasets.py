import numpy as np
import random

def is_float(string):
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

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    # Sort une liste des lignes du fichier, chaque ligne est un élément
    lines = f.readlines()
    random.shuffle(lines)
    data = []
    labels = []
    for line in lines:
        # Nettoie les lignes
        line = line.strip()
        if line == '':
            continue
        # Pour chaque ligne, crée une liste des caractéristique de l'iris et de son label
        parts = line.split(',')
        # Ajoute les caractéristiques de l'iris à data et son label à labels
        data.append([float(x) for x in parts[:-1]])
        labels.append(conversion_labels[parts[-1]])
    data = np.array(data)
    labels = np.array(labels)

    # Sépare les données en un set d'entraînement et un set de test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
	
	
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset
    lines = f.readlines()
    random.shuffle(lines)
    data = []
    labels = []
    for line in lines:
        # Nettoie les lignes
        line = line.strip()
        if line == '':
            continue
        # Pour chaque ligne, crée une liste des caractéristique de l'iris et de son label
        parts = line.split(',')
        # Ajoute les caractéristiques de l'iris à data et son label à labels
        data.append([float(x) for x in parts[:-1]])
        labels.append(parts[-1])
    data = np.array(data)
    labels = np.array(labels)
    # Sépare les données en un set d'entraînement et un set de test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]
	
	# La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.

    lines = f.readlines()
    random.shuffle(lines)
    data = []
    labels = []
    for line in lines:
        # Nettoie les lignes
        line = line.strip()
        if line == '':
            continue
        # Pour chaque ligne, crée une liste des caractéristique de l'iris et de son label
        parts = line.split(',')
        # Ajoute les caractéristiques de l'iris à data et son label à labels
        if parts[0] == 'M':
            data.append(
                 [1, 0, 0] + [float(x) if is_float(x) else x
                              for x in parts[1:-1]]
              )
        elif parts[0] == 'F':
            data.append(
                 [0, 1, 0] + [float(x) if is_float(x) else x
                              for x in parts[1:-1]]
              )
        else:
            data.append(
                 [0, 0, 1] + [float(x) if is_float(x) else x
                              for x in parts[1:-1]]
              )
        labels.append(parts[-1])
    data = np.array(data)
    labels = np.array(labels)
    # Sépare les données en un set d'entraînement et un set de test
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    train_labels = labels[:train_size]
    test = data[train_size:]
    test_labels = labels[train_size:]
    
    return (train, train_labels, test, test_labels)