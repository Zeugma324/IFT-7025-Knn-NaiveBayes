1. Description des Classes

1.1 KNN
La classe KNN comprend un cinstructeur "__init__" qui initialise
l'ensemble des données d'entraînement du classificateur à un array
numpy vide. De même pour l'ensemble des labels d'entraînement. Il
initialise également le k de l'objet à une liste de 2 valeurs
qui représentent les valeurs minimale et maximale de l'ensemble
des valeurs à tester pour k lors de la validation croisée.

On retrouve ensuite la méthode "predict" qui vise à prédire la classe
de l'instance x passée en paramètre selon la méthode décrite dans le
document pdf. Nous avons inversé l'ordre des méthodes "train et
"predict" pour le KNN car comme le modèle ne nécessite pas de phase
d'entraînement, la méthode "train" effectue la validation croisée et
a donc besoin d'utiliser la méthode "predict".

Pour finir, on retrouve la méthode "evaluate" qui permet de calculer
les métriques d'évaluation sur un ensemble de données et de labels
donnés.

1.2 Naive Bayes

La classe BayesNaif comprend un constructeur "__init__" qui initialise les 
éléments nécessaires pour le calcul des probabilités. Elle initialise
les classes du dataset à None, ainsi que trois dictionnaires : 
prior pour les probabilités a priori des classes, mean pour les moyennes 
des attributs par classe, et var pour leurs variances.

On a ensuite la méthode "train", qui est utilisée pour 
entraîner le modèle. Elle calcule les probabilités a priori, les 
moyennes et les variances pour chaque classe en se basant sur les données 
d'entraînement et leurs labels.

La méthode "predict" permet de prédire la classe d'une instance donnée. 
Elle utilise les informations calculées lors de l'entraînement pour 
déterminer la probabilité a posteriori de chaque classe. La classe 
ayant la probabilité maximale est retournée comme résultat.

Enfin, la méthode "evaluate" calcule les métriques d'évaluation 
(accuracy, précision, rappel, F1-score) et génère une matrice de 
confusion pour un ensemble de données et de labels donnés.


2. Répartition des Tâches

Nous nous sommes répartis par modèle. Grégoire a réalisé le KNN
et, Noam et Zineb se sont chargé du Naive Bayes Classifier et
des définitions des métriques.
Nous n'avons pas rencontré de difficulté particulière pour ce
travail.

3. Comparaison

Dataset	    Algorithme    Accuracy	 Precision	 Recall	   F1-score
Iris	   Naïve Bayes	   0.9556	   0.9556	 0.9608	    0.9554
           KNN (k=3)	   0.9556	   0.9608	 0.9556	    0.9554
Wine	   Naïve Bayes	   0.7990	   0.7933	 0.8057	    0.7951
           KNN (k=1)	   0.7879	   0.7813	 0.7779	    0.7794
Ormeaux	   Naïve Bayes	   0.5582	   0.4968	 0.6431	    0.5063
           KNN (k=5)	   0.8174	   0.5941	 0.7159	    0.6307

Dataset Iris : Les deux algorithmes affichent des performances similaires 
avec une accuracy et un F1-score élevés, ce qui traduit de très bonne performances
à classifier dans ce dataset.

Dataset Wine : Naïve Bayes est légèrement meilleur en termes d’accuracy, 
de précision, et de rappel, ce qui peut se traduire par une par une meilleur
capacité à gérer des distributions probabilistes.

Dataset Ormeaux : KNN surpasse largement Naïve Bayes. Cela montre que le KNN, bien paramétré, 
est plus efficace sur des datasets complexes ou non linéaires.