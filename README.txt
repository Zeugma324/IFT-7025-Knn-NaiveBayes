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

1.2

2. Répartition des Tâches

Nous nous sommes répartis par modèle. Grégoire a réalisé le KNN
et, Noam et Zineb se sont chargé du Naive Bayes Classifier et
des définitions des métriques.
Nous n'avons pas rencontré de difficulté particulière pour ce
travail.