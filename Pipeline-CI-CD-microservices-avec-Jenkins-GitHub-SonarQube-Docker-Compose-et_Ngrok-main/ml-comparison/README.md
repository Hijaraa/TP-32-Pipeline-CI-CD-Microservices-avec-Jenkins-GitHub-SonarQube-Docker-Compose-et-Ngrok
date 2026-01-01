# Comparaison d'Algorithmes ML avec Métriques RPU

Ce module permet de comparer différents algorithmes de machine learning avec calcul détaillé des métriques **RPU (Recall, Precision, Uplift)** et génération de **matrices de confusion**.

## 📋 Fonctionnalités

- **7 Algorithmes de Classification** :
  - Random Forest
  - SVM (Support Vector Machine)
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Gradient Boosting

- **Métriques Calculées** :
  - **Recall** (Sensibilité)
  - **Precision** (Précision)
  - **Uplift** (Amélioration par rapport au baseline)
  - Accuracy
  - F1-Score
  - AUC-ROC (pour classification binaire)
  - Cross-validation scores

- **Visualisations** :
  - Matrices de confusion pour tous les algorithmes
  - Comparaison graphique des métriques
  - Export des résultats en JSON et CSV

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
cd ml-comparison
pip install -r requirements.txt
```

## 📖 Utilisation

### 1. Utilisation en ligne de commande

#### Avec données d'exemple (dataset Breast Cancer)

```bash
python run_comparison.py
```

#### Avec vos propres données CSV

```bash
python run_comparison.py --data votre_fichier.csv --target nom_colonne_cible --output mes_resultats
```

**Paramètres disponibles** :
- `--data` : Chemin vers un fichier CSV
- `--target` : Nom de la colonne cible (si non spécifié, dernière colonne utilisée)
- `--output` : Préfixe pour les fichiers de sortie (défaut: `results`)
- `--test-size` : Proportion des données de test (défaut: 0.2)

### 2. Utilisation en tant que module Python

```python
from algorithm_comparison import AlgorithmComparator, load_sample_data
import numpy as np

# Charger des données
X, y = load_sample_data()

# Initialiser le comparateur
comparator = AlgorithmComparator(random_state=42)
comparator.initialize_algorithms()

# Préparer les données
comparator.prepare_data(X, y, test_size=0.2)

# Entraîner et évaluer
results = comparator.train_and_evaluate_all()

# Obtenir le résumé comparatif
summary = comparator.get_comparison_summary()
print(summary)

# Obtenir le meilleur algorithme
best_name, best_result = comparator.get_best_algorithm()
print(f"Meilleur: {best_name}")

# Visualiser les matrices de confusion
comparator.plot_confusion_matrices(save_path='confusion_matrices.png')

# Visualiser la comparaison des métriques
comparator.plot_metrics_comparison(save_path='metrics_comparison.png')

# Exporter les résultats
comparator.export_results('results.json')
```

### 3. Utilisation via API Flask

Démarrer le serveur API :

```bash
python api.py
```

L'API sera accessible sur `http://localhost:5000`

#### Endpoints disponibles :

1. **POST /api/initialize**
   - Initialise le comparateur avec des données
   - Body (optionnel) : `{"X": [[...]], "y": [...]}`
   - Si non fourni, utilise des données d'exemple

2. **POST /api/train**
   - Entraîne et évalue tous les algorithmes
   - Retourne les résultats complets

3. **GET /api/comparison**
   - Retourne le résumé comparatif de tous les algorithmes

4. **GET /api/best**
   - Retourne le meilleur algorithme et ses métriques

5. **GET /api/confusion-matrix/<algorithm>**
   - Retourne la matrice de confusion d'un algorithme spécifique
   - Exemple : `/api/confusion-matrix/Random Forest`

6. **GET /api/rpu-metrics**
   - Retourne toutes les métriques RPU pour tous les algorithmes

#### Exemple d'utilisation de l'API :

```bash
# Initialiser
curl -X POST http://localhost:5000/api/initialize

# Entraîner
curl -X POST http://localhost:5000/api/train

# Obtenir la comparaison
curl http://localhost:5000/api/comparison

# Obtenir le meilleur algorithme
curl http://localhost:5000/api/best

# Obtenir les métriques RPU
curl http://localhost:5000/api/rpu-metrics
```

## 📊 Métriques RPU Expliquées

### Recall (Rappel / Sensibilité)
Mesure la capacité du modèle à identifier tous les cas positifs réels.

```
Recall = TP / (TP + FN)
```

### Precision (Précision)
Mesure la proportion de prédictions positives qui sont correctes.

```
Precision = TP / (TP + FP)
```

### Uplift (Amélioration)
Mesure l'amélioration de l'accuracy par rapport à un modèle baseline (prédiction majoritaire).

```
Uplift = (Accuracy - Baseline Accuracy) / Baseline Accuracy
```

### Matrice de Confusion
Tableau qui montre les prédictions correctes et incorrectes :

```
                Prédit
              Positif  Négatif
Réel Positif    TP      FN
     Négatif     FP      TN
```

Où :
- **TP** (True Positive) : Correctement prédit comme positif
- **TN** (True Negative) : Correctement prédit comme négatif
- **FP** (False Positive) : Incorrectement prédit comme positif
- **FN** (False Negative) : Incorrectement prédit comme négatif

## 📁 Structure des Fichiers Générés

Après exécution, les fichiers suivants sont générés :

- `results_summary.csv` : Résumé comparatif au format CSV
- `results_results.json` : Résultats détaillés au format JSON
- `results_confusion_matrices.png` : Visualisation des matrices de confusion
- `results_metrics_comparison.png` : Comparaison graphique des métriques

## 📈 Exemple de Résultats

### Résumé Comparatif

| Algorithm | Accuracy | Precision | Recall | F1-Score | Uplift | AUC-ROC |
|-----------|----------|-----------|--------|----------|--------|---------|
| Random Forest | 0.9649 | 0.9647 | 0.9649 | 0.9648 | 0.9298 | 0.9987 |
| Gradient Boosting | 0.9649 | 0.9647 | 0.9649 | 0.9648 | 0.9298 | 0.9987 |
| SVM | 0.9474 | 0.9471 | 0.9474 | 0.9473 | 0.8947 | 0.9965 |
| ... | ... | ... | ... | ... | ... | ... |

### Format JSON des Résultats

```json
{
  "Random Forest": {
    "metrics": {
      "Accuracy": 0.9649,
      "Precision": 0.9647,
      "Recall": 0.9649,
      "F1-Score": 0.9648,
      "Uplift": 0.9298,
      "AUC-ROC": 0.9987
    },
    "confusion_matrix": [[71, 2], [2, 39]],
    "cv_mean": 0.9649,
    "cv_std": 0.0123
  }
}
```

## 🔧 Personnalisation

### Ajouter un nouvel algorithme

Modifiez la méthode `initialize_algorithms()` dans `algorithm_comparison.py` :

```python
def initialize_algorithms(self):
    self.models = {
        # ... algorithmes existants ...
        'Nouvel Algorithme': VotreClassifier(
            param1=value1,
            param2=value2
        )
    }
```

### Modifier les métriques calculées

Modifiez la méthode `calculate_rpu_metrics()` pour ajouter d'autres métriques.

## 🐛 Dépannage

### Erreur : "No module named 'sklearn'"
```bash
pip install -r requirements.txt
```

### Erreur lors de la génération des visualisations
Assurez-vous que matplotlib est correctement installé et que vous avez les permissions d'écriture.

### Problème avec les données
- Vérifiez que votre fichier CSV est bien formaté
- Assurez-vous que la colonne cible existe
- Vérifiez qu'il n'y a pas de valeurs manquantes

## 📝 Notes

- Les algorithmes sont entraînés avec des paramètres par défaut optimisés
- Pour de meilleures performances, ajustez les hyperparamètres selon vos données
- Le calcul de l'Uplift utilise un baseline simple (prédiction majoritaire)
- L'AUC-ROC n'est calculé que pour les problèmes de classification binaire

## 🔗 Références

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matrice de Confusion](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Métriques de Classification](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

---

**Auteur** : Équipe de développement  
**Date** : 2024

