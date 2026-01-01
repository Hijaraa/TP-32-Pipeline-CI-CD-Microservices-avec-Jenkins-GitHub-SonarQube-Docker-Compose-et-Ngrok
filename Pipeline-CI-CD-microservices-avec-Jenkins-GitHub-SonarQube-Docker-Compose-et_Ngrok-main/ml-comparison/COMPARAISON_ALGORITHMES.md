# Comparaison Détaillée des Algorithmes ML - Métriques RPU et Matrices de Confusion

## 📊 Vue d'Ensemble

Ce document présente une analyse détaillée de la comparaison de 7 algorithmes de classification avec calcul des métriques **RPU (Recall, Precision, Uplift)** et génération de **matrices de confusion**.

## 🎯 Algorithmes Comparés

### 1. Random Forest
- **Type** : Ensemble Learning (Bagging)
- **Principe** : Combine plusieurs arbres de décision
- **Avantages** : Résistant au surapprentissage, gère bien les données non linéaires
- **Inconvénients** : Peut être lent sur de grands datasets

### 2. Support Vector Machine (SVM)
- **Type** : Classificateur à marge maximale
- **Principe** : Trouve l'hyperplan optimal pour séparer les classes
- **Avantages** : Efficace en haute dimension, bon avec des données non linéaires (kernel trick)
- **Inconvénients** : Sensible à l'échelle des features, lent sur de grands datasets

### 3. Logistic Regression
- **Type** : Modèle linéaire probabiliste
- **Principe** : Utilise une fonction logistique pour modéliser la probabilité
- **Avantages** : Simple, interprétable, rapide
- **Inconvénients** : Assume une relation linéaire, sensible aux outliers

### 4. Naive Bayes
- **Type** : Classificateur probabiliste
- **Principe** : Utilise le théorème de Bayes avec l'hypothèse d'indépendance
- **Avantages** : Très rapide, bon pour les données textuelles
- **Inconvénients** : Hypothèse d'indépendance souvent violée

### 5. K-Nearest Neighbors (KNN)
- **Type** : Instance-based Learning
- **Principe** : Classe selon les k voisins les plus proches
- **Avantages** : Simple, non-paramétrique, adaptatif
- **Inconvénients** : Lent pour les prédictions, sensible à l'échelle

### 6. Decision Tree
- **Type** : Arbre de décision
- **Principe** : Divise récursivement l'espace des features
- **Avantages** : Interprétable, gère les données non linéaires
- **Inconvénients** : Sujet au surapprentissage, instable

### 7. Gradient Boosting
- **Type** : Ensemble Learning (Boosting)
- **Principe** : Combine séquentiellement des modèles faibles
- **Avantages** : Très performant, gère bien les relations complexes
- **Inconvénients** : Peut surajuster, plus lent à entraîner

## 📈 Métriques RPU Expliquées

### Recall (Rappel / Sensibilité)

**Définition** : Proportion de vrais positifs correctement identifiés parmi tous les vrais positifs.

```
Recall = TP / (TP + FN)
```

**Interprétation** :
- **Recall élevé** : Le modèle trouve la plupart des cas positifs
- **Recall faible** : Le modèle manque beaucoup de cas positifs (beaucoup de faux négatifs)

**Quand c'est important** :
- Détection de maladies (on ne veut pas manquer de cas)
- Détection de fraudes
- Sécurité (détection d'intrusions)

### Precision (Précision)

**Définition** : Proportion de prédictions positives qui sont correctes.

```
Precision = TP / (TP + FP)
```

**Interprétation** :
- **Precision élevée** : Quand le modèle prédit positif, c'est généralement correct
- **Precision faible** : Beaucoup de faux positifs

**Quand c'est important** :
- Filtrage de spam (on ne veut pas bloquer des emails légitimes)
- Recommandations (on veut recommander des items pertinents)
- Classification de documents

### Uplift (Amélioration)

**Définition** : Amélioration relative de l'accuracy par rapport à un modèle baseline (prédiction majoritaire).

```
Uplift = (Accuracy - Baseline Accuracy) / Baseline Accuracy
```

**Interprétation** :
- **Uplift > 0** : Le modèle est meilleur que le baseline
- **Uplift élevé** : Le modèle apporte une valeur significative
- **Uplift proche de 0** : Le modèle n'est guère meilleur qu'une prédiction aléatoire

**Exemple** :
- Baseline accuracy : 50%
- Model accuracy : 95%
- Uplift = (95% - 50%) / 50% = 90%

## 🔍 Matrice de Confusion

### Structure

La matrice de confusion est un tableau 2x2 (pour classification binaire) ou NxN (pour classification multi-classe) qui montre :

```
                Prédit
              Positif  Négatif
Réel Positif    TP      FN
     Négatif     FP      TN
```

### Composants

- **TP (True Positive)** : Correctement prédit comme positif
  - Exemple : Un patient malade correctement identifié comme malade

- **TN (True Negative)** : Correctement prédit comme négatif
  - Exemple : Un patient sain correctement identifié comme sain

- **FP (False Positive)** : Incorrectement prédit comme positif (Type I Error)
  - Exemple : Un patient sain incorrectement identifié comme malade

- **FN (False Negative)** : Incorrectement prédit comme négatif (Type II Error)
  - Exemple : Un patient malade incorrectement identifié comme sain

### Interprétation

**Matrice idéale** :
```
[[TP,  0 ]
 [0,  TN ]]
```
Tous les cas sont correctement classés.

**Matrice problématique** :
- **Beaucoup de FN** : Le modèle manque beaucoup de cas positifs (Recall faible)
- **Beaucoup de FP** : Le modèle fait beaucoup de fausses alertes (Precision faible)

## 📊 Exemple de Comparaison Détaillée

### Scénario : Classification Binaire (Maladie Oui/Non)

#### Résultats Hypothétiques

| Algorithme | Accuracy | Precision | Recall | F1-Score | Uplift | AUC-ROC |
|------------|----------|-----------|--------|----------|--------|---------|
| **Random Forest** | 0.965 | 0.965 | 0.965 | 0.965 | 0.930 | 0.999 |
| **Gradient Boosting** | 0.964 | 0.964 | 0.964 | 0.964 | 0.928 | 0.999 |
| **SVM** | 0.947 | 0.947 | 0.947 | 0.947 | 0.895 | 0.997 |
| **Logistic Regression** | 0.956 | 0.956 | 0.956 | 0.956 | 0.912 | 0.996 |
| **Naive Bayes** | 0.912 | 0.912 | 0.912 | 0.912 | 0.824 | 0.982 |
| **KNN** | 0.939 | 0.939 | 0.939 | 0.939 | 0.878 | 0.991 |
| **Decision Tree** | 0.930 | 0.930 | 0.930 | 0.930 | 0.860 | 0.987 |

### Analyse par Algorithme

#### 1. Random Forest (Meilleur)

**Matrice de Confusion** :
```
[[71,  2]   TP=71, FN=2
 [ 2, 39]]  FP=2,  TN=39
```

**Analyse** :
- ✅ **Recall élevé (0.965)** : Identifie 97.3% des cas positifs (71/73)
- ✅ **Precision élevée (0.965)** : 97.3% des prédictions positives sont correctes (71/73)
- ✅ **Uplift excellent (0.930)** : 93% d'amélioration par rapport au baseline
- ✅ **AUC-ROC excellent (0.999)** : Très bonne séparation des classes

**Recommandation** : **Algorithme recommandé** pour ce problème.

#### 2. Gradient Boosting

**Matrice de Confusion** :
```
[[71,  2]
 [ 2, 39]]
```

**Analyse** :
- Performance très similaire à Random Forest
- Légèrement moins performant mais très proche
- Bon compromis si Random Forest est trop lent

#### 3. SVM

**Matrice de Confusion** :
```
[[68,  5]
 [ 1, 40]]
```

**Analyse** :
- ⚠️ **Recall plus faible (0.947)** : Manque 5 cas positifs (68/73 = 93.2%)
- ✅ **Precision très élevée (0.947)** : Peu de faux positifs (1 seul)
- **Trade-off** : Préfère éviter les faux positifs au détriment de quelques faux négatifs

**Recommandation** : Utiliser si les faux positifs sont très coûteux.

#### 4. Logistic Regression

**Matrice de Confusion** :
```
[[70,  3]
 [ 2, 39]]
```

**Analyse** :
- Performance solide et équilibrée
- Avantage : **Très interprétable** (coefficients explicables)
- Rapide à entraîner

**Recommandation** : Bon choix si l'interprétabilité est importante.

#### 5. Naive Bayes

**Matrice de Confusion** :
```
[[65,  8]
 [ 2, 39]]
```

**Analyse** :
- ⚠️ **Recall plus faible (0.912)** : Manque 8 cas positifs (65/73 = 89%)
- Performance inférieure aux autres
- Avantage : **Très rapide**

**Recommandation** : Utiliser pour des données textuelles ou si la vitesse est critique.

#### 6. KNN

**Matrice de Confusion** :
```
[[67,  6]
 [ 1, 40]]
```

**Analyse** :
- Performance moyenne
- ⚠️ **Recall modéré (0.939)** : Manque 6 cas positifs
- ✅ **Precision élevée** : Peu de faux positifs

**Recommandation** : Utiliser si les données sont bien normalisées et le dataset n'est pas trop grand.

#### 7. Decision Tree

**Matrice de Confusion** :
```
[[66,  7]
 [ 1, 40]]
```

**Analyse** :
- ⚠️ **Recall le plus faible (0.930)** : Manque 7 cas positifs (66/73 = 90.4%)
- Avantage : **Très interprétable** (règles explicites)
- Inconvénient : Peut surajuster

**Recommandation** : Utiliser si l'interprétabilité est cruciale, mais préférer Random Forest.

## 🎯 Recommandations par Cas d'Usage

### Cas 1 : Détection de Maladie (Recall Critique)

**Priorité** : Recall > Precision

**Choix recommandés** :
1. **Random Forest** (Recall: 0.965)
2. **Gradient Boosting** (Recall: 0.964)
3. **Logistic Regression** (Recall: 0.956)

**Raison** : On ne veut pas manquer de cas positifs (malades).

### Cas 2 : Filtrage de Spam (Precision Critique)

**Priorité** : Precision > Recall

**Choix recommandés** :
1. **SVM** (Precision: 0.947, peu de FP)
2. **KNN** (Precision: 0.939)
3. **Decision Tree** (Precision: 0.930)

**Raison** : On ne veut pas bloquer des emails légitimes (faux positifs coûteux).

### Cas 3 : Équilibre Optimal

**Priorité** : F1-Score (équilibre Recall/Precision)

**Choix recommandés** :
1. **Random Forest** (F1: 0.965)
2. **Gradient Boosting** (F1: 0.964)
3. **Logistic Regression** (F1: 0.956)

### Cas 4 : Interprétabilité Requise

**Priorité** : Compréhensibilité du modèle

**Choix recommandés** :
1. **Logistic Regression** (coefficients explicables)
2. **Decision Tree** (règles explicites)
3. **Random Forest** (importance des features)

## 📝 Guide d'Interprétation des Matrices de Confusion

### Classification Binaire

**Matrice parfaite** :
```
[[100,   0]   Tous les positifs corrects
 [  0, 100]]  Tous les négatifs corrects
```

**Beaucoup de Faux Négatifs** :
```
[[ 50,  50]   Manque la moitié des positifs
 [  0, 100]]  Tous les négatifs corrects
```
→ **Problème** : Recall faible, le modèle manque des cas positifs

**Beaucoup de Faux Positifs** :
```
[[100,   0]   Tous les positifs corrects
 [ 50,  50]]  Beaucoup de négatifs mal classés
```
→ **Problème** : Precision faible, beaucoup de fausses alertes

### Classification Multi-Classe

Pour 3 classes (A, B, C) :

```
        Prédit
      A    B    C
A   [50,   2,   1]   Classe A bien identifiée
B   [ 1,  45,   2]   Classe B bien identifiée
C   [ 2,   1,  48]   Classe C bien identifiée
```

**Interprétation** :
- Diagonale principale : Prédictions correctes
- Hors diagonale : Erreurs de classification
- Exemple : 2 cas de classe A mal classés comme B

## 🔧 Utilisation du Code

### Exécution Rapide

```bash
cd ml-comparison
pip install -r requirements.txt
python run_comparison.py
```

### Avec Vos Données

```bash
python run_comparison.py --data votre_data.csv --target colonne_cible
```

### Via API

```bash
python api.py
# Puis utiliser les endpoints REST
```

## 📚 Références

- **Recall** : [Wikipedia - Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- **Precision** : [Wikipedia - Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- **Matrice de Confusion** : [scikit-learn - Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- **Uplift Modeling** : [Wikipedia - Uplift Modeling](https://en.wikipedia.org/wiki/Uplift_modelling)

---

## 📊 Résultats Réels Obtenus

**Note** : Les résultats ci-dessus sont des exemples. Les performances réelles ont été calculées sur le dataset Breast Cancer Wisconsin.

### 🎯 Résultats Réels (Dataset Breast Cancer - 569 échantillons)

| Algorithme | Accuracy | Precision | Recall | F1-Score | Uplift | AUC-ROC |
|------------|----------|-----------|--------|----------|--------|---------|
| **SVM** | **98.25%** | **98.25%** | **98.25%** | **98.25%** | **55.56%** | **99.50%** |
| **Logistic Regression** | **98.25%** | **98.25%** | **98.25%** | **98.25%** | **55.56%** | **99.54%** |
| Random Forest | 95.61% | 95.61% | 95.61% | 95.60% | 51.39% | 99.39% |
| K-Nearest Neighbors | 95.61% | 95.61% | 95.61% | 95.60% | 51.39% | 97.88% |
| Gradient Boosting | 95.61% | 95.69% | 95.61% | 95.58% | 51.39% | 99.07% |
| Naive Bayes | 92.98% | 92.98% | 92.98% | 92.98% | 47.22% | 98.68% |
| Decision Tree | 91.23% | 91.61% | 91.23% | 91.30% | 44.44% | 91.57% |

**Meilleur algorithme** : **SVM** et **Logistic Regression** (ex aequo à 98.25%)

**Matrice de confusion - SVM** :
```
                Prédit
              Bénin  Malin
Réel Bénin     41      1
     Malin      1     71
```
Seulement 2 erreurs sur 114 prédictions !

📄 **Voir le document `RESULTATS_REELS.md` pour l'analyse complète des résultats réels.**

