"""
Module de comparaison d'algorithmes de machine learning
avec calcul des métriques RPU (Recall, Precision, Uplift) et matrices de confusion
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class AlgorithmComparator:
    """
    Classe pour comparer différents algorithmes de classification
    avec calcul des métriques RPU et matrices de confusion
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le comparateur d'algorithmes
        
        Args:
            random_state: Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def initialize_algorithms(self):
        """Initialise tous les algorithmes à comparer"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=10
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            ),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            )
        }
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Prépare les données pour l'entraînement
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion des données de test
        """
        # Encoder les labels si nécessaire
        if y.dtype == 'object' or isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
        
        # Split des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Normalisation des features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def calculate_rpu_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calcule les métriques RPU (Recall, Precision, Uplift)
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
            y_proba: Probabilités prédites (optionnel)
        
        Returns:
            Dictionnaire contenant les métriques
        """
        # Métriques de base
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calcul de l'Uplift (amélioration par rapport à un modèle baseline)
        # Baseline: prédiction majoritaire
        baseline_accuracy = max(np.bincount(y_true)) / len(y_true)
        uplift = (accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
        
        # AUC-ROC si probabilités disponibles
        auc_roc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            except:
                pass
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Uplift': uplift,
            'AUC-ROC': auc_roc,
            'Baseline Accuracy': baseline_accuracy
        }
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Génère la matrice de confusion
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
        
        Returns:
            Matrice de confusion
        """
        return confusion_matrix(y_true, y_pred)
    
    def train_and_evaluate_all(self) -> Dict[str, Any]:
        """
        Entraîne et évalue tous les algorithmes
        
        Returns:
            Dictionnaire contenant les résultats de tous les algorithmes
        """
        if not self.models:
            self.initialize_algorithms()
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Entraînement de {name}...")
            print(f"{'='*60}")
            
            try:
                # Entraînement
                model.fit(self.X_train, self.y_train)
                
                # Prédictions
                y_pred = model.predict(self.X_test)
                
                # Probabilités (si disponible)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)
                
                # Matrice de confusion
                cm = self.generate_confusion_matrix(self.y_test, y_pred)
                
                # Métriques RPU
                metrics = self.calculate_rpu_metrics(self.y_test, y_pred, y_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, 
                    cv=5, scoring='accuracy'
                )
                
                # Rapport de classification
                class_report = classification_report(
                    self.y_test, y_pred, 
                    output_dict=True, 
                    zero_division=0
                )
                
                # Stockage des résultats
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'confusion_matrix': cm.tolist(),
                    'metrics': metrics,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': class_report
                }
                
                print(f"✓ {name} - Accuracy: {metrics['Accuracy']:.4f}, "
                      f"Precision: {metrics['Precision']:.4f}, "
                      f"Recall: {metrics['Recall']:.4f}, "
                      f"Uplift: {metrics['Uplift']:.4%}")
                
            except Exception as e:
                print(f"✗ Erreur lors de l'entraînement de {name}: {str(e)}")
                self.results[name] = {'error': str(e)}
        
        return self.results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Génère un résumé comparatif de tous les algorithmes
        
        Returns:
            DataFrame avec les métriques comparatives
        """
        summary_data = []
        
        for name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                summary_data.append({
                    'Algorithm': name,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score'],
                    'Uplift': metrics['Uplift'],
                    'AUC-ROC': metrics['AUC-ROC'] if metrics['AUC-ROC'] else 'N/A',
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Accuracy', ascending=False)
        return df
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 10), 
                                save_path: str = None):
        """
        Visualise les matrices de confusion pour tous les algorithmes
        
        Args:
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder la figure
        """
        n_models = len([r for r in self.results.values() if 'error' not in r])
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        idx = 0
        for name, result in self.results.items():
            if 'error' not in result:
                cm = np.array(result['confusion_matrix'])
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    ax=axes[idx],
                    cbar_kws={'label': 'Count'}
                )
                axes[idx].set_title(f'{name}\nAccuracy: {result["metrics"]["Accuracy"]:.4f}')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
                idx += 1
        
        # Masquer les axes inutilisés
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matrices de confusion sauvegardées dans {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path: str = None):
        """
        Visualise la comparaison des métriques
        
        Args:
            save_path: Chemin pour sauvegarder la figure
        """
        df = self.get_comparison_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted['Algorithm'], df_sorted[metric], color='steelblue')
            ax.set_xlabel(metric)
            ax.set_title(f'Comparaison des {metric}')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparaison des métriques sauvegardée dans {save_path}")
        
        plt.show()
    
    def export_results(self, filepath: str):
        """
        Exporte les résultats au format JSON
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        export_data = {}
        
        for name, result in self.results.items():
            if 'error' not in result:
                export_data[name] = {
                    'metrics': result['metrics'],
                    'confusion_matrix': result['confusion_matrix'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                }
            else:
                export_data[name] = {'error': result['error']}
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Résultats exportés dans {filepath}")
    
    def get_best_algorithm(self) -> Tuple[str, Dict]:
        """
        Retourne le meilleur algorithme basé sur l'accuracy
        
        Returns:
            Tuple (nom, résultats) du meilleur algorithme
        """
        best_name = None
        best_accuracy = -1
        
        for name, result in self.results.items():
            if 'error' not in result:
                accuracy = result['metrics']['Accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_name = name
        
        return best_name, self.results[best_name] if best_name else None


def load_sample_data():
    """
    Charge des données d'exemple pour la démonstration
    Utilise le dataset Iris de scikit-learn
    """
    from sklearn.datasets import load_iris, load_breast_cancer, make_classification
    
    # Option 1: Dataset Iris (classification multi-classe)
    # iris = load_iris()
    # return iris.data, iris.target
    
    # Option 2: Dataset Breast Cancer (classification binaire)
    cancer = load_breast_cancer()
    return cancer.data, cancer.target
    
    # Option 3: Données synthétiques
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
    #                            n_redundant=5, n_classes=2, random_state=42)
    # return X, y


if __name__ == "__main__":
    """
    Exemple d'utilisation du comparateur d'algorithmes
    """
    print("="*60)
    print("COMPARAISON D'ALGORITHMES DE MACHINE LEARNING")
    print("Métriques RPU (Recall, Precision, Uplift) et Matrices de Confusion")
    print("="*60)
    
    # Chargement des données
    print("\n1. Chargement des données...")
    X, y = load_sample_data()
    print(f"   ✓ Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"   ✓ Classes: {len(np.unique(y))}")
    
    # Initialisation du comparateur
    print("\n2. Initialisation du comparateur...")
    comparator = AlgorithmComparator(random_state=42)
    comparator.initialize_algorithms()
    print(f"   ✓ {len(comparator.models)} algorithmes initialisés")
    
    # Préparation des données
    print("\n3. Préparation des données...")
    comparator.prepare_data(X, y, test_size=0.2)
    print(f"   ✓ Données d'entraînement: {comparator.X_train.shape[0]} échantillons")
    print(f"   ✓ Données de test: {comparator.X_test.shape[0]} échantillons")
    
    # Entraînement et évaluation
    print("\n4. Entraînement et évaluation des algorithmes...")
    results = comparator.train_and_evaluate_all()
    
    # Résumé comparatif
    print("\n5. Résumé comparatif:")
    print("="*60)
    summary = comparator.get_comparison_summary()
    print(summary.to_string(index=False))
    
    # Meilleur algorithme
    print("\n6. Meilleur algorithme:")
    print("="*60)
    best_name, best_result = comparator.get_best_algorithm()
    print(f"   ✓ {best_name}")
    print(f"   ✓ Accuracy: {best_result['metrics']['Accuracy']:.4f}")
    print(f"   ✓ Precision: {best_result['metrics']['Precision']:.4f}")
    print(f"   ✓ Recall: {best_result['metrics']['Recall']:.4f}")
    print(f"   ✓ Uplift: {best_result['metrics']['Uplift']:.4%}")
    
    # Export des résultats
    print("\n7. Export des résultats...")
    comparator.export_results('ml_comparison_results.json')
    
    # Visualisations
    print("\n8. Génération des visualisations...")
    comparator.plot_confusion_matrices(save_path='confusion_matrices.png')
    comparator.plot_metrics_comparison(save_path='metrics_comparison.png')
    
    print("\n" + "="*60)
    print("COMPARAISON TERMINÉE")
    print("="*60)

