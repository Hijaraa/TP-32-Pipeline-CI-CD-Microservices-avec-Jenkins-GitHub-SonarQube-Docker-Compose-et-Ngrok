# -*- coding: utf-8 -*-
"""
Module de comparaison d'algorithmes ML pour l'analyse de logs
SafeOps-Logminer - Comparaison des modèles d'analyse de logs
Inclut Log Sergeon SLM et autres modèles d'analyse de logs
"""

import sys
import io

# Fix encoding pour Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Pour les modèles de deep learning (optionnel)
try:
    from sklearn.neural_network import MLPClassifier
    HAS_MLP = True
except:
    HAS_MLP = False


class LogAnalysisComparator:
    """
    Classe pour comparer différents algorithmes d'analyse de logs
    Spécialement conçue pour SafeOps-Logminer
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le comparateur pour l'analyse de logs
        
        Args:
            random_state: Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def initialize_algorithms(self):
        """Initialise tous les algorithmes pour l'analyse de logs"""
        self.models = {
            # Modèles classiques pour l'analyse de logs
            'Log Sergeon SLM (Simulated)': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest (Log Analysis)': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'SVM (Log Classification)': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Logistic Regression (Log Anomaly Detection)': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Naive Bayes (Text-based Logs)': MultinomialNB(
                alpha=1.0
            ),
            'Isolation Forest (Anomaly Detection)': IsolationForest(
                contamination=0.1,
                random_state=self.random_state
            )
        }
        
        # Ajouter MLP si disponible
        if HAS_MLP:
            self.models['MLP (Neural Network)'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            )
    
    def prepare_log_data(self, log_texts: List[str], labels: List[str], 
                        use_tfidf: bool = True, max_features: int = 5000):
        """
        Prépare les données de logs pour l'entraînement
        
        Args:
            log_texts: Liste des messages de logs (texte)
            labels: Labels correspondants (anomaly/normal, error/info, etc.)
            use_tfidf: Utiliser TF-IDF (True) ou Count Vectorizer (False)
            max_features: Nombre maximum de features
        """
        # Vectorisation des logs
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
        
        # Vectoriser les logs
        X = self.vectorizer.fit_transform(log_texts).toarray()
        
        # Encoder les labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Normalisation (sauf pour Naive Bayes qui préfère les comptes)
        # On garde les données brutes pour Naive Bayes
        
    def calculate_rpu_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calcule les métriques RPU pour l'analyse de logs
        
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
        
        # Calcul de l'Uplift
        baseline_accuracy = max(np.bincount(y_true)) / len(y_true)
        uplift = (accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
        
        # AUC-ROC si probabilités disponibles
        auc_roc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            except:
                pass
        
        # Métriques spécifiques pour l'analyse de logs
        # True Positive Rate (détection d'anomalies)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            true_positive_rate = None
            false_positive_rate = None
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Uplift': uplift,
            'AUC-ROC': auc_roc,
            'True Positive Rate': true_positive_rate,
            'False Positive Rate': false_positive_rate,
            'Baseline Accuracy': baseline_accuracy
        }
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Génère la matrice de confusion"""
        return confusion_matrix(y_true, y_pred)
    
    def train_and_evaluate_all(self) -> Dict[str, Any]:
        """Entraîne et évalue tous les algorithmes"""
        if not self.models:
            self.initialize_algorithms()
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Entraînement de {name}...")
            print(f"{'='*60}")
            
            try:
                # Isolation Forest nécessite un traitement spécial
                if 'Isolation Forest' in name:
                    # Isolation Forest retourne -1 pour anomalies, 1 pour normal
                    model.fit(self.X_train)
                    y_pred = model.predict(self.X_test)
                    # Convertir -1 -> 1 (anomaly), 1 -> 0 (normal) pour correspondre aux labels
                    y_pred = np.where(y_pred == -1, 1, 0)
                    y_proba = None
                else:
                    # Entraînement standard
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test)
                
                # Matrice de confusion
                cm = self.generate_confusion_matrix(self.y_test, y_pred)
                
                # Métriques RPU
                metrics = self.calculate_rpu_metrics(self.y_test, y_pred, y_proba)
                
                # Cross-validation score
                if 'Isolation Forest' not in name:
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train, 
                        cv=5, scoring='accuracy'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = None
                    cv_std = None
                
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
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'classification_report': class_report
                }
                
                print(f"[OK] {name} - Accuracy: {metrics['Accuracy']:.4f}, "
                      f"Precision: {metrics['Precision']:.4f}, "
                      f"Recall: {metrics['Recall']:.4f}, "
                      f"Uplift: {metrics['Uplift']:.4%}")
                
            except Exception as e:
                print(f"[ERREUR] Erreur lors de l'entrainement de {name}: {str(e)}")
                self.results[name] = {'error': str(e)}
        
        return self.results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Génère un résumé comparatif"""
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
                    'TPR': metrics['True Positive Rate'] if metrics['True Positive Rate'] else 'N/A',
                    'CV Mean': result['cv_mean'] if result['cv_mean'] else 'N/A',
                    'CV Std': result['cv_std'] if result['cv_std'] else 'N/A'
                })
        
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values('Accuracy', ascending=False)
        return df
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 10), 
                                save_path: str = None):
        """Visualise les matrices de confusion"""
        n_models = len([r for r in self.results.values() if 'error' not in r])
        if n_models == 0:
            print("Aucun résultat à visualiser")
            return
            
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
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
    
    def export_results(self, filepath: str):
        """Exporte les résultats au format JSON"""
        export_data = {}
        
        for name, result in self.results.items():
            if 'error' not in result:
                export_data[name] = {
                    'metrics': {k: (v if v is not None else 'N/A') for k, v in result['metrics'].items()},
                    'confusion_matrix': result['confusion_matrix'],
                    'cv_mean': result['cv_mean'] if result['cv_mean'] else 'N/A',
                    'cv_std': result['cv_std'] if result['cv_std'] else 'N/A'
                }
            else:
                export_data[name] = {'error': result['error']}
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Résultats exportés dans {filepath}")
    
    def get_best_algorithm(self) -> Tuple[str, Dict]:
        """Retourne le meilleur algorithme"""
        best_name = None
        best_accuracy = -1
        
        for name, result in self.results.items():
            if 'error' not in result:
                accuracy = result['metrics']['Accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_name = name
        
        return best_name, self.results[best_name] if best_name else None


def generate_sample_log_data(n_samples: int = 1000):
    """
    Génère des données de logs d'exemple pour la démonstration
    Simule des logs d'application avec anomalies
    """
    np.random.seed(42)
    
    # Patterns de logs normaux
    normal_patterns = [
        "INFO: Request processed successfully",
        "DEBUG: Connection established",
        "INFO: User login successful",
        "DEBUG: Cache hit",
        "INFO: Database query completed",
        "DEBUG: Session created",
        "INFO: API call successful",
        "DEBUG: Response sent",
    ]
    
    # Patterns de logs d'erreur/anomalie
    anomaly_patterns = [
        "ERROR: Connection timeout",
        "ERROR: Database connection failed",
        "CRITICAL: Memory leak detected",
        "ERROR: Authentication failed",
        "WARNING: High CPU usage",
        "ERROR: Invalid request format",
        "CRITICAL: Service unavailable",
        "ERROR: Permission denied",
    ]
    
    log_texts = []
    labels = []
    
    # Générer des logs normaux (70%)
    n_normal = int(n_samples * 0.7)
    for _ in range(n_normal):
        pattern = np.random.choice(normal_patterns)
        # Ajouter de la variabilité
        log_text = f"{pattern} - ID: {np.random.randint(1000, 9999)}"
        log_texts.append(log_text)
        labels.append('normal')
    
    # Générer des logs d'anomalie (30%)
    n_anomaly = n_samples - n_normal
    for _ in range(n_anomaly):
        pattern = np.random.choice(anomaly_patterns)
        log_text = f"{pattern} - ID: {np.random.randint(1000, 9999)}"
        log_texts.append(log_text)
        labels.append('anomaly')
    
    # Mélanger
    indices = np.random.permutation(len(log_texts))
    log_texts = [log_texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return log_texts, labels


if __name__ == "__main__":
    """
    Exemple d'utilisation pour SafeOps-Logminer
    """
    print("="*60)
    print("COMPARAISON D'ALGORITHMES ML POUR L'ANALYSE DE LOGS")
    print("SafeOps-Logminer - Log Sergeon SLM et autres modèles")
    print("="*60)
    
    # Générer des données de logs d'exemple
    print("\n1. Generation de donnees de logs d'exemple...")
    log_texts, labels = generate_sample_log_data(n_samples=1000)
    print(f"   [OK] {len(log_texts)} logs generes")
    print(f"   [OK] Classes: {set(labels)}")
    
    # Initialisation du comparateur
    print("\n2. Initialisation du comparateur...")
    comparator = LogAnalysisComparator(random_state=42)
    comparator.initialize_algorithms()
    print(f"   [OK] {len(comparator.models)} algorithmes initialises")
    
    # Préparation des données
    print("\n3. Preparation des donnees de logs...")
    comparator.prepare_log_data(log_texts, labels, use_tfidf=True, max_features=5000)
    print(f"   [OK] Donnees d'entrainement: {comparator.X_train.shape[0]} echantillons")
    print(f"   [OK] Donnees de test: {comparator.X_test.shape[0]} echantillons")
    print(f"   [OK] Features: {comparator.X_train.shape[1]}")
    
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
    if best_result:
        print(f"   [OK] {best_name}")
        print(f"   [OK] Accuracy: {best_result['metrics']['Accuracy']:.4f}")
        print(f"   [OK] Precision: {best_result['metrics']['Precision']:.4f}")
        print(f"   [OK] Recall: {best_result['metrics']['Recall']:.4f}")
        print(f"   [OK] Uplift: {best_result['metrics']['Uplift']:.4%}")
    
    # Export des résultats
    print("\n7. Export des resultats...")
    comparator.export_results('log_analysis_results.json')
    
    # Visualisations
    print("\n8. Generation des visualisations...")
    comparator.plot_confusion_matrices(save_path='log_analysis_confusion_matrices.png')
    
    print("\n" + "="*60)
    print("COMPARAISON TERMINEE")
    print("="*60)

