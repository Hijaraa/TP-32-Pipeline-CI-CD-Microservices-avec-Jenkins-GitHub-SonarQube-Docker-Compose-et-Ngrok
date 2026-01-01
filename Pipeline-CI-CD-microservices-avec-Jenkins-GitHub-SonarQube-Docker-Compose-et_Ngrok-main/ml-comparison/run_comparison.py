#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour exécuter la comparaison d'algorithmes
"""

import sys
import os
import argparse
from algorithm_comparison import AlgorithmComparator, load_sample_data

# Fix encoding pour Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import matplotlib
    matplotlib.use('Agg')  # Pour éviter les problèmes d'affichage en mode headless
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Comparaison d\'algorithmes ML avec métriques RPU et matrices de confusion'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        help='Chemin vers un fichier CSV de données (optionnel)'
    )
    parser.add_argument(
        '--target', 
        type=str, 
        help='Nom de la colonne cible (si fichier CSV fourni)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Préfixe pour les fichiers de sortie (défaut: results)'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion des données de test (défaut: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("COMPARAISON D'ALGORITHMES DE MACHINE LEARNING")
    print("Métriques RPU (Recall, Precision, Uplift) et Matrices de Confusion")
    print("="*60)
    
    # Chargement des données
    print("\n1. Chargement des données...")
    if args.data:
        try:
            import pandas as pd
            df = pd.read_csv(args.data)
            if args.target:
                X = df.drop(columns=[args.target]).values
                y = df[args.target].values
            else:
                # Dernière colonne par défaut
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            print(f"   [OK] Donnees chargees depuis {args.data}")
        except Exception as e:
            print(f"   [ERREUR] Erreur lors du chargement: {e}")
            print("   -> Utilisation de donnees d'exemple")
            X, y = load_sample_data()
    else:
        X, y = load_sample_data()
        print("   [OK] Donnees d'exemple chargees")
    
    print(f"   [OK] {X.shape[0]} echantillons, {X.shape[1]} features")
    print(f"   [OK] Classes: {len(set(y))}")
    
    # Initialisation du comparateur
    print("\n2. Initialisation du comparateur...")
    comparator = AlgorithmComparator(random_state=42)
    comparator.initialize_algorithms()
    print(f"   [OK] {len(comparator.models)} algorithmes initialises")
    
    # Préparation des données
    print("\n3. Preparation des donnees...")
    comparator.prepare_data(X, y, test_size=args.test_size)
    print(f"   [OK] Donnees d'entrainement: {comparator.X_train.shape[0]} echantillons")
    print(f"   [OK] Donnees de test: {comparator.X_test.shape[0]} echantillons")
    
    # Entraînement et évaluation
    print("\n4. Entraînement et évaluation des algorithmes...")
    results = comparator.train_and_evaluate_all()
    
    # Résumé comparatif
    print("\n5. Résumé comparatif:")
    print("="*60)
    summary = comparator.get_comparison_summary()
    print(summary.to_string(index=False))
    
    # Sauvegarder le résumé
    summary_file = f"{args.output}_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\n   [OK] Resume sauvegarde dans {summary_file}")
    
    # Meilleur algorithme
    print("\n6. Meilleur algorithme:")
    print("="*60)
    best_name, best_result = comparator.get_best_algorithm()
    if best_result:
        print(f"   [OK] {best_name}")
        print(f"   [OK] Accuracy: {best_result['metrics']['Accuracy']:.4f}")
        print(f"   [OK] Precision: {best_result['metrics']['Precision']:.4f}")
        print(f"   [OK] Recall: {best_result['metrics']['Recall']:.4f}")
        print(f"   [OK] F1-Score: {best_result['metrics']['F1-Score']:.4f}")
        print(f"   [OK] Uplift: {best_result['metrics']['Uplift']:.4%}")
        if best_result['metrics']['AUC-ROC']:
            print(f"   [OK] AUC-ROC: {best_result['metrics']['AUC-ROC']:.4f}")
    
    # Export des résultats
    print("\n7. Export des resultats...")
    results_file = f"{args.output}_results.json"
    comparator.export_results(results_file)
    
    # Visualisations
    print("\n8. Generation des visualisations...")
    try:
        comparator.plot_confusion_matrices(save_path=f"{args.output}_confusion_matrices.png")
        comparator.plot_metrics_comparison(save_path=f"{args.output}_metrics_comparison.png")
        print(f"   [OK] Visualisations sauvegardees")
    except Exception as e:
        print(f"   [ATTENTION] Erreur lors de la generation des visualisations: {e}")
    
    print("\n" + "="*60)
    print("COMPARAISON TERMINEE")
    print("="*60)
    print(f"\nFichiers generes:")
    print(f"  - {summary_file}")
    print(f"  - {results_file}")
    print(f"  - {args.output}_confusion_matrices.png")
    print(f"  - {args.output}_metrics_comparison.png")


if __name__ == "__main__":
    main()

