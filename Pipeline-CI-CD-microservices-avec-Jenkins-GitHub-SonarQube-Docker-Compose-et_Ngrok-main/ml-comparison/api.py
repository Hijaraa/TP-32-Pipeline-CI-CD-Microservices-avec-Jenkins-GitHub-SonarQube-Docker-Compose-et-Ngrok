"""
API Flask pour exposer les fonctionnalités de comparaison d'algorithmes
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from algorithm_comparison import AlgorithmComparator, load_sample_data
import json

app = Flask(__name__)
CORS(app)

# Instance globale du comparateur
comparator = None

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({'status': 'healthy', 'service': 'ML Algorithm Comparison'})

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialise le comparateur avec des données"""
    global comparator
    
    try:
        data = request.json
        
        # Si des données sont fournies, les utiliser
        if 'X' in data and 'y' in data:
            X = np.array(data['X'])
            y = np.array(data['y'])
        else:
            # Sinon, utiliser des données d'exemple
            X, y = load_sample_data()
        
        # Initialisation
        comparator = AlgorithmComparator(random_state=42)
        comparator.initialize_algorithms()
        comparator.prepare_data(X, y, test_size=0.2)
        
        return jsonify({
            'status': 'success',
            'message': 'Comparateur initialisé avec succès',
            'data_shape': {
                'samples': X.shape[0],
                'features': X.shape[1],
                'classes': len(np.unique(y))
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train():
    """Entraîne et évalue tous les algorithmes"""
    global comparator
    
    if comparator is None:
        return jsonify({'status': 'error', 'message': 'Comparateur non initialisé'}), 400
    
    try:
        results = comparator.train_and_evaluate_all()
        
        # Préparer les résultats pour JSON
        json_results = {}
        for name, result in results.items():
            if 'error' not in result:
                json_results[name] = {
                    'metrics': result['metrics'],
                    'confusion_matrix': result['confusion_matrix'],
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
            else:
                json_results[name] = {'error': result['error']}
        
        return jsonify({
            'status': 'success',
            'results': json_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    """Retourne le résumé comparatif"""
    global comparator
    
    if comparator is None or not comparator.results:
        return jsonify({'status': 'error', 'message': 'Aucun résultat disponible'}), 400
    
    try:
        summary = comparator.get_comparison_summary()
        return jsonify({
            'status': 'success',
            'comparison': summary.to_dict('records')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/best', methods=['GET'])
def get_best():
    """Retourne le meilleur algorithme"""
    global comparator
    
    if comparator is None or not comparator.results:
        return jsonify({'status': 'error', 'message': 'Aucun résultat disponible'}), 400
    
    try:
        best_name, best_result = comparator.get_best_algorithm()
        
        if best_result:
            return jsonify({
                'status': 'success',
                'best_algorithm': best_name,
                'metrics': best_result['metrics'],
                'confusion_matrix': best_result['confusion_matrix']
            })
        else:
            return jsonify({'status': 'error', 'message': 'Aucun résultat valide'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/confusion-matrix/<algorithm>', methods=['GET'])
def get_confusion_matrix(algorithm):
    """Retourne la matrice de confusion d'un algorithme spécifique"""
    global comparator
    
    if comparator is None or not comparator.results:
        return jsonify({'status': 'error', 'message': 'Aucun résultat disponible'}), 400
    
    if algorithm not in comparator.results:
        return jsonify({'status': 'error', 'message': f'Algorithme {algorithm} non trouvé'}), 404
    
    result = comparator.results[algorithm]
    if 'error' in result:
        return jsonify({'status': 'error', 'message': result['error']}), 400
    
    return jsonify({
        'status': 'success',
        'algorithm': algorithm,
        'confusion_matrix': result['confusion_matrix'],
        'metrics': result['metrics']
    })

@app.route('/api/rpu-metrics', methods=['GET'])
def get_rpu_metrics():
    """Retourne toutes les métriques RPU pour tous les algorithmes"""
    global comparator
    
    if comparator is None or not comparator.results:
        return jsonify({'status': 'error', 'message': 'Aucun résultat disponible'}), 400
    
    try:
        rpu_metrics = {}
        for name, result in comparator.results.items():
            if 'error' not in result:
                rpu_metrics[name] = {
                    'Recall': result['metrics']['Recall'],
                    'Precision': result['metrics']['Precision'],
                    'Uplift': result['metrics']['Uplift'],
                    'Accuracy': result['metrics']['Accuracy'],
                    'F1-Score': result['metrics']['F1-Score']
                }
        
        return jsonify({
            'status': 'success',
            'rpu_metrics': rpu_metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("API ML Algorithm Comparison")
    print("="*60)
    print("\nEndpoints disponibles:")
    print("  POST /api/initialize - Initialise le comparateur")
    print("  POST /api/train - Entraîne tous les algorithmes")
    print("  GET  /api/comparison - Résumé comparatif")
    print("  GET  /api/best - Meilleur algorithme")
    print("  GET  /api/confusion-matrix/<algorithm> - Matrice de confusion")
    print("  GET  /api/rpu-metrics - Métriques RPU")
    print("\nDémarrage du serveur sur http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

