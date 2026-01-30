"""
BBBP Experiment: BBBP 데이터셋으로 SDT 학습 및 평가
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import time

from src.ontology import MolecularFeatureExtractor, MolecularInstance
from src.sdt import SDTLearner
from src.utils import ModelEvaluator, calculate_probability_from_tree
from src.utils.scaffold_splitter import scaffold_split


def load_bbbp_data(csv_path: str):
    """BBBP 데이터 로드"""
    print(f"Loading BBBP data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['p_np'].value_counts()}")
    
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    데이터 전처리 및 train/test 분할 (Scaffold Split 적용)
    """
    print("\nPreprocessing data...")
    
    # SMILES와 label 추출
    smiles_list = df['smiles'].tolist()
    labels = df['p_np'].values
    
    # Train/Test 분할 (Scaffold Split)
    train_smiles, test_smiles, train_labels, test_labels = scaffold_split(
        smiles_list, labels, test_size=test_size, seed=random_state
    )
    
    print(f"Train samples: {len(train_smiles)}")
    print(f"Test samples: {len(test_smiles)}")
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    
    return train_smiles, test_smiles, train_labels, test_labels


def extract_molecular_features(smiles_list, labels, feature_extractor):
    """
    SMILES로부터 분자 특성 추출 및 MolecularInstance 생성
    """
    instances = []
    failed_count = 0
    
    for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
        try:
            features = feature_extractor.extract_features(smiles)
            mol_instance = MolecularInstance(
                mol_id=f"Mol_{i}",
                smiles=smiles,
                label=int(label),
                features=features
            )
            instances.append(mol_instance)
        except Exception as e:
            failed_count += 1
            # print(f"Failed to process SMILES {i}: {smiles} - {str(e)}")
    
    if failed_count > 0:
        print(f"Warning: {failed_count} molecules failed to process")
    
    return instances


def train_sdt(train_instances, max_depth=10, min_samples_split=10, min_samples_leaf=5):
    """
    SDT 학습
    """
    print("\n" + "="*70)
    print("Training Semantic Decision Tree")
    print("="*70)
    
    learner = SDTLearner(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        verbose=True
    )
    
    start_time = time.time()
    tree = learner.fit(train_instances)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Total nodes: {len(tree.nodes)}")
    print(f"Max depth reached: {max(node.depth for node in tree.nodes)}")
    
    return tree


def evaluate_model(tree, test_instances, output_dir):
    """
    모델 평가
    """
    print("\n" + "="*70)
    print("Evaluating Model")
    print("="*70)
    
    # 예측
    y_true = np.array([inst.label for inst in test_instances])
    y_pred = tree.predict_batch(test_instances)
    
    # 확률 예측 (AUC-ROC 계산용)
    y_pred_proba = np.array([
        calculate_probability_from_tree(tree, inst) 
        for inst in test_instances
    ])
    
    # 평가
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
    evaluator.print_metrics()
    
    # 시각화
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    evaluator.plot_roc_curve(
        y_true, y_pred_proba,
        save_path=os.path.join(output_dir, 'roc_curve.png')
    )
    
    # Feature importance
    feature_importance = tree.get_feature_importance()
    if feature_importance:
        print("\nTop 10 Important Features:")
        sorted_importance = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_importance:
            print(f"  {feature:30s}: {importance:.4f}")
        
        evaluator.plot_feature_importance(
            feature_importance,
            top_n=15,
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )
    
    return metrics


def save_results(metrics, tree, output_dir):
    """
    결과 저장
    """
    print("\nSaving results...")
    
    # 메트릭 저장
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("BBBP Experiment Results\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        
        if 'auc_roc' in metrics:
            f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
        
        f.write(f"\nTree Structure:\n")
        f.write(f"Total nodes: {len(tree.nodes)}\n")
        f.write(f"Max depth: {max(node.depth for node in tree.nodes)}\n")
        
        # Feature importance
        feature_importance = tree.get_feature_importance()
        if feature_importance:
            f.write(f"\nFeature Importance:\n")
            sorted_importance = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance:
                f.write(f"  {feature:30s}: {importance:.4f}\n")
    
    print(f"Results saved to {output_dir}")


def main():
    """메인 실험 함수"""
    # 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'bbbp', 'BBBP.csv')
    output_dir = os.path.join(project_root, 'output', 'bbbp_results')
    
    # 1. 데이터 로드
    df = load_bbbp_data(data_path)
    
    # 2. 데이터 전처리
    train_smiles, test_smiles, train_labels, test_labels = preprocess_data(df)
    
    # 3. 분자 특성 추출
    print("\nExtracting molecular features...")
    feature_extractor = MolecularFeatureExtractor()
    
    print("Processing training data...")
    train_instances = extract_molecular_features(
        train_smiles, train_labels, feature_extractor
    )
    print(f"Train instances created: {len(train_instances)}")
    
    print("Processing test data...")
    test_instances = extract_molecular_features(
        test_smiles, test_labels, feature_extractor
    )
    print(f"Test instances created: {len(test_instances)}")
    
    # 4. SDT 학습
    tree = train_sdt(
        train_instances,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    # 5. 모델 평가
    metrics = evaluate_model(tree, test_instances, output_dir)
    
    # 6. 결과 저장
    save_results(metrics, tree, output_dir)
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
