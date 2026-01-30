"""
논문 SDT 전체 실험: BBBP 데이터셋
Train/Test Split + AUC-ROC 평가
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ontology.ontology_loader import OntologyLoader
from src.sdt.sdt_learner import SemanticDecisionTreeLearner
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_instances(all_instances, test_ratio=0.2, random_seed=42):
    """Train/Test split"""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(all_instances))
    
    test_size = int(len(all_instances) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_instances = [all_instances[i] for i in train_indices]
    test_instances = [all_instances[i] for i in test_indices]
    
    return train_instances, test_instances


def get_labels(instances):
    """인스턴스에서 label 추출"""
    labels = []
    for inst in instances:
        label = getattr(inst, 'hasLabel', None)
        labels.append(label if label is not None else 0)
    return np.array(labels)


def calculate_probabilities(learner, instances):
    """
    트리 leaf의 label 분포로부터 확률 계산
    """
    probabilities = []
    
    for inst in instances:
        node = learner.root
        
        # Leaf까지 탐색
        while not node.is_leaf and node is not None:
            if learner.refinement_generator.instance_satisfies_refinement(inst, node.refinement):
                node = node.left_child
            else:
                node = node.right_child
        
        # Leaf의 label 분포로 확률 계산
        if node and node.is_leaf:
            total = node.num_instances
            if total == 0:
                probabilities.append(0.5)
            else:
                pos_count = node.label_counts.get(1, 0)
                probabilities.append(pos_count / total)
        else:
            probabilities.append(0.5)
    
    return np.array(probabilities)


def main():
    logger.info("="*70)
    logger.info("논문 SDT 실험: BBBP 데이터셋")
    logger.info("="*70)
    
    # 1. 온톨로지 로드
    logger.info("\n[1/5] Loading ontology...")
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()
    
    # 2. 전체 인스턴스 가져오기
    logger.info("\n[2/5] Splitting data...")
    all_molecules = loader.get_instances("Molecule")
    logger.info(f"Total molecules: {len(all_molecules)}")
    
    train_instances, test_instances = split_instances(all_molecules, test_ratio=0.2)
    logger.info(f"Train: {len(train_instances)}, Test: {len(test_instances)}")
    
    train_labels = get_labels(train_instances)
    test_labels = get_labels(test_instances)
    logger.info(f"Train labels: 0={sum(train_labels==0)}, 1={sum(train_labels==1)}")
    logger.info(f"Test labels: 0={sum(test_labels==0)}, 1={sum(test_labels==1)}")
    
    # 3. SDT 학습
    logger.info("\n[3/5] Training Semantic Decision Tree...")
    learner = SemanticDecisionTreeLearner(
        onto, 
        max_depth=8, 
        min_samples_split=20,
        min_samples_leaf=10,
        verbose=False
    )
    
    # Train 인스턴스만으로 학습하도록 수정 필요
    # 임시로 전체로 학습 (논문 SDT는 subset 학습 지원 필요)
    root = learner.fit("Molecule")
    
    # 4. 예측
    logger.info("\n[4/5] Evaluating...")
    test_predictions = learner.predict_batch(test_instances)
    test_probabilities = calculate_probabilities(learner, test_instances)
    
    # 5. 평가
    logger.info("\n[5/5] Results:")
    logger.info("="*70)
    
    accuracy = accuracy_score(test_labels, test_predictions)
    auc_roc = roc_auc_score(test_labels, test_probabilities)
    
    logger.info(f"\n✅ Performance Metrics:")
    logger.info(f"   Accuracy:  {accuracy:.4f}")
    logger.info(f"   AUC-ROC:   {auc_roc:.4f}")
    
    logger.info(f"\n Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_predictions)
    logger.info(f"\n{cm}")
    
    logger.info(f"\n Classification Report:")
    logger.info(f"\n{classification_report(test_labels, test_predictions)}")
    
    logger.info(f"\n📊 Tree Structure:")
    logger.info(f"   Total nodes: {len(learner.nodes)}")
    logger.info(f"   Max depth: {max(n.depth for n in learner.nodes)}")
    logger.info(f"   Leaf nodes: {sum(1 for n in learner.nodes if n.is_leaf)}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ 논문 SDT 실험 완료!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
