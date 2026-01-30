
"""
Benchmark Runner
Executes Semantic Decision Tree experiments across multiple datasets.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.benchmark_config import DATASET_CONFIG
from src.ontology import MolecularFeatureExtractor, MolecularInstance
from src.sdt import SDTLearner
from src.utils import ModelEvaluator, calculate_probability_from_tree
from src.utils.scaffold_splitter import scaffold_split
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(dataset_name):
    """Load dataset from config"""
    config = DATASET_CONFIG[dataset_name]
    path = config['path']
    logger.info(f"Loading {dataset_name} from {path}...")
    df = pd.read_csv(path)
    return df, config


def get_targets(df, config):
    """Get list of targets for the dataset"""
    targets = config['targets']
    if targets == 'ALL_EXCEPT_SMILES':
        smiles_col = config.get('smiles_col', 'smiles')
        targets = [col for col in df.columns if col != smiles_col and col != 'mol_id']
    return targets


def preprocess_data(df, target_col, split_type, test_size=0.2, seed=42, smiles_col='smiles'):
    """Preprocess and split data for a specific target"""
    
    # 1. Filter valid data (remove NaNs in target)
    df_valid = df.dropna(subset=[target_col]).copy()
    
    # Check if we have enough data
    if len(df_valid) < 50:
        logger.warning(f"  Skipping {target_col}: Not enough valid samples ({len(df_valid)})")
        return None
    
    # 2. Extract arrays
    smiles_list = df_valid[smiles_col].tolist()
    labels = df_valid[target_col].values.astype(int)
    
    # 3. Split
    if split_type == 'scaffold':
        logger.info(f"  Using Random Scaffold Split for {target_col}")
        train_s, test_s, train_l, test_l = scaffold_split(
            smiles_list, labels, test_size=test_size, seed=seed
        )
    else:
        logger.info(f"  Using Random Split for {target_col}")
        train_s, test_s, train_l, test_l = train_test_split(
            smiles_list, labels, test_size=test_size, random_state=seed, stratify=labels
        )
        
    return train_s, test_s, train_l, test_l


def extract_features(smiles_list, labels, extractor):
    """Extract molecular features using shared extractor"""
    instances = []
    # Batch processing could be optimized but keeping simple for now
    for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
        try:
            features = extractor.extract_features(smiles)
            if features:
                instances.append(MolecularInstance(
                    mol_id=f"Mol_{i}", smiles=smiles, label=int(label), features=features
                ))
        except Exception:
            pass # Skip failed conversions
    return instances


def run_experiment(dataset_name, target, train_data, test_data, output_dir):
    """Run SDT training and evaluation for a single target"""
    train_instances, test_instances = train_data, test_data
    
    logger.info(f"  Training SDT on {len(train_instances)} samples...")
    # Use balanced class weights to optimize AUC-ROC
    learner = SDTLearner(
        max_depth=10, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        class_weight='balanced', 
        verbose=False
    )
    
    start_time = time.time()
    tree = learner.fit(train_instances)
    duration = time.time() - start_time
    logger.info(f"  Training completed in {duration:.2f}s. Nodes: {len(tree.nodes)}")
    
    # Evaluate
    y_true = np.array([inst.label for inst in test_instances])
    y_pred = tree.predict_batch(test_instances)
    y_pred_proba = np.array([calculate_probability_from_tree(tree, inst) for inst in test_instances])
    
    evaluator = ModelEvaluator()
    try:
        metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
    except ValueError as e:
        logger.error(f"  Evaluation failed for {target}: {e}")
        return None

    # Save results
    target_dir = os.path.join(output_dir, target)
    os.makedirs(target_dir, exist_ok=True)
    
    with open(os.path.join(target_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\nTarget: {target}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"AUC_ROC: {metrics.get('auc_roc', 'N/A')}\n")
        f.write(f"F1: {metrics['f1_score']:.4f}\n")
    
    # Plotting (optional, maybe skip for mass run to save time/space, or keep)
    # evaluator.plot_roc_curve(y_true, y_pred_proba, save_path=os.path.join(target_dir, 'roc.png'))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SDT Benchmark Runner")
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name or "all"')
    parser.add_argument('--result_dir', type=str, default='output/benchmark', help='Output directory')
    args = parser.parse_args()
    
    datasets_to_run = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    feature_extractor = MolecularFeatureExtractor()
    
    all_results = []

    for ds_name in datasets_to_run:
        if ds_name not in DATASET_CONFIG:
            logger.error(f"Unknown dataset: {ds_name}")
            continue
            
        logger.info(f"Processing {ds_name}...")
        try:
            df, config = load_dataset(ds_name)
            targets = get_targets(df, config)
            smiles_col = config.get('smiles_col', 'smiles')
            
            ds_output_dir = os.path.join(args.result_dir, ds_name)
            os.makedirs(ds_output_dir, exist_ok=True)
            
            for target in targets:
                logger.info(f"Target: {target}")
                
                # Preprocess
                split_data = preprocess_data(df, target, config['split'], smiles_col=smiles_col)
                if not split_data:
                    continue
                    
                train_s, test_s, train_l, test_l = split_data
                
                # Feature Extraction
                train_inst = extract_features(train_s, train_l, feature_extractor)
                test_inst = extract_features(test_s, test_l, feature_extractor)
                
                if len(train_inst) == 0 or len(test_inst) == 0:
                    logger.warning(f"  No valid instances extracted for {target}")
                    continue

                # Run Experiment
                metrics = run_experiment(ds_name, target, train_inst, test_inst, ds_output_dir)
                if metrics:
                    res = {'dataset': ds_name, 'target': target}
                    res.update(metrics)
                    all_results.append(res)
                    
        except Exception as e:
            logger.error(f"Failed to process {ds_name}: {e}", exc_info=True)

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.result_dir, 'benchmark_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Benchmark complete. Summary saved to {summary_path}")
        print("\nBenchmark Summary:")
        print(summary_df[['dataset', 'target', 'auc_roc', 'accuracy']].to_string())
    else:
        logger.warning("No results to save.")
        
    # Aggregate results
    try:
        from src.utils.aggregate_results import aggregate_results
        agg_output = os.path.join(args.result_dir, 'benchmark_summary_averaged.csv')
        aggregate_results(summary_path, agg_output)
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")

if __name__ == "__main__":
    main()
