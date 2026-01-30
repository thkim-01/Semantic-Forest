
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_learner import LogicSDTLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_bbbp():
    logger.info("Starting BBBP Logic SDT Optimization...")

    # Load Data
    try:
        df = pd.read_csv("data/bbbp/BBBP.csv")
    except FileNotFoundError:
        logger.error("BBBP.csv not found")
        return

    # Use full dataset
    df_sample = df
    logger.info(f"Loaded {len(df_sample)} samples.")

    # Split
    train_df, test_df = train_test_split(df_sample, test_size=0.3, random_state=42)
    
    # Ontology Setup
    onto_path = "ontology/temp_opt_bbbp.owl"
    if os.path.exists(onto_path): os.remove(onto_path)
    onto = MoleculeOntology(onto_path)
    extractor = MolecularFeatureExtractor()

    def populate_ontology(dataframe, name):
        instances = []
        for idx, row in dataframe.iterrows():
            try:
                mol_id = f"Mol_{name}_{idx}"
                feats = extractor.extract_features(row['smiles'])
                inst = onto.add_molecule_instance(mol_id, feats, label=int(row['p_np']))
                instances.append(inst)
            except: pass
        return instances

    logger.info("Populating Ontology...")
    train_instances = populate_ontology(train_df, "Train")
    test_instances = populate_ontology(test_df, "Test")
    
    # Grid Search / Optimization
    # Testing deeper tree + conjunctions
    depths = [10, 15]
    best_auc = 0
    best_depth = 0
    
    for d in depths:
        logger.info(f"Training Logic SDT with max_depth={d}...")
        learner = LogicSDTLearner(onto, max_depth=d, min_samples_split=20, min_samples_leaf=5, class_weight='balanced', verbose=False)
        tree = learner.fit(train_instances) # Conjunction is enabled in refinement generator
        
        # Predict
        probs = []
        for inst in test_instances:
            node = tree.root
            while not node.is_leaf:
                if learner.refinement_generator.instance_satisfies_refinement(inst, node.refinement):
                    node = node.left_child
                else:
                    node = node.right_child
            
            total = sum(node.label_counts.values())
            prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
            probs.append(prob1)
            
        test_labels = [inst.hasLabel[0] for inst in test_instances]
        auc = roc_auc_score(test_labels, probs)
        logger.info(f"Depth {d} Result -> AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_depth = d
            
    logger.info(f"Optimization Complete. Best Depth: {best_depth}, Best AUC: {best_auc:.4f}")
    
    with open("optimization_results.txt", "w") as f:
        f.write(f"Best Depth: {best_depth}\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")

if __name__ == "__main__":
    optimize_bbbp()
