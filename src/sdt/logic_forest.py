
from typing import List, Optional
import numpy as np
from src.sdt.logic_learner import LogicSDTLearner
from sklearn.utils import resample

class SemanticRandomForest:
    """
    Random Forest ensemble of Logic-Based Semantic Decision Trees.
    Combines rule-based explainability with ensemble performance.
    """
    def __init__(self, ontology_manager, n_estimators: int = 20, 
                 max_depth: int = 10, min_samples_split: int = 10, 
                 min_samples_leaf: int = 5, class_weight: str = 'balanced',
                 verbose: bool = False):
        self.ontology_manager = ontology_manager
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.verbose = verbose
        
        self.trees = []
    
    def fit(self, instances: List):
        """
        Train the forest on provided instances.
        Uses bootstrap aggregating (Bagging).
        """
        self.trees = []
        n_samples = len(instances)
        
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Training Tree {i+1}/{self.n_estimators}...")
                
            # Bootstrap sample
            # Since instances are objects, we resample the list indices or list itself
            # resample handles lists
            sample_instances = resample(instances, n_samples=n_samples, random_state=i)
            
            # Create and train learner
            learner = LogicSDTLearner(
                self.ontology_manager,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                verbose=False # Reduce noise
            )
            
            tree = learner.fit(sample_instances)
            self.trees.append((tree, learner)) # Store learner too for refinement generator if needed
            
        print(f"Semantic Random Forest training completed. {len(self.trees)} trees built.")
        
    def predict_proba(self, instances: List) -> List[float]:
        """
        Predict probability of positive class (label 1).
        Averages probabilities from all trees.
        """
        all_probs = []
        
        for i, (tree, learner) in enumerate(self.trees):
            # Predict for this tree
            tree_probs = []
            for inst in instances:
                node = tree.root
                while not node.is_leaf:
                    # Use learner's generator to check refinement
                    if learner.refinement_generator.instance_satisfies_refinement(inst, node.refinement):
                        node = node.left_child
                    else:
                        node = node.right_child
                
                # Leaf probability
                total = sum(node.label_counts.values())
                prob1 = node.label_counts.get(1, 0) / total if total > 0 else 0.0
                tree_probs.append(prob1)
            all_probs.append(tree_probs)
            
        # Average across trees
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs.tolist()
        
    def predict(self, instances: List, threshold: float = 0.5) -> List[int]:
        """
        Predict class labels.
        """
        probs = self.predict_proba(instances)
        return [1 if p >= threshold else 0 for p in probs]
