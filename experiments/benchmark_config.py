
"""
Benchmark Configuration
Defines dataset paths, target class columns, and splitting strategies.
"""

import os

# Base data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

DATASET_CONFIG = {
    'bbbp': {
        'path': os.path.join(DATA_DIR, 'bbbp', 'BBBP.csv'),
        'targets': ['p_np'],
        'split': 'scaffold',
        'smiles_col': 'smiles'
    },
    'bace': {
        'path': os.path.join(DATA_DIR, 'bace', 'bace.csv'),
        'targets': ['Class'],
        'split': 'scaffold',
        'smiles_col': 'smiles'
    },
    'hiv': {
        'path': os.path.join(DATA_DIR, 'hiv', 'HIV.csv'),
        'targets': ['HIV_active'],
        'split': 'scaffold',
        'smiles_col': 'smiles'
    },
    'tox21': {
        'path': os.path.join(DATA_DIR, 'tox21', 'tox21.csv'),
        'targets': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
        'split': 'random',
        'smiles_col': 'smiles'
    },
    'clintox': {
        'path': os.path.join(DATA_DIR, 'clintox', 'clintox.csv'),
        'targets': ['FDA_APPROVED', 'CT_TOX'],
        'split': 'random',
        'smiles_col': 'smiles'
    },
    'sider': {
        'path': os.path.join(DATA_DIR, 'sider', 'sider.csv'),
        'targets': 'ALL_EXCEPT_SMILES',  # Special marker to be handled in runner
        'split': 'random',
        'smiles_col': 'smiles'
    }
}
