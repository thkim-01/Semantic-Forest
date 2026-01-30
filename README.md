# Semantic Decision Tree (SDT) for Molecular Classification

## Overview
SMILES 기반 분자 구조를 화학 온톨로지로 변환하고, Semantic Decision Tree를 사용하여 Blood-Brain Barrier Penetration (BBBP) 예측을 수행합니다.

## Project Structure
```
SDT/
├── data/
│   └── bbbp/
│       └── BBBP.csv
├── src/
│   ├── ontology/
│   │   ├── __init__.py
│   │   ├── molecule_ontology.py    # 온톨로지 구축
│   │   └── smiles_converter.py     # SMILES → 온톨로지 변환
│   ├── sdt/
│   │   ├── __init__.py
│   │   ├── refinement.py           # Refinement 연산자
│   │   ├── tree.py                 # SDT 구조
│   │   └── learner.py              # SDT 학습 알고리즘
│   └── utils/
│       ├── __init__.py
│       └── evaluation.py           # 평가 지표 (AUC-ROC)
├── experiments/
│   └── bbbp_experiment.py          # 메인 실행 스크립트
├── output/
│   └── (결과 저장 디렉토리)
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python experiments/bbbp_experiment.py
```

## Evaluation Metric
- AUC-ROC (Area Under the ROC Curve)
