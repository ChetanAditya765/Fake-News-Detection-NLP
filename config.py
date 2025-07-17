
# Configuration file for Fake News Detection System
import os

# Data Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Text Processing Configuration
MAX_FEATURES = 10000
MAX_LENGTH = 512
EMBEDDING_DIM = 300

# Model Parameters
MODELS_CONFIG = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': RANDOM_STATE
    },
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 32
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    }
}

# Explainable AI Configuration
LIME_CONFIG = {
    'num_features': 20,
    'num_samples': 1000
}

SHAP_CONFIG = {
    'max_evals': 100,
    'sample_size': 100
}

# Web App Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

STREAMLIT_CONFIG = {
    'port': 8501
}

# Dataset URLs (for automatic download)
DATASET_URLS = {
    'welfake': 'https://zenodo.org/records/4561253/files/WELFake_Dataset.csv',
    'liar': 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip',
    'isot': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset'
}
