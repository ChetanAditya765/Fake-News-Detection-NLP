
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }

    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        print(f"Training {model_name}...")

        model = self.models[model_name]
        model.fit(X_train, y_train)

        # Store trained model
        self.trained_models[model_name] = model

        return model

    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"{model_name} - Accuracy: {accuracy:.4f}")

        return accuracy, report, cm

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        self.initialize_models()

        print("Training all models...")
        print("-" * 50)

        for model_name in self.models.keys():
            # Train model
            model = self.train_model(model_name, X_train, y_train)

            # Evaluate model
            self.evaluate_model(model_name, model, X_test, y_test)

            print("-" * 50)

        # Display results summary
        self.display_results_summary()

    def display_results_summary(self):
        """Display summary of all model results"""
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)

        results_df = pd.DataFrame([
            {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['classification_report']['1']['precision'],
                'Recall': results['classification_report']['1']['recall'],
                'F1-Score': results['classification_report']['1']['f1-score']
            }
            for model_name, results in self.results.items()
        ])

        # Sort by accuracy
        results_df = results_df.sort_values('Accuracy', ascending=False)

        print(results_df.to_string(index=False, float_format='%.4f'))

        # Save results
        results_df.to_csv('results/model_comparison.csv', index=False)

        return results_df

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Fake', 'Real'], 
                       yticklabels=['Fake', 'Real'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

    def perform_hyperparameter_tuning(self, model_name, X_train, y_train, param_grid):
        """Perform hyperparameter tuning for a specific model"""
        print(f"Performing hyperparameter tuning for {model_name}...")

        model = self.models[model_name]

        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def cross_validate_models(self, X_train, y_train, cv=5):
        """Perform cross-validation for all models"""
        print("\n🔄 CROSS-VALIDATION RESULTS")
        print("=" * 50)

        cv_results = {}

        for model_name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_results[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }

            print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results

    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.trained_models.items():
            model_path = f'models/{model_name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"✅ {model_name} saved to {model_path}")

    def load_model(self, model_name):
        """Load a trained model"""
        model_path = f'models/{model_name}_model.pkl'
        try:
            model = joblib.load(model_path)
            print(f"✅ {model_name} loaded from {model_path}")
            return model
        except:
            print(f"❌ Failed to load {model_name} from {model_path}")
            return None

    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            print("No models trained yet!")
            return None

        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['accuracy'])
        best_model = self.trained_models[best_model_name]

        print(f"🏆 Best model: {best_model_name} "
              f"(Accuracy: {self.results[best_model_name]['accuracy']:.4f})")

        return best_model_name, best_model

def hyperparameter_grids():
    """Define hyperparameter grids for tuning"""
    return {
        'logistic_regression': {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }

if __name__ == "__main__":
    print("🚀 Starting model training...")

    # Load preprocessed data
    df = pd.read_csv('data/processed/fake_news_cleaned.csv')
  # or your correct file path

    # Ensure target and features exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Expected 'text' and 'label' columns in dataset")

    # Convert text to features
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Save the vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Save models
    trainer.save_models()

    # Optionally plot confusion matrices
    trainer.plot_confusion_matrices()

import sys
sys.exit(0)
