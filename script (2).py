# Create the model training module
model_training_py = """
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
        \"\"\"Initialize different ML models\"\"\"
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        
    def train_model(self, model_name, X_train, y_train):
        \"\"\"Train a specific model\"\"\"
        print(f"Training {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        \"\"\"Evaluate model performance\"\"\"
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
        \"\"\"Train and evaluate all models\"\"\"
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
        \"\"\"Display summary of all model results\"\"\"
        print("\\n📊 MODEL PERFORMANCE SUMMARY")
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
        \"\"\"Plot confusion matrices for all models\"\"\"
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
        \"\"\"Perform hyperparameter tuning for a specific model\"\"\"
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
        \"\"\"Perform cross-validation for all models\"\"\"
        print("\\n🔄 CROSS-VALIDATION RESULTS")
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
        \"\"\"Save all trained models\"\"\"
        for model_name, model in self.trained_models.items():
            model_path = f'models/{model_name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"✅ {model_name} saved to {model_path}")
    
    def load_model(self, model_name):
        \"\"\"Load a trained model\"\"\"
        model_path = f'models/{model_name}_model.pkl'
        try:
            model = joblib.load(model_path)
            print(f"✅ {model_name} loaded from {model_path}")
            return model
        except:
            print(f"❌ Failed to load {model_name} from {model_path}")
            return None
    
    def get_best_model(self):
        \"\"\"Get the best performing model\"\"\"
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
    \"\"\"Define hyperparameter grids for tuning\"\"\"
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
    # This will be called from the main training script
    print("Model training module loaded successfully!")
"""

# Save model training module
with open("src/model_training.py", "w") as f:
    f.write(model_training_py)

print("✅ Model training module created!")

# Create the explainable AI module
explainable_ai_py = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import shap
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ExplainableAI:
    def __init__(self, model, vectorizer, class_names=['Fake', 'Real']):
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names
        self.lime_explainer = LimeTextExplainer(class_names=class_names)
        
    def create_prediction_pipeline(self):
        \"\"\"Create a pipeline for LIME explanations\"\"\"
        def predict_proba(texts):
            # Transform texts using the same vectorizer
            text_vectors = self.vectorizer.transform(texts)
            return self.model.predict_proba(text_vectors)
        
        return predict_proba
    
    def explain_with_lime(self, text, num_features=20, num_samples=1000):
        \"\"\"Generate LIME explanation for a single text\"\"\"
        print("Generating LIME explanation...")
        
        # Create prediction function
        predict_fn = self.create_prediction_pipeline()
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def explain_with_shap(self, texts, max_evals=100):
        \"\"\"Generate SHAP explanations for texts\"\"\"
        print("Generating SHAP explanations...")
        
        # Create prediction function for SHAP
        def predict_proba(texts):
            if isinstance(texts, str):
                texts = [texts]
            text_vectors = self.vectorizer.transform(texts)
            return self.model.predict_proba(text_vectors)
        
        # Create SHAP explainer
        explainer = shap.Explainer(predict_proba, texts)
        
        # Generate explanations
        shap_values = explainer(texts, max_evals=max_evals)
        
        return explainer, shap_values
    
    def visualize_lime_explanation(self, explanation, save_path=None):
        \"\"\"Visualize LIME explanation\"\"\"
        # Get explanation as list
        exp_list = explanation.as_list()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate positive and negative contributions
        words = [item[0] for item in exp_list]
        contributions = [item[1] for item in exp_list]
        
        # Create color map
        colors = ['red' if contrib < 0 else 'green' for contrib in contributions]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(words)), contributions, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Contribution to Prediction')
        ax.set_title('LIME Explanation: Word Contributions')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        ax.text(0.02, 0.98, 'Red: Supports Fake\\nGreen: Supports Real', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def visualize_shap_explanation(self, shap_values, texts, save_path=None):
        \"\"\"Visualize SHAP explanation\"\"\"
        # Create SHAP waterfall plot for first text
        if len(texts) > 0:
            shap.waterfall_plot(shap_values[0], max_display=20)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def get_feature_importance(self, feature_names=None):
        \"\"\"Get feature importance from the model\"\"\"
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_[0])
        else:
            print("Model doesn't support feature importance extraction")
            return None
        
        if feature_names is None:
            feature_names = self.vectorizer.get_feature_names_out()
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def visualize_feature_importance(self, top_n=20, save_path=None):
        \"\"\"Visualize top feature importance\"\"\"
        feature_importance_df = self.get_feature_importance()
        
        if feature_importance_df is None:
            return
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color='skyblue', alpha=0.7)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def explain_prediction(self, text, use_lime=True, use_shap=False, 
                         num_features=20, visualize=True):
        \"\"\"Comprehensive explanation of a single prediction\"\"\"
        print(f"\\n🔍 EXPLAINING PREDICTION FOR TEXT:")
        print(f"'{text[:100]}...'")
        print("=" * 80)
        
        # Make prediction
        text_vector = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vector)[0]
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(text_vector)[0]
            print(f"Prediction: {self.class_names[prediction]} "
                  f"(Confidence: {probability[prediction]:.2f})")
        else:
            print(f"Prediction: {self.class_names[prediction]}")
        
        explanations = {}
        
        # LIME explanation
        if use_lime:
            lime_explanation = self.explain_with_lime(text, num_features)
            explanations['lime'] = lime_explanation
            
            if visualize:
                self.visualize_lime_explanation(lime_explanation)
        
        # SHAP explanation
        if use_shap:
            shap_explainer, shap_values = self.explain_with_shap([text])
            explanations['shap'] = (shap_explainer, shap_values)
            
            if visualize:
                self.visualize_shap_explanation(shap_values, [text])
        
        return explanations
    
    def generate_explanation_report(self, text, save_path=None):
        \"\"\"Generate a comprehensive explanation report\"\"\"
        explanations = self.explain_prediction(text, use_lime=True, 
                                             use_shap=False, visualize=False)
        
        # Create report
        report = {
            'text': text,
            'prediction': None,
            'confidence': None,
            'lime_explanation': None,
            'top_supporting_words': [],
            'top_opposing_words': []
        }
        
        # Get prediction
        text_vector = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vector)[0]
        report['prediction'] = self.class_names[prediction]
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(text_vector)[0]
            report['confidence'] = probability[prediction]
        
        # Process LIME explanation
        if 'lime' in explanations:
            lime_exp = explanations['lime']
            exp_list = lime_exp.as_list()
            
            # Separate supporting and opposing words
            supporting = [(word, score) for word, score in exp_list if score > 0]
            opposing = [(word, score) for word, score in exp_list if score < 0]
            
            report['top_supporting_words'] = supporting[:10]
            report['top_opposing_words'] = opposing[:10]
        
        # Save report if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

if __name__ == "__main__":
    print("Explainable AI module loaded successfully!")
"""

# Save explainable AI module
with open("src/explainable_ai.py", "w") as f:
    f.write(explainable_ai_py)

print("✅ Explainable AI module created!")