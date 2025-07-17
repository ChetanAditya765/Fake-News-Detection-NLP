
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
        """Create a pipeline for LIME explanations"""
        def predict_proba(texts):
            # Transform texts using the same vectorizer
            text_vectors = self.vectorizer.transform(texts)
            return self.model.predict_proba(text_vectors)

        return predict_proba

    def explain_with_lime(self, text, num_features=20, num_samples=1000):
        """Generate LIME explanation for a single text"""
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
        """Generate SHAP explanations for texts"""
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
        """Visualize LIME explanation"""
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
        ax.text(0.02, 0.98, 'Red: Supports Fake\nGreen: Supports Real', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def visualize_shap_explanation(self, shap_values, texts, save_path=None):
        """Visualize SHAP explanation"""
        # Create SHAP waterfall plot for first text
        if len(texts) > 0:
            shap.waterfall_plot(shap_values[0], max_display=20)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()

    def get_feature_importance(self, feature_names=None):
        """Get feature importance from the model"""
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
        """Visualize top feature importance"""
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
        """Comprehensive explanation of a single prediction"""
        print(f"\n🔍 EXPLAINING PREDICTION FOR TEXT:")
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
        """Generate a comprehensive explanation report"""
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
