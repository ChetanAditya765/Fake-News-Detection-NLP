
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            print("NLTK data download failed. Some features may not work.")

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        if pd.isna(text):
            return ""

        # Clean text
        text = self.clean_text(text)

        # Tokenize and lemmatize
        text = self.tokenize_and_lemmatize(text)

        return text

    def load_and_preprocess_data(self, file_path, text_column='text', label_column='label'):
        """Load and preprocess dataset"""
        print(f"Loading data from {file_path}...")

        # Load data
        df = pd.read_csv(file_path)

        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Handle missing values
        df = df.dropna(subset=[text_column, label_column])

        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]

        print(f"After preprocessing: {df.shape}")
        print(f"Label distribution:\n{df[label_column].value_counts()}")

        return df

    def create_features(self, texts, fit_vectorizer=True):
        """Create TF-IDF features"""
        if fit_vectorizer:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)

        return tfidf_features

    def split_data(self, df, text_column='processed_text', label_column='label', 
                   test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X = df[text_column]
        y = df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = {
        'text': [
            "Breaking: Scientists discover new planet in our solar system with signs of life",
            "Local weather forecast shows sunny skies expected tomorrow with temperatures reaching 75°F",
            "SHOCKING: Celebrities are secretly aliens from Mars, government covers it up",
            "Stock market shows steady growth in technology sector this quarter",
            "Miracle cure: Drink this one ingredient to lose 50 pounds in a week",
            "New research shows benefits of regular exercise on mental health",
            "Politicians are lizard people controlling the world through mind control waves",
            "City council approves new infrastructure budget for road improvements",
            "Vaccines contain microchips for government surveillance of citizens",
            "University study finds positive correlation between reading and cognitive function"
        ],
        'label': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
    }

    df = pd.DataFrame(sample_data)
    return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # ✅ Load and preprocess the real dataset
    processed_df = preprocessor.load_and_preprocess_data(
        "data/raw/WELFake_Dataset.csv", 
        text_column="text", 
        label_column="label"
    )

    # ✅ Save the cleaned data for training
    processed_df.to_csv("data/processed/fake_news_cleaned.csv", index=False)

    print("✅ Full dataset preprocessing complete and saved.")
