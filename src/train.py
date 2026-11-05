import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from textblob import TextBlob
from tensorflow.keras.utils import to_categorical

from src.model_architecture import DeepCSAT_ANN

def compute_sentiment_polarity(text):
    """Calculate sentiment polarity score using TextBlob."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

def preprocess_and_engineer_features(df):
    """
    Data preprocessing & Feature engineering pipeline:
    1. SLA response_time_mins calculation / feature handling
    2. Sentiment polarity calculation on customer remarks
    3. TF-IDF feature extraction on textual feedback
    """
    df = df.copy()

    # 1. SLA response_time_mins engineering
    if 'response_time_mins' not in df.columns:
        if 'response_time' in df.columns:
            df['response_time_mins'] = pd.to_numeric(df['response_time'], errors='coerce')
        elif 'created_at' in df.columns and 'responded_at' in df.columns:
            created = pd.to_datetime(df['created_at'])
            responded = pd.to_datetime(df['responded_at'])
            df['response_time_mins'] = (responded - created).dt.total_seconds() / 60.0
        else:
            df['response_time_mins'] = 0.0
    df['response_time_mins'] = df['response_time_mins'].fillna(df['response_time_mins'].median())

    # 2. Sentiment polarity feature engineering
    text_col = None
    for col in ['customer_remark', 'remarks', 'feedback', 'review', 'comment']:
        if col in df.columns:
            text_col = col
            break

    if text_col:
        df['sentiment_polarity'] = df[text_col].apply(compute_sentiment_polarity)
    else:
        df['sentiment_polarity'] = 0.0
        text_col = 'dummy_text'
        df[text_col] = ''

    # 3. TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna('')).toarray()

    return df, tfidf_matrix, vectorizer

def run_training_pipeline(data_path=None, df=None):
    """
    End-to-end training pipeline:
    - Preprocessing & Feature Engineering (SLA response_time_mins, sentiment polarity, TF-IDF)
    - Stratified 70/15/15 train/val/test split
    - SMOTE training oversampling
    - DeepCSAT_ANN compilation, training with EarlyStopping
    - Model evaluation on test split
    """
    if df is None:
        if data_path:
            df = pd.read_csv(data_path)
        else:
            # Benchmark synthetic dataset
            np.random.seed(42)
            n = 1000
            df = pd.DataFrame({
                'response_time_mins': np.random.exponential(scale=20, size=n),
                'customer_remark': np.random.choice([
                    "Fast response and very helpful support!",
                    "Delayed resolution, unsatisfied with item delivery.",
                    "Average experience, solved my issue.",
                    "Customer service was terrible and unhelpful.",
                    "Excellent resolution, five stars!"
                ], size=n),
                'csat_score': np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.15, 0.2, 0.25, 0.25])
            })

    # Prepare targets (mapped to 0..num_classes-1)
    target_col = 'csat_score' if 'csat_score' in df.columns else 'CSAT'
    y = df[target_col].values
    min_val = y.min()
    y_indexed = y - min_val if min_val > 0 else y
    num_classes = len(np.unique(y_indexed))

    # Preprocessing & Feature Engineering
    df_processed, tfidf_feats, vectorizer = preprocess_and_engineer_features(df)
    num_feats = df_processed[['response_time_mins', 'sentiment_polarity']].fillna(0).values

    scaler = StandardScaler()
    num_feats_scaled = scaler.fit_transform(num_feats)

    X = np.hstack([num_feats_scaled, tfidf_feats])

    # Stratified 70/15/15 Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_indexed, test_size=0.30, stratify=y_indexed, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # SMOTE Training Oversampling
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # One-hot encode targets for Softmax loss
    y_train_cat = to_categorical(y_train_res, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Build DeepCSAT_ANN model
    ann_wrapper = DeepCSAT_ANN(input_dim=X_train_res.shape[1], num_classes=num_classes)
    model = ann_wrapper.model
    early_stopping = ann_wrapper.get_early_stopping(monitor='val_loss', patience=10)

    # Train model
    history = model.fit(
        X_train_res, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Model Evaluation Pipeline
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=== DeepCSAT Model Evaluation Pipeline Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    return model, history, acc, vectorizer, scaler

if __name__ == '__main__':
    run_training_pipeline()
