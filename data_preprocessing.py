import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

# ============================================================
# NLTK SETUP (FIXES punkt_tab ERROR PERMANENTLY)
# ============================================================
REQUIRED_NLTK_RESOURCES = [
    "stopwords",
    "punkt",
    "punkt_tab"
]

for resource in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "data_preprocessing.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# TEXT TRANSFORMATION FUNCTION
# ============================================================
def transform_text(text):
    """
    Normalize text:
    - lowercase
    - tokenize
    - remove stopwords & punctuation
    - stemming
    """
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)

    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word.isalnum() and word not in STOP_WORDS
    ]

    return " ".join(cleaned_tokens)

# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================
def preprocess_df(df, text_column="text", target_column="target"):
    try:
        logger.debug("Starting preprocessing for DataFrame")

        if text_column not in df.columns:
            raise KeyError(f"Missing text column: {text_column}")

        if target_column not in df.columns:
            raise KeyError(f"Missing target column: {target_column}")

        # Encode target
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        # Remove duplicates
        df = df.drop_duplicates()
        logger.debug("Duplicates removed")

        # Text normalization
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")

        return df

    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise

# ============================================================
# MAIN PIPELINE
# ============================================================
def main(text_column="text", target_column="target"):
    try:
        # Load data
        train_path = "./data/raw/train.csv"
        test_path = "./data/raw/test.csv"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Train or test file not found in data/raw")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug("Data loaded properly")

        # Preprocess
        train_processed = preprocess_df(train_data, text_column, target_column)
        test_processed = preprocess_df(test_data, text_column, target_column)

        # Save processed data
        output_dir = "./data/interim"
        os.makedirs(output_dir, exist_ok=True)

        train_processed.to_csv(
            os.path.join(output_dir, "train_processed.csv"),
            index=False
        )
        test_processed.to_csv(
            os.path.join(output_dir, "test_processed.csv"),
            index=False
        )

        logger.debug("Processed data saved successfully")

    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        raise

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
