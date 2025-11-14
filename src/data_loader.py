"""
Data loading and preprocessing utilities
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def load_wine_data(white_wine_path="data/winequality-white.csv", 
                   red_wine_path="data/winequality-red.csv"):
    """
    Load and combine white and red wine datasets
    
    Returns:
        pd.DataFrame: Combined wine dataset
    """
    white_wine = pd.read_csv(white_wine_path, sep=";")
    red_wine = pd.read_csv(red_wine_path, sep=",")
    
    # Add indicator variable
    red_wine['is_red'] = 1
    white_wine['is_red'] = 0
    
    # Combine datasets
    data = pd.concat([red_wine, white_wine], axis=0)
    
    # Remove spaces from column names
    data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
    
    return data

def prepare_classification_data(data, quality_threshold=7):
    """
    Convert quality scores to binary classification
    
    Args:
        data: Wine dataset
        quality_threshold: Threshold for high quality (default: 7)
    
    Returns:
        pd.DataFrame: Data with binary quality labels
    """
    data = data.copy()
    high_quality = (data.quality >= quality_threshold).astype(int)
    data.quality = high_quality
    return data

def split_data(data, train_size=0.6, random_state=123):
    """
    Split data into train, validation, and test sets
    
    Args:
        data: Wine dataset
        train_size: Proportion for training (default: 0.6)
        random_state: Random seed
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = data.drop(["quality"], axis=1)
    y = data.quality
    
    # Split out the training data
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    
    # Split the remaining data equally into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test