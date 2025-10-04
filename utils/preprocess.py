#!/usr/bin/env python3
"""
Data Preprocessing Module for ML-Based Intrusion Detection System
Handles data cleaning, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    """
    Data preprocessing class for intrusion detection dataset
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Cleaning dataset...")
        
        # Remove duplicate rows
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values with median for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill missing values with mode for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Handle infinite values in numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values in numerical columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        print(f"Dataset cleaned. Final shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df (pd.DataFrame): Input dataset
            fit (bool): Whether to fit the encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        print("Encoding categorical features...")
        
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    # Fit new encoder
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"Fitted encoder for {col}")
                else:
                    # Use existing encoder
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        unique_values = set(df[col].astype(str).unique())
                        known_values = set(le.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            print(f"Warning: Unknown values in {col}: {unknown_values}")
                            # Replace unknown values with most frequent known value
                            df[col] = df[col].astype(str)
                            for val in unknown_values:
                                df[col] = df[col].replace(val, le.classes_[0])
                        
                        df[col] = le.transform(df[col].astype(str))
                    else:
                        print(f"Warning: No encoder found for {col}")
        
        return df
    
    def normalize_features(self, df, feature_columns=None, fit=True):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataset
            feature_columns (list): List of columns to normalize
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        print("Normalizing features...")
        
        if feature_columns is None:
            # Exclude target columns and categorical columns
            exclude_cols = ['attack_type', 'is_attack']
            feature_columns = [col for col in df.columns 
                             if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # Ensure we only normalize numeric columns
        numeric_columns = []
        for col in feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        if not numeric_columns:
            print("Warning: No numeric columns found for normalization")
            return df
        
        if fit:
            # Fit new scaler
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            self.is_fitted = True
            print(f"Fitted scaler for {len(numeric_columns)} features")
        else:
            # Use existing scaler
            if self.is_fitted:
                df[numeric_columns] = self.scaler.transform(df[numeric_columns])
                print(f"Applied existing scaler to {len(numeric_columns)} features")
            else:
                print("Warning: Scaler not fitted yet")
        
        return df
    
    def split_data(self, df, target_column='is_attack', test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data with test_size={test_size}")
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'attack_type'] if 'attack_type' in df.columns else [target_column])
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to the CSV file
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor)
        """
        print("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Normalize features (only numeric columns)
        df = self.normalize_features(df, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df, test_size=test_size, random_state=random_state
        )
        
        print("Preprocessing pipeline completed successfully!")
        return X_train, X_test, y_train, y_test, self
    
    def save_preprocessor(self, file_path):
        """
        Save the preprocessor (encoders and scaler) to disk
        
        Args:
            file_path (str): Path to save the preprocessor
        """
        try:
            preprocessor_data = {
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }
            joblib.dump(preprocessor_data, file_path)
            print(f"Preprocessor saved to {file_path}")
        except Exception as e:
            print(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, file_path):
        """
        Load the preprocessor (encoders and scaler) from disk
        
        Args:
            file_path (str): Path to load the preprocessor from
        """
        try:
            preprocessor_data = joblib.load(file_path)
            self.label_encoders = preprocessor_data['label_encoders']
            self.scaler = preprocessor_data['scaler']
            self.is_fitted = preprocessor_data['is_fitted']
            print(f"Preprocessor loaded from {file_path}")
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
    
    def transform_new_data(self, df):
        """
        Transform new data using fitted preprocessor
        
        Args:
            df (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            print("Error: Preprocessor not fitted yet")
            return None
        
        # Clean data
        df = self.clean_data(df.copy())
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Normalize features
        feature_columns = [col for col in df.columns 
                          if col not in ['attack_type', 'is_attack'] and pd.api.types.is_numeric_dtype(df[col])]
        df = self.normalize_features(df, feature_columns, fit=False)
        
        return df

def main():
    """Test the preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Test with sample dataset
    file_path = '/home/kali/Desktop/NeuroShield IDS/data/sample_dataset.csv'
    
    if os.path.exists(file_path):
        result = preprocessor.preprocess_pipeline(file_path)
        if result:
            X_train, X_test, y_train, y_test, preprocessor = result
            print("Preprocessing test completed successfully!")
            
            # Save preprocessor
            preprocessor.save_preprocessor('/home/kali/Desktop/NeuroShield IDS/models/preprocessor.pkl')
    else:
        print(f"Dataset file not found: {file_path}")

if __name__ == "__main__":
    main()
