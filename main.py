#!/usr/bin/env python3
"""
Main Entry Point for ML-Based Intrusion Detection System (IDS)
This script provides the main interface for training and testing the IDS
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.train_model import IDSModelTrainer
from utils.preprocess import DataPreprocessor

class IDSManager:
    """
    Main manager class for the Intrusion Detection System
    """
    
    def __init__(self):
        self.trainer = IDSModelTrainer()
        self.preprocessor = DataPreprocessor()
        self.models_dir = '/home/kali/Desktop/NeuroShield IDS/models'
        self.data_dir = '/home/kali/Desktop/NeuroShield IDS/data'
        
    def train_system(self, dataset_path=None):
        """
        Train the complete IDS system
        
        Args:
            dataset_path (str): Path to the dataset file
        """
        print("="*60)
        print("ML-BASED INTRUSION DETECTION SYSTEM - TRAINING")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Use default dataset if not provided
        if dataset_path is None:
            dataset_path = os.path.join(self.data_dir, 'sample_dataset.csv')
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            print("Please ensure the dataset file exists or provide a valid path")
            return False
        
        print(f"Using dataset: {dataset_path}")
        
        try:
            # Run complete training pipeline
            results = self.trainer.train_complete_pipeline(dataset_path, self.models_dir)
            
            if results:
                print("\n✅ Training completed successfully!")
                print("📁 Models saved to:", self.models_dir)
                print("📊 Evaluation plots saved to:", os.path.join(self.models_dir, 'plots'))
                return True
            else:
                print("\n❌ Training failed!")
                return False
                
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            return False
    
    def test_system(self, test_data_path=None):
        """
        Test the trained IDS system
        
        Args:
            test_data_path (str): Path to test data file
        """
        print("="*60)
        print("ML-BASED INTRUSION DETECTION SYSTEM - TESTING")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if models exist
        model_path = os.path.join(self.models_dir, 'ids_model.pkl')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the system first using: python main.py --train")
            return False
        
        if not os.path.exists(preprocessor_path):
            print(f"Error: Preprocessor not found at {preprocessor_path}")
            print("Please train the system first using: python main.py --train")
            return False
        
        # Load model and preprocessor
        try:
            print("Loading trained model and preprocessor...")
            model = joblib.load(model_path)
            self.preprocessor.load_preprocessor(preprocessor_path)
            print("✅ Model and preprocessor loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model/preprocessor: {e}")
            return False
        
        # Use default test data if not provided
        if test_data_path is None:
            test_data_path = os.path.join(self.data_dir, 'sample_dataset.csv')
        
        if not os.path.exists(test_data_path):
            print(f"Error: Test data not found at {test_data_path}")
            return False
        
        try:
            # Load and preprocess test data
            print(f"Loading test data from: {test_data_path}")
            df = self.preprocessor.load_data(test_data_path)
            
            if df is None:
                print("❌ Failed to load test data")
                return False
            
            # Clean and transform data
            df = self.preprocessor.clean_data(df)
            df = self.preprocessor.encode_categorical_features(df, fit=False)
            
            # Separate features and target
            if 'is_attack' in df.columns:
                X_test = df.drop(columns=['is_attack', 'attack_type'] if 'attack_type' in df.columns else ['is_attack'])
                y_test = df['is_attack']
                
                # Normalize features
                feature_columns = [col for col in X_test.columns if pd.api.types.is_numeric_dtype(X_test[col])]
                X_test = self.preprocessor.normalize_features(X_test, feature_columns, fit=False)
                
                # Make predictions
                print("Making predictions...")
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                
                # Display results
                print("\n📊 TEST RESULTS:")
                print("-" * 40)
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                print("\nConfusion Matrix:")
                print(cm)
                
                # Show prediction distribution
                unique, counts = np.unique(y_pred, return_counts=True)
                print(f"\nPrediction Distribution:")
                for val, count in zip(unique, counts):
                    label = "Attack" if val == 1 else "Normal"
                    print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")
                
                print("\n✅ Testing completed successfully!")
                return True
                
            else:
                # No target column - just make predictions
                print("No target column found. Making predictions on features only...")
                
                # Normalize features
                feature_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                X_test = self.preprocessor.normalize_features(df, feature_columns, fit=False)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Show prediction distribution
                unique, counts = np.unique(y_pred, return_counts=True)
                print(f"\nPrediction Distribution:")
                for val, count in zip(unique, counts):
                    label = "Attack" if val == 1 else "Normal"
                    print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")
                
                print("\n✅ Testing completed successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Testing failed with error: {e}")
            return False
    
    def predict_single(self, features_dict):
        """
        Make prediction on a single sample
        
        Args:
            features_dict (dict): Dictionary of feature values
            
        Returns:
            tuple: (prediction, probability)
        """
        try:
            # Load model and preprocessor
            model_path = os.path.join(self.models_dir, 'ids_model.pkl')
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                return None, None
            
            model = joblib.load(model_path)
            self.preprocessor.load_preprocessor(preprocessor_path)
            
            # Convert to DataFrame
            df = pd.DataFrame([features_dict])
            
            # Transform data
            df = self.preprocessor.transform_new_data(df)
            
            if df is None:
                return None, None
            
            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0, 1] if hasattr(model, 'predict_proba') else None
            
            return prediction, probability
            
        except Exception as e:
            print(f"Error in single prediction: {e}")
            return None, None
    
    def show_system_info(self):
        """
        Display system information and status
        """
        print("="*60)
        print("ML-BASED INTRUSION DETECTION SYSTEM - INFO")
        print("="*60)
        
        # Check system status
        model_path = os.path.join(self.models_dir, 'ids_model.pkl')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        dataset_path = os.path.join(self.data_dir, 'sample_dataset.csv')
        
        print("📁 System Status:")
        print(f"  Dataset: {'✅ Found' if os.path.exists(dataset_path) else '❌ Not found'}")
        print(f"  Model: {'✅ Found' if os.path.exists(model_path) else '❌ Not found'}")
        print(f"  Preprocessor: {'✅ Found' if os.path.exists(preprocessor_path) else '❌ Not found'}")
        
        if os.path.exists(model_path):
            try:
                # Load model metadata
                metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    print(f"\n🤖 Model Information:")
                    print(f"  Best Model: {metadata.get('best_model_name', 'Unknown')}")
                    print(f"  Available Models: {', '.join(metadata.get('model_names', []))}")
                
                # Check model file size
                model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                print(f"  Model Size: {model_size:.2f} MB")
                
            except Exception as e:
                print(f"  Error reading model info: {e}")
        
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                print(f"\n📊 Dataset Information:")
                print(f"  Shape: {df.shape}")
                print(f"  Features: {df.shape[1] - 2}")  # Excluding target columns
                print(f"  Samples: {df.shape[0]}")
                
                if 'is_attack' in df.columns:
                    attack_ratio = df['is_attack'].mean()
                    print(f"  Attack Ratio: {attack_ratio:.2%}")
                
            except Exception as e:
                print(f"  Error reading dataset info: {e}")
        
        print("\n🚀 Usage:")
        print("  Train:    python main.py --train")
        print("  Test:     python main.py --test")
        print("  Info:     python main.py --info")
        print("  Dashboard: streamlit run dashboard/app.py")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='ML-Based Intrusion Detection System (IDS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    # Train the system
  python main.py --test                     # Test the system
  python main.py --train --dataset data.csv # Train with custom dataset
  python main.py --test --data test.csv     # Test with custom data
  python main.py --info                     # Show system information
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train the IDS system')
    parser.add_argument('--test', action='store_true', help='Test the IDS system')
    parser.add_argument('--info', action='store_true', help='Show system information')
    parser.add_argument('--dataset', type=str, help='Path to training dataset')
    parser.add_argument('--data', type=str, help='Path to test data')
    
    args = parser.parse_args()
    
    # Create IDS manager
    ids_manager = IDSManager()
    
    # Execute based on arguments
    if args.train:
        success = ids_manager.train_system(args.dataset)
        sys.exit(0 if success else 1)
        
    elif args.test:
        success = ids_manager.test_system(args.data)
        sys.exit(0 if success else 1)
        
    elif args.info:
        ids_manager.show_system_info()
        
    else:
        # No arguments provided - show help and system info
        parser.print_help()
        print("\n")
        ids_manager.show_system_info()

if __name__ == "__main__":
    main()
