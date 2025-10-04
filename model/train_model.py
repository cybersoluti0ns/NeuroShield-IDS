#!/usr/bin/env python3
"""
Model Training Module for ML-Based Intrusion Detection System
Trains and saves machine learning models for intrusion detection
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import DataPreprocessor

class IDSModelTrainer:
    """
    Model trainer class for intrusion detection system
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_importance = None
        
    def load_preprocessed_data(self, file_path):
        """
        Load and preprocess data for training
        
        Args:
            file_path (str): Path to the dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor)
        """
        print("Loading and preprocessing data...")
        
        self.preprocessor = DataPreprocessor()
        result = self.preprocessor.preprocess_pipeline(file_path)
        
        if result is None:
            print("Error: Failed to preprocess data")
            return None
        
        X_train, X_test, y_train, y_test, preprocessor = result
        return X_train, X_test, y_train, y_test, preprocessor
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models for comparison
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print("Training multiple models...")
        
        # Define models to train
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} training completed")
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation results for all models
        """
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate AUC if probabilities are available
            auc = None
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def select_best_model(self, results):
        """
        Select the best model based on F1 score
        
        Args:
            results (dict): Evaluation results
            
        Returns:
            str: Name of the best model
        """
        print("Selecting best model...")
        
        best_f1 = 0
        best_model_name = None
        
        for name, metrics in results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"Best model: {best_model_name} (F1 Score: {best_f1:.4f})")
        return best_model_name
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='RandomForest'):
        """
        Perform hyperparameter tuning for the best model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            model_name (str): Name of the model to tune
        """
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'DecisionTree':
            param_grid = {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            base_model = DecisionTreeClassifier(random_state=42)
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            base_model = SVC(random_state=42, probability=True)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def get_feature_importance(self, model_name='RandomForest'):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Feature importance scores
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            self.feature_importance = sorted_importance
            return sorted_importance
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def save_models(self, models_dir='/home/kali/Desktop/NeuroShield IDS/models'):
        """
        Save all trained models and preprocessor
        
        Args:
            models_dir (str): Directory to save models
        """
        print("Saving models...")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{name.lower()}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save the best model as the main IDS model
        if self.best_model is not None:
            main_model_path = os.path.join(models_dir, 'ids_model.pkl')
            joblib.dump(self.best_model, main_model_path)
            print(f"Saved best model ({self.best_model_name}) to {main_model_path}")
        
        # Save preprocessor
        if self.preprocessor is not None:
            preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
            self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'model_names': list(self.models.keys()),
            'feature_importance': self.feature_importance
        }
        
        metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"Saved model metadata to {metadata_path}")
    
    def plot_results(self, results, save_path='/home/kali/Desktop/NeuroShield IDS/models'):
        """
        Plot evaluation results and confusion matrices
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save plots
        """
        print("Creating evaluation plots...")
        
        # Create plots directory
        plots_dir = os.path.join(save_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Model comparison
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [results[name][metric] for name in model_names]
            
            bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Confusion matrices
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices', fontsize=16)
        
        for i, (name, metrics) in enumerate(results.items()):
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Feature importance (if available)
        if self.feature_importance:
            top_features = dict(list(self.feature_importance.items())[:15])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importance = list(top_features.values())
            
            plt.barh(features, importance, color='skyblue')
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def train_complete_pipeline(self, file_path, models_dir='/home/kali/Desktop/NeuroShield IDS/models'):
        """
        Complete training pipeline
        
        Args:
            file_path (str): Path to the dataset
            models_dir (str): Directory to save models
        """
        print("Starting complete training pipeline...")
        
        # Load and preprocess data
        data_result = self.load_preprocessed_data(file_path)
        if data_result is None:
            return None
        
        X_train, X_test, y_train, y_test, preprocessor = data_result
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Select best model
        best_model_name = self.select_best_model(results)
        
        # Hyperparameter tuning for best model
        self.hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # Re-evaluate after tuning
        print("Re-evaluating after hyperparameter tuning...")
        results = self.evaluate_models(X_test, y_test)
        best_model_name = self.select_best_model(results)
        
        # Get feature importance
        self.get_feature_importance(best_model_name)
        
        # Save models
        self.save_models(models_dir)
        
        # Create plots
        self.plot_results(results, models_dir)
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
            if metrics['auc']:
                print(f"  AUC:       {metrics['auc']:.4f}")
        
        print(f"\nBest Model: {self.best_model_name}")
        print("="*50)
        
        return results

def main():
    """Main function to run the training pipeline"""
    trainer = IDSModelTrainer()
    
    # Path to the dataset
    dataset_path = '/home/kali/Desktop/NeuroShield IDS/data/sample_dataset.csv'
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please run generate_dataset.py first to create the dataset")
        return
    
    # Run complete training pipeline
    results = trainer.train_complete_pipeline(dataset_path)
    
    if results:
        print("\nTraining completed successfully!")
        print("Models and evaluation results saved to the models directory")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()


