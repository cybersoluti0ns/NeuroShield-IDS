#!/usr/bin/env python3
"""
Demo script for NeuroShield IDS
Demonstrates the key features of the ML-Based Intrusion Detection System
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import IDSManager

def demo_single_prediction():
    """Demonstrate single prediction functionality"""
    print("🔍 DEMO: Single Prediction")
    print("=" * 50)
    
    # Initialize IDS manager
    ids_manager = IDSManager()
    
    # Create a sample network connection
    sample_connection = {
        'duration': 1.5,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 1024,
        'dst_bytes': 2048,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 10,
        'srv_count': 5,
        'serror_rate': 0.1,
        'srv_serror_rate': 0.05,
        'rerror_rate': 0.02,
        'srv_rerror_rate': 0.01,
        'same_srv_rate': 0.8,
        'diff_srv_rate': 0.2,
        'srv_diff_host_rate': 0.1,
        'dst_host_count': 20,
        'dst_host_srv_count': 10,
        'dst_host_same_srv_rate': 0.7,
        'dst_host_diff_srv_rate': 0.3,
        'dst_host_same_src_port_rate': 0.6,
        'dst_host_srv_diff_host_rate': 0.15,
        'dst_host_serror_rate': 0.05,
        'dst_host_srv_serror_rate': 0.02,
        'dst_host_rerror_rate': 0.01,
        'dst_host_srv_rerror_rate': 0.005
    }
    
    # Make prediction
    prediction, probability = ids_manager.predict_single(sample_connection)
    
    if prediction is not None:
        result = "🚨 ATTACK DETECTED" if prediction == 1 else "✅ NORMAL TRAFFIC"
        confidence = f"{probability:.1%}" if probability is not None else "N/A"
        
        print(f"Sample Connection Analysis:")
        print(f"  Protocol: {sample_connection['protocol_type']}")
        print(f"  Service: {sample_connection['service']}")
        print(f"  Duration: {sample_connection['duration']} seconds")
        print(f"  Result: {result}")
        print(f"  Confidence: {confidence}")
    else:
        print("❌ Prediction failed - model not loaded")
    
    print()

def demo_batch_prediction():
    """Demonstrate batch prediction functionality"""
    print("📊 DEMO: Batch Prediction")
    print("=" * 50)
    
    # Create sample batch data
    np.random.seed(42)
    n_samples = 5
    
    batch_data = {
        'duration': np.random.exponential(1.0, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples),
        'src_bytes': np.random.lognormal(6, 2, n_samples).astype(int),
        'dst_bytes': np.random.lognormal(6, 2, n_samples).astype(int),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.poisson(0.1, n_samples),
        'urgent': np.random.poisson(0.05, n_samples),
        'hot': np.random.poisson(0.1, n_samples),
        'num_failed_logins': np.random.poisson(0.1, n_samples),
        'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'num_compromised': np.random.poisson(0.05, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'num_root': np.random.poisson(0.1, n_samples),
        'num_file_creations': np.random.poisson(0.1, n_samples),
        'num_shells': np.random.poisson(0.01, n_samples),
        'num_access_files': np.random.poisson(0.1, n_samples),
        'num_outbound_cmds': np.random.poisson(0.01, n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'count': np.random.poisson(10, n_samples),
        'srv_count': np.random.poisson(5, n_samples),
        'serror_rate': np.random.beta(1, 9, n_samples),
        'srv_serror_rate': np.random.beta(1, 9, n_samples),
        'rerror_rate': np.random.beta(1, 9, n_samples),
        'srv_rerror_rate': np.random.beta(1, 9, n_samples),
        'same_srv_rate': np.random.beta(5, 5, n_samples),
        'diff_srv_rate': np.random.beta(1, 9, n_samples),
        'srv_diff_host_rate': np.random.beta(1, 9, n_samples),
        'dst_host_count': np.random.poisson(20, n_samples),
        'dst_host_srv_count': np.random.poisson(10, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5, 5, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(1, 9, n_samples),
        'dst_host_same_src_port_rate': np.random.beta(5, 5, n_samples),
        'dst_host_srv_diff_host_rate': np.random.beta(1, 9, n_samples),
        'dst_host_serror_rate': np.random.beta(1, 9, n_samples),
        'dst_host_srv_serror_rate': np.random.beta(1, 9, n_samples),
        'dst_host_rerror_rate': np.random.beta(1, 9, n_samples),
        'dst_host_srv_rerror_rate': np.random.beta(1, 9, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(batch_data)
    
    # Save to temporary CSV
    temp_file = '/tmp/demo_batch.csv'
    df.to_csv(temp_file, index=False)
    
    print(f"Created batch dataset with {n_samples} samples")
    print("Sample data:")
    print(df[['protocol_type', 'service', 'duration', 'src_bytes', 'dst_bytes']].head())
    
    # Test the system
    ids_manager = IDSManager()
    success = ids_manager.test_system(temp_file)
    
    if success:
        print("✅ Batch prediction completed successfully!")
    else:
        print("❌ Batch prediction failed")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print()

def demo_system_info():
    """Demonstrate system information"""
    print("ℹ️ DEMO: System Information")
    print("=" * 50)
    
    ids_manager = IDSManager()
    ids_manager.show_system_info()
    print()

def main():
    """Run all demonstrations"""
    print("🛡️ NeuroShield IDS - Demo Script")
    print("=" * 60)
    print()
    
    # Check if system is ready
    model_path = '/home/kali/Desktop/NeuroShield IDS/models/ids_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the system first:")
        print("   python3 main.py --train")
        return
    
    try:
        # Run demonstrations
        demo_system_info()
        demo_single_prediction()
        demo_batch_prediction()
        
        print("🎉 Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Launch the dashboard: streamlit run dashboard/app.py")
        print("2. Upload your own CSV files for analysis")
        print("3. Explore the model performance visualizations")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")

if __name__ == "__main__":
    main()
