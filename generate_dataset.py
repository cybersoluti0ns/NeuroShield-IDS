#!/usr/bin/env python3
"""
Dataset Generator for ML-Based Intrusion Detection System
Generates synthetic network traffic data in NSL-KDD format
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_dataset(n_samples=10000):
    """
    Generate synthetic network traffic dataset for intrusion detection
    Based on NSL-KDD dataset structure
    """
    np.random.seed(42)
    random.seed(42)
    
    # Protocol types
    protocols = ['tcp', 'udp', 'icmp']
    
    # Service types
    services = ['http', 'ftp', 'smtp', 'ssh', 'telnet', 'dns', 'pop3', 'imap', 'smtp', 'other']
    
    # Flag types
    flags = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'S1', 'S2', 'S3', 'OTH']
    
    # Attack types
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    
    data = []
    
    for i in range(n_samples):
        # Basic connection features
        duration = np.random.exponential(1.0) if random.random() < 0.8 else np.random.exponential(10.0)
        protocol_type = random.choice(protocols)
        service = random.choice(services)
        flag = random.choice(flags)
        
        # Network features
        src_bytes = int(np.random.lognormal(6, 2))
        dst_bytes = int(np.random.lognormal(6, 2))
        
        # Connection features
        land = 0 if random.random() < 0.99 else 1
        wrong_fragment = int(np.random.poisson(0.1))
        urgent = int(np.random.poisson(0.05))
        
        # Host features
        hot = int(np.random.poisson(0.1))
        num_failed_logins = int(np.random.poisson(0.1))
        logged_in = 1 if random.random() < 0.7 else 0
        num_compromised = int(np.random.poisson(0.05))
        root_shell = 1 if random.random() < 0.01 else 0
        su_attempted = 1 if random.random() < 0.01 else 0
        num_root = int(np.random.poisson(0.1))
        num_file_creations = int(np.random.poisson(0.1))
        num_shells = int(np.random.poisson(0.01))
        num_access_files = int(np.random.poisson(0.1))
        num_outbound_cmds = int(np.random.poisson(0.01))
        is_host_login = 1 if random.random() < 0.1 else 0
        is_guest_login = 1 if random.random() < 0.05 else 0
        
        # Count features
        count = int(np.random.poisson(10))
        srv_count = int(np.random.poisson(5))
        serror_rate = np.random.beta(1, 9)
        srv_serror_rate = np.random.beta(1, 9)
        rerror_rate = np.random.beta(1, 9)
        srv_rerror_rate = np.random.beta(1, 9)
        same_srv_rate = np.random.beta(5, 5)
        diff_srv_rate = 1 - same_srv_rate
        srv_diff_host_rate = np.random.beta(1, 9)
        
        # Time-based features
        dst_host_count = int(np.random.poisson(20))
        dst_host_srv_count = int(np.random.poisson(10))
        dst_host_same_srv_rate = np.random.beta(5, 5)
        dst_host_diff_srv_rate = 1 - dst_host_same_srv_rate
        dst_host_same_src_port_rate = np.random.beta(5, 5)
        dst_host_srv_diff_host_rate = np.random.beta(1, 9)
        dst_host_serror_rate = np.random.beta(1, 9)
        dst_host_srv_serror_rate = np.random.beta(1, 9)
        dst_host_rerror_rate = np.random.beta(1, 9)
        dst_host_srv_rerror_rate = np.random.beta(1, 9)
        
        # Determine if this is an attack (20% of data)
        is_attack = random.random() < 0.2
        
        if is_attack:
            # Modify features to simulate attack patterns
            attack_type = random.choice(['dos', 'probe', 'r2l', 'u2r'])
            
            if attack_type == 'dos':
                # DoS attacks: high connection count, low service variety
                count = int(np.random.poisson(100))
                srv_count = int(np.random.poisson(2))
                same_srv_rate = np.random.beta(8, 2)
                dst_host_count = int(np.random.poisson(200))
                dst_host_same_srv_rate = np.random.beta(8, 2)
                
            elif attack_type == 'probe':
                # Probe attacks: many failed connections, high error rates
                num_failed_logins = int(np.random.poisson(5))
                serror_rate = np.random.beta(5, 5)
                srv_serror_rate = np.random.beta(5, 5)
                dst_host_serror_rate = np.random.beta(5, 5)
                
            elif attack_type == 'r2l':
                # R2L attacks: many file access attempts
                num_access_files = int(np.random.poisson(10))
                num_file_creations = int(np.random.poisson(5))
                num_shells = int(np.random.poisson(2))
                
            elif attack_type == 'u2r':
                # U2R attacks: privilege escalation attempts
                num_root = int(np.random.poisson(5))
                root_shell = 1 if random.random() < 0.3 else 0
                su_attempted = 1 if random.random() < 0.2 else 0
        else:
            attack_type = 'normal'
        
        # Create row
        row = {
            'duration': duration,
            'protocol_type': protocol_type,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land,
            'wrong_fragment': wrong_fragment,
            'urgent': urgent,
            'hot': hot,
            'num_failed_logins': num_failed_logins,
            'logged_in': logged_in,
            'num_compromised': num_compromised,
            'root_shell': root_shell,
            'su_attempted': su_attempted,
            'num_root': num_root,
            'num_file_creations': num_file_creations,
            'num_shells': num_shells,
            'num_access_files': num_access_files,
            'num_outbound_cmds': num_outbound_cmds,
            'is_host_login': is_host_login,
            'is_guest_login': is_guest_login,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': srv_rerror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': dst_host_rerror_rate,
            'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate,
            'attack_type': attack_type,
            'is_attack': 1 if is_attack else 0
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    """Generate and save the synthetic dataset"""
    print("Generating synthetic intrusion detection dataset...")
    
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=10000)
    
    # Save to CSV
    output_path = '/home/kali/Desktop/NeuroShield IDS/data/sample_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print(f"\nAttack type distribution:")
    print(df['attack_type'].value_counts())
    print(f"\nAttack vs Normal distribution:")
    print(df['is_attack'].value_counts())

if __name__ == "__main__":
    main()