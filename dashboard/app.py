#!/usr/bin/env python3
"""
Streamlit Dashboard for ML-Based Intrusion Detection System
Interactive web interface for uploading data and detecting intrusions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="NeuroShield IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .attack-prediction {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .normal-prediction {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

class IDSDashboard:
    """
    Streamlit dashboard for the Intrusion Detection System
    """
    
    def __init__(self):
        self.models_dir = '/home/kali/Desktop/NeuroShield IDS/models'
        self.model = None
        self.preprocessor = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            model_path = os.path.join(self.models_dir, 'ids_model.pkl')
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                return False
            
            self.model = joblib.load(model_path)
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict_data(self, df):
        """Make predictions on uploaded data"""
        if not self.model_loaded:
            return None, None
        
        try:
            # Transform the data
            df_transformed = self.preprocessor.transform_new_data(df)
            
            if df_transformed is None:
                return None, None
            
            # Make predictions
            predictions = self.model.predict(df_transformed)
            probabilities = self.model.predict_proba(df_transformed)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            return predictions, probabilities
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return None, None
    
    def create_confusion_matrix_plot(self, y_true, y_pred):
        """Create confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Normal', 'Attack'],
            y=['Normal', 'Attack'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        return fig
    
    def create_metrics_chart(self, metrics):
        """Create metrics comparison chart"""
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f'{v:.3f}' for v in metric_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            width=600,
            height=400
        )
        
        return fig
    
    def create_prediction_distribution(self, predictions):
        """Create prediction distribution chart"""
        unique, counts = np.unique(predictions, return_counts=True)
        labels = ['Normal' if x == 0 else 'Attack' for x in unique]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.3,
                marker_colors=['#4caf50', '#f44336']
            )
        ])
        
        fig.update_layout(
            title="Prediction Distribution",
            width=400,
            height=400
        )
        
        return fig

def main():
    """Main dashboard function"""
    
    # Initialize dashboard
    dashboard = IDSDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ NeuroShield IDS</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Machine Learning Based Intrusion Detection System</h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 System Status")
    
    # Load model
    if dashboard.load_model():
        st.sidebar.success("✅ Model Loaded Successfully")
        
        # Show model info
        try:
            metadata_path = os.path.join(dashboard.models_dir, 'model_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                st.sidebar.info(f"**Best Model:** {metadata.get('best_model_name', 'Unknown')}")
        except:
            pass
    else:
        st.sidebar.error("❌ Model Not Found")
        st.sidebar.warning("Please train the model first using: `python main.py --train`")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Upload & Predict", "📈 Model Performance", "🔍 Live Analysis", "ℹ️ System Info"])
    
    with tab1:
        st.header("📊 Upload Data and Detect Intrusions")
        
        # File upload section
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with network traffic features for intrusion detection"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display sample data
                st.subheader("📋 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Make predictions
                if st.button("🔍 Detect Intrusions", type="primary"):
                    with st.spinner("Analyzing data..."):
                        predictions, probabilities = dashboard.predict_data(df)
                        
                        if predictions is not None:
                            # Add predictions to dataframe
                            df_results = df.copy()
                            df_results['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in predictions]
                            
                            if probabilities is not None:
                                df_results['Attack_Probability'] = probabilities
                            
                            # Display results
                            st.subheader("🎯 Detection Results")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Samples", len(predictions))
                            
                            with col2:
                                attack_count = np.sum(predictions)
                                st.metric("Attacks Detected", attack_count)
                            
                            with col3:
                                normal_count = len(predictions) - attack_count
                                st.metric("Normal Traffic", normal_count)
                            
                            with col4:
                                attack_percentage = (attack_count / len(predictions)) * 100
                                st.metric("Attack Rate", f"{attack_percentage:.1f}%")
                            
                            # Prediction distribution chart
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_dist = dashboard.create_prediction_distribution(predictions)
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with col2:
                                # Show detailed results
                                st.subheader("📊 Detailed Results")
                                st.dataframe(df_results, use_container_width=True)
                            
                            # Download results
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"ids_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Show individual predictions with probabilities
                            if probabilities is not None:
                                st.subheader("🔍 Individual Predictions")
                                
                                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                                    if i < 10:  # Show first 10 predictions
                                        if pred == 1:
                                            st.markdown(f"""
                                            <div class="attack-prediction">
                                                <strong>Sample {i+1}:</strong> 🚨 <strong>ATTACK DETECTED</strong><br>
                                                <strong>Confidence:</strong> {prob:.2%}
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="normal-prediction">
                                                <strong>Sample {i+1}:</strong> ✅ <strong>NORMAL TRAFFIC</strong><br>
                                                <strong>Confidence:</strong> {1-prob:.2%}
                                            </div>
                                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to make predictions. Please check your data format.")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab2:
        st.header("📈 Model Performance")
        
        # Load evaluation results if available
        try:
            # Try to load model metadata
            metadata_path = os.path.join(dashboard.models_dir, 'model_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                
                st.subheader("🤖 Model Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Best Model:** {metadata.get('best_model_name', 'Unknown')}")
                    st.info(f"**Available Models:** {', '.join(metadata.get('model_names', []))}")
                
                with col2:
                    # Show feature importance if available
                    if 'feature_importance' in metadata and metadata['feature_importance']:
                        st.subheader("🎯 Top Feature Importance")
                        importance = metadata['feature_importance']
                        top_features = dict(list(importance.items())[:10])
                        
                        fig_importance = go.Figure(data=[
                            go.Bar(
                                y=list(top_features.keys()),
                                x=list(top_features.values()),
                                orientation='h',
                                marker_color='lightblue'
                            )
                        ])
                        
                        fig_importance.update_layout(
                            title="Top 10 Most Important Features",
                            xaxis_title="Importance Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
            
            # Check for evaluation plots
            plots_dir = os.path.join(dashboard.models_dir, 'plots')
            if os.path.exists(plots_dir):
                st.subheader("📊 Evaluation Plots")
                
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                
                if plot_files:
                    for plot_file in plot_files:
                        plot_path = os.path.join(plots_dir, plot_file)
                        st.image(plot_path, caption=plot_file.replace('.png', '').replace('_', ' ').title())
                else:
                    st.info("No evaluation plots found. Run training to generate plots.")
            else:
                st.info("No evaluation plots found. Run training to generate plots.")
                
        except Exception as e:
            st.error(f"Error loading model performance data: {e}")
    
    with tab3:
        st.header("🔍 Live Analysis")
        
        st.subheader("📊 Real-time Network Analysis")
        st.info("This feature would integrate with live network monitoring tools like Scapy for real-time packet analysis.")
        
        # Simulate live analysis
        if st.button("🎯 Simulate Live Analysis"):
            with st.spinner("Analyzing live network traffic..."):
                # Generate some sample live data
                np.random.seed(42)
                n_samples = 50
                
                # Create sample live data
                live_data = {
                    'duration': np.random.exponential(1.0, n_samples),
                    'protocol_type': np.random.choice([0, 1, 2], n_samples),
                    'service': np.random.choice(range(10), n_samples),
                    'flag': np.random.choice(range(9), n_samples),
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
                
                live_df = pd.DataFrame(live_data)
                
                # Make predictions
                predictions, probabilities = dashboard.predict_data(live_df)
                
                if predictions is not None:
                    # Display live results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Live Detection Results")
                        
                        attack_count = np.sum(predictions)
                        normal_count = len(predictions) - attack_count
                        
                        st.metric("Attacks Detected", attack_count, delta=f"{attack_count/len(predictions)*100:.1f}%")
                        st.metric("Normal Traffic", normal_count, delta=f"{normal_count/len(predictions)*100:.1f}%")
                        
                        # Live prediction chart
                        fig_live = dashboard.create_prediction_distribution(predictions)
                        st.plotly_chart(fig_live, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Recent Alerts")
                        
                        # Show recent predictions with timestamps
                        for i, (pred, prob) in enumerate(zip(predictions[-10:], probabilities[-10:] if probabilities is not None else [None]*10)):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            
                            if pred == 1:
                                st.error(f"🚨 {timestamp} - ATTACK DETECTED (Confidence: {prob:.1%})" if prob else f"🚨 {timestamp} - ATTACK DETECTED")
                            else:
                                st.success(f"✅ {timestamp} - Normal Traffic (Confidence: {1-prob:.1%})" if prob else f"✅ {timestamp} - Normal Traffic")
    
    with tab4:
        st.header("ℹ️ System Information")
        
        # System status
        st.subheader("🔧 System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Model Status:** ✅ Loaded and Ready")
            st.info("**Preprocessor:** ✅ Loaded and Ready")
            
            # Check dataset
            dataset_path = '/home/kali/Desktop/NeuroShield IDS/data/sample_dataset.csv'
            if os.path.exists(dataset_path):
                st.info("**Dataset:** ✅ Available")
                try:
                    df_info = pd.read_csv(dataset_path)
                    st.info(f"**Dataset Size:** {df_info.shape[0]} samples, {df_info.shape[1]} features")
                except:
                    st.warning("**Dataset:** ⚠️ Found but unreadable")
            else:
                st.warning("**Dataset:** ❌ Not found")
        
        with col2:
            # Model information
            try:
                metadata_path = os.path.join(dashboard.models_dir, 'model_metadata.pkl')
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    st.info(f"**Best Model:** {metadata.get('best_model_name', 'Unknown')}")
                    st.info(f"**Model Type:** Random Forest Classifier")
                else:
                    st.info("**Model Type:** Random Forest Classifier")
            except:
                st.info("**Model Type:** Random Forest Classifier")
        
        # Feature information
        st.subheader("📋 Feature Information")
        
        feature_info = """
        The IDS system analyzes the following network traffic features:
        
        **Basic Features:**
        - Duration: Connection duration
        - Protocol Type: TCP, UDP, ICMP
        - Service: Network service type
        - Flag: Connection status flag
        
        **Traffic Features:**
        - Source/Destination bytes
        - Connection counts
        - Error rates
        - Service rates
        
        **Host Features:**
        - Failed login attempts
        - File access patterns
        - Shell access attempts
        - Root access attempts
        
        **Time-based Features:**
        - Same service rates
        - Different service rates
        - Host connection patterns
        """
        
        st.markdown(feature_info)
        
        # Usage instructions
        st.subheader("🚀 Usage Instructions")
        
        usage_info = """
        1. **Upload Data:** Use the "Upload & Predict" tab to upload CSV files with network traffic data
        2. **View Results:** The system will analyze the data and show intrusion detection results
        3. **Download Results:** Save the prediction results as CSV files
        4. **Monitor Performance:** Check model performance metrics in the "Model Performance" tab
        5. **Live Analysis:** Use the "Live Analysis" tab for real-time monitoring simulation
        
        **Data Format:** Upload CSV files with the same feature structure as the training data.
        The system will automatically preprocess and normalize the data before making predictions.
        """
        
        st.markdown(usage_info)
        
        # Contact information
        st.subheader("📞 Support")
        st.info("For technical support or questions about the IDS system, please refer to the documentation or contact the system administrator.")

if __name__ == "__main__":
    main()

