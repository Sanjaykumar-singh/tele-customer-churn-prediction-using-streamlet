# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
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
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .feature-importance {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ChurnPredictorApp:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.load_model()
    
    def load_model(self):
        """Load the trained model and artifacts"""
        try:
            with open('optimized_churn_model.pkl', 'rb') as file:
                artifact = pickle.load(file)
            
            self.model = artifact['model']
            self.feature_names = artifact['feature_names']
            self.optimal_threshold = artifact.get('optimal_threshold', 0.5)
            self.performance = artifact.get('performance', {})
            
            st.sidebar.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Model file 'optimized_churn_model.pkl' not found. Please ensure it's in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
    
    def display_welcome(self):
        """Display welcome section and overview"""
        st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Welcome to Churn Prediction System</h3>
            <p>This interactive dashboard helps you predict customer churn probability using our optimized machine learning model.</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>📊 Single customer prediction</li>
                <li>📁 Batch prediction from CSV files</li>
                <li>📈 Model performance insights</li>
                <li>🎚️ Adjustable prediction threshold</li>
                <li>📤 Download predictions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def display_model_info(self):
        """Display model information in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏆 Model Information")
        
        if hasattr(self, 'performance'):
            st.sidebar.metric("AUC Score", f"{self.performance.get('auc_score', 'N/A'):.3f}")
            st.sidebar.metric("F1-Score (Churn)", f"{self.performance.get('f1_score_minority', 'N/A'):.3f}")
            st.sidebar.metric("Optimal Threshold", f"{self.optimal_threshold:.3f}")
        
        st.sidebar.markdown("### 🔍 Top 10 Features")
        for i, feature in enumerate(self.feature_names, 1):
            st.sidebar.write(f"{i}. {feature}")
    
    def single_prediction_section(self):
        """Section for single customer prediction"""
        st.markdown('<h2 class="sub-header">👤 Single Customer Prediction</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("single_prediction_form"):
            st.markdown("### Enter Customer Details")
            
            # Create columns for better organization
            col1, col2 = st.columns(2)
            
            customer_data = {}
            
            # Generate input fields for all features
            for i, feature in enumerate(self.feature_names):
                if i % 2 == 0:
                    with col1:
                        customer_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.1,
                            key=f"input_{feature}"
                        )
                else:
                    with col2:
                        customer_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.1,
                            key=f"input_{feature}"
                        )
            
            # Threshold adjustment
            st.markdown("### ⚙️ Prediction Settings")
            threshold = st.slider(
                "Prediction Threshold",
                min_value=0.1,
                max_value=0.9,
                value=float(self.optimal_threshold),
                step=0.05,
                help="Higher threshold = more conservative predictions (fewer false positives)"
            )
            
            submitted = st.form_submit_button("🔮 Predict Churn Probability")
            
            if submitted:
                self.predict_single_customer(customer_data, threshold)
    
    def predict_single_customer(self, customer_data, threshold):
        """Predict churn for a single customer"""
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([customer_data])
            
            # Ensure correct feature order
            input_df = input_df[self.feature_names]
            
            # Make prediction
            probability = self.model.predict_proba(input_df)[0, 1]
            prediction = probability >= threshold
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{probability:.1%}")
            
            with col2:
                status = "🚨 HIGH RISK" if prediction else "✅ LOW RISK"
                st.metric("Churn Status", status)
            
            with col3:
                st.metric("Threshold Used", f"{threshold:.1%}")
            
            # Visual indicators
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [probability], color='red' if prediction else 'green', alpha=0.7)
            ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.1%})')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Churn Probability')
            ax.set_yticks([])
            ax.legend()
            st.pyplot(fig)
            
            # Recommendation
            if prediction:
                st.markdown("""
                <div class="warning-box">
                <h4>🎯 Recommended Actions:</h4>
                <ul>
                    <li>📞 Proactive customer outreach</li>
                    <li>🎁 Offer retention incentives</li>
                    <li>🔍 Conduct satisfaction survey</li>
                    <li>⭐ Assign to dedicated account manager</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Customer Status: Stable</h4>
                <p>This customer shows low churn risk. Continue with regular engagement strategies.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    
    def batch_prediction_section(self):
        """Section for batch predictions from CSV files"""
        st.markdown('<h2 class="sub-header">📁 Batch Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Upload CSV File Requirements:</h4>
        <ul>
            <li>File must contain all 10 features used by the model</li>
            <li>Features should be in the same order or have matching column names</li>
            <li>Supported formats: CSV, Excel</li>
            <li>Maximum file size: 200MB</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data for batch prediction"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records")
                
                # Display preview
                st.markdown("### 📄 Data Preview")
                st.dataframe(df.head())
                
                # Check if required features are present
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    st.error(f"❌ Missing features: {list(missing_features)}")
                    return
                
                # Threshold selection for batch
                batch_threshold = st.slider(
                    "Batch Prediction Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=float(self.optimal_threshold),
                    step=0.05,
                    key="batch_threshold"
                )
                
                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Processing predictions..."):
                        results = self.predict_batch(df, batch_threshold)
                    
                    # Display results
                    self.display_batch_results(results, batch_threshold)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    def predict_batch(self, df, threshold):
        """Predict churn for batch data"""
        try:
            # Select and reorder features
            input_data = df[self.feature_names]
            
            # Predict probabilities
            probabilities = self.model.predict_proba(input_data)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            # Create results DataFrame
            results = df.copy()
            results['Churn_Probability'] = probabilities
            results['Churn_Prediction'] = predictions
            results['Risk_Level'] = np.where(
                probabilities >= 0.7, 'High',
                np.where(probabilities >= 0.4, 'Medium', 'Low')
            )
            
            return results
            
        except Exception as e:
            st.error(f"❌ Batch prediction error: {e}")
            return None
    
    def display_batch_results(self, results, threshold):
        """Display batch prediction results"""
        st.markdown("### 📈 Batch Prediction Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(results)
        churn_count = results['Churn_Prediction'].sum()
        churn_rate = churn_count / total_customers
        
        with col1:
            st.metric("Total Customers", total_customers)
        with col2:
            st.metric("Predicted Churns", churn_count)
        with col3:
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with col4:
            st.metric("Threshold Used", f"{threshold:.1%}")
        
        # Risk distribution
        risk_counts = results['Risk_Level'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart for risk levels
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Customer Risk Distribution')
        
        # Histogram of probabilities
        ax2.hist(results['Churn_Probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.1%})')
        ax2.set_xlabel('Churn Probability')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Churn Probability Distribution')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display results table
        st.markdown("### 📋 Detailed Results")
        st.dataframe(results)
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"churn_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    def model_insights_section(self):
        """Section showing model insights and feature importance"""
        st.markdown('<h2 class="sub-header">🔍 Model Insights</h2>', unsafe_allow_html=True)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance in Churn Prediction')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Model performance explanation
        st.markdown("### 🎯 How to Interpret Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>📈 Probability Ranges:</h4>
            <ul>
                <li><strong>0.0 - 0.3:</strong> Low Risk 🟢</li>
                <li><strong>0.3 - 0.7:</strong> Medium Risk 🟡</li>
                <li><strong>0.7 - 1.0:</strong> High Risk 🔴</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>⚙️ Threshold Guidance:</h4>
            <ul>
                <li><strong>Low (0.1-0.3):</strong> Catch all churners (more false positives)</li>
                <li><strong>Medium (0.4-0.6):</strong> Balanced approach</li>
                <li><strong>High (0.7-0.9):</strong> Conservative (fewer false positives)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main method to run the Streamlit app"""
        # Display welcome section
        self.display_welcome()
        
        # Display model info in sidebar
        self.display_model_info()
        
        # Create navigation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Navigation")
        app_section = st.sidebar.radio(
            "Go to:",
            ["Single Prediction", "Batch Prediction", "Model Insights"]
        )
        
        # Display selected section
        if app_section == "Single Prediction":
            self.single_prediction_section()
        elif app_section == "Batch Prediction":
            self.batch_prediction_section()
        elif app_section == "Model Insights":
            self.model_insights_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Customer Churn Prediction System • Built with Streamlit • "
            "Model: Random Forest with Top 10 Features"
            "</div>",
            unsafe_allow_html=True
        )

# Create requirements.txt content
requirements_content = """
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pickle-mixin>=1.0.0
"""

# Create deployment guide in the app
def show_deployment_guide():
    """Show deployment guide in an expander"""
    with st.sidebar.expander("🚀 Deployment Guide"):
        st.markdown("""
        ### Step 1: Install Dependencies
        ```bash
        pip install -r requirements.txt
        ```
        
        ### Step 2: Run the App
        ```bash
        streamlit run app.py
        ```
        
        ### Step 3: Access the App
        Open http://localhost:8501 in your browser
        
        ### Required Files:
        - `app.py` (this file)
        - `optimized_churn_model.pkl` (trained model)
        - `requirements.txt` (dependencies)
        """)

# Main execution
if __name__ == "__main__":
    # Show deployment guide
    show_deployment_guide()
    
    # Initialize and run the app
    app = ChurnPredictorApp()
    app.run()
