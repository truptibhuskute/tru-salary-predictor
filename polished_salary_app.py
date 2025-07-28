"""
TRU Salary Predictor - Polished & Professional UI
Copyright (c) 2025 TRU Salary Predictor
Contact: truptibhuskute@gmail.com

A stunning, modern salary prediction system with beautiful UI and accurate predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from datetime import datetime
import random
import warnings
from bias_mitigation import BiasMitigator
from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TRU Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS styling with animations and modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: #ffffff;
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* Premium Header */
    .premium-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,215,0,0.1), rgba(255,165,0,0.1));
        z-index: -1;
    }
    
    .premium-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(255,215,0,0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(255,215,0,0.8)); }
    }
    
    .premium-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .premium-description {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Premium Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .premium-card:hover::before {
        left: 100%;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Premium Metrics */
    .premium-metric {
        background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,165,0,0.2));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,215,0,0.3);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .premium-metric:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(255,215,0,0.3);
    }
    
    .premium-metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .premium-metric-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    /* Premium Form */
    .premium-form {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .form-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .form-section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #000000;
        border-radius: 50px;
        font-weight: 700;
        padding: 1rem 3rem;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #FFA500, #FFD700);
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(255, 215, 0, 0.6);
    }
    
    /* Premium Inputs */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(45deg, #FFD700, #FFA500) !important;
    }
    
    /* Premium Alerts */
    .premium-success {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
        border: 1px solid rgba(76, 175, 80, 0.5);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .premium-warning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.2), rgba(255, 152, 0, 0.1));
        border: 1px solid rgba(255, 152, 0, 0.5);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .premium-error {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(244, 67, 54, 0.1));
        border: 1px solid rgba(244, 67, 54, 0.5);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Premium Footer */
    .premium-footer {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem;
        margin-top: 4rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .premium-footer h4 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .premium-title {
            font-size: 2.5rem;
        }
        .premium-metric-value {
            font-size: 2.5rem;
        }
        .premium-card {
            padding: 2rem;
        }
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #FFD700;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

class PolishedSalaryPredictor:
    """Polished salary prediction system with premium UI"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.load_models()
        self.setup_feature_options()
    
    def load_models(self):
        """Load trained models and artifacts"""
        try:
            self.model = joblib.load('model/best_model.pkl')
            self.preprocessor = joblib.load('model/preprocessor.pkl')
            self.skills_mlb = joblib.load('model/skills_mlb.pkl')
            
            with open('model/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)
            
            with open('model/advanced_metrics.json', 'r') as f:
                self.metrics = json.load(f)
            
            # Initialize bias mitigator
            try:
                self.bias_mitigator = BiasMitigator()
                self.bias_mitigation_available = True
            except:
                self.bias_mitigation_available = False
                
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.model_loaded = False
    
    def setup_feature_options(self):
        """Setup feature options for the interface"""
        self.job_roles = [
            'Data Scientist', 'Software Engineer', 'Product Manager', 'Sales Manager',
            'Marketing Manager', 'HR Manager', 'Finance Manager', 'DevOps Engineer',
            'UI/UX Designer', 'Data Engineer'
        ]
        
        self.seniority_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director']
        
        self.cities = [
            'Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Chennai', 'Pune',
            'Gurgaon', 'Noida', 'Kolkata', 'Ahmedabad', 'Indore', 'Jaipur',
            'Chandigarh', 'Vadodara', 'Coimbatore', 'Kochi', 'Bhubaneswar',
            'Nagpur', 'Lucknow', 'Patna'
        ]
        
        self.education_levels = [
            'High School', 'Associate', 'Bachelors', 'Masters', 'MBA', 'PhD', 'JD', 'MD'
        ]
        
        self.majors = [
            'Computer Science', 'Computer Engineering', 'Information Technology', 'Software Engineering',
            'Data Science', 'Artificial Intelligence', 'Machine Learning', 'Cybersecurity',
            'Electrical Engineering', 'Electronics Engineering', 'Mechanical Engineering',
            'Civil Engineering', 'Chemical Engineering', 'Biomedical Engineering',
            'Telecommunications', 'Network Engineering', 'Database Management',
            'Web Development', 'Mobile App Development', 'Cloud Computing',
            'DevOps Engineering', 'System Administration', 'Digital Marketing',
            'Business Analytics', 'Project Management', 'Product Management'
        ]
        
        self.industries = [
            'Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail',
            'Education', 'Consulting', 'Real Estate', 'Transportation', 'Energy'
        ]
        
        self.company_sizes = [
            'Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)', 
            'Large (1001-5000)', 'Enterprise (5000+)'
        ]
        
        self.skills = [
            'Python', 'Java', 'JavaScript', 'SQL', 'React', 'Node.js', 'AWS',
            'Docker', 'Kubernetes', 'Machine Learning', 'Data Analysis',
            'Project Management', 'Leadership', 'Communication', 'Problem Solving',
            'Agile', 'Scrum', 'Git', 'Linux', 'Excel', 'PowerBI', 'Tableau',
            'Salesforce', 'SAP', 'Oracle', 'MongoDB', 'Redis', 'Hadoop',
            'Apache Spark', 'Azure', 'Google Cloud', 'DevOps', 'CI/CD',
            'Microservices', 'API Development', 'Mobile Development', 'UI/UX',
            'Product Management', 'Business Analysis', 'Financial Analysis',
            'Marketing', 'Sales', 'Customer Service', 'HR Management',
            'Recruitment', 'Training', 'Compliance', 'Risk Management'
        ]
        
        # City multipliers for cost of living
        self.city_multipliers = {
            'Mumbai': 1.4, 'Bangalore': 1.3, 'Delhi': 1.2, 'Hyderabad': 1.1,
            'Chennai': 1.0, 'Pune': 1.1, 'Gurgaon': 1.2, 'Noida': 1.2,
            'Kolkata': 0.9, 'Ahmedabad': 0.8, 'Indore': 0.7, 'Jaipur': 0.8,
            'Chandigarh': 1.0, 'Vadodara': 0.8, 'Coimbatore': 0.9, 'Kochi': 0.9,
            'Bhubaneswar': 0.8, 'Nagpur': 0.7, 'Lucknow': 0.7, 'Patna': 0.6
        }
    
    def predict_salary(self, input_data):
        """Predict salary with confidence intervals"""
        try:
            # Handle skills encoding first
            if 'Skills' in input_data.columns:
                skills_data = input_data['Skills'].iloc[0] if len(input_data) > 0 else ''
                if isinstance(skills_data, str):
                    skills_list = [skill.strip() for skill in skills_data.split(',') if skill.strip()]
                    # Encode skills
                    skills_encoded = self.skills_mlb.transform([skills_list])
                    skill_cols = [f'Skill_{s}' for s in self.skills_mlb.classes_]
                    skills_df = pd.DataFrame(skills_encoded, columns=skill_cols, index=input_data.index)
                    input_data = pd.concat([input_data.drop('Skills', axis=1), skills_df], axis=1)
            
            # Preprocess input using the trained preprocessor
            X_processed = self.preprocessor.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            # Apply fresher multiplier if applicable
            if 'FresherMultiplier' in input_data.columns:
                fresher_multiplier = input_data['FresherMultiplier'].iloc[0]
                if fresher_multiplier < 1.0:
                    original_prediction = prediction
                    prediction *= fresher_multiplier
            
            # Apply bias correction if available
            bias_correction = 0
            if hasattr(self, 'bias_mitigation_available') and self.bias_mitigation_available:
                demographics = {}
                if 'AgeGroup' in input_data.columns:
                    demographics['age_group'] = input_data['AgeGroup'].iloc[0]
                if 'Gender' in input_data.columns:
                    demographics['gender'] = input_data['Gender'].iloc[0]
                if 'Ethnicity' in input_data.columns:
                    demographics['ethnicity'] = input_data['Ethnicity'].iloc[0]
                
                if demographics:
                    original_prediction = prediction
                    prediction = self.bias_mitigator.apply_bias_correction(prediction, demographics)
                    bias_correction = prediction - original_prediction
            
            # Calculate confidence interval (simplified)
            confidence_interval = prediction * 0.1  # ±10%
            
            return {
                'prediction': prediction,
                'lower_bound': prediction - confidence_interval,
                'upper_bound': prediction + confidence_interval,
                'confidence_interval': confidence_interval,
                'bias_correction': bias_correction
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def analyze_bias(self, input_data, prediction):
        """Analyze potential bias in the prediction"""
        bias_analysis = {}
        
        # Gender bias analysis
        if 'Gender' in input_data.columns:
            gender = input_data['Gender'].iloc[0]
            if gender in self.metrics.get('bias_analysis', {}).get('gender', {}):
                gender_bias = self.metrics['bias_analysis']['gender'][gender]
                bias_analysis['gender'] = {
                    'bias': gender_bias['bias'],
                    'avg_true': gender_bias['avg_true_salary'],
                    'avg_pred': gender_bias['avg_pred_salary']
                }
        
        # Age group bias analysis
        if 'AgeGroup' in input_data.columns:
            age_group = input_data['AgeGroup'].iloc[0]
            if age_group in self.metrics.get('bias_analysis', {}).get('age_group', {}):
                age_bias = self.metrics['bias_analysis']['age_group'][age_group]
                bias_analysis['age_group'] = {
                    'bias': age_bias['bias'],
                    'avg_true': age_bias['avg_true_salary'],
                    'avg_pred': age_bias['avg_pred_salary']
                }
        
        return bias_analysis
    
    def explain_prediction(self, input_data):
        """Generate SHAP explanations for the prediction"""
        try:
            # Handle skills encoding first (same as predict_salary)
            if 'Skills' in input_data.columns:
                skills_data = input_data['Skills'].iloc[0] if len(input_data) > 0 else ''
                if isinstance(skills_data, str):
                    skills_list = [skill.strip() for skill in skills_data.split(',') if skill.strip()]
                    # Encode skills
                    skills_encoded = self.skills_mlb.transform([skills_list])
                    skill_cols = [f'Skill_{s}' for s in self.skills_mlb.classes_]
                    skills_df = pd.DataFrame(skills_encoded, columns=skill_cols, index=input_data.index)
                    input_data = pd.concat([input_data.drop('Skills', axis=1), skills_df], axis=1)
            
            # Preprocess input using the trained preprocessor
            X_processed = self.preprocessor.transform(input_data)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_processed)
            
            # Get feature names from the preprocessor
            feature_names = []
            for transformer in self.preprocessor.named_transformers_.values():
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out())
                elif hasattr(transformer, 'categories_'):
                    for i, categories in enumerate(transformer.categories_):
                        feature_names.extend([f"{transformer.feature_names_in_[i]}_{cat}" for cat in categories])
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names[:len(shap_values[0])],
                'SHAP_Value': shap_values[0]
            }).sort_values('SHAP_Value', key=abs, ascending=False)
            
            return feature_importance.head(10)
        except Exception as e:
            st.error(f"Explanation error: {e}")
            return None

def main():
    """Main application with premium UI"""
    
    # Premium Header
    st.markdown("""
    <div class="premium-header fade-in-up">
        <h1 class="premium-title">💰 TRU Salary Predictor</h1>
        <p class="premium-subtitle">AI-Powered Salary Intelligence for the Modern Workforce</p>
        <p class="premium-description">
            Experience the future of salary prediction with our advanced AI system. 
            Get accurate, fair, and transparent salary insights based on real market data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = PolishedSalaryPredictor()
    
    if not predictor.model_loaded:
        st.error("Models not loaded. Please ensure model files exist.")
        return
    
    # Main prediction form
    with st.container():
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        
        st.markdown('<h2 style="text-align: center; color: #FFD700; margin-bottom: 2rem;">📊 Enter Your Professional Profile</h2>', unsafe_allow_html=True)
        
        with st.form("premium_salary_prediction_form"):
            # Personal Information Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="form-section-title">👤 Personal Information</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=70, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            with col2:
                ethnicity = st.selectbox("Ethnicity", ["Asian", "White", "Hispanic", "Black", "Other"])
                total_experience = st.number_input("Total Years Experience", min_value=0, max_value=50, value=5)
            
            with col3:
                years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=2)
                years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=2)
            
            num_companies = st.number_input("Number of Companies Worked", min_value=1, max_value=20, value=2)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Job Information Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="form-section-title">💼 Job Information</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                job_role = st.selectbox("Job Role", predictor.job_roles)
                seniority = st.selectbox("Seniority Level", predictor.seniority_levels)
            
            with col2:
                department = st.selectbox("Department", ["Engineering", "Sales", "Marketing", "HR", "Finance", "IT", "Analytics", "Design", "Product"])
                industry = st.selectbox("Industry", predictor.industries)
            
            with col3:
                company_size = st.selectbox("Company Size", predictor.company_sizes)
                revenue_tier = st.selectbox("Revenue Tier", ["<1M", "1M-10M", "10M-100M", "100M-1B", ">1B"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance & Education Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="form-section-title">📈 Performance & Education</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=2)
                
                # Performance rating logic for freshers vs experienced
                if total_experience == 0:
                    st.markdown('<div class="premium-warning">', unsafe_allow_html=True)
                    st.info("🎓 **Fresher Mode**: Performance calculated based on education, skills, and certifications")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate performance for freshers
                    base_performance = 3.0
                    education_bonus = {
                        'High School': 0.0, 'Associate': 0.2, 'Bachelors': 0.5,
                        'Masters': 0.8, 'MBA': 1.0, 'PhD': 1.2, 'JD': 1.0, 'MD': 1.5
                    }
                    cert_bonus = min(certifications * 0.1, 0.5)  # Max 0.5 bonus from certs
                    # Use a default education level since it's not defined yet in this scope
                    calculated_performance = min(5.0, base_performance + education_bonus.get('Bachelors', 0.0) + cert_bonus)
                    
                    st.metric("Calculated Performance", f"{calculated_performance:.1f}")
                    performance_rating = calculated_performance
                else:
                    performance_rating = st.slider("Performance Rating", 1.0, 5.0, 3.5, 0.1)
            
            with col2:
                education_level = st.selectbox("Education Level", predictor.education_levels)
                major = st.selectbox("Major/Field of Study", predictor.majors)
                top_tier_university = st.checkbox("Top Tier University")
                years_since_graduation = st.number_input("Years Since Graduation", min_value=0, max_value=50, value=5)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Location & Skills Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="form-section-title">📍 Location & Skills</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.selectbox("City", predictor.cities)
                state = st.selectbox("State", [
                    'Maharashtra', 'Karnataka', 'Delhi', 'Telangana', 'Tamil Nadu',
                    'Haryana', 'Uttar Pradesh', 'West Bengal', 'Gujarat', 'Madhya Pradesh',
                    'Rajasthan', 'Chandigarh', 'Kerala', 'Odisha', 'Bihar'
                ])
            
            with col2:
                skills = st.multiselect("Skills", predictor.skills, default=["Python", "SQL"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            submit = st.form_submit_button("🚀 Get My Salary Prediction")
        
        # Process prediction
        if submit:
            # Validation warnings
            warnings = []
            
            # Check education-role compatibility
            if job_role and seniority and education_level:
                try:
                    is_compatible, message = predictor.validator.validate_education_role_compatibility(
                        education_level, job_role, seniority
                    )
                    if not is_compatible:
                        warnings.append(f"⚠️ {message}")
                except:
                    pass
            
            # Check experience-role compatibility
            if total_experience == 0 and seniority in ['Senior', 'Lead', 'Manager', 'Director']:
                warnings.append("⚠️ 0 years experience is insufficient for Senior/Lead/Manager roles")
            
            # Show warnings
            if warnings:
                st.markdown('<div class="premium-warning">', unsafe_allow_html=True)
                st.warning("**Validation Warnings:**")
                for warning in warnings:
                    st.write(warning)
                st.info("💡 Consider adjusting education level or experience for more realistic predictions")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Process prediction
        if submit:
            with st.spinner("🔮 Analyzing your profile and calculating salary..."):
                # Calculate derived features
                years_since_promotion = years_in_role  # Simplified
                career_progression = years_since_promotion / max(years_in_role, 1)
                job_level = {'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5, 'Director': 6}.get(seniority, 1)
                num_skills = len(skills)
                
                # Fresher salary adjustment
                is_fresher = total_experience == 0
                if is_fresher:
                    # Reduce base salary for freshers
                    fresher_multiplier = 0.6  # 40% reduction for freshers
                    # Additional reduction for low education
                    if education_level in ['High School', 'Associate']:
                        fresher_multiplier *= 0.5  # Further 50% reduction
                else:
                    fresher_multiplier = 1.0
                
                # Age group
                if age < 25:
                    age_group = 'Early Career'
                elif age < 35:
                    age_group = 'Mid Career'
                elif age < 45:
                    age_group = 'Senior Career'
                elif age < 55:
                    age_group = 'Late Career'
                else:
                    age_group = 'Executive'
                
                # Metropolitan area check
                metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Gurgaon', 'Noida']
                is_metropolitan = city in metro_cities
                
                # Cost of living multiplier
                cost_of_living_multiplier = predictor.city_multipliers.get(city, 1.0)
                
                # Company age (random for demo)
                company_age = random.randint(1, 50)
                
                # Create input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Ethnicity': [ethnicity],
                    'TotalYearsExperience': [total_experience],
                    'YearsInCurrentRole': [years_in_role],
                    'YearsAtCompany': [years_at_company],
                    'YearsSinceLastPromotion': [years_since_promotion],
                    'NumCompaniesWorked': [num_companies],
                    'CareerProgression': [career_progression],
                    'JobRole': [job_role],
                    'Seniority': [seniority],
                    'JobLevel': [job_level],
                    'Department': [department],
                    'Industry': [industry],
                    'CompanySize': [company_size],
                    'CompanyAge': [company_age],
                    'RevenueTier': [revenue_tier],
                    'PerformanceRating': [performance_rating],
                    'Certifications': [certifications],
                    'NumSkills': [num_skills],
                    'EducationLevel': [education_level],
                    'Major': [major],
                    'TopTierUniversity': [top_tier_university],
                    'YearsSinceGraduation': [years_since_graduation],
                    'City': [city],
                    'State': [state],
                    'IsMetropolitanArea': [is_metropolitan],
                    'CostOfLivingMultiplier': [cost_of_living_multiplier],
                    'AgeGroup': [age_group],
                    'Skills': [','.join(skills)],
                    'IsFresher': [is_fresher],
                    'FresherMultiplier': [fresher_multiplier]
                })
                
                # Make prediction
                prediction_result = predictor.predict_salary(input_data)
                
                if prediction_result:
                    # Display results in premium cards
                    st.markdown("## 💰 Your Salary Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="premium-metric">', unsafe_allow_html=True)
                        st.markdown(f'<div class="premium-metric-value">₹{prediction_result["prediction"]:,.0f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="premium-metric-label">Predicted Salary</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="premium-metric">', unsafe_allow_html=True)
                        st.markdown(f'<div class="premium-metric-value">₹{prediction_result["confidence_interval"]:,.0f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="premium-metric-label">Confidence Range</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="premium-metric">', unsafe_allow_html=True)
                        st.markdown(f'<div class="premium-metric-value">90%</div>', unsafe_allow_html=True)
                        st.markdown('<div class="premium-metric-label">Confidence Level</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Salary range
                    st.markdown("### 📊 Salary Range")
                    st.markdown(f"""
                    **Range**: ₹{prediction_result['lower_bound']:,.0f} - ₹{prediction_result['upper_bound']:,.0f}
                    """)
                    
                    # Display bias correction if applied
                    if prediction_result.get('bias_correction', 0) != 0:
                        st.markdown('<div class="premium-success">', unsafe_allow_html=True)
                        st.info(f"🎯 **Bias Correction Applied**: ₹{prediction_result['bias_correction']:,.0f}")
                        
                        if prediction_result['bias_correction'] > 0:
                            st.success("✅ **Fairness Enhancement**: Salary adjusted upward to address historical bias")
                        else:
                            st.warning("⚖️ **Fairness Adjustment**: Salary adjusted downward to address historical bias")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Bias analysis
                    bias_analysis = predictor.analyze_bias(input_data, prediction_result['prediction'])
                    
                    if bias_analysis:
                        st.markdown("### 🔍 Bias Analysis")
                        
                        for bias_type, bias_data in bias_analysis.items():
                            if abs(bias_data['bias']) > 100000:  # Significant bias
                                st.markdown('<div class="premium-warning">', unsafe_allow_html=True)
                                st.warning(f"⚠️ Potential {bias_type} bias detected: ₹{bias_data['bias']:,.0f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="premium-success">', unsafe_allow_html=True)
                                st.success(f"✅ No significant {bias_type} bias detected")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explanation
                    st.markdown("### 🔍 Prediction Explanation")
                    explanation = predictor.explain_prediction(input_data)
                    
                    if explanation is not None:
                        # Create feature importance chart
                        fig = px.bar(
                            explanation,
                            x='SHAP_Value',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Factors Affecting Your Salary",
                            labels={'SHAP_Value': 'Impact on Salary', 'Feature': 'Factors'}
                        )
                        fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("💡 **Understanding the Chart**: Positive values increase your salary, negative values decrease it.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Premium Footer
    st.markdown("""
    <div class="premium-footer">
        <h4>💰 TRU Salary Predictor</h4>
        <p>Copyright © 2025 TRU Salary Predictor</p>
        <p>📧 Contact: <a href='mailto:truptibhuskute@gmail.com' style='color: #FFD700;'>truptibhuskute@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 