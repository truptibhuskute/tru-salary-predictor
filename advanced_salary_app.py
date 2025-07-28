"""
TRU Salary Predictor - Advanced Salary Prediction System
Copyright (c) 2025 TRU Salary Predictor
Contact: truptibhuskute@gmail.com

A highly accurate and interpretable machine learning model that predicts employee salaries,
enabling organizations to make data-driven decisions for compensation, recruitment, and retention,
and helping individuals understand their market value.

This system includes bias detection and mitigation for fair predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
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

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stButton > button {
        background-color: #00d4aa;
        color: #0e1117;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5em 2em;
        border: none;
    }
    .stButton > button:hover {
        background-color: #00b894;
        color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
    }
    .bias-alert {
        background-color: #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #51cf66;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedSalaryPredictor:
    """Advanced salary prediction system with bias detection and explainability"""
    
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
            'Technology', 'Finance', 'Healthcare', 'Consulting', 'Manufacturing',
            'Retail', 'Education', 'Non-profit', 'Government'
        ]
        
        self.company_sizes = [
            'Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)',
            'Large (1001-10000)', 'Enterprise (10000+)'
        ]
        
        # Cost of living multipliers by Indian city
        self.city_multipliers = {
            'Mumbai': 1.4, 'Bangalore': 1.3, 'Delhi': 1.2, 'Hyderabad': 1.1,
            'Chennai': 1.0, 'Pune': 0.95, 'Gurgaon': 1.15, 'Noida': 1.1,
            'Kolkata': 0.9, 'Ahmedabad': 0.85, 'Indore': 0.8, 'Jaipur': 0.85,
            'Chandigarh': 0.9, 'Vadodara': 0.8, 'Coimbatore': 0.85, 'Kochi': 0.9,
            'Bhubaneswar': 0.8, 'Nagpur': 0.8, 'Lucknow': 0.8, 'Patna': 0.75
        }
        
        self.skills = list(self.skills_mlb.classes_) if hasattr(self, 'skills_mlb') else [
            'Python', 'SQL', 'Java', 'JavaScript', 'React', 'Machine Learning',
            'Data Analysis', 'Leadership', 'Communication', 'Project Management'
        ]
    
    def validate_input(self, data):
        """Validate input data for realism"""
        errors = []
        
        # Get the first row since we're validating single input
        row = data.iloc[0]
        
        # Age validation
        if row['Age'] < 18 or row['Age'] > 70:
            errors.append("Age must be between 18 and 70 years")
        
        # Experience validation
        if row['TotalYearsExperience'] < 0 or row['TotalYearsExperience'] > 50:
            errors.append("Total years experience must be between 0 and 50")
        
        # Age vs experience validation
        if row['Age'] < (18 + row['TotalYearsExperience']):
            errors.append("Age must be at least 18 + total years experience")
        
        # Company years validation
        if row['YearsAtCompany'] > row['TotalYearsExperience']:
            errors.append("Years at company cannot exceed total experience")
        
        # Performance validation
        if row['PerformanceRating'] < 1.0 or row['PerformanceRating'] > 5.0:
            errors.append("Performance rating must be between 1.0 and 5.0")
        
        return errors
    
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
                    st.info(f"🎓 **Fresher Adjustment**: Applied {fresher_multiplier:.1f}x multiplier for entry-level position")
            
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
            
            # Add numeric feature names
            if 'num' in self.preprocessor.named_transformers_:
                feature_names.extend(self.preprocessor.named_transformers_['num'].get_feature_names_out())
            
            # Add categorical feature names
            if 'cat' in self.preprocessor.named_transformers_:
                cat_transformer = self.preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                    onehot = cat_transformer.named_steps['onehot']
                    if hasattr(onehot, 'get_feature_names_out'):
                        feature_names.extend(onehot.get_feature_names_out())
            
            # Create explanation
            explanation = {}
            for i, feature in enumerate(feature_names):
                if i < len(shap_values[0]):
                    explanation[feature] = shap_values[0][i]
            
            return explanation
        except Exception as e:
            st.warning(f"Could not generate explanation: {e}")
            return {}

def main():
    """Main application"""
    st.title("💰 TRU Salary Predictor")
    st.markdown("Industry-level salary prediction with bias detection and explainability")
    
    # Initialize predictor
    predictor = AdvancedSalaryPredictor()
    
    if not predictor.model_loaded:
        st.error("Models not loaded. Please ensure model files exist.")
        return
    
    # Show main prediction page directly
    single_prediction_page(predictor)

def single_prediction_page(predictor):
    """Single prediction page with comprehensive features"""
    st.header("Single Employee Salary Prediction")
    
    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["White", "Asian", "Hispanic", "Black", "Other"])
            
            st.subheader("Experience")
            total_experience = st.number_input("Total Years Experience", min_value=0, max_value=50, value=5)
            years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=2)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=2)
            num_companies = st.number_input("Number of Companies Worked", min_value=1, max_value=20, value=2)
            
        with col2:
            st.subheader("Job Information")
            job_role = st.selectbox("Job Role", predictor.job_roles)
            seniority = st.selectbox("Seniority Level", predictor.seniority_levels)
            department = st.selectbox("Department", ["Engineering", "Sales", "Marketing", "HR", "Finance", "IT", "Analytics", "Design", "Product"])
            
            st.subheader("Company Information")
            industry = st.selectbox("Industry", predictor.industries)
            company_size = st.selectbox("Company Size", predictor.company_sizes)
            revenue_tier = st.selectbox("Revenue Tier", ["<1M", "1M-10M", "10M-100M", "100M-1B", ">1B"])
            
            st.subheader("Performance")
            
            certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=2)
            
            # Performance rating logic for freshers vs experienced
            if total_experience == 0:
                st.info("🎓 **Fresher Mode**: Performance based on education, skills, and certifications")
                # Calculate performance for freshers
                base_performance = 3.0
                education_bonus = {
                    'High School': 0.0, 'Associate': 0.2, 'Bachelors': 0.5,
                    'Masters': 0.8, 'MBA': 1.0, 'PhD': 1.2, 'JD': 1.0, 'MD': 1.5
                }
                cert_bonus = min(certifications * 0.1, 0.5)  # Max 0.5 bonus from certs
                # Use a default education level if not defined yet
                current_education = education_level if 'education_level' in locals() else 'Bachelors'
                calculated_performance = min(5.0, base_performance + education_bonus.get(current_education, 0.0) + cert_bonus)
                
                st.metric("Calculated Performance", f"{calculated_performance:.1f}")
                performance_rating = calculated_performance
            else:
                performance_rating = st.slider("Performance Rating", 1.0, 5.0, 3.5, 0.1)
        
        st.subheader("Education")
        col3, col4 = st.columns(2)
        with col3:
            education_level = st.selectbox("Education Level", predictor.education_levels)
            
            # Show education requirements for selected job role
            if job_role and seniority:
                try:
                    is_compatible, message = predictor.validator.validate_education_role_compatibility(
                        education_level, job_role, seniority
                    )
                    if not is_compatible:
                        st.warning(f"⚠️ {message}")
                        st.info(f"💡 For {seniority} {job_role}, consider: Bachelors, Masters, or PhD")
                except:
                    pass
            
            major = st.text_input("Major/Field of Study", "Computer Science")
            top_tier_university = st.checkbox("Top Tier University")
        
        with col4:
            years_since_graduation = st.number_input("Years Since Graduation", min_value=0, max_value=50, value=5)
        
        st.subheader("Location")
        col5, col6 = st.columns(2)
        with col5:
            city = st.selectbox("City", predictor.cities)
            state = st.selectbox("State", [
                'Maharashtra', 'Karnataka', 'Delhi', 'Telangana', 'Tamil Nadu',
                'Haryana', 'Uttar Pradesh', 'West Bengal', 'Gujarat', 'Madhya Pradesh',
                'Rajasthan', 'Chandigarh', 'Kerala', 'Odisha', 'Bihar'
            ])
        
        with col6:
            is_metropolitan = st.checkbox("Metropolitan Area", value=True)
        
        st.subheader("Skills")
        skills = st.multiselect("Skills", predictor.skills, default=["Python", "SQL"])
        
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
            st.warning("**Validation Warnings:**")
            for warning in warnings:
                st.write(warning)
            st.info("💡 Consider adjusting education level or experience for more realistic predictions")
        
        submit = st.form_submit_button("Predict Salary")
    
    if submit:
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
        elif age < 50:
            age_group = 'Senior Career'
        else:
            age_group = 'Late Career'
        
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
            'CompanyAge': [random.randint(1, 50)],  # Random company age
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
            'CostOfLivingMultiplier': [predictor.city_multipliers.get(city, 1.0)],  # Get from city mapping
            'AgeGroup': [age_group],
            'Skills': [','.join(skills)],
            'IsFresher': [is_fresher],
            'FresherMultiplier': [fresher_multiplier]
        })
        
        # Validate input
        validation_errors = predictor.validate_input(input_data)
        
        if validation_errors:
            st.error("Validation errors found:")
            for error in validation_errors:
                st.write(f"• {error}")
        else:
            # Make prediction
            prediction_result = predictor.predict_salary(input_data)
            
            if prediction_result:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Salary",
                        f"₹{prediction_result['prediction']:,.0f}",
                        f"±₹{prediction_result['confidence_interval']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Range",
                        f"₹{prediction_result['lower_bound']:,.0f} - ₹{prediction_result['upper_bound']:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Confidence",
                        "90%"
                    )
                
                # Display bias correction if applied
                if prediction_result.get('bias_correction', 0) != 0:
                    st.info(f"🎯 **Bias Correction Applied**: ₹{prediction_result['bias_correction']:,.0f}")
                    
                    if prediction_result['bias_correction'] > 0:
                        st.success("✅ **Fairness Enhancement**: Salary adjusted upward to address historical bias")
                    else:
                        st.warning("⚖️ **Fairness Adjustment**: Salary adjusted downward to address historical bias")
                
                # Bias analysis
                bias_analysis = predictor.analyze_bias(input_data, prediction_result['prediction'])
                
                if bias_analysis:
                    st.subheader("🔍 Bias Analysis")
                    
                    for bias_type, bias_data in bias_analysis.items():
                        if abs(bias_data['bias']) > 100000:  # Significant bias
                            st.warning(f"⚠️ Potential {bias_type} bias detected: ₹{bias_data['bias']:,.0f}")
                        else:
                            st.success(f"✅ No significant {bias_type} bias detected")
                
                # Explanation
                st.subheader("🔍 Prediction Explanation")
                explanation = predictor.explain_prediction(input_data)
                
                if explanation:
                    # Top 10 features
                    top_features = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    
                    fig = go.Figure()
                    features, values = zip(*top_features)
                    
                    colors = ['red' if v < 0 else 'green' for v in values]
                    
                    fig.add_trace(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=colors
                    ))
                    
                    fig.update_layout(
                        title="Feature Impact on Prediction",
                        xaxis_title="SHAP Value (Impact on Salary)",
                        yaxis_title="Features",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
    
    # Footer with contact information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #262730; border-radius: 8px;'>
        <h4>💰 TRU Salary Predictor</h4>
        <p>Copyright © 2025 TRU Salary Predictor</p>
        <p>📧 Contact: <a href='mailto:truptibhuskute@gmail.com'>truptibhuskute@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True) 