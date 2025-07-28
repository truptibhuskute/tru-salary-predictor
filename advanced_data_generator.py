"""
Advanced Data Generator for TRU Salary Prediction
Copyright (c) 2025 TRU Salary Predictor
Contact: contact@trusalaryprediction.com

Generates realistic and diverse salary datasets for machine learning model training.
Specifically designed for the Indian job market with realistic salary ranges and demographics.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from data_validation import DataValidator

class AdvancedSalaryDataGenerator:
    """
    Industry-level salary data generator with comprehensive features
    """
    
    def __init__(self):
        # Initialize validator
        self.validator = DataValidator()
        
        # Realistic salary ranges by role and seniority (INR - Indian Rupees)
        self.salary_ranges = {
            'Data Scientist': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Manager': {'min': 5000000, 'max': 8000000}
            },
            'Software Engineer': {
                'Junior': {'min': 300000, 'max': 800000},  # More realistic entry-level
                'Mid': {'min': 800000, 'max': 1500000},
                'Senior': {'min': 1500000, 'max': 2500000},
                'Lead': {'min': 2500000, 'max': 4000000},
                'Manager': {'min': 4000000, 'max': 6000000}
            },
            'Product Manager': {
                'Junior': {'min': 800000, 'max': 1500000},
                'Mid': {'min': 1500000, 'max': 2500000},
                'Senior': {'min': 2500000, 'max': 4000000},
                'Lead': {'min': 4000000, 'max': 6000000},
                'Director': {'min': 6000000, 'max': 12000000}
            },
            'Sales Manager': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Director': {'min': 5000000, 'max': 8000000}
            },
            'Marketing Manager': {
                'Junior': {'min': 500000, 'max': 1000000},
                'Mid': {'min': 1000000, 'max': 1800000},
                'Senior': {'min': 1800000, 'max': 3000000},
                'Lead': {'min': 3000000, 'max': 4500000},
                'Director': {'min': 4500000, 'max': 7000000}
            },
            'HR Manager': {
                'Junior': {'min': 400000, 'max': 800000},
                'Mid': {'min': 800000, 'max': 1500000},
                'Senior': {'min': 1500000, 'max': 2500000},
                'Lead': {'min': 2500000, 'max': 4000000},
                'Director': {'min': 4000000, 'max': 6000000}
            },
            'Finance Manager': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Director': {'min': 5000000, 'max': 8000000}
            },
            'DevOps Engineer': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Manager': {'min': 5000000, 'max': 8000000}
            },
            'UI/UX Designer': {
                'Junior': {'min': 400000, 'max': 800000},
                'Mid': {'min': 800000, 'max': 1500000},
                'Senior': {'min': 1500000, 'max': 2500000},
                'Lead': {'min': 2500000, 'max': 4000000},
                'Manager': {'min': 4000000, 'max': 6000000}
            },
            'Data Engineer': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Manager': {'min': 5000000, 'max': 8000000}
            }
        }
        
        # Cost of living multipliers by Indian city (using validator's valid locations)
        self.city_multipliers = self.validator.valid_locations.copy()
        # Apply multipliers
        multipliers = {
            'Mumbai': 1.4, 'Bangalore': 1.3, 'Delhi': 1.2, 'Hyderabad': 1.1,
            'Chennai': 1.0, 'Pune': 0.95, 'Gurgaon': 1.15, 'Noida': 1.1,
            'Kolkata': 0.9, 'Ahmedabad': 0.85, 'Indore': 0.8, 'Jaipur': 0.85,
            'Chandigarh': 0.9, 'Vadodara': 0.8, 'Coimbatore': 0.85, 'Kochi': 0.9,
            'Bhubaneswar': 0.8, 'Nagpur': 0.8, 'Lucknow': 0.8, 'Patna': 0.75
        }
        for city in self.city_multipliers:
            if city in multipliers:
                self.city_multipliers[city] = multipliers[city]
            else:
                self.city_multipliers[city] = 1.0
        
        # Education multipliers (more realistic for Indian market)
        self.education_multipliers = {
            'High School': 0.4, 'Associate': 0.6, 'Bachelors': 1.0,
            'Masters': 1.4, 'MBA': 1.6, 'PhD': 1.8, 'JD': 1.7, 'MD': 2.2
        }
        
        # Industry multipliers
        self.industry_multipliers = {
            'Technology': 1.3, 'Finance': 1.2, 'Healthcare': 1.1,
            'Consulting': 1.2, 'Manufacturing': 0.9, 'Retail': 0.8,
            'Education': 0.7, 'Non-profit': 0.6, 'Government': 0.8
        }
        
        # Company size multipliers
        self.company_size_multipliers = {
            'Startup (1-50)': 0.9, 'Small (51-200)': 1.0, 'Medium (201-1000)': 1.1,
            'Large (1001-10000)': 1.2, 'Enterprise (10000+)': 1.3
        }
        
        # Performance multipliers
        self.performance_multipliers = {
            1.0: 0.7, 1.5: 0.8, 2.0: 0.85, 2.5: 0.9, 3.0: 1.0,
            3.5: 1.1, 4.0: 1.2, 4.5: 1.3, 5.0: 1.4
        }
        
        # Base salary ranges are already in INR (Indian Rupees)
        self.usd_to_inr = 1.0
        
        # Job roles and seniority levels (using validator's valid roles)
        self.job_roles = list(self.validator.valid_job_roles.keys())
        self.seniority_levels = list(set([level for levels in self.validator.valid_job_roles.values() for level in levels]))
        
    def generate_experience_features(self, age, total_experience):
        """Generate comprehensive experience-related features"""
        # Years in current role (0 to total experience)
        years_in_role = min(total_experience, random.randint(0, total_experience))
        
        # Years at company (0 to total experience)
        years_at_company = min(total_experience, random.randint(0, total_experience))
        
        # Years since last promotion (0 to years in role)
        years_since_promotion = random.randint(0, years_in_role)
        
        # Number of companies worked (1 to reasonable number)
        max_companies = min(total_experience // 2 + 1, 10)
        num_companies = random.randint(1, max_companies)
        
        # Career progression indicator
        career_progression = years_since_promotion / max(years_in_role, 1)
        
        return {
            'TotalYearsExperience': total_experience,
            'YearsInCurrentRole': years_in_role,
            'YearsAtCompany': years_at_company,
            'YearsSinceLastPromotion': years_since_promotion,
            'NumCompaniesWorked': num_companies,
            'CareerProgression': career_progression
        }
    
    def generate_education_features(self):
        """Generate education-related features"""
        education_levels = list(self.education_multipliers.keys())
        education = random.choice(education_levels)
        
        # Major/Field of study
        majors = {
            'High School': ['General'],
            'Associate': ['Business', 'Technology', 'Arts'],
            'Bachelors': ['Computer Science', 'Engineering', 'Business', 'Mathematics', 'Economics', 'Psychology'],
            'Masters': ['Computer Science', 'Business Administration', 'Data Science', 'Engineering', 'Finance'],
            'MBA': ['Business Administration', 'Finance', 'Marketing', 'Operations'],
            'PhD': ['Computer Science', 'Engineering', 'Mathematics', 'Economics', 'Psychology'],
            'JD': ['Law'],
            'MD': ['Medicine']
        }
        
        major = random.choice(majors.get(education, ['General']))
        
        # Top tier university (20% chance)
        top_tier_university = random.random() < 0.2
        
        # Years since graduation
        if education in ['High School', 'Associate']:
            years_since_graduation = random.randint(5, 30)
        elif education in ['Bachelors']:
            years_since_graduation = random.randint(2, 25)
        elif education in ['Masters', 'MBA']:
            years_since_graduation = random.randint(1, 20)
        else:  # PhD, JD, MD
            years_since_graduation = random.randint(0, 15)
        
        return {
            'EducationLevel': education,
            'Major': major,
            'TopTierUniversity': top_tier_university,
            'YearsSinceGraduation': years_since_graduation
        }
    
    def generate_job_features(self):
        """Generate job-related features"""
        job_role = random.choice(self.job_roles)
        
        # Seniority level based on job role (using validator's valid combinations)
        valid_seniorities = self.validator.valid_job_roles[job_role]
        seniority = random.choice(valid_seniorities)
        
        # Department/Function (using validator's valid departments)
        department = random.choice(self.validator.valid_departments)
        
        # Industry (using validator's valid industries)
        industry = random.choice(self.validator.valid_industries)
        
        # Job level (numeric)
        level_map = {'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5, 'Director': 6}
        job_level = level_map.get(seniority, 1)
        
        return {
            'JobRole': job_role,
            'Seniority': seniority,
            'JobLevel': job_level,
            'Department': department,
            'Industry': industry
        }
    
    def generate_location_features(self):
        """Generate location-based features"""
        cities = list(self.validator.valid_locations.keys())
        city = random.choice(cities)
        
        # Get state from validator's valid locations
        state = self.validator.valid_locations[city]
        
        # Metropolitan area indicator
        metro_cities = ['Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Chennai', 'Pune']
        is_metropolitan = city in metro_cities
        
        return {
            'City': city,
            'State': state,
            'IsMetropolitanArea': is_metropolitan,
            'CostOfLivingMultiplier': self.city_multipliers[city]
        }
    
    def generate_company_features(self):
        """Generate company-specific features"""
        company_sizes = list(self.company_size_multipliers.keys())
        company_size = random.choice(company_sizes)
        
        industries = list(self.industry_multipliers.keys())
        industry = random.choice(industries)
        
        # Company age (years since founding)
        company_age = random.randint(1, 50)
        
        # Revenue tier (proxy for financial health)
        revenue_tiers = ['<1M', '1M-10M', '10M-100M', '100M-1B', '>1B']
        revenue_tier = random.choice(revenue_tiers)
        
        return {
            'CompanySize': company_size,
            'Industry': industry,
            'CompanyAge': company_age,
            'RevenueTier': revenue_tier
        }
    
    def generate_performance_features(self):
        """Generate performance and skill-based features"""
        # Performance rating (1-5 scale)
        performance_rating = round(random.uniform(2.5, 4.5), 1)
        
        # Number of certifications
        certifications = random.randint(0, 8)
        
        # Skills (comprehensive list)
        all_skills = [
            'Python', 'SQL', 'Java', 'JavaScript', 'React', 'Angular', 'Vue.js',
            'Node.js', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'Machine Learning',
            'Deep Learning', 'Data Analysis', 'Statistics', 'Excel', 'PowerBI',
            'Tableau', 'Salesforce', 'SAP', 'Oracle', 'MongoDB', 'Redis',
            'Leadership', 'Communication', 'Project Management', 'Agile',
            'Scrum', 'Kanban', 'Git', 'JIRA', 'Confluence', 'Slack',
            'Microsoft Office', 'Google Workspace', 'Customer Service',
            'Problem Solving', 'Critical Thinking', 'Teamwork', 'Time Management'
        ]
        
        # Select skills based on role
        role_skills = {
            'Data Scientist': ['Python', 'SQL', 'Machine Learning', 'Statistics', 'Data Analysis'],
            'Software Engineer': ['Java', 'Python', 'JavaScript', 'SQL', 'Git'],
            'Product Manager': ['Product Management', 'Agile', 'Communication', 'Leadership'],
            'Sales Manager': ['Sales', 'CRM', 'Communication', 'Leadership'],
            'Marketing Manager': ['Marketing', 'Digital Marketing', 'Analytics'],
            'HR Manager': ['HR Management', 'Recruitment', 'Communication'],
            'Finance Manager': ['Financial Analysis', 'Excel', 'Accounting'],
            'DevOps Engineer': ['Docker', 'Kubernetes', 'AWS', 'Linux'],
            'UI/UX Designer': ['Figma', 'Adobe XD', 'Prototyping', 'User Research'],
            'Data Engineer': ['Python', 'SQL', 'Apache Spark', 'Hadoop']
        }
        
        role = self.generate_job_features()['JobRole']
        base_skills = role_skills.get(role, ['Communication', 'Problem Solving'])
        
        # Add some additional skills
        num_skills = random.randint(3, 8)
        available_skills = [skill for skill in all_skills if skill not in base_skills]
        additional_skills_count = min(num_skills - len(base_skills), len(available_skills))
        
        if additional_skills_count > 0:
            additional_skills = random.sample(available_skills, additional_skills_count)
            skills = base_skills + additional_skills
        else:
            skills = base_skills
            
        skills = list(set(skills))  # Remove duplicates
        
        return {
            'PerformanceRating': performance_rating,
            'Certifications': certifications,
            'Skills': ','.join(skills),
            'NumSkills': len(skills)
        }
    
    def generate_demographic_features(self, age):
        """Generate demographic features (for bias analysis)"""
        gender = random.choice(['Male', 'Female', 'Other'])
        
        # Ethnicity (for bias detection)
        ethnicities = ['White', 'Asian', 'Hispanic', 'Black', 'Other']
        ethnicity = random.choice(ethnicities)
        
        # Age group
        if age < 25:
            age_group = 'Early Career'
        elif age < 35:
            age_group = 'Mid Career'
        elif age < 50:
            age_group = 'Senior Career'
        else:
            age_group = 'Late Career'
        
        return {
            'Gender': gender,
            'Ethnicity': ethnicity,
            'AgeGroup': age_group
        }
    
    def calculate_realistic_salary(self, features):
        """Calculate realistic salary based on all features"""
        role = features['JobRole']
        seniority = features['Seniority']
        
        # Base salary from role and seniority
        base_range = self.salary_ranges[role][seniority]
        base_salary = random.uniform(base_range['min'], base_range['max'])
        
        # Apply experience multiplier (5% per year)
        exp_multiplier = 1 + (features['TotalYearsExperience'] * 0.05)
        base_salary *= exp_multiplier
        
        # Apply education multiplier
        education_mult = self.education_multipliers.get(features['EducationLevel'], 1.0)
        base_salary *= education_mult
        
        # Apply location multiplier
        base_salary *= features['CostOfLivingMultiplier']
        
        # Apply industry multiplier
        industry_mult = self.industry_multipliers.get(features['Industry'], 1.0)
        base_salary *= industry_mult
        
        # Apply company size multiplier
        company_mult = self.company_size_multipliers.get(features['CompanySize'], 1.0)
        base_salary *= company_mult
        
        # Apply performance multiplier
        perf_mult = self.performance_multipliers.get(features['PerformanceRating'], 1.0)
        base_salary *= perf_mult
        
        # Apply top tier university bonus
        if features['TopTierUniversity']:
            base_salary *= 1.1
        
        # Apply skills bonus
        skills_bonus = 1 + (features['NumSkills'] * 0.02)
        base_salary *= skills_bonus
        
        # Apply certification bonus
        cert_bonus = 1 + (features['Certifications'] * 0.03)
        base_salary *= cert_bonus
        
        # Add some randomness (±10%)
        random_factor = random.uniform(0.9, 1.1)
        base_salary *= random_factor
        
        # Convert to INR
        salary_inr = base_salary * self.usd_to_inr
        
        return round(salary_inr, 0)
    
    def generate_dataset(self, n_samples=5000):
        """Generate comprehensive salary dataset"""
        np.random.seed(42)
        random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Generate age (22-65, normally distributed around 35)
            age = int(np.random.normal(35, 8))
            age = max(22, min(65, age))
            
            # Generate total experience (0 to age-22, with most people having 2-20 years)
            max_exp = age - 22
            total_experience = int(np.random.exponential(6))
            total_experience = min(total_experience, max_exp)
            
            # Generate all feature sets
            experience_features = self.generate_experience_features(age, total_experience)
            education_features = self.generate_education_features()
            job_features = self.generate_job_features()
            location_features = self.generate_location_features()
            company_features = self.generate_company_features()
            performance_features = self.generate_performance_features()
            demographic_features = self.generate_demographic_features(age)
            
            # Combine all features
            record = {
                'ID': i + 1,
                'Age': age,
                **experience_features,
                **education_features,
                **job_features,
                **location_features,
                **company_features,
                **performance_features,
                **demographic_features
            }
            
            # Calculate salary
            record['Salary'] = self.calculate_realistic_salary(record)
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['YearsSinceLastPromotion'] = df['YearsInCurrentRole'] - df['YearsSinceLastPromotion']
        df['CareerProgression'] = df['YearsSinceLastPromotion'] / df['YearsInCurrentRole'].replace(0, 1)
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                               labels=['Early Career', 'Mid Career', 'Senior Career', 'Late Career', 'Executive'])
        df['CompanyAge'] = random.randint(1, 50)
        df['CostOfLivingMultiplier'] = df['City'].map(self.city_multipliers)
        
        # Final validation of the complete dataset
        print("\n🔍 Validating generated dataset...")
        validation_report = self.validator.validate_dataset(df)
        
        print(f"✅ Valid rows: {validation_report['valid_rows']}")
        print(f"❌ Invalid rows: {validation_report['invalid_rows']}")
        print(f"📊 Validation success rate: {validation_report['valid_rows']/len(df)*100:.1f}%")
        
        if validation_report['invalid_rows'] > 0:
            print("\n⚠️ Validation issues found:")
            for field, stats in validation_report['field_validation_summary'].items():
                if stats['error_rate'] > 0:
                    print(f"  - {field}: {stats['error_rate']:.1f}% error rate")
        
        # Clean and fix any remaining issues
        if validation_report['invalid_rows'] > 0:
            print("\n🔧 Cleaning dataset...")
            df_clean = self.validator.clean_and_fix_dataset(df)
            df = df_clean
        
        # Validate the generated data
        print("=== Advanced Dataset Validation ===")
        print(f"Total records: {len(df)}")
        print(f"Salary range: ₹{df['Salary'].min():,.0f} - ₹{df['Salary'].max():,.0f}")
        print(f"Average salary: ₹{df['Salary'].mean():,.0f}")
        print(f"Median salary: ₹{df['Salary'].median():,.0f}")
        
        print(f"\nAge range: {df['Age'].min()} - {df['Age'].max()}")
        print(f"Experience range: {df['TotalYearsExperience'].min()} - {df['TotalYearsExperience'].max()}")
        
        print(f"\nTop 5 job roles by average salary:")
        role_salary = df.groupby('JobRole')['Salary'].mean().sort_values(ascending=False)
        for role, salary in role_salary.head().items():
            print(f"  {role}: ₹{salary:,.0f}")
        
        print(f"\nTop 5 cities by average salary:")
        city_salary = df.groupby('City')['Salary'].mean().sort_values(ascending=False)
        for city, salary in city_salary.head().items():
            print(f"  {city}: ₹{salary:,.0f}")
        
        return df

def main():
    """Generate and save advanced salary dataset"""
    print("Generating advanced employee salary dataset...")
    
    generator = AdvancedSalaryDataGenerator()
    df = generator.generate_dataset(n_samples=5000)
    
    # Save to CSV
    filename = f"Advanced_Salary_Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\nDataset saved to: {filename}")
    print(f"Records: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    # Show sample data
    print("\n=== Sample Data ===")
    print(df.head(3).to_string(index=False))
    
    # Show feature importance insights
    print(f"\n=== Feature Insights ===")
    print(f"Education levels: {df['EducationLevel'].value_counts().to_dict()}")
    print(f"Industries: {df['Industry'].value_counts().to_dict()}")
    print(f"Company sizes: {df['CompanySize'].value_counts().to_dict()}")

if __name__ == "__main__":
    main() 