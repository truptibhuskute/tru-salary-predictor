"""
Data Validation Module for TRU Salary Prediction
Copyright (c) 2025 TRU Salary Predictor
Contact: truptibhuskute@gmail.com

Comprehensive validation system for salary prediction data including location validation,
demographic checks, and business logic validation.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """
    Comprehensive data validator for salary prediction datasets
    """
    
    def __init__(self):
        # Valid Indian cities and their states
        self.valid_locations = {
            'Mumbai': 'Maharashtra',
            'Bangalore': 'Karnataka', 
            'Delhi': 'Delhi',
            'Hyderabad': 'Telangana',
            'Chennai': 'Tamil Nadu',
            'Pune': 'Maharashtra',
            'Gurgaon': 'Haryana',
            'Noida': 'Uttar Pradesh',
            'Kolkata': 'West Bengal',
            'Ahmedabad': 'Gujarat',
            'Indore': 'Madhya Pradesh',
            'Jaipur': 'Rajasthan',
            'Chandigarh': 'Chandigarh',
            'Vadodara': 'Gujarat',
            'Coimbatore': 'Tamil Nadu',
            'Kochi': 'Kerala',
            'Bhubaneswar': 'Odisha',
            'Nagpur': 'Maharashtra',
            'Lucknow': 'Uttar Pradesh',
            'Patna': 'Bihar'
        }
        
        # Valid job roles and their seniority levels
        self.valid_job_roles = {
            'Data Scientist': ['Junior', 'Mid', 'Senior', 'Lead', 'Manager'],
            'Software Engineer': ['Junior', 'Mid', 'Senior', 'Lead', 'Manager'],
            'Product Manager': ['Junior', 'Mid', 'Senior', 'Lead', 'Director'],
            'Sales Manager': ['Junior', 'Mid', 'Senior', 'Lead', 'Director'],
            'Marketing Manager': ['Junior', 'Mid', 'Senior', 'Lead', 'Director'],
            'HR Manager': ['Junior', 'Mid', 'Senior', 'Lead', 'Director'],
            'Finance Manager': ['Junior', 'Mid', 'Senior', 'Lead', 'Director'],
            'DevOps Engineer': ['Junior', 'Mid', 'Senior', 'Lead', 'Manager'],
            'UI/UX Designer': ['Junior', 'Mid', 'Senior', 'Lead', 'Manager'],
            'Data Engineer': ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
        }
        
        # Valid education levels and majors
        self.valid_education = {
            'High School': ['General'],
            'Associate': ['Computer Science', 'Business', 'Engineering'],
            'Bachelors': ['Computer Science', 'Engineering', 'Business', 'Mathematics', 'Statistics', 'Economics', 'Marketing', 'Finance', 'Human Resources'],
            'Masters': ['Computer Science', 'Data Science', 'Business Administration', 'Engineering', 'Statistics', 'Economics'],
            'MBA': ['General Management', 'Finance', 'Marketing', 'Human Resources', 'Operations', 'Technology Management'],
            'PhD': ['Computer Science', 'Data Science', 'Statistics', 'Economics', 'Business'],
            'JD': ['Law'],
            'MD': ['Medicine']
        }
        
        # Valid industries
        self.valid_industries = [
            'Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Retail',
            'Consulting', 'Education', 'Government', 'Non-Profit', 'Media',
            'Real Estate', 'Transportation', 'Energy', 'Telecommunications'
        ]
        
        # Valid departments
        self.valid_departments = [
            'Engineering', 'Data Science', 'Product', 'Sales', 'Marketing',
            'Human Resources', 'Finance', 'Operations', 'Legal', 'Customer Support'
        ]
        
        # Valid company sizes
        self.valid_company_sizes = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
        
        # Valid skills
        self.valid_skills = [
            'Python', 'Java', 'JavaScript', 'SQL', 'R', 'Machine Learning',
            'Data Analysis', 'Statistics', 'AWS', 'Azure', 'Docker', 'Kubernetes',
            'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
            'Tableau', 'Power BI', 'Excel', 'PowerPoint', 'Word',
            'Communication', 'Leadership', 'Project Management', 'Agile',
            'Scrum', 'Sales', 'Marketing', 'Customer Service', 'Negotiation',
            'Financial Modeling', 'Accounting', 'HR Management', 'Recruitment',
            'UI/UX Design', 'Graphic Design', 'Adobe Creative Suite', 'Figma',
            'DevOps', 'CI/CD', 'Linux', 'Git', 'Jenkins', 'Ansible',
            'Big Data', 'Hadoop', 'Spark', 'Kafka', 'MongoDB', 'PostgreSQL',
            'Redis', 'Elasticsearch', 'TensorFlow', 'PyTorch', 'Scikit-learn',
            'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly'
        ]
        
        # Validation rules
        self.validation_rules = {
            'age': {'min': 18, 'max': 70},
            'total_years_experience': {'min': 0, 'max': 25},
            'years_in_current_role': {'min': 0, 'max': 15},
            'years_at_company': {'min': 0, 'max': 20},
            'years_since_graduation': {'min': 0, 'max': 30},
            'performance_rating': {'min': 1.0, 'max': 5.0},
            'salary': {'min': 270000, 'max': 45000000},  # 2.7L to 4.5Cr
            'num_skills': {'min': 3, 'max': 8},
            'certifications': {'min': 0, 'max': 5}
        }
    
    def validate_location(self, city: str, state: str) -> Tuple[bool, str]:
        """
        Validate city and state combination
        """
        if city not in self.valid_locations:
            return False, f"Invalid city: {city}. Valid cities: {list(self.valid_locations.keys())}"
        
        expected_state = self.valid_locations[city]
        if state != expected_state:
            return False, f"City {city} should be in {expected_state}, not {state}"
        
        return True, "Location is valid"
    
    def validate_job_role(self, role: str, seniority: str) -> Tuple[bool, str]:
        """
        Validate job role and seniority combination
        """
        if role not in self.valid_job_roles:
            return False, f"Invalid job role: {role}. Valid roles: {list(self.valid_job_roles.keys())}"
        
        valid_seniorities = self.valid_job_roles[role]
        if seniority not in valid_seniorities:
            return False, f"Invalid seniority '{seniority}' for role '{role}'. Valid seniorities: {valid_seniorities}"
        
        return True, "Job role and seniority are valid"
    
    def validate_education_role_compatibility(self, education: str, role: str, seniority: str) -> Tuple[bool, str]:
        """
        Validate education level compatibility with job role and seniority
        """
        # Education requirements for different roles
        role_education_requirements = {
            'Data Scientist': {
                'Junior': ['Bachelors', 'Masters', 'PhD'],
                'Mid': ['Bachelors', 'Masters', 'PhD'],
                'Senior': ['Bachelors', 'Masters', 'PhD'],
                'Lead': ['Masters', 'PhD'],
                'Manager': ['Masters', 'PhD']
            },
            'Software Engineer': {
                'Junior': ['Bachelors', 'Masters', 'PhD'],
                'Mid': ['Bachelors', 'Masters', 'PhD'],
                'Senior': ['Bachelors', 'Masters', 'PhD'],
                'Lead': ['Bachelors', 'Masters', 'PhD'],
                'Manager': ['Bachelors', 'Masters', 'PhD']
            },
            'Product Manager': {
                'Junior': ['Bachelors', 'Masters', 'MBA'],
                'Mid': ['Bachelors', 'Masters', 'MBA'],
                'Senior': ['Masters', 'MBA'],
                'Lead': ['Masters', 'MBA'],
                'Director': ['MBA', 'Masters']
            },
            'Sales Manager': {
                'Junior': ['Bachelors', 'Masters', 'MBA'],
                'Mid': ['Bachelors', 'Masters', 'MBA'],
                'Senior': ['Bachelors', 'Masters', 'MBA'],
                'Lead': ['Masters', 'MBA'],
                'Director': ['MBA', 'Masters']
            },
            'Marketing Manager': {
                'Junior': ['Bachelors', 'Masters', 'MBA'],
                'Mid': ['Bachelors', 'Masters', 'MBA'],
                'Senior': ['Bachelors', 'Masters', 'MBA'],
                'Lead': ['Masters', 'MBA'],
                'Director': ['MBA', 'Masters']
            },
            'HR Manager': {
                'Junior': ['Bachelors', 'Masters', 'MBA'],
                'Mid': ['Bachelors', 'Masters', 'MBA'],
                'Senior': ['Bachelors', 'Masters', 'MBA'],
                'Lead': ['Masters', 'MBA'],
                'Director': ['MBA', 'Masters']
            },
            'Finance Manager': {
                'Junior': ['Bachelors', 'Masters', 'MBA'],
                'Mid': ['Bachelors', 'Masters', 'MBA'],
                'Senior': ['Bachelors', 'Masters', 'MBA'],
                'Lead': ['Masters', 'MBA'],
                'Director': ['MBA', 'Masters']
            },
            'DevOps Engineer': {
                'Junior': ['Bachelors', 'Masters', 'PhD'],
                'Mid': ['Bachelors', 'Masters', 'PhD'],
                'Senior': ['Bachelors', 'Masters', 'PhD'],
                'Lead': ['Bachelors', 'Masters', 'PhD'],
                'Manager': ['Bachelors', 'Masters', 'PhD']
            },
            'UI/UX Designer': {
                'Junior': ['Bachelors', 'Masters'],
                'Mid': ['Bachelors', 'Masters'],
                'Senior': ['Bachelors', 'Masters'],
                'Lead': ['Bachelors', 'Masters'],
                'Manager': ['Bachelors', 'Masters']
            },
            'Data Engineer': {
                'Junior': ['Bachelors', 'Masters', 'PhD'],
                'Mid': ['Bachelors', 'Masters', 'PhD'],
                'Senior': ['Bachelors', 'Masters', 'PhD'],
                'Lead': ['Bachelors', 'Masters', 'PhD'],
                'Manager': ['Bachelors', 'Masters', 'PhD']
            }
        }
        
        if role in role_education_requirements and seniority in role_education_requirements[role]:
            required_education = role_education_requirements[role][seniority]
            if education not in required_education:
                return False, f"Education level '{education}' is insufficient for {seniority} {role}. Required: {required_education}"
        
        return True, "Education level is compatible with job role and seniority"
    
    def validate_education(self, level: str, major: str) -> Tuple[bool, str]:
        """
        Validate education level and major combination
        """
        if level not in self.valid_education:
            return False, f"Invalid education level: {level}. Valid levels: {list(self.valid_education.keys())}"
        
        valid_majors = self.valid_education[level]
        if major not in valid_majors:
            return False, f"Invalid major '{major}' for education level '{level}'. Valid majors: {valid_majors}"
        
        return True, "Education level and major are valid"
    
    def validate_numeric_field(self, value: float, field_name: str) -> Tuple[bool, str]:
        """
        Validate numeric fields against defined ranges
        """
        if field_name not in self.validation_rules:
            return True, f"No validation rules for {field_name}"
        
        rules = self.validation_rules[field_name]
        min_val = rules['min']
        max_val = rules['max']
        
        if not isinstance(value, (int, float)) or np.isnan(value):
            return False, f"{field_name} must be a valid number"
        
        if value < min_val or value > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}, got {value}"
        
        return True, f"{field_name} is valid"
    
    def validate_experience_consistency(self, age: int, total_experience: int, 
                                      years_in_role: int, years_at_company: int) -> Tuple[bool, str]:
        """
        Validate experience consistency
        """
        # Age should be >= 18 + total experience
        min_age = 18 + total_experience
        if age < min_age:
            return False, f"Age {age} is too young for {total_experience} years of experience. Minimum age should be {min_age}"
        
        # Years in role should be <= total experience
        if years_in_role > total_experience:
            return False, f"Years in current role ({years_in_role}) cannot exceed total experience ({total_experience})"
        
        # Years at company should be <= total experience
        if years_at_company > total_experience:
            return False, f"Years at company ({years_at_company}) cannot exceed total experience ({total_experience})"
        
        # Years in role should be <= years at company
        if years_in_role > years_at_company:
            return False, f"Years in current role ({years_in_role}) cannot exceed years at company ({years_at_company})"
        
        return True, "Experience consistency is valid"
    
    def validate_salary_realistic(self, salary: float, role: str, seniority: str, 
                                 city: str, experience: int) -> Tuple[bool, str]:
        """
        Validate if salary is realistic for given parameters
        """
        # Get expected salary range for role and seniority
        if role not in self.valid_job_roles or seniority not in self.valid_job_roles[role]:
            return False, f"Cannot validate salary for invalid role/seniority: {role}/{seniority}"
        
        # Define expected salary ranges (in INR)
        salary_ranges = {
            'Data Scientist': {
                'Junior': {'min': 600000, 'max': 1200000},
                'Mid': {'min': 1200000, 'max': 2000000},
                'Senior': {'min': 2000000, 'max': 3500000},
                'Lead': {'min': 3500000, 'max': 5000000},
                'Manager': {'min': 5000000, 'max': 8000000}
            },
            'Software Engineer': {
                'Junior': {'min': 500000, 'max': 1000000},
                'Mid': {'min': 1000000, 'max': 1800000},
                'Senior': {'min': 1800000, 'max': 3000000},
                'Lead': {'min': 3000000, 'max': 4500000},
                'Manager': {'min': 4500000, 'max': 7000000}
            }
        }
        
        if role in salary_ranges and seniority in salary_ranges[role]:
            expected_range = salary_ranges[role][seniority]
            min_salary = expected_range['min']
            max_salary = expected_range['max']
            
            # Apply city multiplier
            city_multipliers = {
                'Mumbai': 1.4, 'Bangalore': 1.3, 'Delhi': 1.2, 'Hyderabad': 1.1,
                'Chennai': 1.0, 'Pune': 0.95, 'Gurgaon': 1.15, 'Noida': 1.1,
                'Kolkata': 0.9, 'Ahmedabad': 0.85, 'Indore': 0.8, 'Jaipur': 0.85,
                'Chandigarh': 0.9, 'Vadodara': 0.8, 'Coimbatore': 0.85, 'Kochi': 0.9,
                'Bhubaneswar': 0.8, 'Nagpur': 0.8, 'Lucknow': 0.8, 'Patna': 0.75
            }
            
            multiplier = city_multipliers.get(city, 1.0)
            adjusted_min = min_salary * multiplier
            adjusted_max = max_salary * multiplier
            
            # Allow some flexibility (±20%)
            tolerance = 0.2
            min_allowed = adjusted_min * (1 - tolerance)
            max_allowed = adjusted_max * (1 + tolerance)
            
            if salary < min_allowed or salary > max_allowed:
                return False, f"Salary ₹{salary:,.0f} is outside realistic range ₹{min_allowed:,.0f} - ₹{max_allowed:,.0f} for {seniority} {role} in {city}"
        
        return True, "Salary is realistic"
    
    def validate_skills(self, skills: List[str]) -> Tuple[bool, str]:
        """
        Validate skills list
        """
        if not isinstance(skills, list):
            return False, "Skills must be a list"
        
        if len(skills) < 3 or len(skills) > 8:
            return False, f"Number of skills ({len(skills)}) must be between 3 and 8"
        
        invalid_skills = [skill for skill in skills if skill not in self.valid_skills]
        if invalid_skills:
            return False, f"Invalid skills: {invalid_skills}. Valid skills: {self.valid_skills[:10]}..."
        
        return True, "Skills are valid"
    
    def validate_demographics(self, age: int, gender: str, ethnicity: str) -> Tuple[bool, str]:
        """
        Validate demographic information
        """
        valid_genders = ['Male', 'Female', 'Other']
        valid_ethnicities = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
        
        if gender not in valid_genders:
            return False, f"Invalid gender: {gender}. Valid genders: {valid_genders}"
        
        if ethnicity not in valid_ethnicities:
            return False, f"Invalid ethnicity: {ethnicity}. Valid ethnicities: {valid_ethnicities}"
        
        return True, "Demographics are valid"
    
    def validate_entire_row(self, row: pd.Series) -> Dict[str, Tuple[bool, str]]:
        """
        Validate an entire data row
        """
        validation_results = {}
        
        # Location validation
        city = row.get('City', '')
        state = row.get('State', '')
        validation_results['location'] = self.validate_location(city, state)
        
        # Job role validation
        role = row.get('JobRole', '')
        seniority = row.get('Seniority', '')
        validation_results['job_role'] = self.validate_job_role(role, seniority)
        
        # Education-role compatibility validation
        education = row.get('EducationLevel', '')
        validation_results['education_role_compatibility'] = self.validate_education_role_compatibility(education, role, seniority)
        
        # Education validation
        education = row.get('EducationLevel', '')
        major = row.get('Major', '')
        validation_results['education'] = self.validate_education(education, major)
        
        # Numeric field validations
        numeric_fields = ['Age', 'TotalYearsExperience', 'YearsInCurrentRole', 
                         'YearsAtCompany', 'PerformanceRating', 'Salary']
        for field in numeric_fields:
            if field in row:
                validation_results[field] = self.validate_numeric_field(row[field], field.lower())
        
        # Experience consistency
        if all(field in row for field in ['Age', 'TotalYearsExperience', 'YearsInCurrentRole', 'YearsAtCompany']):
            validation_results['experience_consistency'] = self.validate_experience_consistency(
                row['Age'], row['TotalYearsExperience'], row['YearsInCurrentRole'], row['YearsAtCompany']
            )
        
        # Salary realism
        if all(field in row for field in ['Salary', 'JobRole', 'Seniority', 'City', 'TotalYearsExperience']):
            validation_results['salary_realistic'] = self.validate_salary_realistic(
                row['Salary'], row['JobRole'], row['Seniority'], row['City'], row['TotalYearsExperience']
            )
        
        # Demographics
        if all(field in row for field in ['Age', 'Gender', 'Ethnicity']):
            validation_results['demographics'] = self.validate_demographics(
                row['Age'], row['Gender'], row['Ethnicity']
            )
        
        return validation_results
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate entire dataset and return comprehensive report
        """
        validation_report = {
            'total_rows': len(df),
            'valid_rows': 0,
            'invalid_rows': 0,
            'validation_errors': {},
            'field_validation_summary': {},
            'recommendations': []
        }
        
        for idx, row in df.iterrows():
            row_validations = self.validate_entire_row(row)
            
            # Check if row is completely valid
            row_is_valid = all(is_valid for is_valid, _ in row_validations.values())
            
            if row_is_valid:
                validation_report['valid_rows'] += 1
            else:
                validation_report['invalid_rows'] += 1
                validation_report['validation_errors'][idx] = {
                    field: message for field, (is_valid, message) in row_validations.items() 
                    if not is_valid
                }
        
        # Generate field-level summary
        for field in ['location', 'job_role', 'education', 'experience_consistency', 
                     'salary_realistic', 'demographics']:
            field_errors = sum(1 for errors in validation_report['validation_errors'].values() 
                             if field in errors)
            validation_report['field_validation_summary'][field] = {
                'total_errors': field_errors,
                'error_rate': field_errors / len(df) * 100
            }
        
        # Generate recommendations
        if validation_report['invalid_rows'] > 0:
            validation_report['recommendations'].append(
                f"Found {validation_report['invalid_rows']} invalid rows. Please review and fix data quality issues."
            )
        
        error_rates = validation_report['field_validation_summary']
        for field, stats in error_rates.items():
            if stats['error_rate'] > 10:
                validation_report['recommendations'].append(
                    f"High error rate in {field}: {stats['error_rate']:.1f}%. Consider data quality improvements."
                )
        
        return validation_report
    
    def clean_and_fix_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to clean and fix common data issues
        """
        df_clean = df.copy()
        fixes_applied = []
        
        # Fix location mismatches
        for idx, row in df_clean.iterrows():
            city = row.get('City', '')
            state = row.get('State', '')
            
            if city in self.valid_locations and state != self.valid_locations[city]:
                old_state = state
                df_clean.at[idx, 'State'] = self.valid_locations[city]
                fixes_applied.append(f"Row {idx}: Fixed state from '{old_state}' to '{self.valid_locations[city]}' for city '{city}'")
        
        # Fix experience inconsistencies
        for idx, row in df_clean.iterrows():
            age = row.get('Age', 0)
            total_exp = row.get('TotalYearsExperience', 0)
            years_in_role = row.get('YearsInCurrentRole', 0)
            years_at_company = row.get('YearsAtCompany', 0)
            
            # Fix years in role if it exceeds total experience
            if years_in_role > total_exp:
                df_clean.at[idx, 'YearsInCurrentRole'] = total_exp
                fixes_applied.append(f"Row {idx}: Fixed YearsInCurrentRole from {years_in_role} to {total_exp}")
            
            # Fix years at company if it exceeds total experience
            if years_at_company > total_exp:
                df_clean.at[idx, 'YearsAtCompany'] = total_exp
                fixes_applied.append(f"Row {idx}: Fixed YearsAtCompany from {years_at_company} to {total_exp}")
            
            # Fix years in role if it exceeds years at company
            if years_in_role > years_at_company:
                df_clean.at[idx, 'YearsInCurrentRole'] = years_at_company
                fixes_applied.append(f"Row {idx}: Fixed YearsInCurrentRole from {years_in_role} to {years_at_company}")
        
        # Fix age if it's too young for experience
        for idx, row in df_clean.iterrows():
            age = row.get('Age', 0)
            total_exp = row.get('TotalYearsExperience', 0)
            min_age = 18 + total_exp
            
            if age < min_age:
                df_clean.at[idx, 'Age'] = min_age
                fixes_applied.append(f"Row {idx}: Fixed Age from {age} to {min_age}")
        
        # Fix salary outliers (cap at reasonable maximum)
        max_salary = 45000000  # 4.5Cr
        for idx, row in df_clean.iterrows():
            salary = row.get('Salary', 0)
            if salary > max_salary:
                df_clean.at[idx, 'Salary'] = max_salary
                fixes_applied.append(f"Row {idx}: Capped Salary from {salary:,.0f} to {max_salary:,.0f}")
        
        print(f"Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied[:10]:  # Show first 10 fixes
            print(f"  - {fix}")
        if len(fixes_applied) > 10:
            print(f"  ... and {len(fixes_applied) - 10} more fixes")
        
        return df_clean

def main():
    """
    Test the validation system
    """
    validator = DataValidator()
    
    # Test location validation
    print("Testing location validation:")
    print(validator.validate_location("Mumbai", "Maharashtra"))  # Should be valid
    print(validator.validate_location("Mumbai", "Karnataka"))    # Should be invalid
    print(validator.validate_location("InvalidCity", "Maharashtra"))  # Should be invalid
    print()
    
    # Test job role validation
    print("Testing job role validation:")
    print(validator.validate_job_role("Data Scientist", "Senior"))  # Should be valid
    print(validator.validate_job_role("Data Scientist", "Invalid"))  # Should be invalid
    print(validator.validate_job_role("InvalidRole", "Senior"))  # Should be invalid
    print()
    
    # Test experience consistency
    print("Testing experience consistency:")
    print(validator.validate_experience_consistency(30, 8, 3, 5))  # Should be valid
    print(validator.validate_experience_consistency(25, 10, 3, 5))  # Should be invalid (too young)
    print(validator.validate_experience_consistency(30, 8, 10, 5))  # Should be invalid (role > total exp)
    print()

if __name__ == "__main__":
    main() 