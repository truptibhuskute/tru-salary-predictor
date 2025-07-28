"""
Bias Mitigation Module for TRU Salary Prediction
Copyright (c) 2025 TRU Salary Predictor
Contact: truptibhuskute@gmail.com

Addresses age, gender, and ethnicity biases detected in the model.
Ensures fair and equitable salary predictions across all demographic groups.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class BiasMitigator:
    def __init__(self, model_path='model/model.pkl', preprocessor_path='model/preprocessor.pkl'):
        """Initialize bias mitigator with trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.bias_corrections = self._calculate_bias_corrections()
        
    def _calculate_bias_corrections(self):
        """Calculate bias correction factors based on detected biases"""
        # Based on the bias analysis from model_report.md
        corrections = {
            'age_group': {
                'Early Career': 269531,  # Add ₹2.7L to correct underprediction
                'Mid Career': -18432,    # Subtract ₹18K to correct overprediction
                'Senior Career': -52752,  # Subtract ₹53K to correct overprediction
                'Late Career': -124810    # Subtract ₹1.2L to correct overprediction
            },
            'gender': {
                'Female': 51005,         # Add ₹51K to correct underprediction
                'Male': 4590,            # Add ₹5K to correct underprediction
                'Other': -86352          # Subtract ₹86K to correct overprediction
            },
            'ethnicity': {
                'Black': 147025,         # Add ₹1.5L to correct underprediction
                'Other': 16478,          # Add ₹16K to correct underprediction
                'White': -28899,         # Subtract ₹29K to correct overprediction
                'Asian': -97601,         # Subtract ₹98K to correct overprediction
                'Hispanic': -86484       # Subtract ₹86K to correct overprediction
            }
        }
        return corrections
    
    def apply_bias_correction(self, prediction, demographics):
        """
        Apply bias correction to salary prediction
        
        Args:
            prediction (float): Original salary prediction
            demographics (dict): Dictionary with age_group, gender, ethnicity keys
            
        Returns:
            float: Bias-corrected salary prediction
        """
        correction = 0
        
        # Apply age group correction
        if 'age_group' in demographics and demographics['age_group'] in self.bias_corrections['age_group']:
            correction += self.bias_corrections['age_group'][demographics['age_group']]
        
        # Apply gender correction
        if 'gender' in demographics and demographics['gender'] in self.bias_corrections['gender']:
            correction += self.bias_corrections['gender'][demographics['gender']]
        
        # Apply ethnicity correction
        if 'ethnicity' in demographics and demographics['ethnicity'] in self.bias_corrections['ethnicity']:
            correction += self.bias_corrections['ethnicity'][demographics['ethnicity']]
        
        # Apply correction (ensure salary doesn't go negative)
        corrected_prediction = max(0, prediction + correction)
        
        return corrected_prediction
    
    def get_bias_report(self):
        """Generate a detailed bias report with recommendations"""
        report = """
# Bias Mitigation Report for TRU Salary Prediction

## Detected Biases

### Age Group Bias
- **Early Career**: Underpredicted by ₹2.7L (102 people affected)
- **Mid Career**: Slightly overpredicted by ₹18K (394 people affected)
- **Senior Career**: Overpredicted by ₹53K (468 people affected)
- **Late Career**: Overpredicted by ₹1.2L (36 people affected)

### Gender Bias
- **Female**: Underpredicted by ₹51K (347 people affected)
- **Male**: Slightly underpredicted by ₹5K (327 people affected)
- **Other**: Overpredicted by ₹86K (326 people affected)

### Ethnicity Bias
- **Black**: Underpredicted by ₹1.5L (197 people affected)
- **Other**: Slightly underpredicted by ₹16K (225 people affected)
- **White**: Overpredicted by ₹29K (186 people affected)
- **Asian**: Overpredicted by ₹98K (211 people affected)
- **Hispanic**: Overpredicted by ₹86K (181 people affected)

## Bias Correction Factors Applied

### Age Group Corrections
- Early Career: +₹2,69,531
- Mid Career: -₹18,432
- Senior Career: -₹52,752
- Late Career: -₹1,24,810

### Gender Corrections
- Female: +₹51,005
- Male: +₹4,590
- Other: -₹86,352

### Ethnicity Corrections
- Black: +₹1,47,025
- Other: +₹16,478
- White: -₹28,899
- Asian: -₹97,601
- Hispanic: -₹86,484

## Recommendations

### 1. Data Collection Improvements
- Ensure balanced representation across all demographic groups
- Collect more data for underrepresented groups (Early Career, Black, Female)
- Include intersectional analysis (e.g., Black women, Asian seniors)

### 2. Model Improvements
- Use fairness-aware algorithms (e.g., Adversarial Debiasing)
- Implement demographic parity constraints
- Regular bias audits and retraining

### 3. Business Practices
- Review hiring and promotion practices
- Implement blind resume screening
- Regular diversity and inclusion training

### 4. Monitoring
- Continuous bias monitoring in production
- Regular model retraining with updated data
- Stakeholder feedback collection

## Impact Assessment

### Positive Impact
- Fairer salary predictions for underrepresented groups
- Reduced discrimination in compensation decisions
- Better compliance with anti-discrimination laws
- Improved organizational reputation

### Considerations
- Slight reduction in overall model accuracy
- Need for regular bias monitoring
- Potential resistance to change in existing processes
- Legal and ethical compliance requirements

## Implementation Notes

This bias mitigation system:
1. Automatically applies corrections based on detected biases
2. Ensures predictions remain positive
3. Provides transparency in bias correction
4. Can be easily updated as new bias patterns emerge
"""
        return report
    
    def save_bias_report(self, filename='bias_mitigation_report.md'):
        """Save bias report to file"""
        report = self.get_bias_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Bias mitigation report saved to {filename}")

def main():
    """Test the bias mitigation system"""
    try:
        mitigator = BiasMitigator()
        
        # Test bias correction
        test_cases = [
            {
                'prediction': 5000000,  # ₹50L
                'demographics': {'age_group': 'Early Career', 'gender': 'Female', 'ethnicity': 'Black'},
                'description': 'Early Career Black Female'
            },
            {
                'prediction': 3000000,  # ₹30L
                'demographics': {'age_group': 'Late Career', 'gender': 'Male', 'ethnicity': 'Asian'},
                'description': 'Late Career Asian Male'
            },
            {
                'prediction': 4000000,  # ₹40L
                'demographics': {'age_group': 'Mid Career', 'gender': 'Other', 'ethnicity': 'White'},
                'description': 'Mid Career Other White'
            }
        ]
        
        print("=== Bias Mitigation Test Results ===\n")
        
        for case in test_cases:
            original = case['prediction']
            corrected = mitigator.apply_bias_correction(original, case['demographics'])
            difference = corrected - original
            
            print(f"Case: {case['description']}")
            print(f"Original Prediction: ₹{original:,}")
            print(f"Corrected Prediction: ₹{corrected:,}")
            print(f"Bias Correction: ₹{difference:,}")
            print(f"Correction Percentage: {(difference/original)*100:.2f}%\n")
        
        # Save bias report
        mitigator.save_bias_report()
        
    except Exception as e:
        print(f"Error testing bias mitigation: {e}")

if __name__ == "__main__":
    main() 