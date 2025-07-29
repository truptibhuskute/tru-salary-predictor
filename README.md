# 💼 TRU Salary Prediction Pro

**A highly accurate and interpretable machine learning model that predicts employee salaries, enabling organizations to make data-driven decisions for compensation, recruitment, and retention, and helping individuals understand their market value.**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Quality](#data-quality)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Overview

TRU Salary Prediction Pro is an advanced salary prediction system specifically designed for the Indian job market. It combines cutting-edge machine learning algorithms with ethical AI practices to provide accurate, fair, and interpretable salary predictions.

### Key Highlights:
- **90.82% Model Accuracy** (LightGBM)
- **Bias Detection & Mitigation** for fair predictions
- **Indian Market Localization** with 20+ cities
- **SHAP Explanations** for interpretability
- **Real-time Validation** and error detection
- **Batch Processing** capabilities

---

## ✨ Features

### 🎯 Core Functionality
- **Single Salary Prediction**: Individual employee salary estimation
- **Batch Processing**: Bulk salary predictions for multiple employees
- **Real-time Validation**: Input data quality checks
- **Confidence Intervals**: Salary range predictions with uncertainty

### 🔍 Advanced Analytics
- **Bias Analysis**: Demographic fairness assessment
- **SHAP Explanations**: Feature importance for individual predictions
- **Model Insights**: Performance metrics and feature importance
- **Data Validation**: Comprehensive quality assurance

### 🏢 Indian Market Focus
- **20 Indian Cities**: Mumbai, Bangalore, Delhi, Hyderabad, Chennai, Pune, Gurgaon, Noida, Kolkata, Ahmedabad, Indore, Jaipur, Chandigarh, Vadodara, Coimbatore, Kochi, Bhubaneswar, Nagpur, Lucknow, Patna
- **Realistic Salary Ranges**: ₹4L - ₹1.2Cr based on Indian market
- **Cost of Living Adjustments**: City-specific multipliers
- **Indian Job Roles**: 10 major roles with realistic hierarchies

### 🎯 Bias Detection & Mitigation
- **Automatic Bias Detection**: Identifies age, gender, and ethnicity biases
- **Fairness Corrections**: Real-time bias mitigation
- **Transparency Reports**: Detailed bias analysis
- **Compliance Ready**: Anti-discrimination law compliance

---

## 📁 Project Structure

```
TRU_Salary_Prediction/
├── 📊 Data Generation
│   └── advanced_data_generator.py          # Generate realistic Indian salary data
├── 🤖 Model Training
│   └── advanced_model_trainer.py           # Train and evaluate ML models
├── 🎯 Bias Mitigation
│   └── bias_mitigation.py                  # Bias detection and correction
├── 🌐 Web Application
│   └── advanced_salary_app.py              # Streamlit web interface
├── 📈 Model Artifacts
│   ├── model/
│   │   ├── model.pkl                       # Trained LightGBM model
│   │   ├── preprocessor.pkl                # Data preprocessing pipeline
│   │   ├── skills_mlb.pkl                  # Skills encoding
│   │   ├── metrics.json                    # Model performance metrics
│   │   └── feature_names.json              # Feature names mapping
│   └── model_report.md                     # Detailed model analysis
├── 📊 Generated Data
│   └── Advanced_Salary_Dataset_*.csv       # Generated datasets
├── 📋 Documentation
│   ├── README.md                           # Project documentation
│   ├── requirements.txt                    # Python dependencies
│   └── bias_mitigation_report.md           # Bias analysis report
└── 🎯 Reports
    └── model_report.md                     # Model performance report
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/TRU-Salary-Prediction.git
cd TRU-Salary-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Data
```bash
python advanced_data_generator.py
```

### Step 4: Train Model
```bash
python advanced_model_trainer.py
```

### Step 5: Run Application
```bash
streamlit run advanced_salary_app.py
```

The application will be available at: **http://localhost:8507**

---

## 📖 Usage

### 🎯 Single Prediction
1. **Navigate to Single Prediction tab**
2. **Fill in employee details**:
   - Demographics (Age, Gender, Ethnicity)
   - Experience (Total years, Current role, Company tenure)
   - Job details (Role, Seniority, Department, Industry)
   - Education (Level, Major, University tier)
   - Location (City, State)
   - Skills (Technical and soft skills)
   - Performance metrics

3. **Submit and get results**:
   - Predicted salary with confidence interval
   - Bias correction applied (if any)
   - SHAP explanation of factors
   - Validation status

### 📊 Batch Processing
1. **Navigate to Batch Prediction tab**
2. **Upload CSV file** with employee data
3. **Validate data** for quality and consistency
4. **Process predictions** for all employees
5. **Download results** with bias corrections

### 🔍 Bias Analysis
1. **Navigate to Bias Analysis tab**
2. **View demographic bias** across age, gender, ethnicity
3. **Analyze fairness metrics** and recommendations
4. **Review bias correction** strategies

### 📈 Model Insights
1. **Navigate to Model Insights tab**
2. **View feature importance** rankings
3. **Analyze model performance** metrics
4. **Review cross-validation** results

---

## 📊 Data Quality

### ✅ Validation Rules
- **Age**: 18-70 years
- **Experience**: 0-25 years (must be ≤ age - 18)
- **Performance Rating**: 1.0-5.0
- **Salary Range**: ₹2.7L - ₹4.5Cr (realistic Indian market)
- **Skills**: 3-8 skills per employee
- **Education**: Valid education levels and majors

### 🎯 Salary Ranges by Role & Seniority

| Role | Junior | Mid | Senior | Lead | Manager/Director |
|------|--------|-----|--------|------|------------------|
| **Data Scientist** | ₹6-12L | ₹12-20L | ₹20-35L | ₹35-50L | ₹50-80L |
| **Software Engineer** | ₹5-10L | ₹10-18L | ₹18-30L | ₹30-45L | ₹45-70L |
| **Product Manager** | ₹8-15L | ₹15-25L | ₹25-40L | ₹40-60L | ₹60-120L |
| **Sales Manager** | ₹6-12L | ₹12-20L | ₹20-35L | ₹35-50L | ₹50-80L |
| **Marketing Manager** | ₹5-10L | ₹10-18L | ₹18-30L | ₹30-45L | ₹45-70L |
| **HR Manager** | ₹4-8L | ₹8-15L | ₹15-25L | ₹25-40L | ₹40-60L |
| **Finance Manager** | ₹6-12L | ₹12-20L | ₹20-35L | ₹35-50L | ₹50-80L |
| **DevOps Engineer** | ₹6-12L | ₹12-20L | ₹20-35L | ₹35-50L | ₹50-80L |
| **UI/UX Designer** | ₹4-8L | ₹8-15L | ₹15-25L | ₹25-40L | ₹40-60L |
| **Data Engineer** | ₹6-12L | ₹12-20L | ₹20-35L | ₹35-50L | ₹50-80L |

### 🏙️ Cost of Living Multipliers
- **Mumbai**: 1.4x (highest)
- **Bangalore**: 1.3x (tech hub)
- **Delhi**: 1.2x (capital)
- **Chennai**: 1.0x (baseline)
- **Pune**: 0.95x
- **Tier 2/3 Cities**: 0.75x-0.9x

---

## 🎯 Model Performance

### 📊 Overall Performance
- **Best Model**: LightGBM
- **R² Score**: 90.82%
- **RMSE**: ₹17,01,186
- **MAE**: ₹10,09,448
- **Cross-Validation R²**: 89.00% (±1.53%)

### 🏆 Model Comparison
| Model | R² Score | RMSE | MAE | CV R² |
|-------|----------|------|-----|-------|
| **LightGBM** | 90.82% | ₹17.01L | ₹10.09L | 89.00% |
| **XGBoost** | 87.87% | ₹19.55L | ₹12.10L | 84.79% |
| **Gradient Boosting** | 85.11% | ₹21.66L | ₹13.06L | 83.22% |
| **Random Forest** | 78.64% | ₹25.94L | ₹16.07L | 75.45% |
| **Linear Regression** | 78.75% | ₹25.87L | ₹17.62L | 77.15% |

### 🔍 Top 10 Features
1. **Job Level** (338 points)
2. **Cost of Living Multiplier** (256 points)
3. **Total Years Experience** (187 points)
4. **Performance Rating** (149 points)
5. **Certifications** (134 points)
6. **Years Since Graduation** (109 points)
7. **Company Age** (105 points)
8. **Education Level (MD)** (97 points)
9. **Years at Company** (85 points)
10. **Industry (Technology)** (81 points)

---

## 🔍 Bias Analysis

### 📊 Detected Biases
- **Early Career**: Underpredicted by ₹2.7L (102 people)
- **Female**: Underpredicted by ₹51K (347 people)
- **Black**: Underpredicted by ₹1.5L (197 people)
- **Late Career**: Overpredicted by ₹1.2L (36 people)
- **Asian**: Overpredicted by ₹98K (211 people)

### 🎯 Bias Correction Applied
- **Automatic Correction**: Real-time fairness adjustments
- **Transparency**: Clear reporting of bias corrections
- **Compliance**: Anti-discrimination law adherence
- **Monitoring**: Continuous bias detection

---

## 📚 API Documentation

### Core Classes

#### `AdvancedSalaryPredictor`
Main prediction class with bias detection and correction.

```python
predictor = AdvancedSalaryPredictor()

# Single prediction
result = predictor.predict_salary(input_data)
# Returns: {'prediction': salary, 'bias_correction': amount, ...}

# Bias analysis
bias = predictor.analyze_bias(input_data, prediction)

# SHAP explanation
explanation = predictor.explain_prediction(input_data)
```

#### `BiasMitigator`
Handles bias detection and correction.

```python
mitigator = BiasMitigator()

# Apply bias correction
corrected_salary = mitigator.apply_bias_correction(
    prediction, demographics
)

# Generate bias report
report = mitigator.get_bias_report()
```

### Key Methods

#### `predict_salary(input_data)`
- **Input**: DataFrame with employee features
- **Output**: Dictionary with prediction and metadata
- **Features**: Automatic bias correction, confidence intervals

#### `analyze_bias(input_data, prediction)`
- **Input**: Employee data and prediction
- **Output**: Bias analysis across demographics
- **Features**: Gender, age, ethnicity bias detection

#### `explain_prediction(input_data)`
- **Input**: Employee data
- **Output**: SHAP explanation with feature importance
- **Features**: Individual prediction interpretability

---

## 🤝 Contributing

We welcome contributions to improve TRU Salary Prediction Pro!

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- **Code Style**: Follow PEP 8 standards
- **Documentation**: Add docstrings and comments
- **Testing**: Include unit tests for new features
- **Bias Awareness**: Ensure new features don't introduce bias

### Areas for Improvement
- **Additional Models**: Neural networks, ensemble methods
- **More Features**: Industry-specific factors, market trends
- **API Development**: REST API for integration
- **Mobile App**: React Native or Flutter application
- **Advanced Analytics**: Time series analysis, market predictions

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Copyright Notice
```
Copyright (c) 2025 TRU Salary Predictor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

### Project Maintainer
- **Name**: TRU Salary Prediction Team
- **Email**: bhuskutetrupti@gmail.com
- **GitHub**: [github.com/trusalaryprediction](https://github.com/trusalaryprediction)



- **Bug Reports**: bhuskutetrupti@gmail.com


---

## 🙏 Acknowledgments

- **Data Sources**: Inspired by real-world salary surveys and market data
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, SHAP
- **Web Framework**: Streamlit for the user interface
- **Ethics Research**: Fairness in AI and bias mitigation techniques
- **Indian Market Research**: Local salary trends and cost of living data

