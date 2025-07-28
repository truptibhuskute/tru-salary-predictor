"""
Advanced Model Trainer for TRU Salary Prediction
Copyright (c) 2025 TRU Salary Predictor
Contact: contact@trusalaryprediction.com

Trains and evaluates multiple machine learning models for salary prediction.
Includes bias analysis, feature importance, and model comparison.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedSalaryModelTrainer:
    """
    Advanced model trainer with multiple algorithms and bias detection
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.bias_analysis = {}
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.skills_mlb = MultiLabelBinarizer()
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the advanced dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Separate features and target
        target_col = 'Salary'
        feature_cols = [col for col in df.columns if col not in ['ID', target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle skills column separately
        if 'Skills' in categorical_cols:
            skills_col = 'Skills'
            categorical_cols.remove(skills_col)
            
            # Process skills
            X['Skills'] = X['Skills'].fillna('')
            X['Skills'] = X['Skills'].apply(lambda x: [skill.strip() for skill in x.split(',') if skill.strip()])
            
            # Fit and transform skills
            skills_encoded = self.skills_mlb.fit_transform(X['Skills'])
            skill_cols = [f'Skill_{s}' for s in self.skills_mlb.classes_]
            skills_df = pd.DataFrame(skills_encoded, columns=skill_cols, index=X.index)
            
            # Remove original skills column and add encoded
            X = X.drop('Skills', axis=1)
            X = pd.concat([X, skills_df], axis=1)
            
            # Update numeric columns
            numeric_cols.extend(skill_cols)
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, labels=False)
        )
        
        # Fit preprocessor
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = numeric_cols.copy()
        
        # Add categorical feature names
        if 'cat' in preprocessor.named_transformers_:
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                onehot = cat_transformer.named_steps['onehot']
                if hasattr(onehot, 'categories_'):
                    for i, col in enumerate(categorical_cols):
                        if i < len(onehot.categories_):
                            for val in onehot.categories_[i]:
                                feature_names.append(f"{col}_{val}")
        
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        return X_train_processed, X_test_processed, y_train, y_test, X_train, X_test
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and compare performance"""
        print("Training multiple models...")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42),
            'SVR': SVR()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(self.feature_names, model.feature_importances_))
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:,.0f}")
            print(f"  MAE: {mae:,.0f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """Perform hyperparameter tuning for the best model"""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def analyze_bias(self, X_train, X_test, y_train, y_test, original_X_train, original_X_test):
        """Analyze model bias across different demographic groups"""
        print("Analyzing model bias...")
        
        # Use the best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_model = self.models[best_model_name]['model']
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        bias_analysis = {}
        
        # Analyze bias by gender
        if 'Gender' in original_X_train.columns:
            gender_bias = self._analyze_group_bias(
                original_X_test, y_test, y_pred_test, 'Gender'
            )
            bias_analysis['gender'] = gender_bias
        
        # Analyze bias by ethnicity
        if 'Ethnicity' in original_X_train.columns:
            ethnicity_bias = self._analyze_group_bias(
                original_X_test, y_test, y_pred_test, 'Ethnicity'
            )
            bias_analysis['ethnicity'] = ethnicity_bias
        
        # Analyze bias by age group
        if 'AgeGroup' in original_X_train.columns:
            age_bias = self._analyze_group_bias(
                original_X_test, y_test, y_pred_test, 'AgeGroup'
            )
            bias_analysis['age_group'] = age_bias
        
        self.bias_analysis = bias_analysis
        return bias_analysis
    
    def _analyze_group_bias(self, X, y_true, y_pred, group_col):
        """Analyze bias for a specific group"""
        groups = X[group_col].unique()
        bias_results = {}
        
        for group in groups:
            mask = X[group_col] == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if len(group_y_true) > 0:
                # Calculate metrics for this group
                mae = mean_absolute_error(group_y_true, group_y_pred)
                rmse = np.sqrt(mean_squared_error(group_y_true, group_y_pred))
                r2 = r2_score(group_y_true, group_y_pred)
                
                # Calculate bias (average prediction error)
                bias = np.mean(group_y_pred - group_y_true)
                
                bias_results[group] = {
                    'count': len(group_y_true),
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'bias': bias,
                    'avg_true_salary': group_y_true.mean(),
                    'avg_pred_salary': group_y_pred.mean()
                }
        
        return bias_results
    
    def create_visualizations(self, X_test, y_test, y_pred):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Salary')
        axes[0, 0].set_ylabel('Predicted Salary')
        axes[0, 0].set_title('Predicted vs Actual Salary')
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Salary')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Residuals distribution
        axes[0, 2].hist(residuals, bins=50, alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        
        # 4. Feature importance (for best model)
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        if best_model_name in self.feature_importance:
            importance = self.feature_importance[best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, scores = zip(*top_features)
            axes[1, 0].barh(range(len(features)), scores)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title(f'Top 10 Features ({best_model_name})')
        
        # 5. Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        axes[1, 1].bar(model_names, r2_scores)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Bias analysis
        if self.bias_analysis:
            bias_data = []
            for group_type, groups in self.bias_analysis.items():
                for group, metrics in groups.items():
                    bias_data.append({
                        'Group': f"{group_type}: {group}",
                        'Bias': metrics['bias'],
                        'Count': metrics['count']
                    })
            
            if bias_data:
                bias_df = pd.DataFrame(bias_data)
                axes[1, 2].bar(bias_df['Group'], bias_df['Bias'])
                axes[1, 2].set_ylabel('Bias (Predicted - Actual)')
                axes[1, 2].set_title('Bias Analysis by Group')
                axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_artifacts(self, model_name='XGBoost'):
        """Save the best model and all artifacts"""
        print("Saving model and artifacts...")
        
        # Save best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_model = self.models[best_model_name]['model']
        
        # Save model
        joblib.dump(best_model, 'model/best_model.pkl')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'model/preprocessor.pkl')
        
        # Save skills binarizer
        joblib.dump(self.skills_mlb, 'model/skills_mlb.pkl')
        
        # Save feature names
        with open('model/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save model metrics
        metrics = {
            'best_model': best_model_name,
            'models': {name: {k: v for k, v in data.items() if k != 'model'} 
                      for name, data in self.models.items()},
            'feature_importance': self.feature_importance,
            'bias_analysis': self.bias_analysis
        }
        
        with open('model/advanced_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Best model ({best_model_name}) and artifacts saved to 'model/' directory")
    
    def generate_report(self):
        """Generate comprehensive model report"""
        print("Generating model report...")
        
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_metrics = self.models[best_model_name]
        
        report = f"""
# Advanced Salary Prediction Model Report

## Model Performance Summary

**Best Model**: {best_model_name}
**R² Score**: {best_metrics['r2']:.4f}
**RMSE**: {best_metrics['rmse']:,.0f}
**MAE**: {best_metrics['mae']:,.0f}
**Cross-Validation R²**: {best_metrics['cv_mean']:.4f} (+/- {best_metrics['cv_std'] * 2:.4f})

## All Models Performance

"""
        
        for name, metrics in self.models.items():
            report += f"""
### {name}
- R² Score: {metrics['r2']:.4f}
- RMSE: {metrics['rmse']:,.0f}
- MAE: {metrics['mae']:,.0f}
- CV R²: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})

"""
        
        # Feature importance
        if best_model_name in self.feature_importance:
            report += "## Top 10 Most Important Features\n\n"
            importance = self.feature_importance[best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for feature, score in top_features:
                report += f"- {feature}: {score:.4f}\n"
        
        # Bias analysis
        if self.bias_analysis:
            report += "\n## Bias Analysis\n\n"
            for group_type, groups in self.bias_analysis.items():
                report += f"### {group_type.title()}\n\n"
                for group, metrics in groups.items():
                    report += f"- **{group}**: Bias = {metrics['bias']:,.0f}, Count = {metrics['count']}\n"
        
        # Save report
        with open('model_report.md', 'w') as f:
            f.write(report)
        
        print("Model report saved to 'model_report.md'")
        return report

def main():
    """Main training pipeline"""
    print("Starting Advanced Salary Model Training Pipeline...")
    
    # Initialize trainer
    trainer = AdvancedSalaryModelTrainer()
    
    # Load and preprocess data
    try:
        # Try to find the most recent advanced dataset
        import glob
        advanced_files = glob.glob('Advanced_Salary_Dataset_*.csv')
        if advanced_files:
            latest_file = max(advanced_files, key=lambda x: os.path.getctime(x))
            print(f"Using dataset: {latest_file}")
            X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = trainer.load_and_preprocess_data(latest_file)
        else:
            raise FileNotFoundError("No advanced dataset found")
    except FileNotFoundError:
        print("Advanced dataset not found. Please run advanced_data_generator.py first.")
        return
    
    # Train multiple models
    results = trainer.train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\nBest model: {best_model_name}")
    
    # Analyze bias
    bias_analysis = trainer.analyze_bias(X_train, X_test, y_train, y_test, X_train_orig, X_test_orig)
    
    # Create visualizations
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)
    trainer.create_visualizations(X_test, y_test, y_pred)
    
    # Save model and artifacts
    trainer.save_model_and_artifacts()
    
    # Generate report
    report = trainer.generate_report()
    
    print("\n=== Training Complete ===")
    print(f"Best model: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['r2']:.4f}")
    print(f"RMSE: {results[best_model_name]['rmse']:,.0f}")

if __name__ == "__main__":
    main() 