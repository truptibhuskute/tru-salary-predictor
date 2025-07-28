# 🚀 TRU Salary Predictor - Deployment Guide

## 📋 Prerequisites

- Python 3.8+
- Git
- GitHub account
- Deployment platform account (Streamlit Cloud, Heroku, etc.)

## 🎯 Quick Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/truptibhuskute/tru-salary-predictor.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

### Option 2: Heroku

1. **Install Heroku CLI:**
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy:**
   ```bash
   heroku login
   heroku create tru-salary-predictor
   git push heroku main
   ```

### Option 3: Railway

1. **Connect to Railway:**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Set environment variables if needed
   - Deploy automatically

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
tru-salary-predictor/
├── streamlit_app.py          # Main deployment entry point
├── polished_salary_app.py    # Premium UI version
├── modern_salary_app.py      # Modern UI version
├── advanced_salary_app.py    # Advanced features version
├── advanced_data_generator.py # Data generation
├── advanced_model_trainer.py # Model training
├── data_validation.py        # Data validation
├── bias_mitigation.py        # Bias detection
├── requirements.txt          # Dependencies
├── Procfile                 # Heroku deployment
├── setup.sh                 # Setup script
├── runtime.txt              # Python version
├── app.json                 # App metadata
└── README.md               # Documentation
```

## 🌐 Environment Variables

Set these in your deployment platform:

- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## 🔒 Security Notes

- Model files are in `.gitignore` to prevent large file uploads
- Data files are excluded for privacy
- Environment variables for sensitive data

## 📞 Support

- **Email**: bhuskutetrupti@gmail.com
- **Copyright**: © 2025 TRU Salary Predictor

## 🎉 Success!

Your TRU Salary Predictor will be live at your deployment URL! 