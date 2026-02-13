# Crop Yield Predictor

A machine learning web application that predicts crop yield based on field conditions such as soil type, climate, and nutrient levels. Built with Python, Scikit-learn, and Streamlit.

**Live App:** https://maxwelltebi-cropyield-prediction.streamlit.app/


## Overview

This project applies **Linear Regression** to predict agricultural crop yield based on a combination of environmental and soil input features. The goal is to demonstrate how machine learning can support data-driven decision-making in agriculture and food security.

The model was trained on a crop yield dataset covering 11 crop types across varying soil conditions, climate readings, and fertilizer applications.

---

## Features

- Predicts crop yield from user-provided field conditions
- Clean, interactive web interface built with Streamlit
- Supports 10 crop types, 5 soil types, and 7 fertilizer options
- Instant predictions with no setup required for end users

---

## Model Details

| Property | Detail |
|---|---|
| Algorithm | Linear Regression |
| Library | Scikit-learn |
| Encoding | One-Hot Encoding for categorical variables |
| Scaling | StandardScaler |
| Evaluation Metrics | R² Score, MAE, RMSE |
| R² Score | 0.8398 (model trained without sugarcane outlier) |

### Features Used

| Feature | Type |
|---|---|
| Temperature | Numeric |
| Humidity | Numeric |
| Moisture | Numeric |
| Nitrogen | Numeric |
| Potassium | Numeric |
| Phosphorus | Numeric |
| Soil Type | Categorical |
| Crop Type | Categorical |
| Fertilizer Name | Categorical |

---

## Key Findings

- **Sugarcane** was identified as a significant outlier, with yields far exceeding all other crop types (~8,000 units vs. ~1,500-5,000 for others). Including it inflated the R² to 0.9251 by giving the model an easy signal to exploit, while the model trained *without* sugarcane achieved a more honest R² of **0.8398** — reflecting genuine predictive power across staple crops.
- Nutrient levels (Nitrogen, Potassium, Phosphorus) and soil type were among the strongest predictors of yield variation.

---

## Tech Stack

- **Python 3**
- **Pandas and NumPy** — data manipulation
- **Scikit-learn** — model training and preprocessing
- **Matplotlib and Seaborn** — exploratory data analysis
- **Streamlit** — web application and deployment
- **Joblib** — model serialization

---

## Project Structure

```
crop_yield_app/
│
├── app.py                  # Streamlit dashboard
├── crop_yield_model.pkl    # Trained Linear Regression model
├── crop_yield_scaler.pkl   # Fitted StandardScaler
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/crop-yield-app.git
cd crop-yield-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## Future Improvements

- Expand to more crop types including sugarcane as a separate dedicated model
- Experiment with Ridge and Lasso regression to handle multicollinearity among nutrient features
- Add confidence intervals to predictions
- Incorporate real-world geospatial data (region, elevation, season)
- Explore non-linear models such as Random Forest and Gradient Boosting for performance comparison

---

## Author

**Your Name**
- GitHub: https://github.com/Maxwelltebi
- LinkedIn: www.linkedin.com/in/maxwelltebi

---

## License

This project is open source and available under the [MIT License](LICENSE).
