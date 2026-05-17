# 🚗 CarDekho Used Car Price Prediction Project

## 📌 Project Overview

This project focuses on analyzing used car market data from CarDekho and building a machine learning model to predict the selling price of used cars based on different features such as:

- Car Name / Brand
- Manufacturing Year
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission Type
- Ownership History
- Mileage
- Engine Capacity
- Maximum Power
- Number of Seats

The main goal of this project is to understand how different vehicle attributes affect resale value and create a predictive system that estimates a fair selling price.

---

# 🎯 Problem Statement

Buying or selling a used car often creates confusion about the correct market price.

The price depends on multiple factors:

- Age of the car
- Brand value
- Fuel type
- Mileage
- Vehicle condition
- Engine power
- Ownership history
- Transmission type

Manually estimating the price is difficult.

This project solves that problem using **Data Analysis + Machine Learning**.

---

# 📂 Dataset Information

Dataset Name: **CarDekho Used Cars Dataset**

Source: CarDekho marketplace dataset

File Used:
```bash
cardekho.csv
```

Dataset Shape:

```python
8128 rows × 12 columns
```

---

# 📊 Dataset Features

| Feature Name | Description |
|------------|-------------|
| name | Car brand and model name |
| year | Manufacturing year |
| selling_price | Target variable (car selling price) |
| km_driven | Total kilometers driven |
| fuel | Fuel type (Petrol / Diesel / CNG / LPG / Electric) |
| seller_type | Individual / Dealer / Trustmark Dealer |
| transmission | Manual / Automatic |
| owner | Ownership history |
| mileage(km/ltr/kg) | Mileage efficiency |
| engine | Engine displacement in CC |
| max_power | Maximum power output |
| seats | Number of seats |

---

# 🛠️ Technologies Used

## Programming Language
- Python 3.x

## Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib / Pickle

Optional:
- Streamlit (for deployment)
- Flask / FastAPI (for API deployment)

Installation:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

If deploying app:

```bash
pip install streamlit
```

---

# 📁 Project Structure

```bash
Car-Price-Prediction/
│
├── cardekho.csv
├── notebook.ipynb
├── data_preprocessing.py
├── model_training.py
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

# 🔍 Project Workflow

## 1. Data Collection

The dataset contains used car listings collected from CarDekho.

Example:

```python
df = pd.read_csv("cardekho.csv")
```

---

## 2. Data Understanding

Basic exploration:

```python
df.head()
df.info()
df.describe()
df.shape
```

Check missing values:

```python
df.isnull().sum()
```

---

## 3. Data Cleaning

Raw data may contain:

- Missing values
- Mixed data types
- Duplicate records
- String formatting issues

Cleaning steps:

### Missing Value Handling

```python
df.dropna(inplace=True)
```

OR

```python
df.fillna(method='ffill')
```

---

### Removing Duplicates

```python
df.drop_duplicates(inplace=True)
```

---

### Feature Formatting

Columns like:

```python
max_power
mileage(km/ltr/kg)
```

may require datatype conversion.

Example:

```python
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
```

---

# 📈 Exploratory Data Analysis (EDA)

EDA helps understand trends and relationships.

## Univariate Analysis

Examples:

- Fuel type distribution
- Transmission distribution
- Ownership count

```python
sns.countplot(x='fuel', data=df)
```

---

## Bivariate Analysis

Examples:

- Price vs Year
- Price vs Kilometers Driven
- Price vs Engine

```python
sns.scatterplot(x='year', y='selling_price', data=df)
```

---

## Correlation Analysis

```python
sns.heatmap(df.corr(), annot=True)
```

Helps identify important numeric relationships.

---

# ⚙️ Feature Engineering

Useful engineered features:

## Car Age

Instead of year:

```python
current_year = 2026
df['car_age'] = current_year - df['year']
```

This improves model learning.

---

## Brand Extraction

From car name:

```python
df['brand'] = df['name'].apply(lambda x: x.split()[0])
```

Useful because brand strongly affects resale price.

---

# 🔄 Data Preprocessing

Machine learning requires numeric inputs.

Categorical columns:

- fuel
- seller_type
- transmission
- owner
- brand

Encoding:

```python
pd.get_dummies()
```

OR

```python
LabelEncoder()
```

Feature scaling if needed:

```python
StandardScaler()
```

---

# 🤖 Model Building

Target:

```python
selling_price
```

Features:

All relevant processed features except target.

Train-test split:

```python
from sklearn.model_selection import train_test_split
```

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# 🧠 Machine Learning Models

Models you can try:

## Linear Regression

Good baseline model.

```python
LinearRegression()
```

---

## Decision Tree Regressor

Captures non-linear relationships.

```python
DecisionTreeRegressor()
```

---

## Random Forest Regressor

Usually strong for tabular data.

```python
RandomForestRegressor()
```

---

## Gradient Boosting

Better performance in many cases.

```python
GradientBoostingRegressor()
```

---

## XGBoost (Optional)

Advanced high-performance model.

---

# 📏 Model Evaluation

Regression metrics:

## MAE

Mean Absolute Error

```python
mean_absolute_error()
```

---

## MSE

Mean Squared Error

```python
mean_squared_error()
```

---

## RMSE

Root Mean Squared Error

Measures prediction error in actual units.

---

## R² Score

Best for overall model performance.

```python
r2_score()
```

---

Example:

```python
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

---

# 💾 Model Saving

Save trained model:

```python
import pickle

pickle.dump(model, open('model.pkl', 'wb'))
```

Load later:

```python
model = pickle.load(open('model.pkl', 'rb'))
```

---

# 🌐 Deployment

Possible deployment options:

## Streamlit App

Run:

```bash
streamlit run app.py
```

Features:

- User input form
- Predict selling price
- Interactive UI

---

## FastAPI Deployment

Run API:

```bash
uvicorn app:app --reload
```

---

# 🚀 Example Prediction Inputs

Example:

- Brand: Hyundai
- Year: 2018
- Fuel: Diesel
- Transmission: Manual
- KM Driven: 55,000
- Owner: First Owner
- Engine: 1493
- Max Power: 115
- Seats: 5

Output:

```bash
Predicted Price: ₹6,75,000
```

---

# 📌 Key Insights

Possible insights:

- Newer cars sell at higher prices.
- Lower kilometers driven increases value.
- Diesel cars may have higher resale in some segments.
- Automatic cars often cost more.
- Premium brands retain better value.
- Multiple ownership reduces selling price.

---

# 🔮 Future Improvements

Planned enhancements:

- Hyperparameter tuning
- Cross-validation
- Advanced feature engineering
- XGBoost/CatBoost models
- Model explainability (SHAP)
- Streamlit web app
- FastAPI backend deployment
- Docker containerization
- Cloud deployment

---

# 📚 Learning Outcomes

This project helps practice:

✅ Data Cleaning  
✅ EDA  
✅ Feature Engineering  
✅ Regression Modeling  
✅ Model Evaluation  
✅ Deployment Basics  
✅ End-to-End ML Workflow

---

# ▶️ How to Run This Project

Clone repository:

```bash
git clone https://github.com/yourusername/car-price-prediction.git
```

Move into folder:

```bash
cd car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebook or app:

```bash
jupyter notebook
```

OR

```bash
streamlit run app.py
```

---

# 👨‍💻 Author

Your Name

GitHub:
```bash
https://github.com/yourusername
```

LinkedIn:
```bash
https://linkedin.com/in/yourprofile
```

---

# ⭐ If You Like This Project

If this project helped you learn something useful:

⭐ Star the repository  
🍴 Fork the project  
📢 Share with others  

---
