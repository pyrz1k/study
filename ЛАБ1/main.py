"""Module providing a function printing python version."""
# Загрузка дата сета и разбиение данных
import pandas as pd
from sklearn.model_selection import train_test_split

File_Path = '/Users/ivanscherbakov/Documents/ВУЗ/ИТиП/ЛАБ1/Laptop_price.csv'
df = pd.read_csv(File_Path)
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Создание пайплайна для предобработки данных и обучения модели
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

pipeline = Pipeline([
    ('prepocessor', preprocessor),
    ('model', XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 5))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, '/Users/ivanscherbakov/Documents/ВУЗ/ИТиП/ЛАБ1/Laptop_price_model.pkl')

