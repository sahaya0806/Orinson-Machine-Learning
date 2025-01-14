import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
file_path = '/mnt/data/house_prediction_dataset/Housing.csv'
data = pd.read_csv('Housing.csv')
X = data.drop('price', axis=1)
y = data['price']
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_preprocessor = OneHotEncoder(drop='first', handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('cat', categorical_preprocessor, categorical_columns)], remainder='passthrough')
model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
("The predicted results for the test sets")
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
