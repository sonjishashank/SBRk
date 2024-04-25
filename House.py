from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app)

# PostgreSQL connection parameters
hostname = "dpg-col27k8l5elc73djd4b0-a.singapore-postgres.render.com"
port = 5432
database = "dishuu"
username = "dishuu_user"
password = "dkuwOzMtN38Unv5J3umsmSTuezn5XEUa"
table_name = "your_table_name"  # Replace with your table name

# Create connection string
connection_str = f"postgresql://{username}:{password}@{hostname}:{port}/{database}"

# Create SQLAlchemy engine
engine = create_engine(connection_str)

# Fetch data from PostgreSQL table
query = f"SELECT * FROM {table_name}"
dataset = pd.read_sql(query, engine)

# Data analysis and modeling
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
regression = LinearRegression()
regression.fit(X_train, y_train)
reg_pred = regression.predict(X_test)

# Random Forest Regression
forest = RandomForestRegressor(max_depth=5, n_estimators=80, random_state=99)
forest.fit(X_train, y_train)
y_pred_rf = forest.predict(X_test)

# Model evaluation
mse_linear = mean_squared_error(y_test, reg_pred)
r2_linear = r2_score(y_test, reg_pred)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Plotting (assuming you still want to plot)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, reg_pred, label='Linear Regression', color='r', alpha=0.5)
plt.scatter(y_test, y_pred_rf, label='Random Forest Regression', color='g', alpha=0.5)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.legend()
plt.title("Model Comparison: Linear vs. Random Forest Regression")
plt.grid(True)

@app.route('/predictions')
def get_predictions():
    return jsonify({
        "Linear Regression Predictions": reg_pred.tolist(),
        "Random Forest Regression Predictions": y_pred_rf.tolist(),
        "Linear Regression Metrics": {
            "MSE": mse_linear,
            "R²": r2_linear
        },
        "Random Forest Regression Metrics": {
            "MSE": mse_rf,
            "R²": r2_rf
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
