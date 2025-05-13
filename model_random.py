# STEP 1: Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pymongo import MongoClient


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["AQI"]
collection = db["china_data"]


# Load data from MongoDB
air = pd.DataFrame(list(collection.find()))


# Drop unwanted columns
air.drop(columns=["_id", "station"], inplace=True, errors="ignore")


# Convert time column to datetime
air["hour"] = pd.to_datetime(air["hour"], errors="coerce")


# Make a copy of the DataFrame to modify
air_cleaned = air.copy()

# Loop through each numeric column
for column in air_cleaned.select_dtypes(include=['number']).columns:
    Q1 = air_cleaned[column].quantile(0.25)
    Q3 = air_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    air_cleaned[column] = air_cleaned[column].apply(
        lambda x: x if lower_bound <= x <= upper_bound else np.nan
    )


## check for missing values using .isnull() and sum()

air_cleaned.isnull().sum()

## calculate the % of missing values . Round the % up to 4 char

round(air_cleaned.isnull().sum()/len(air_cleaned.index), 4)*100


#Missing value treatment
air_cleaned['PM25'] = air_cleaned['PM25'].fillna(air_cleaned['PM25'].mean())
air_cleaned['PM10'] = air_cleaned['PM10'].fillna(air_cleaned['PM10'].mean())
air_cleaned['SO2'] = air_cleaned['SO2'].fillna(air_cleaned['SO2'].mean())
air_cleaned['NO2'] = air_cleaned['NO2'].fillna(air_cleaned['NO2'].mean())
air_cleaned['CO'] = air_cleaned['CO'].fillna(air_cleaned['CO'].mean())
air_cleaned['O3'] = air_cleaned['O3'].fillna(air_cleaned['O3'].mean())
air_cleaned['TEMP'] = air_cleaned['TEMP'].fillna(air_cleaned['TEMP'].mean())
air_cleaned['PRES'] = air_cleaned['PRES'].fillna(air_cleaned['PRES'].mean())
air_cleaned['DEWP'] = air_cleaned['DEWP'].fillna(air_cleaned['DEWP'].mean())
air_cleaned['RAIN'] = air_cleaned['RAIN'].fillna(air_cleaned['RAIN'].mean())
air_cleaned['WSPM'] = air_cleaned['WSPM'].fillna(air_cleaned['WSPM'].mean())
air_cleaned['wd'] = air_cleaned['wd'].fillna(air_cleaned['wd'].mode()[0])

## check for missing values using .isnull() and sum()

air_cleaned.isnull().sum()

# Function to calculate AQI sub-index for each pollutant
def calculate_sub_index(concentration, breakpoints):
    if pd.isna(concentration):
        return None

    for bp in breakpoints:
        Clow, Chigh, Ilow, Ihigh = bp
        if Clow <= concentration <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (concentration - Clow) + Ilow
            return round(aqi, 2)
    
    # Return None if concentration is outside defined breakpoints
    return None


# AQI Breakpoints
pm25_bp = [(0.0, 12.0, 0, 50),(12.1, 35.4, 51, 100),(35.5, 55.4, 101, 150),(55.5, 150.4, 151, 200),(150.5, 250.4, 201, 300),(250.5, 350.4, 301, 400),(350.5, 500.4, 401, 500)]
pm10_bp = [(0, 54, 0, 50),(55, 154, 51, 100),(155, 254, 101, 150),(255, 354, 151, 200),(355, 424, 201, 300),(425, 504, 301, 400),(505, 604, 401, 500)]
no2_bp = [(0, 53, 0, 50),(54, 100, 51, 100),(101, 360, 101, 150),(361, 649, 151, 200),(650, 1249, 201, 300),(1250, 1649, 301, 400),(1650, 2049, 401, 500)]
so2_bp = [(0, 35, 0, 50),(36, 75, 51, 100),(76, 185, 101, 150),(186, 304, 151, 200),(305, 604, 201, 300),(605, 804, 301, 400),(805, 1004, 401, 500)]
co_bp = [(0.0, 4.4, 0, 50),(4.5, 9.4, 51, 100),(9.5, 12.4, 101, 150),(12.5, 15.4, 151, 200),(15.5, 30.4, 201, 300),(30.5, 40.4, 301, 400),(40.5, 50.4, 401, 500)]
o3_bp = [(0, 54, 0, 50),(55, 70, 51, 100),(71, 85, 101, 150),(86, 105, 151, 200),(106, 200, 201, 300),(201, 300, 301, 400),(301, 400, 401, 500)]


  # Calculate AQI for each row
def calculate_aqi(row):
    sub_indices = [
        calculate_sub_index(row['PM25'], pm25_bp),
        calculate_sub_index(row['PM10'], pm10_bp),
        calculate_sub_index(row['NO2'], no2_bp),
        calculate_sub_index(row['SO2'], so2_bp),
        calculate_sub_index(row['CO'], co_bp),
        calculate_sub_index(row['O3'], o3_bp)
    ]
    return max(filter(None, sub_indices), default=None)  # Take the maximum valid sub-index as AQI

# Apply AQI calculation
air_cleaned['AQI'] = air_cleaned.apply(calculate_aqi, axis=1)

# Print AQI statistics to check if values are being calculated
print("AQI Calculation Summary:")
print(air_cleaned['AQI'].describe())
print("First few AQI values:")
print(air_cleaned[['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']].head())

# Drop NaN AQI values before training
air_cleaned.dropna(subset=['AQI'], inplace=True)

# Features and target selection
X = air_cleaned[['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]  # Features
y = air_cleaned['AQI']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test have no NaNs before scaling
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Ensure y_train has no NaNs
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure there are no NaN values in the scaled features
if np.isnan(X_train_scaled).sum() > 0 or np.isnan(X_test_scaled).sum() > 0:
    print("Warning: NaN values detected in scaled features. Filling with mean values.")
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmean(X_train_scaled))
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmean(X_test_scaled))




param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Removed 'auto'
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='r2',
    verbose=2
)

# Fit the model on a subset (ensure X_train_scaled and y_train are defined)
grid_search.fit(X_train_scaled[:50000], y_train[:50000])

# Print best parameters
print("Best Parameters:", grid_search.best_params_)



# STEP 5: Make Predictions
# Predict on test data
rf_model = grid_search.best_estimator_  # Assign the best model found by GridSearchCV
y_pred = rf_model.predict(X_test_scaled)


# STEP 6: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# STEP 7: Visualize Results


# plt.scatter(y_test, y_pred, color='blue', label="Predicted vs Actual")
# plt.plot(y_test, y_test, color='red', linestyle='dashed', label="Perfect Fit")  # Reference line
# plt.xlabel("Actual AQI")
# plt.ylabel("Predicted AQI")
# plt.title("Random Forest AQI Prediction")
# plt.legend()
# plt.show()


# Save model and scaler
joblib.dump(rf_model, "models/ran_model.pkl")
joblib.dump(scaler, "models/ran_scaler.pkl")

print("✅ Model and scaler saved successfully!")



