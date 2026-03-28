import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('air_quality_data.csv')
    
    X = df[['PM2.5', 'PM10', 'Temperature', 'Humidity', 'CO', 'NO2']]
    y = df['AQI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training ANN Model...")
    # MLP Regressor to predict AQI value
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                         max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Complete. MSE: {mse:.2f}, R2 Score: {r2:.4f}")
    
    joblib.dump(model, 'ann_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler saved successfully.")

if __name__ == '__main__':
    main()
