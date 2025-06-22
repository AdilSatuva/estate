import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Connect to database
        conn = sqlite3.connect('real_estate.db')
        df = pd.read_sql_query("SELECT city, rooms, floor, area, total_floors, amenities, price FROM apartments WHERE status = 'active'", conn)
        conn.close()

        if df.empty:
            logger.warning("No data to train model")
            return None

        # Data cleaning
        df = df.dropna()
        if df.empty:
            logger.warning("No valid data after cleaning")
            return None

        df = df[df['price'] > 0]
        df = df[df['area'] > 10]
        df = df[df['rooms'].isin([1, 2, 3, 4])]
        df = df[df['floor'] > 0]
        df = df[df['total_floors'] > 0]

        # Encode city
        le = LabelEncoder()
        df['city_encoded'] = le.fit_transform(df['city'])

        # Process amenities
        amenities_list = ['Парковка', 'Балкон', 'Лифт', 'Мебель', 'Кондиционер', 'Охрана']
        for amenity in amenities_list:
            df[amenity] = df['amenities'].apply(lambda x: 1 if x and amenity in x.split(',') else 0)

        # Features and target
        X = df[['city_encoded', 'rooms', 'floor', 'area', 'total_floors'] + amenities_list]
        y = df['price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logger.debug(f"Best parameters: {grid_search.best_params_}")

        # Evaluate model
        y_pred = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logger.debug(f"Model evaluation - RMSE: {rmse:.2f}, R²: {r2:.2f}")

        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump({
                'model': best_model,
                'label_encoder': le,
                'features': X.columns.tolist(),
                'rmse': rmse,
                'r2': r2
            }, f)
        logger.debug("Model trained and saved")
        return {
            'model': best_model,
            'label_encoder': le,
            'features': X.columns.tolist(),
            'rmse': rmse,
            'r2': r2
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

if __name__ == '__main__':
    train_model()