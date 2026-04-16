from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_models(df):
    models = {}

    # ---- Regression (Rating Prediction) ----
    if 'Aggregate rating' in df.columns:
        X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Aggregate rating'], errors='ignore')
        y = df['Aggregate rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        joblib.dump(reg_model, "models/rating_model.pkl")

        models["regression"] = reg_model

    # ---- Classification (Cuisine Classification) ----
    if 'Cuisines' in df.columns:
        df['Cuisines'] = df['Cuisines'].astype('category').cat.codes

        X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Cuisines'], errors='ignore')
        y = df['Cuisines']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf_model = RandomForestClassifier()
        clf_model.fit(X_train, y_train)

        joblib.dump(clf_model, "models/cuisine_model.pkl")

        models["classification"] = clf_model

    return models, X_test, y_test