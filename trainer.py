# trainer.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test, scaler, encoders = load_and_preprocess_data("healthcare-dataset-stroke-data.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model, scaler, and encoders
    joblib.dump(model, "stroke_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "encoders.pkl")

if __name__ == "__main__":
    train_and_save_model()
