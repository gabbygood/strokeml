# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop 'id' column if present
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Fill missing values
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split features and labels
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, label_encoders
