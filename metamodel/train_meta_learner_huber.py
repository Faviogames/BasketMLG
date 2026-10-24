import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_meta_learner():
    print("="*70)
    print("ENTRENAMIENTO META-LEARNER (The Judge)")
    print("="*70)
    
    # 1. Load Data
    df = pd.read_csv('data/training_dataset_meta_v1.csv')
    
    # 2. Define Features
    # Removing Adjuster (Noise)
    # Removing Bookie Line (Mostly 0s)
    features = [
        'base_model_prediction',
        'pace_poisson_projection'
    ]
    target = 'final_total'
    
    # Drop rows with missing target or features
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Evaluate Individual Models
    print("\n[1] Evaluando Componentes Individuales (Test Set):")
    mae_base = mean_absolute_error(y_test, X_test['base_model_prediction'])
    mae_pacer = mean_absolute_error(y_test, X_test['pace_poisson_projection'])
    
    print(f"   Base Model MAE:     {mae_base:.2f}")
    print(f"   Pacer Model MAE:    {mae_pacer:.2f}")
    
    # 5. Train Huber Regressor
    print("\n[2] Entrenando Huber Regressor...")
    # epsilon=1.35 is standard, smaller makes it more robust to outliers
    meta_model = HuberRegressor(epsilon=1.35, max_iter=1000)
    meta_model.fit(X_train, y_train)
    
    # 6. Evaluate Meta Model
    y_pred = meta_model.predict(X_test)
    mae_meta = mean_absolute_error(y_test, y_pred)
    
    print(f"\n[3] Resultados Meta-Learner:")
    print(f"   Meta-Learner MAE:   {mae_meta:.2f}")
    print(f"   Mejora vs Base:     {mae_base - mae_meta:.2f} pts")
    
    # 7. Coefficients
    print("\n[4] Coeficientes (Importancia):")
    coefs = meta_model.coef_
    intercept = meta_model.intercept_
    
    for name, coef in zip(features, coefs):
        print(f"   {name:25s}: {coef:.4f}")
    print(f"   Intercept:                {intercept:.4f}")
    
    # 8. Save Model
    os.makedirs('metamodel/models', exist_ok=True)
    joblib.dump(meta_model, 'metamodel/models/meta_learner_v1_final.joblib')
    print(f"\nModelo guardado: metamodel/models/meta_learner_v1_final.joblib")
    
    # Save coefficients for easy reading
    with open('metamodel/models/meta_learner_coefs.txt', 'w') as f:
        f.write(f"Intercept: {intercept}\n")
        for name, coef in zip(features, coefs):
            f.write(f"{name}: {coef}\n")

if __name__ == "__main__":
    train_meta_learner()
