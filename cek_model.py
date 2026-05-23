import joblib

model_ml = joblib.load("Nutri_RF.pkl")
print("Parameter Model:", model_ml.get_params())

print("Jumlah Pohon Keputusan:", len(model_ml.estimators_))