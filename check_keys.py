import joblib

encoders = joblib.load("encoders.pkl")

print("📌 encoders에 들어 있는 키 목록:")
for key in encoders.keys():
    print("-", key)
