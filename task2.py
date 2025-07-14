# ✅ Step 1–5: All-in-One Titanic Prediction API
from flask import Flask, request, jsonify
import pandas as pd
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Step 1: Load and preprocess data
df = sns.load_dataset("titanic").dropna(subset=["sex", "age", "fare", "embarked", "survived"])
df = df[["sex", "age", "fare", "embarked", "survived"]]
df["sex"] = LabelEncoder().fit_transform(df["sex"])
df["embarked"] = LabelEncoder().fit_transform(df["embarked"])

# ✅ Step 2: Train/test split
X = df.drop("survived", axis=1)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 3: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with accuracy: {acc:.2f}")

# ✅ Step 4: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ✅ Step 5: Flask API setup
app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "🎯 Titanic Survival Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        sex = 1 if data["sex"].lower() == "male" else 0
        age = float(data["age"])
        fare = float(data["fare"])
        embarked_dict = {"c": 0, "q": 1, "s": 2}
        embarked = embarked_dict.get(data["embarked"].lower(), 2)

        input_data = [[sex, age, fare, embarked]]
        pred = model.predict(input_data)[0]
        return jsonify({
            "prediction": "Survived" if pred == 1 else "Did not survive"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
