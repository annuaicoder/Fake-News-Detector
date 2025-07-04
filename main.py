import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
    
# Load dataset
df = pd.read_csv("news.csv")

# Map labels to binary
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# Interactive Prediction
while True:
    headline = input("\n📰 Enter a news headline (or 'exit'): ").strip()
    if headline.lower() == "exit":
        break
    vec = vectorizer.transform([headline])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec).max() * 100
    label = "REAL ✅" if pred == 1 else "FAKE ❌"
    print(f"🧠 This headline is: {label} (Confidence: {conf:.1f}%)")
