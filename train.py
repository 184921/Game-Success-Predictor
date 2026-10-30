import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

games = pd.read_csv("games.csv")

games["Release date"] = pd.to_datetime(games["Release date"], errors="coerce")
games = games[games["Release date"] <= pd.Timestamp.today()].copy()

games["Min owners"] = games["Estimated owners"].str.split(" - ").str[0].astype(int)
games["Successful"] = (games["Min owners"] >= 100000).astype(int)

for col in ["Windows", "Mac", "Linux"]:
    games[col] = games[col].astype(int)

genres = games["Genres"].fillna("").str.get_dummies(sep=",")

X = pd.concat([games[["Price", "Windows", "Mac", "Linux"]], genres], axis=1)
y = games["Successful"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=0,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Antall spill:", len(games))
print("Antall features:", X.shape[1])
print("\nResultater:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions).T[::-1, ::-1])

joblib.dump({"model": model, "columns": X.columns.tolist()}, "model.pkl")
print("\nModellen er lagret i model.pkl")