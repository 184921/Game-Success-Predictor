import streamlit as st
import pandas as pd
import joblib

data = joblib.load("model.pkl")
model = data["model"]
columns = data["columns"]

st.title("Game Success Predictor")
st.write("Predict whether your Steam game could become popular.")

price = st.number_input("Game price ($)", min_value=0.0, value=9.99)

genre_columns = [col for col in columns if col not in ["Price", "Windows", "Mac", "Linux"]]
genres = st.multiselect("Genres", genre_columns)

st.write("Supported operating systems")
windows = st.checkbox("Windows", value=True)
mac = st.checkbox("Mac")
linux = st.checkbox("Linux")

if st.button("Predict success"):
    game = pd.DataFrame(0.0, index=[0], columns=columns)

    game["Price"] = price
    game["Windows"] = int(windows)
    game["Mac"] = int(mac)
    game["Linux"] = int(linux)

    for genre in genres:
        game[genre] = 1

    prediction = model.predict(game)[0]
    probability = model.predict_proba(game)[0][1]

    if prediction == 1:
        st.success("This game is predicted to be popular!")
    else:
        st.warning("This game is predicted to be less popular.")

    st.write(f"Model score for popularity: {probability:.1%}")
    st.caption("Popular means at least 100,000 estimated Steam owners.")