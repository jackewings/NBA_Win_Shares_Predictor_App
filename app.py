import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# Define project root and model path
project_root = '/Users/jackewings/Desktop/Personal_Projects/Apps/Win_Shares_Predictor'
model_path = os.path.join(project_root, 'Models', 'lasso_model.pkl')

# Load the model
model = joblib.load(model_path)

st.title("Win Shares Predictor")

st.markdown("""
### What is Win Shares per 48?
**Win Shares per 48 minutes (WS/48)** is an advanced basketball statistic that estimates a player's contribution to their team's wins, scaled to a per-48-minute basis (a full game length).

**Interpretation**:
- 🔴 **Below 0.00**: **Hurting** the team’s chances of winning (rare, poor performers).
- 🟡 **0.00 – 0.10**: **Below average to average** players.
- 🟢 **Above 0.10**: **Good to elite** performance levels.

For reference:
- **League average WS/48 is ~0.10**.
- **Top-tier players** can have WS/48 of **0.20 or higher**.

Use the sliders below to enter a player's stats and see their predicted WS/48!
""")

# Create input widgets for the features your model needs
age = st.number_input("Age", min_value=18, max_value=40, value=25, step = 1)
pts_per_game = st.number_input("Points per Game", min_value=0.0, max_value=50.0, value=10.0, step = .5)
trb_per_game = st.number_input("Total Rebounds per Game", min_value=0.0, max_value=30.0, value=5.0, step = .5)
ast_per_game = st.number_input("Assists per Game", min_value=0.0, max_value=20.0, value=3.0, step = .5)
x2p_percent = st.number_input("2PT%", min_value=0.0, max_value=1.0, value=0.5, step = .05)
x3p_percent = st.number_input("3PT%", min_value=0.0, max_value=1.0, value=0.35, step = .05)
dbpm = st.number_input("Defensive BPM", min_value=-10.0, max_value=10.0, value=0.0, step = .2)
is_backcourt = st.checkbox("Backcourt Player?")

feature_names = ['age', 'pts_per_game', 'trb_per_game', 'ast_per_game', 'x2p_percent', 'x3p_percent', 'dbpm', 'is_backcourt']

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.4em 1em;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

if st.button("Predict Win Shares"):
    input_df = pd.DataFrame([[
        age, pts_per_game, trb_per_game, ast_per_game,
        x2p_percent, x3p_percent, dbpm, int(is_backcourt)
    ]], columns=feature_names)

    prediction = model.predict(input_df)[0]

    # Determine color based on predicted win shares
    if prediction < 0:
        box_color = "#b30000"  # dark red
    elif prediction < 0.1:
        box_color = "#b38f00"  # dark yellow
    else:
        box_color = "#006600"  # dark green

    # Display result with styling
    st.markdown(
        f"""
        <div style="
            background-color: {box_color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            ">
            Predicted Win Shares: {prediction:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
