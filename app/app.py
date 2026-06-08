import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EstateX", layout="wide")

# ----------------- UI STYLING -----------------

st.markdown("""
<style>

.stApp {
    background-color: #020617;
}

/* LABEL COLORS */
label {
    color: white !important;
    font-weight: 500;
}

/* dropdown selected value (Bandra etc) */
div[data-baseweb="select"] span {
    color: white !important;
    font-weight: 500;
}

/* dropdown background */
div[data-baseweb="select"] > div {
    background-color:#1e293b;
}

/* input boxes */
input {
    background-color:#1e293b !important;
    color:white !important;
}

/* header */
.header {
    font-size:38px;
    font-weight:700;
    color:white;
}

.subheader {
    color:#94a3b8;
    margin-bottom:30px;
}

/* section title */
.section-title {
    color:white;
    font-size:24px;
    margin-top:30px;
}

/* buttons */
.stButton button {
    background-color:#111827;
    color:white;
    border-radius:8px;
    padding:10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------

st.markdown('<div class="header">🏠 EstateX</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter property details to estimate the house price.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Enter Property Details</div>', unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------

model = pickle.load(open("../model/model.pkl","rb"))
columns = pickle.load(open("../model/columns.pkl","rb"))

# ----------------- INPUTS -----------------

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (sq ft)", 500, 5000, 1200)

with col2:
    bhk = st.selectbox("Bedrooms", [1,2,3,4,5])

with col3:
    furnishing = st.selectbox("Furnishing", ["Furnished","Semi","Unfurnished"])


col4, col5, col6 = st.columns(3)

with col4:
    location = st.selectbox(
        "Location",
        ["Churchgate","Dadar","Bandra","Andheri","Mira Bhayander"]
    )

with col5:
    floor = st.number_input("Floor Number",1,50)

with col6:
    total_floors = st.number_input("Total Floors",1,60)


col7, col8 = st.columns(2)

with col7:
    age = st.number_input("Property Age",0,30)

with col8:
    parking = st.selectbox("Parking",[0,1])


predict = st.button("Predict House Price")

# ----------------- PREDICTION -----------------

if predict:

    data = {
        "Location":[location],
        "Carpet_Area":[area],
        "BHK":[bhk],
        "Floor_Number":[floor],
        "Total_Floors":[total_floors],
        "Property_Age":[age],
        "Parking":[parking],
        "Furnishing":[furnishing]
    }

    df = pd.DataFrame(data)

    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)

    price = int(prediction[0])

    st.success(f"Estimated Property Price: ₹ {price:,}")


# ----------------- DASHBOARD VISUALIZATION -----------------

st.markdown('<div class="section-title">Market Insights Dashboard</div>', unsafe_allow_html=True)

data = pd.read_csv("../dataset/mumbai_housing.csv")

# ---- METRICS ----

m1,m2,m3,m4 = st.columns(4)

m1.metric("Total Properties", len(data))
m2.metric("Average Price", f"₹ {int(data['Price'].mean()):,}")
m3.metric("Highest Price", f"₹ {int(data['Price'].max()):,}")
m4.metric("Average Area", f"{int(data['Carpet_Area'].mean())} sqft")

st.markdown("---")

# ---- ROW 1 ----

c1,c2 = st.columns(2)

with c1:

    fig1 = px.line(
        data.sort_values("Carpet_Area"),
        x="Carpet_Area",
        y="Price",
        color="Location",
        title="Price Growth vs Area"
    )

    fig1.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617"
    )

    st.plotly_chart(fig1, use_container_width=True)

with c2:

    location_price = data.groupby("Location")["Price"].mean().reset_index()

    fig2 = px.bar(
        location_price,
        x="Location",
        y="Price",
        title="Average Price by Location",
        color="Location"
    )

    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---- ROW 2 ----

c3,c4 = st.columns(2)

with c3:

    fig3 = px.pie(
        data,
        names="Location",
        title="Property Distribution by Location",
        hole=0.5
    )

    fig3.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617"
    )

    st.plotly_chart(fig3, use_container_width=True)

with c4:

    fig4 = px.histogram(
        data,
        x="Price",
        title="Property Price Distribution",
        nbins=15
    )

    fig4.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617"
    )

    st.plotly_chart(fig4, use_container_width=True)