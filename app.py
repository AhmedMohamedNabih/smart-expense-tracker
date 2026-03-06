import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from pipeline import build_features, generate_final_recommendation, prepare_for_clustering


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Expense Tracker",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("💰 Smart Expense Tracker")
st.write("أدخل بياناتك لمعرفة تحليلك المالي وتوصياتك.")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("robust_scaler(3).joblib")
    kmeans = joblib.load("kmeans_model.joblib")
    cluster_baseline = joblib.load("cluster_baseline.joblib")
    return scaler, kmeans, cluster_baseline


scaler, kmeans, cluster_baseline = load_models()

# -------------------------------
# User Inputs
# -------------------------------
salary = st.number_input("المرتب الشهري *", min_value=0.0, step=100.0)

food = st.number_input("الأكل", min_value=0.0)
drink = st.number_input("المشروبات", min_value=0.0)
shopping = st.number_input("التسوق", min_value=0.0)
transport = st.number_input("المواصلات", min_value=0.0)
bills = st.number_input("الفواتير", min_value=0.0)
health = st.number_input("الصحة", min_value=0.0)
entertainment = st.number_input("الترفيه", min_value=0.0)

# -------------------------------
# Button
# -------------------------------
if st.button("تحليل الآن"):

    if salary <= 0:
        st.error("⚠️ لازم تدخل المرتب الشهري الأول علشان نقدر نحلل مصروفاتك.")
        st.stop()

    # -------------------------------
    # Expenses Dictionary
    # -------------------------------
    spend_dict = {
        "food": float(food),
        "drink": float(drink),
        "shopping": float(shopping),
        "transport": float(transport),
        "bills": float(bills),
        "health": float(health),
        "entertainment": float(entertainment)
    }

    total_spend = sum(spend_dict.values())

    if total_spend > salary:
        st.warning("⚠️ مصروفاتك أكبر من مرتبك هذا الشهر.")

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    new_user_df = build_features(salary, spend_dict)

    X_new = prepare_for_clustering(new_user_df)

    X_scaled = scaler.transform(X_new)

    cluster = kmeans.predict(X_scaled)[0]

    new_user_df['cluster'] = cluster

    # -------------------------------
    # Recommendation
    # -------------------------------
    recommendation = generate_final_recommendation(
        new_user_df.iloc[0],
        cluster_baseline
    )

    st.info(recommendation)

    # -------------------------------
    # Chart Data
    # -------------------------------
    chart_data = pd.DataFrame({
        "الفئة": [
            "الأكل",
            "المشروبات",
            "التسوق",
            "المواصلات",
            "الفواتير",
            "الصحة",
            "الترفيه"
        ],
        "المبلغ": [
            spend_dict["food"],
            spend_dict["drink"],
            spend_dict["shopping"],
            spend_dict["transport"],
            spend_dict["bills"],
            spend_dict["health"],
            spend_dict["entertainment"]
        ]
    })

    # -------------------------------
    # Chart Title
    # -------------------------------
    st.subheader("📊 توزيع المصروفات")

    # -------------------------------
    # Plotly Chart
    # -------------------------------
    fig = px.bar(
        chart_data,
        x="الفئة",
        y="المبلغ",
        text="المبلغ",
        color="الفئة"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis_title="",
        yaxis_title="المبلغ",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)