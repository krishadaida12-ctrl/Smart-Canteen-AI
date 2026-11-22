
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether
)
from reportlab.lib import colors as pdf_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Smart Canteen AI",
    page_icon="🌿",
    layout="wide"
)
# Improve text visibility
st.markdown("""
<style>


/* Dark Sage Green Sidebar */
section[data-testid="stSidebar"] {
    background-color: #294B38 !important;
}

section[data-testid="stSidebar"] * {
    color: #DCE9D5 !important;
}

/* Option 3 - Sage Green Theme */
.stApp {
    background-color: #E8EFE6 !important;
    color: #30483B;
}

[data-testid="stMainBlockContainer"] {
    background-color: #FDFBF5;
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(48, 72, 59, 0.08);
}

h1, h2, h3 {
    color: #426B55 !important;
}

/* Prediction button */
div.stButton > button {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

div.stButton > button p {
    color: #FFFFFF !important;
}

/* Success message */
div[data-testid="stAlert"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Smart recommendations */
div[data-testid="stAlert"] {
    font-weight: 600 !important;
}

/* Dropdown text */
div[data-baseweb="select"] {
    color: #FFFFFF !important;
}

/* General captions on white backgrounds */
div[data-testid="stCaptionContainer"] {
    color: #374151 !important;
}

</style>
""", unsafe_allow_html=True)

# Custom colours and design
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg, #F6F8F2, #E9F2E9, #FFF3E7
    );
    color: #244B3E;
}

/* Main heading */
h1 {
    color: #205C4A !important;
    font-size: 38px !important;
    font-weight: 800 !important;
}

/* Subheadings */
h2, h3 {
    color: #205C4A !important;
}

/* Normal text */
p, label {
    color: #355E50;
}

/* Hero banner */
.hero {
    background: linear-gradient(
        120deg, #205C4A, #39836A
    );
    padding: 30px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(32,92,74,0.15);
}

.hero-title {
    color: white;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.2;
}

.hero-text {
    color: #E0F2E8;
    font-size: 15px;
    margin-top: 12px;
}

/* Section heading */
.section-heading {
    background: #FFFFFF;
    padding: 16px 20px;
    border-left: 5px solid #39836A;
    border-radius: 12px;
    margin: 20px 0;
    font-size: 20px;
    font-weight: 700;
    color: #205C4A;
}

/* Input boxes */
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    color: #244B3E !important;
}

/* Prediction button */
div.stButton > button {
    background: linear-gradient(
        90deg, #205C4A, #39836A
    );
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
    min-height: 50px;
    box-shadow: 0 5px 15px rgba(32,92,74,0.2);
}

div.stButton > button:hover {
    background: #174C3B;
    color: white !important;
    border: none;
}

/* Footer */
.footer {
    text-align: center;
    color: #587A68;
    font-size: 13px;
    margin-top: 35px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)
# Load the trained Random Forest model
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "food_waste_model.pkl"
)

xgb_model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")
# Load Food Wastage Prediction Model
waste_model = joblib.load(BASE_DIR / "models" / "waste_model.pkl")
waste_columns = joblib.load(BASE_DIR / "models" / "waste_columns.pkl")
feature_columns = joblib.load(
    BASE_DIR / "models" / "feature_columns.pkl"
)
# Header

# Function to generate a professional PDF report
def create_pdf_report(day, meal, menu, students, buffer,
                      predicted_food, recommended_food,
                      planned_extra, predicted_waste,
                      waste_percentage):

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=(595, 842),
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=pdf_colors.HexColor("#205C4A"),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=pdf_colors.HexColor("#587A68"),
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=pdf_colors.HexColor("#205C4A"),
        spaceBefore=18,
        spaceAfter=10
    )

    story = []

    # Report heading
    story.append(
        Paragraph("SMART CANTEEN AI", title_style)
    )

    story.append(
        Paragraph(
            "Food Waste Prediction Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph("Canteen Details", heading_style)
    )

    # Input details table
    details = [
        ["Parameter", "Value"],
        ["Day", day],
        ["Meal Type", meal],
        ["Menu Type", menu],
        ["Expected Students", str(students)],
        ["Preparation Buffer", f"{buffer}%"]
    ]

    details_table = Table(
        details,
        colWidths=[240, 240]
    )
    
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
        pdf_colors.HexColor("#205C4A")),

        ("TEXTCOLOR", (0, 0), (-1, 0),
        pdf_colors.white),

        ("BACKGROUND", (0, 1), (-1, -1),
        pdf_colors.HexColor("#F1F7F2")),

        ("TEXTCOLOR", (0, 1), (-1, -1),
         pdf_colors.HexColor("#244B3E")),

        ("GRID", (0, 0), (-1, -1),
        0.5, pdf_colors.HexColor("#D6E5DA")),

        ("PADDING", (0, 0), (-1, -1), 10)
   ]))

    story.append(details_table)

    story.append(
        Paragraph("Prediction Results", heading_style)
    )

    results = [
        ["Prediction", "Quantity"],
        ["Predicted Consumption",
         f"{predicted_food:.2f} kg"],
        ["Recommended Preparation",
         f"{recommended_food:.2f} kg"],
        ["Planned Extra Food",
         f"{planned_extra:.2f} kg"],
        ["Predicted Food Wastage",
         f"{predicted_waste:.2f} kg"],
        ["Predicted Wastage Percentage",
         f"{waste_percentage:.2f}%"]
    ]

    results_table = Table(
        results,
        colWidths=[240, 240]
    )

    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         pdf_colors.HexColor("#39836A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), pdf_colors.white),
        ("BACKGROUND", (0, 1), (-1, -1),
         pdf_colors.HexColor("#FFF4E8")),
        ("TEXTCOLOR", (0, 1), (-1, -1),
         pdf_colors.HexColor("#244B3E")),
        ("GRID", (0, 0), (-1, -1),
         0.5, pdf_colors.HexColor("#E9DCCB")),
        ("PADDING", (0, 0), (-1, -1), 10)
    ]))

    story.append(results_table)

    # Comparison graph
    story.append(
        Paragraph("Food Preparation Analysis", heading_style)
    )

    fig, ax = plt.subplots(figsize=(7, 3))

    bars = ax.bar(
        ["Predicted Consumption", "Recommended Preparation"],
        [predicted_food, recommended_food],
        color=["#39836A", "#E8A65B"],
        width=0.5
    )

    ax.set_ylabel("Food Quantity (kg)")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f} kg",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(
        0,
        max(predicted_food, recommended_food) * 1.2 + 1
    )

    fig.tight_layout()

    graph_buffer = BytesIO()
    fig.savefig(
        graph_buffer,
        format="png",
        dpi=150
    )
    plt.close(fig)
    graph_buffer.seek(0)

    story.append(
        Image(graph_buffer, width=480, height=205)
    )

    # Recommendation
    story.append(
        Paragraph("Smart Recommendation", heading_style)
    )

    recommendation = (
        f"For {students} students, the predicted food "
        f"consumption is {predicted_food:.2f} kg. "
        f"With a {buffer}% preparation buffer, "
        f"the recommended food quantity is "
        f"{recommended_food:.2f} kg."
    )

    story.append(
        Paragraph(recommendation, styles["Normal"])
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Note: Predictions are based on simulated "
            "training data. Planned extra food is not "
            "actual measured food waste.",
            styles["Italic"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Smart Canteen AI | Predict Smarter. Waste Less.",
            subtitle_style
        )
    )

    doc.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


import base64

canteen_file = Path(__file__).resolve().parent / "images" / "cateen.png"

with open(canteen_file, "rb") as image_file:
    canteen_base64 = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
<div style="display:flex; align-items:center; gap:20px;
            background:#DCE8D5; border-radius:24px;
            padding:25px; margin-bottom:25px;">
    <div style="flex:1; color:#30483B;">
        <h1 style="color:#426B55;">🌿 Smart Canteen AI</h1>
        <h3 style="color:#30483B;">Predict Smarter. Waste Less.</h3>
        <p>AI-powered food planning for a greener campus.</p>
    </div>
    <img src="data:image/png;base64,{canteen_base64}"
         style="width:48%; border-radius:18px; object-fit:cover;">
</div>
""", unsafe_allow_html=True)

# Introduction
st.title("🍽️ Food Waste Predictor")

st.write(
    "Plan meals intelligently using Machine Learning "
    "and help your college canteen reduce food wastage."
)

st.markdown(
    "🌱 **Eco Friendly** &nbsp;&nbsp; "
    "🤖 **AI Powered** &nbsp;&nbsp; "
    "📊 **Smart Planning**"
)

# Sage Green Sidebar
with st.sidebar:
    st.markdown("## 🍲 Smart Canteen AI")
    st.markdown("### 🌿 Green Campus")
    st.divider()
    st.markdown("🏠 **Home**")
    st.markdown("📊 **Predict Waste**")
    st.markdown("📈 **Visualizations**")
    st.markdown("🌱 **Recommendations**")
    st.markdown("ℹ️ **About Project**")

# College Canteen Banner

from pathlib import Path


# Main layout: Gallery on left, Prediction Form on right
left_panel, right_panel = st.columns([1.2, 1], gap="large")

# Food Gallery Section
import base64

left_panel.markdown("## 🍽️ What's Cooking Today?")
left_panel.markdown("Explore some delicious meals from our college canteen!")

def food_card(image_path, food_name):
    with open(image_path, "rb") as file:
        image_data = base64.b64encode(file.read()).decode()

    st.markdown(f"""
    <div style="text-align:center; padding:8px;
                background:#E8EFE6; border-radius:16px;">
        <img src="data:image/jpeg;base64,{image_data}"
             style="width:100%; height:145px;
                    object-fit:cover; border-radius:12px;">
        <p style="font-weight:600; color:#30483B;
                  margin:8px 0 4px;">
            {food_name}
        </p>
    </div>
    """, unsafe_allow_html=True)

food1, food2, food3, food4 = left_panel.columns(4, gap="small")

with food1:
    food_card("images/uttapam.jfif", "🥞 Uttapam")

with food2:
    food_card("images/veg pulao.jfif", "🍚 Veg Pulao")

with food3:
    food_card("images/chapati.jfif", "🫓 Chapati & Curry")

with food4:
    food_card("images/meals.jfif", "🍛 Canteen Meals")

# Why Smart Canteen AI Section
left_panel.markdown("""
<div style="
    background: linear-gradient(135deg, #E1EDE0, #D4E6D8);
    padding: 18px;
    border-radius: 16px;
    margin-top: 12px;
    text-align: center;
">
    <h3 style="color:#30483B; margin:0 0 16px 0;">
        🌿 Why Smart Canteen AI?
    </h3>
    <div style="display:flex; justify-content:space-around; gap:10px; color:#30483B; font-weight:600;">
        <div style="flex:1;">🌱<p>Reduce Food<br>Waste</p></div>
        <div style="flex:1;">💰<p>Save Canteen<br>Costs</p></div>
        <div style="flex:1;">📊<p>Smarter Meal<br>Planning</p></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Input section
right_panel.markdown("""
<div class="section-heading">
    📝 Enter Canteen Details
</div>
""", unsafe_allow_html=True)

# Two-column layout
col1, col2 = right_panel.columns(2)

with col1:
    day = st.selectbox(
        "📅 Select Day",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )
    
meal_options = ["Breakfast", "Lunch", "Snacks"]

food_menu = {
    "Breakfast": [
        "Puri", "Idly", "Vada", "Upma", "Uttapam"
    ],
    "Lunch": [
        "Veg Biryani", "Chicken Biryani",
        "Veg Fried Rice", "Paneer Biryani",
        "Mixed Rice"
    ],
    "Snacks": [
        "Tea", "Coffee", "Green Tea",
        "Mirchi", "Samosa", "Kachori",
        "Ice Cream"
    ]
}

meal = right_panel.selectbox(
    "🍴 Select Meal Type",
    meal_options
)

menu = right_panel.selectbox(
    "🍲 Select Menu",
    food_menu[meal]
)


# Student attendance
students = right_panel.number_input(
    "👨‍🎓 Expected Number of Students",
    min_value=1,
    value=200,
    step=1
)
# Extra food preparation percentage
buffer = right_panel.number_input(
    "🌿 Extra Preparation Buffer (%)",
    min_value=0,
    max_value=20,
    value=5
)

st.write("")
right_panel.selectbox(
    "Choose Machine Learning Model",
    ["Random Forest", "XGBoost"],
    key="selected_model"
)
# Prediction button
if right_panel.button("✨ Predict Food Requirement"):
    # Keep the original encoded inputs for both models.
    raw_input = pd.DataFrame({
        "Day": [day],
        "Meal_Type": [meal],
        "Menu_Type": [menu],
        "Students_Present": [students]
    })
    encoded_input = pd.get_dummies(
        raw_input, columns=["Day", "Meal_Type", "Menu_Type"], dtype=int
    )

    selected_model = st.session_state.get("selected_model", "Random Forest")
    if selected_model == "XGBoost":
        xgb_columns = xgb_model.get_booster().feature_names
        if xgb_columns is None:
            st.error("The XGBoost model has no saved feature names.")
            st.stop()
        model_input = encoded_input.reindex(columns=xgb_columns, fill_value=0)
        prediction = xgb_model.predict(model_input)[0]
    else:
        model_input = encoded_input.reindex(columns=feature_columns, fill_value=0)
        prediction = model.predict(model_input)[0]

    predicted_food = max(0, float(prediction))
    recommended_food = predicted_food * (1 + buffer / 100)
    planned_extra = recommended_food - predicted_food

    waste_input = pd.DataFrame([{
        "Day": day,
        "Meal_Type": meal,
        "Menu_Type": menu,
        "Students_Present": students,
        "Food_Prepared_kg": recommended_food
    }])
    waste_input = pd.get_dummies(
        waste_input, columns=["Day", "Meal_Type", "Menu_Type"], dtype=int
    )
    waste_input = waste_input.reindex(columns=waste_columns, fill_value=0)
    predicted_waste = max(0, float(waste_model.predict(waste_input)[0]))
    waste_percentage = (
        predicted_waste / recommended_food * 100 if recommended_food > 0 else 0
    )
    expected_leftover = max(0, recommended_food - predicted_food)
    leftover_percentage = (
        expected_leftover / recommended_food * 100 if recommended_food > 0 else 0
    )

    # Save results so downloads remain available after Streamlit reruns.
    st.session_state["last_prediction"] = {
        "day": day, "meal": meal, "menu": menu,
        "students": students, "buffer": buffer,
        "selected_model": selected_model,
        "predicted_food": predicted_food,
        "recommended_food": recommended_food,
        "planned_extra": planned_extra,
        "predicted_waste": predicted_waste,
        "waste_percentage": waste_percentage,
        "expected_leftover": expected_leftover,
        "leftover_percentage": leftover_percentage
    }

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    day = result["day"]
    meal = result["meal"]
    menu = result["menu"]
    students = result["students"]
    buffer = result["buffer"]
    selected_model = result["selected_model"]
    predicted_food = result["predicted_food"]
    recommended_food = result["recommended_food"]
    planned_extra = result["planned_extra"]
    predicted_waste = result["predicted_waste"]
    waste_percentage = result["waste_percentage"]
    expected_leftover = result["expected_leftover"]
    leftover_percentage = result["leftover_percentage"]

    st.markdown("""
    <div class="section-heading">📊 Smart Prediction Results</div>
    """, unsafe_allow_html=True)
    st.success("✨ Prediction completed successfully!")
    st.caption(f"Consumption model: {selected_model}")

    
    # Four prediction result cards in one row
    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("🍽️ Estimated Food Wastage", f"{predicted_waste:.2f} kg")

    with metric2:
        st.metric("📊 Wastage Percentage", f"{waste_percentage:.2f}%")

    with metric3:
        st.metric("📦 Expected Leftover", f"{expected_leftover:.2f} kg")

    with metric4:
        st.metric("📉 Leftover Percentage", f"{leftover_percentage:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:#E4F2E7;padding:22px;border-radius:16px;text-align:center;border:1px solid #C4DFC9;">
          <p style="color:#376A4C;">🍽️ Predicted Consumption</p>
          <h2 style="color:#205C4A !important;">{predicted_food:.2f} kg</h2>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:#FFF0DD;padding:22px;border-radius:16px;text-align:center;border:1px solid #F2D8B5;">
          <p style="color:#93622D;">🌿 Recommended Preparation</p>
          <h2 style="color:#9A5B20 !important;">{recommended_food:.2f} kg</h2>
        </div>""", unsafe_allow_html=True)

    st.metric("📦 Planned Extra Food", f"{planned_extra:.2f} kg")
    st.subheader("📈 Food Preparation Analysis")
    labels = ["Predicted Consumption", "Recommended Preparation"]
    values = [predicted_food, recommended_food]
    fig, ax = plt.subplots(figsize=(6,2.5 ))
    bars = ax.bar(labels, values, color=["#39836A", "#E8A65B"], width=0.5)
    ax.set_ylabel("Food Quantity (kg)")
    ax.set_title("Food Demand vs Preparation")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{height:.2f} kg", ha="center", va="bottom")
    ax.set_ylim(0, max(values) * 1.2 + 1)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("💡 Smart Canteen Recommendations")
    st.info(f"For {students} students, the estimated food demand is {predicted_food:.2f} kg.")
    st.success(f"Recommended preparation: {recommended_food:.2f} kg.")
    if buffer <= 5:
        st.write("🌱 You selected a small preparation buffer. Monitor actual demand carefully.")
    elif buffer <= 10:
        st.write("🍃 You selected a moderate preparation buffer to allow for demand variation.")
    else:
        st.warning("You selected a relatively large preparation buffer. This may increase leftover food.")
    st.caption("These are planning recommendations based on simulated training data. "
               "Predicted wastage is a separate model estimate; actual waste has not been measured. "
               "The separate consumption and wastage models may give inconsistent estimates.")

    st.subheader("📥 Download Prediction Report")
    report = pd.DataFrame({
        "Day": [day], "Meal_Type": [meal], "Menu_Type": [menu],
        "Students": [students], "Consumption_Model": [selected_model],
        "Predicted_Consumption_kg": [round(predicted_food, 2)],
        "Preparation_Buffer_Percent": [buffer],
        "Recommended_Preparation_kg": [round(recommended_food, 2)],
        "Planned_Extra_Food_kg": [round(planned_extra, 2)],
        "Predicted_Wastage_kg": [round(predicted_waste, 2)],
        "Predicted_Wastage_Percent": [round(waste_percentage, 2)],
        "Expected_Leftover_kg": [round(expected_leftover, 2)],
        "Expected_Leftover_Percent": [round(leftover_percentage, 2)]
    })
    csv = report.to_csv(index=False)
    st.download_button(
        label="📥 Download Report as CSV", data=csv,
        file_name="food_waste_prediction_report.csv", mime="text/csv",
        key="csv_prediction_download_fixed"
    )

    pdf_data = create_pdf_report(
        day, meal, menu, students, buffer, predicted_food,
        recommended_food, planned_extra, predicted_waste, waste_percentage
    )
    st.download_button(
        label="📄 Download Beautiful PDF Report", data=pdf_data,
        file_name="Smart_Canteen_Prediction_Report.pdf",
        mime="application/pdf", key="pdf_prediction_download_fixed"
    )

# Footer
st.markdown("""
<div class="footer">
    🌿 Smart Canteen AI
    <br>
    Reducing Food Waste • Building a Sustainable Future
</div>
""", unsafe_allow_html=True)
