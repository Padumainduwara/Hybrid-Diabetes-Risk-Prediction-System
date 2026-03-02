import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- 1. System Configuration ---
st.set_page_config(
    page_title="Medical AI", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Custom Dark Theme CSS ---
st.markdown("""
    <style>
    /* Force Dark Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1cb5e0 0%, #000851 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.2);
        border: 1px solid #333;
    }
    
    /* Input Section Containers (Dark Cards) */
    .section-container {
        background-color: #262730;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #1cb5e0;
        border: 1px solid #3d3d3d;
    }
    
    h3 {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    /* Input Fields Styling Override */
    div[data-baseweb="input"] {
        background-color: #1e1e1e !important; 
    }
    
    /* Button Styling (Neon Glow) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.8);
    }
    
    /* Result Box Styling */
    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        animation: fadeIn 1s;
        color: white;
    }
    .high-risk { 
        background-color: rgba(255, 75, 75, 0.15); 
        border: 2px solid #ff4b4b; 
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.3);
    }
    .low-risk { 
        background-color: rgba(0, 200, 81, 0.15); 
        border: 2px solid #00c851; 
        box-shadow: 0 0 15px rgba(0, 200, 81, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Models ---
@st.cache_resource
def load_system():
    # Load Traditional ML
    xgb_pipeline = joblib.load('Models/xgb_model.pkl')
    
    # Load Deep Learning Models
    ann_model = tf.keras.models.load_model('Models/ann_model.keras')
    resnet_model = tf.keras.models.load_model('Models/resnet_model.keras')
    
    # Load Preprocessor
    dl_preprocessor = joblib.load('Models/preprocessor.pkl')
    
    return xgb_pipeline, ann_model, resnet_model, dl_preprocessor

try:
    xgb_model, ann_model, resnet_model, preprocessor = load_system()
except Exception as e:
    st.error(f"⚠️ System Error: Model files not found. Please ensure files are in 'Models/' directory.")
    st.stop()

# --- 4. Feature Engineering Logic ---
def process_input(data_df):
    df = data_df.copy()
    df['Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['MAP'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
    df['Cholesterol_HDL_Ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['BMI_Age_Risk'] = df['bmi'] * df['age']
    return df

# --- 5. Main Application Layout ---

# Header
st.markdown("""
    <div class="main-header">
        <h1>🧬 AI Health System</h1>
        <p>Advanced Hybrid Ensemble Neural Network</p>
    </div>
""", unsafe_allow_html=True)

# Input Form
with st.form("medical_form"):
    
    # Section 1: Demographics
    st.markdown('<div class="section-container"><h3>👤 Patient Demographics</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age (Years)", 18, 90, 45)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
    with c4:
        family_hist = st.selectbox("Family History", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    # Section 2: Vitals
    st.markdown('<div class="section-container"><h3>⚖️ Vitals & Body Composition</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bmi = st.number_input("BMI Index", 15.0, 60.0, 25.5)
    with c2:
        waist_hip = st.slider("Waist/Hip Ratio", 0.5, 1.2, 0.85)
    with c3:
        sys_bp = st.number_input("Systolic BP", 90, 220, 120)
    with c4:
        dia_bp = st.number_input("Diastolic BP", 60, 140, 80)

    # Section 3: Lab Results
    st.markdown('<div class="section-container"><h3>🩸 Clinical Lab Data</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        chol_total = st.slider("Total Cholesterol", 100, 350, 180)
    with c2:
        hdl = st.slider("HDL Cholesterol", 20, 120, 50)
    with c3:
        triglycerides = st.slider("Triglycerides", 50, 400, 150)

    # Submit Button
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🚀 ANALYZE RISK NOW")

# --- 6. Prediction Engine ---
if submit_button:
    # Prepare Data
    input_data = {
        'age': age,
        'alcohol_consumption_per_week': 2,
        'physical_activity_minutes_per_week': 120,
        'diet_score': 5.0,
        'sleep_hours_per_day': 7.0,
        'screen_time_hours_per_day': 4.0,
        'bmi': bmi,
        'waist_to_hip_ratio': waist_hip,
        'systolic_bp': sys_bp,
        'diastolic_bp': dia_bp,
        'heart_rate': 75,
        'cholesterol_total': chol_total,
        'hdl_cholesterol': hdl,
        'ldl_cholesterol': 100,
        'triglycerides': triglycerides,
        'gender': gender,
        'ethnicity': ethnicity,
        'education_level': 'Highschool',
        'income_level': 'Middle',
        'smoking_status': 'Never',
        'employment_status': 'Employed',
        'family_history_diabetes': family_hist,
        'hypertension_history': 0,
        'cardiovascular_history': 0
    }

    df_raw = pd.DataFrame([input_data])
    df_processed = process_input(df_raw)

    with st.spinner('🤖 AI Engines Running...'):
        try:
            # Inference
            pred_xgb = xgb_model.predict_proba(df_processed)[0][1]
            dl_input = preprocessor.transform(df_processed)
            pred_ann = ann_model.predict(dl_input, verbose=0)[0][0]
            pred_resnet = resnet_model.predict(dl_input, verbose=0)[0][0]
            
            # Weighted Ensemble
            final_prob = (0.4 * pred_xgb) + (0.3 * pred_ann) + (0.3 * pred_resnet)
            
            # --- Results Display ---
            st.markdown("---")
            
            # Verdict Logic
            risk_class = "HIGH RISK DETECTED" if final_prob > 0.5 else "LOW RISK"
            css_class = "high-risk" if final_prob > 0.5 else "low-risk"
            icon = "⚠️" if final_prob > 0.5 else "✅"
            
            # Main Result Card
            st.markdown(f"""
                <div class="result-box {css_class}">
                    <h2 style="margin:0; color:white;">{icon} {risk_class}</h2>
                    <h1 style="font-size: 4em; margin: 10px 0; color:white;">{final_prob:.1%}</h1>
                    <p style="font-size: 1.2em; color:#ddd;">Diabetes Risk Probability</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("<br>### 🧠 System Consensus", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("XGBoost", f"{pred_xgb:.1%}")
            c2.metric("Neural Network", f"{pred_ann:.1%}")
            c3.metric("ResNet", f"{pred_resnet:.1%}")
            
            # Final Note based on Prediction
            if final_prob > 0.5:
                st.error("⚠️ Medical Alert: High probability of Type 2 Diabetes. Clinical screening recommended.")
            else:
                st.success("✅ Medical Clearance: Biomarkers indicate a healthy metabolic profile.")

            # --- EXPLAINABILITY & SUGGESTIONS SECTION ---
            st.markdown("---")
            st.subheader("🔍 Why is this result?")
            st.markdown("Analysis of key risk drivers compared to standard medical thresholds:")

            # 1. Comparison Logic
            factors = []
            
            # Check BMI
            factors.append({"Metric": "BMI", "Your Value": bmi, "Limit": 25.0, "Status": "High" if bmi > 25 else "Normal"})
            # Check BP
            factors.append({"Metric": "Systolic BP", "Your Value": sys_bp, "Limit": 130.0, "Status": "High" if sys_bp > 130 else "Normal"})
            # Check Cholesterol
            factors.append({"Metric": "Cholesterol", "Your Value": chol_total, "Limit": 200.0, "Status": "High" if chol_total > 200 else "Normal"})
            
            # Create Dataframe for Chart
            risk_df = pd.DataFrame(factors)
            
            # 2. Display Bar Chart
            col_chart, col_text = st.columns([2, 1])
            
            with col_chart:
                risk_df['Risk Score'] = (risk_df['Your Value'] / risk_df['Limit']) * 100
                st.caption("Risk Factor Contribution (Values > 100% exceed healthy limits)")
                st.bar_chart(risk_df.set_index("Metric")['Risk Score'], color="#ff4b4b")

            # 3. Dynamic Suggestions
            with col_text:
                st.markdown("### 💡 AI Suggestions")
                suggestions_found = False
                
                if bmi > 25:
                    st.warning(f"**Weight:** BMI ({bmi}) is high. Aim for caloric deficit.")
                    suggestions_found = True
                
                if sys_bp > 130:
                    st.warning(f"**BP:** Systolic BP ({sys_bp}) is elevated. Reduce sodium.")
                    suggestions_found = True
                
                if chol_total > 200:
                    st.warning(f"**Cholesterol:** Level ({chol_total}) is high. Increase fiber.")
                    suggestions_found = True
                
                # Check for other factors if primary ones are normal but risk is high
                if not suggestions_found:
                    if final_prob < 0.5:
                        st.success("✅ Great! All primary risk factors are within healthy ranges. Maintain your current lifestyle.")
                    else:
                        st.info("ℹ️ Note: While BMI and Blood Pressure are normal, other factors (Age, Genetics, or Ethnicity) are contributing to the High Risk score.")

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("<div style='text-align: center; margin-top: 50px; color: #555;'>AI Labs • Powered by TensorFlow & XGBoost</div>", unsafe_allow_html=True)