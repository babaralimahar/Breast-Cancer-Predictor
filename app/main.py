import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION MUST BE THE FIRST COMMAND ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.title("🧬 Nuclei Measurements")
    st.sidebar.markdown("Update the cytology lab measurements below to simulate a prediction.")
    
    data = get_clean_data()
    input_dict = {}

    # Grouping the sliders to make the UI look advanced and clean
    slider_groups = {
        "Mean Values": [
            ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
            ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
            ("Smoothness (mean)", "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
            ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave points_mean"),
            ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)", "fractal_dimension_mean")
        ],
        "Standard Error (SE)": [
            ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"),
            ("Perimeter (se)", "perimeter_se"), ("Area (se)", "area_se"),
            ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
            ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave points_se"),
            ("Symmetry (se)", "symmetry_se"), ("Fractal dimension (se)", "fractal_dimension_se")
        ],
        "Worst Values": [
            ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
            ("Perimeter (worst)", "perimeter_worst"), ("Area (worst)", "area_worst"),
            ("Smoothness (worst)", "smoothness_worst"), ("Compactness (worst)", "compactness_worst"),
            ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave points_worst"),
            ("Symmetry (worst)", "symmetry_worst"), ("Fractal dimension (worst)", "fractal_dimension_worst")
        ]
    }

    # Create expanders for each group
    for group_name, sliders in slider_groups.items():
        with st.sidebar.expander(group_name, expanded=(group_name == "Mean Values")):
            for label, key in sliders:
                input_dict[key] = st.slider(
                    label,
                    min_value=float(0),
                    max_value=float(data[key].max()),
                    value=float(data[key].mean())
                )
                
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 'Concavity', 
                  'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories, fill='toself', name='Mean Value', line_color='#1f77b4'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], 
           input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'], 
           input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
           input_data['fractal_dimension_se']],
        theta=categories, fill='toself', name='Standard Error', line_color='#ff7f0e'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories, fill='toself', name='Worst Value', line_color='#2ca02c'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)' # Transparent background for modern look
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    
    st.markdown("### Diagnosis Result")
    
    if prediction[0] == 0:
        st.markdown("<div class='diagnosis benign'>Status: <b>Benign</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='diagnosis malicious'>Status: <b>Malignant</b></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Confidence Metrics")
    
    benign_prob = model.predict_proba(input_array_scaled)[0][0]
    malignant_prob = model.predict_proba(input_array_scaled)[0][1]
    
    # Advanced UI: Using Streamlit Columns and Metrics for probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Benign Probability", value=f"{benign_prob * 100:.2f}%")
    with col2:
        st.metric(label="Malignant Probability", value=f"{malignant_prob * 100:.2f}%")
        
    st.info("⚠️ **Disclaimer:** This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    # Load custom CSS
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    # Header Section
    st.title("🔬 Breast Cancer Predictor ")
    st.markdown("""
        Welcome to the diagnostic assistant dashboard. Please connect this app to your cytology lab to help diagnose breast cancer from your tissue samples. 
        This app uses a Machine Learning model (Logistic Regression) to predict whether a breast mass is **benign** or **malignant** based on measurements received from the lab. 
        Use the sidebar to manually adjust measurements and observe real-time predictions.
    """)
    st.markdown("---")
    
    # Main Dashboard Layout
    col1, col2 = st.columns([5, 3], gap="large")
    
    with col1:
        st.markdown("### Cellular Profile (Radar Chart)")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
        
    with col2:
        add_predictions(input_data)

    # Footer
    st.markdown("<div class='footer'>Developed by: <b> Babar Ali </b> </div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
