# ML and GenAI in Geopolitical Conflict Analysis and Mitigation - Web App Implementaion using streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import google.generativeai as genai
import time

# Set page configuration
st.set_page_config(
    page_title="Conflict Data Analysis and Mitigation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess the data
def preprocess_data(df):
    try:
        required_columns = ['Region', 'event_type', 'fatalities']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns: {', '.join(required_columns)}")
            return None, None, None, None, None

        region_encoder = LabelEncoder()
        event_encoder = LabelEncoder()
        df['region_encoded'] = region_encoder.fit_transform(df['Region'])
        df['event_encoded'] = event_encoder.fit_transform(df['event_type'])
        df['log_fatalities'] = np.log1p(df['fatalities'])
        df['region_event_interaction'] = df['region_encoded'] * df['event_encoded']
        X = df[['fatalities', 'log_fatalities', 'event_encoded', 'region_event_interaction']]
        y = df['region_encoded']
        return df, X, y, region_encoder, event_encoder
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None, None, None, None, None

# Train Random Forest model with hyperparameter tuning
def train_rf_model(X, y, n_estimators=100, max_depth=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Display progress bar
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)  # Simulate progress
            progress_bar.progress(i)

        # Allow users to adjust hyperparameters
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        predictions = rf_model.predict(X_test_scaled)
        return rf_model, X_test_scaled, y_test, predictions
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

# Plot feature importance
def plot_feature_importance(rf_model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature')['importance'])
    st.write(feature_importance)

# Enhanced visualization for conflict trends
def plot_conflict_trends(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='event_type', order=df['event_type'].value_counts().index)
    plt.title("Number of Conflicts by Event Type")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Generate conflict mitigation strategy prompt
def conflict_mitigation_prompt(conflict_type, region):
    return f"""
    Generate a detailed and actionable mitigation strategy for addressing {conflict_type} in the {region}. The strategy should include:
    - Immediate response actions
    - Medium-term interventions
    - Long-term prevention measures
    - Key stakeholders involved
    - Potential challenges in implementation
    """

# Generate conflict mitigation strategy using Generative AI
def generate_conflict_mitigation_strategy(api_key, conflict_type, region):
    try:
        genai.configure(api_key=api_key)
        prompt_text = conflict_mitigation_prompt(conflict_type, region)
        response = genai.generate_text(prompt=prompt_text, model='gemini-1.5-flash', max_output_tokens=800, temperature=0.3, top_p=1)
        if response and 'candidates' in response and len(response['candidates']) > 0:
            return response['candidates'][0]['output']
        else:
            return "No strategy generated. Please check your input or API key."
    except Exception as e:
        st.error(f"Error generating mitigation strategy: {e}")
        return "An error occurred while generating the mitigation strategy."

# Rule-based fallback strategy generation
def generate_rule_based_strategy(conflict_type, region):
    strategies = {
        'armed conflict': 'Immediate ceasefire, emergency aid, and peacekeeping.',
        'protest': 'De-escalation through dialogue and reform implementation.',
        'terrorism': 'Increased security, intelligence gathering, and community engagement.',
    }
    return strategies.get(conflict_type.lower(), "No predefined strategy available for this conflict type.")

# Streamlit app layout
st.title('Conflict Data Analysis and Mitigation Strategies')
st.markdown("Analyze conflict data and generate mitigation strategies using machine learning and generative AI.")

# Help Section
with st.expander("Help"):
    st.markdown("""
    ### App Features:
    1. **Data Upload**: Upload a CSV file containing conflict data with columns like Region, event_type, and fatalities.
    2. **Data Preprocessing**: Automatically encodes and preprocesses the data for model training.
    3. **Model Training**: Train a Random Forest model with customizable hyperparameters.
    4. **Visualization**: View conflict trends and feature importance charts.
    5. **Mitigation Strategy Generation**: Generate conflict mitigation strategies using AI or predefined rules.
    6. **API Key**: Enter your API key to enable Generative AI features.

    For detailed instructions, hover over inputs or refer to the documentation.
    """)

# Sidebar layout
st.sidebar.title('Settings')
st.sidebar.info("Upload your data and adjust model settings.")
api_key = st.sidebar.text_input("Enter your API key for Generative AI:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your conflict data CSV file", type="csv")
n_estimators = st.sidebar.slider('Number of Trees (n_estimators):', min_value=10, max_value=200, value=100)
max_depth = st.sidebar.slider('Max Tree Depth (max_depth):', min_value=1, max_value=20, value=None)

# App sections
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        processed_df, X, y, region_encoder, event_encoder = preprocess_data(df)
        if processed_df is not None:
            rf_model, X_test_scaled, y_test, predictions = train_rf_model(X, y, n_estimators, max_depth)
            if rf_model:
                # Model evaluation
                with st.expander('Model Evaluation'):
                    st.text(classification_report(y_test, predictions))
                # Feature importance
                with st.expander('Feature Importance'):
                    plot_feature_importance(rf_model, X)
                # Conflict trends visualization
                with st.expander('Conflict Trends'):
                    plot_conflict_trends(df)
                # Generate mitigation strategy
                with st.expander('Generate Mitigation Strategy'):
                    region = st.selectbox('Select Region:', df['Region'].unique())
                    conflict_type = st.selectbox('Select Conflict Type:', df['event_type'].unique())
                    if st.button('Generate Strategy'):
                        if api_key:
                            strategy = generate_conflict_mitigation_strategy(api_key, conflict_type, region)
                            if "No strategy generated" in strategy:
                                strategy = generate_rule_based_strategy(conflict_type, region)
                            st.text_area('Generated Strategy:', value=strategy, height=300)
                        else:
                            st.error("Please enter your API key.")
