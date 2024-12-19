# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from gpt4all import GPT4All  # Offline LLM for mitigation strategies
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Conflict Data Analysis and Mitigation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to preprocess data
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

# Function to train Random Forest model
def train_rf_model(X, y, n_estimators=100, max_depth=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        predictions = rf_model.predict(X_test_scaled)
        return rf_model, X_test_scaled, y_test, predictions
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

# Function to generate mitigation strategy using offline LLM
def generate_mitigation_strategy_llm(conflict_type, region):
    try:
        model = GPT4All("gpt4all-lora-unfiltered-quantized", model_path="./models")  # Ensure model file exists
        prompt = f"""
        Generate a detailed and actionable mitigation strategy for addressing {conflict_type} in the {region}. 
        Include:
        - Immediate response actions
        - Medium-term interventions
        - Long-term prevention measures
        - Key stakeholders involved
        - Potential challenges in implementation
        """
        response = model.generate(prompt)
        return response
    except Exception as e:
        st.error(f"Error generating strategy with LLM: {e}")
        return "An error occurred while generating the mitigation strategy."

# Streamlit app layout
st.title('Conflict Data Analysis and Mitigation Strategies')

# Step 1: Select region before uploading data
region = st.selectbox("Select the region for your data:", ["USA/Canada", "Africa", "Asia", "Europe", "Middle East", "Latin America"])
st.write(f"You selected: {region}")

# Step 2: Upload data
uploaded_file = st.file_uploader("Upload your conflict data CSV file", type="csv")

if uploaded_file:
    # Load data
    df = load_data(uploaded_file)

    if df is not None:
        # Step 3: Preprocess the data
        processed_df, X, y, region_encoder, event_encoder = preprocess_data(df)

        if processed_df is not None:
            # Preview preprocessed data
            st.write("Preprocessed Data Preview:")
            st.dataframe(processed_df.head())

            # Download preprocessed data
            buffer = BytesIO()
            processed_df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Preprocessed Data",
                data=buffer,
                file_name="preprocessed_data.csv",
                mime="text/csv"
            )

            # Step 4: Train Random Forest model
            rf_model, X_test_scaled, y_test, predictions = train_rf_model(X, y)

            if rf_model:
                # Model Evaluation
                st.write("### Model Evaluation")
                evaluation_report = classification_report(y_test, predictions, output_dict=True)
                st.json(evaluation_report)  # Structured for readability

                # Feature Importance
                st.write("### Feature Importance")
                feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
                st.bar_chart(feature_importance.set_index('feature')['importance'])

                # Step 5: EDA using PandasAI
                st.write("### Exploratory Data Analysis with PandasAI")
                api_key = st.text_input("Enter your PandasAI API Key:", type="password")

                if api_key:
                    llm = OpenAI(api_token=api_key)
                    pandas_ai = PandasAI(llm)

                    # Loop for user queries
                    query = st.text_input("Ask a question about your data (e.g., 'What is the average fatalities?'):")
                    if query:
                        try:
                            result = pandas_ai.run(processed_df, prompt=query)
                            st.write("Query Result:")
                            st.write(result)
                        except Exception as e:
                            st.error(f"Error during EDA: {e}")

                # Step 6: Mitigation Strategy Generation
                st.write("### Generate Mitigation Strategy")
                conflict_type = st.selectbox('Select Conflict Type:', df['event_type'].unique())
                if st.button('Generate Strategy'):
                    strategy = generate_mitigation_strategy_llm(conflict_type, region)
                    st.text_area("Generated Mitigation Strategy:", value=strategy, height=300)
