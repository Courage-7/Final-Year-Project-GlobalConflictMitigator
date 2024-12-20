# GeopoliticalConflictMitigator
## Overview
Geopolitical Conflict Analysis Mitigator is a comprehensive tool designed to analyze conflict data and generate actionable mitigation strategies. The application leverages machine learning models, including a Random Forest and Auto ARIMA model, alongside a Large Language Model (LLM) to provide insights and strategies for resolving conflicts across different regions of the world.

This project was developed as part of my final year project, focusing on the intersection of data science, machine learning, and AI-driven decision-making in the context of global conflict resolution.

## Features
Data Analysis: Perform in-depth analysis of conflict data, including event types, fatalities, and regional patterns.

## Predictive Modeling:
Random Forest Model: Classifies regions based on conflict data using a robust ensemble method.

ARIMA Model: Predicts conflict outcomes by analyzing sequences and temporal patterns in the data.

LLM-Powered Strategy Generation: Generate detailed and tailored conflict mitigation strategies using an LLM based on the model's predictions.

User-Friendly Interface: Interact with the data and models through a streamlined Streamlit application.

Customizable Models: Adjust model parameters to refine predictions and strategies.

## Project Structure
├── ACLED data/              # Directory to store datasets

├── EDA with PandasAI/       # Jupyter notebooks for data analysis using Pandasai.

├── streamlit_app/          # Streamlit app code and configurations

├── models/                 # Trained models and scripts for training, Randomforest and Auto ARIMA.

├── requirements.txt        # Python dependencies

├── README.md               # Project documentation



## Usage ( Streamlit web app Implementation )
Upload Conflict Data
Upload your conflict data CSV file via the sidebar in the Streamlit app. The data should contain columns for regions, event types, and fatalities.

## Analyze Data ( PandasAI - users are able to query their dataset using natural language )
Once the data is uploaded, the users will perform several tasks like preprocessing, exploratory data analysis and display key insights about the data, examples including:

Distribution of conflict events by region.
Analysis of fatalities and event types.
Show the trend of conflicts over the past decade... to mention a few.

## Model Training and Evaluation
Choose between the Random Forest and ARIMA models to classify regions or predict conflict outcomes.

Random Forest Model: The app will display a classification report and feature importance analysis after training.
ARIMA Model: The app will display a classification report after training.
Generate Mitigation Strategies
Using the trained model's predictions, generate conflict mitigation strategies tailored to specific regions. The strategies are generated using an LLM and cover immediate, medium-term, and long-term measures.

## Dependencies
Python 3.8+
Streamlit
Scikit-learn
TensorFlow
Pandas
NumPy
Google Generative AI API
transformers
gpt4all

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Special thanks to ACLED for providing the conflict data used in this project.
Thanks to the Streamlit, Scikit-learn, and TensorFlow communities for providing excellent tools for data science and machine learning.
