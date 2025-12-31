import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Liver Cirrhosis Stage Classification", layout="wide")
st.title("🫁 Integrating ML and SHAP Explainability for Reliable Liver Cirrhosis Stage Classification")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("liver_cirrhosis.csv")
    
    # Encode categorical variables
    categorical_cols = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Fill missing values with median (if any)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df, le_dict

df, label_encoders = load_and_preprocess_data()

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a section", 
                        ["🏠 Home", "📊 Data Exploration", "📈 Model Performance & ROC", "🔮 Prediction with SHAP"])

# ------------------- Home -------------------
if page == "🏠 Home":
    st.markdown("""
    ### Project Overview
    This application demonstrates the use of multiple machine learning models to predict the **stage of liver cirrhosis** 
    (Stage 1, 2, or 3) using clinical and laboratory features.
    
    **Key Features:**
    - Multiple classifiers: Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, KNN
    - Model performance comparison with classification reports
    - ROC curves visualization (based on reported metrics)
    - **Interactive predictions** with **SHAP explanations** for interpretability
    
    SHAP (SHapley Additive exPlanations) helps understand which features drive each prediction.
    """)

# ------------------- Data Exploration -------------------
elif page == "📊 Data Exploration":
    st.header("Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Target Distribution (Stage)")
    fig, ax = plt.subplots()
    df['Stage'].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Cirrhosis Stage")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Liver Cirrhosis Stages")
    st.pyplot(fig)

# ------------------- Model Performance & ROC -------------------
elif page == "📈 Model Performance & ROC":
    st.header("Model Training & Performance")
    
    X = df.drop('Stage', axis=1)
    y = df['Stage']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }
    
    results = {}
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            if name in ["SVM", "Logistic Regression"]:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
        results[name] = model
    
    st.success("All models trained successfully!")
    
    st.subheader("Classification Reports")
    for name, model in results.items():
        with st.expander(f"{name}"):
            if name in ["SVM", "Logistic Regression"]:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(pd.DataFrame(report).transpose())
    
    st.subheader("ROC Curves (from Research Table 10)")
    # Hardcoded values from your notebook
    model_names = ['LR', 'DT', 'RF', 'SVM', 'NB', 'KNN']
    sensitivity = np.array([50.95, 97.08, 98.83, 61.33, 48.46, 97.48]) / 100
    fpr = np.array([18.10, 1.03, 0.53, 20.83, 19.94, 1.24]) / 100
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()
    
    for i, model in enumerate(model_names):
        roc_auc = auc([0, fpr[i], 1], [0, sensitivity[i], 1])
        axs[i].plot([0, fpr[i], 1], [0, sensitivity[i], 1], color='blue', lw=2,
                    label=f'{model} (AUC = {roc_auc:.2f})')
        axs[i].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
        axs[i].set_title(f'ROC Curve - {model}')
        axs[i].legend(loc="lower right")
        axs[i].text(0.5, -0.2, f'({chr(97+i)})', transform=axs[i].transAxes, 
                    ha='center', va='top', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)

# ------------------- Prediction with SHAP -------------------
elif page == "🔮 Prediction with SHAP":
    st.header("Make a New Prediction")
    
    X = df.drop('Stage', axis=1)
    y = df['Stage']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Retrain best model (Random Forest recommended for SHAP)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    
    # Input form
    st.write("Enter patient details:")
    input_data = {}
    
    cols = st.columns(3)
    features = X.columns
    for i, feature in enumerate(features):
        with cols[i % 3]:
            if feature in ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']:
                unique_vals = df[feature].unique()
                mapping = {v: label_encoders[feature].transform([v])[0] for v in unique_vals}
                display_val = st.selectbox(feature, options=list(mapping.keys()), key=feature)
                input_data[feature] = mapping[display_val]
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                median_val = float(df[feature].median())
                input_data[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, 
                                                      value=median_val, key=feature)
    
    if st.button("Predict Stage & Explain with SHAP"):
        input_df = pd.DataFrame([input_data])
        
        # Prediction
        pred = rf_model.predict(input_df)[0]
        probas = rf_model.predict_proba(input_df)[0]
        
        st.success(f"**Predicted Stage: {int(pred)}**")
        st.write("Class probabilities:")
        for i, prob in enumerate(probas):
            st.write(f"Stage {i+1}: {prob:.2%}")
        
        # SHAP Explanation
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(input_df)
            
            # For multi-class, SHAP returns list; use the one for predicted class
            if isinstance(shap_values, list):
                shap_val = shap_values[pred - 1]  # Stage starts from 1
            else:
                shap_val = shap_values
            
            st.subheader("SHAP Force Plot (Why this prediction?)")
            shap.initjs()
            fig_force = shap.force_plot(explainer.expected_value[pred-1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                        shap_val[0], input_df.iloc[0], matplotlib=True, show=False)
            st.pyplot(fig_force)
            
            st.subheader("SHAP Feature Importance (Bar Plot)")
            fig_bar, ax = plt.subplots()
            shap.summary_plot(shap_val, input_df, plot_type="bar", show=False)
            st.pyplot(fig_bar)

# Footer
st.markdown("---")
st.caption("Liver Cirrhosis Stage Classification App | Built with Streamlit, scikit-learn & SHAP")