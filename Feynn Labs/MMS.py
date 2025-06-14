# streamlit_medical_segmentation_app.py
import requests
from PIL import Image
from io import BytesIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np

# Set up Streamlit layout
st.set_page_config(page_title="Medical Segmentation Dashboard")

@st.cache_data

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.capitalize()
    if 'Age' in df.columns:
        age_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        age_labels = ['10–20', '20–30', '30–40', '40–50', '50–60', '60–70', '70–80', '80–90']
        df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    return df

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA & Clustering", "ML Model Predictions"])

uploaded_file = st.sidebar.file_uploader("Upload your healthcare_dataset.csv", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    if page == "EDA & Clustering":
        st.title("Medical Segmentation: EDA & Clustering")
        st.sidebar.header("Filters")
        
        image_url = "https://raw.githubusercontent.com/ANIKETGUP3838/Medical-Market-Segmentation/main/h1.jpg"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, width=750)
            
        selected_gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].dropna().unique(), default=df['Gender'].dropna().unique())
        selected_age = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (30, 70))

        df_filtered = df[(df['Gender'].isin(selected_gender)) & (df['Age'].between(*selected_age))]

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("1. Demographic & Billing Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Patient Count by Age Group and Gender**")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_filtered, x='Age Group', hue='Gender', ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.markdown("**Average Billing by Age Group and Gender**")
            fig2, ax2 = plt.subplots()
            sns.barplot(data=df_filtered, x='Age Group', y='Billing Amount', hue='Gender', ci='sd', ax=ax2)
            st.pyplot(fig2)

        st.subheader("2. Admission Type & Conditions")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Admission Type by Gender**")
            fig3, ax3 = plt.subplots()
            sns.countplot(data=df_filtered, x='Admission Type', hue='Gender', ax=ax3)
            st.pyplot(fig3)

        with col4:
            st.markdown("**Top Medical Conditions by Gender**")
            if 'Medical Condition' in df_filtered.columns:
                top_conditions = df_filtered['Medical Condition'].value_counts().nlargest(6).index
                condition_df = df_filtered[df_filtered['Medical Condition'].isin(top_conditions)]
                fig4, ax4 = plt.subplots()
                sns.countplot(data=condition_df, y='Medical Condition', hue='Gender', ax=ax4)
                st.pyplot(fig4)

        st.subheader("3. Unsupervised Segmentation (Clustering)")
        clustering_features = ['Age', 'Billing Amount']
        st.markdown("**PCA + KMeans Clustering using Age and Billing Amount**")
        k = st.slider("Number of clusters (K)", 2, 10, 4)

        X = df_filtered[clustering_features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_pca)

        df_filtered = df_filtered.copy()
        df_filtered['Cluster'] = clusters
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = clusters

        fig5, ax5 = plt.subplots()
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax5)
        ax5.set_title("Patient Clusters Based on Age & Billing")
        st.pyplot(fig5)

        st.subheader("4. Cluster Profiles")
        cluster_summary = df_filtered.groupby('Cluster')[['Age', 'Billing Amount']].mean().round(1)
        cluster_summary['Count'] = df_filtered['Cluster'].value_counts()
        st.dataframe(cluster_summary)

    elif page == "ML Model Predictions":
        st.title("🧠 ML Model Predictions: Test Results Classification")
        st.markdown("This page builds and evaluates classification models to predict **Test Results**.")
    
        df_ml = df.copy()
        df_ml.drop(columns=['Name', 'Date of Admission', 'Discharge Date'], errors='ignore', inplace=True)
    
        # Map target labels to numeric
        df_ml['Test Results'] = df_ml['Test Results'].replace({'Normal': 1, 'Inconclusive': 0, 'Abnormal': 2})
    
        # Drop rows with missing target
        X = df_ml.drop(columns=['Test Results'])
        y = df_ml['Test Results']
        
        # Convert to numeric with coercion
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        
        # Select columns with at least one non-NaN value
        valid_cols = [col for col in X_numeric.columns if X_numeric[col].notna().any()]
        X_valid = X_numeric[valid_cols]
        
        imputer = SimpleImputer(strategy='median')
        X_imputed_array = imputer.fit_transform(X_valid)
        
        X_imputed = pd.DataFrame(X_imputed_array, columns=valid_cols)
        X_imputed.index = X_numeric.index  # preserve index
        
        # Combine with target
        combined = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)
        
        if combined.empty or combined.shape[0] == 0:
            st.error("No data available after preprocessing. Please check the uploaded file or preprocessing steps.")
            st.stop()
        
        X = combined.drop(columns=['Test Results'])
        y = combined['Test Results']
    
        st.write("Preprocessed features shape:", X.shape)
        st.write("Preprocessed target shape:", y.shape)
    
        # 70% train, 30% test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
    
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            results[name] = acc
    
            st.markdown(f"**{name}** - Accuracy: {acc:.3f}")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Abnormal', 'Inconclusive', 'Normal'],
                        yticklabels=['Abnormal', 'Inconclusive', 'Normal'])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
    
        best_model = max(results, key=results.get)
        st.success(f"Best performing model: {best_model} with accuracy {results[best_model]:.3f}")


else:
    st.title("🩺 Medical Segmentation Dashboard")
    image_url = "https://raw.githubusercontent.com/ANIKETGUP3838/Medical-Market-Segmentation/main/h1.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=750)
    st.info("Please upload a CSV file to get started.")
