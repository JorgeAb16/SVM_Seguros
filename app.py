import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Evaluación de Riesgo Actuarial",
    page_icon="🛡️",
    layout="centered"
)

# Título de la app
st.title("🛡️ Sistema de Evaluación de Riesgo Actuarial")
st.markdown("Clasificación de riesgo mediante K-Means y Máquinas de Vectores de Soporte (SVM).")
st.markdown("---")

# Función para cargar los modelos de forma segura
@st.cache_resource
def load_models():
    try:
        kmeans_model = joblib.load(Path("models/kmeans_riesgo_actuarial.pkl"))
        svm_model = joblib.load(Path("models/svm_riesgo_actuarial.pkl"))
        return kmeans_model, svm_model
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

kmeans, svm = load_models()

if kmeans and svm:
    # --- INTERFAZ DE USUARIO / FORMULARIO ---
    st.sidebar.header("📋 Datos del Asegurado")
    
    age = st.sidebar.slider("Edad (age)", min_value=18, max_value=100, value=35)
    sex = st.sidebar.selectbox("Sexo (sex)", options=["male", "female"])
    bmi = st.sidebar.number_input("Índice de Masa Corporal (bmi)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.sidebar.slider("Hijos / Dependientes (children)", min_value=0, max_value=5, value=0)
    smoker = st.sidebar.selectbox("¿Fuma? (smoker)", options=["yes", "no"])
    region = st.sidebar.selectbox("Región (region)", options=["southwest", "southeast", "northwest", "northeast"])
    charges = st.sidebar.number_input("Cargos Médicos Estimados (charges)", min_value=1000.0, max_value=70000.0, value=13000.0, step=500.0)

    # Crear el DataFrame con la misma estructura que el 'insurance.csv' original
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex.lower().strip(),
        'bmi': bmi,
        'children': children,
        'smoker': smoker.lower().strip(),
        'region': region.lower().strip(),
        'charges': charges
    }])

    # Mostrar los datos ingresados en la pantalla principal
    st.subheader("Datos Ingresados para el Análisis")
    st.dataframe(input_data)

    # --- PREDICCIÓN ---
    if st.button("📊 Evaluar Riesgo Actuarial"):
        with st.spinner("Procesando datos y calculando riesgo..."):
            try:
                # 1. Predicción de Segmentación (K-Means)
                # Nota: Si tu modelo requiere que los datos estén preprocesados (OneHotEncoder/Scaler), 
                # asegúrate de que tu archivo .pkl sea el Pipeline completo.
                cluster_pred = kmeans.predict(input_data)[0]
                
                # 2. Predicción de Clasificación de Riesgo (SVM)
                # Si el SVM depende del resultado del cluster, puedes agregarlo aquí:
                # input_data['cluster'] = cluster_pred
                risk_pred = svm.predict(input_data)[0]
                
                # --- MOSTRAR RESULTADOS ---
                st.markdown("---")
                st.subheader("🚀 Resultados del Modelo")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Segmento de Cliente (Clúster)", value=f"Grupo {cluster_pred}")
                    st.info(f"El cliente pertenece al perfil de comportamiento del Grupo {cluster_pred}.")

                with col2:
                    # Formatear el color dependiendo de la salida de tu SVM (ej. si es Riesgo Alto/Bajo o 0/1)
                    st.metric(label="Nivel de Riesgo Actuarial", value=f"{risk_pred}")
                    if str(risk_pred).lower() in ['high', 'alto', '1']:
                        st.error("⚠️ Alerta: Este perfil presenta un Riesgo Actuarial Elevado.")
                    else:
                        st.success("✅ Perfil Seguro: Riesgo Actuarial Bajo / Moderado.")
                        
            except Exception as e:
                st.error(f"Ocurrió un error al realizar la predicción. Verifica que el archivo .pkl incluya el preprocesamiento (Pipeline): {e}")
else:
    st.warning("⚠️ Esperando que los modelos se carguen correctamente...")
