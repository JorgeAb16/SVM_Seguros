# ==========================================
# app.py - Interfaz en Streamlit para el Modelo Entrenado
# ==========================================
import os
import joblib
import pandas as pd
import streamlit as pd_st # (Evitar conflicto de nombres)
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Evaluación de Riesgo Actuarial",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    """Carga el pipeline entrenado y extrae el estimador correcto si viene en un dict/GridSearchCV."""
    ruta_svm = 'models/svm_riesgo_actuarial.pkl'
    if not os.path.exists(ruta_svm):
        st.error(f"❌ No se encontró el archivo del modelo en '{ruta_svm}'. Asegúrate de haber ejecutado tu script de entrenamiento primero.")
        return None
    
    objeto_cargado = joblib.load(ruta_svm)
    
    # --- AQUÍ ESTÁ EL TRUCO PARA EVITAR EL ATTRIBUTEERROR ---
    # Si por error se guardó el objeto GridSearchCV o un diccionario de metadatos/parámetros
    if isinstance(objeto_cargado, dict):
        # Intentamos extraer el mejor estimador si se guardó la estructura de grid_search
        if 'best_estimator_' in objeto_cargado:
            return objeto_cargado['best_estimator_']
        elif 'classifier' in objeto_cargado: 
            # Si guardaste un dict personalizado, busca si la clave contiene el pipeline
            return objeto_cargado['classifier']
        else:
            st.error("❌ El archivo .pkl contiene un diccionario, pero no se encontró un modelo/pipeline válido dentro de él.")
            return None
            
    # Si el objeto tiene el atributo 'best_estimator_' (es un GridSearchCV directo)
    if hasattr(objeto_cargado, 'best_estimator_'):
        return objeto_cargado.best_estimator_
        
    return objeto_cargado

# Título y descripción en la interfaz web
st.title("📊 Sistema Actuarial de Evaluación de Riesgo")
st.markdown("Introduzca los datos del asegurado en el panel inferior para estimar su nivel de riesgo de manera automatizada.")

# Cargar el modelo entrenado
pipeline_svm = cargar_modelo()

if pipeline_svm is not None:
    st.subheader("Datos del Cliente / Asegurado")
    
    # Crear dos columnas en la interfaz para que se vea ordenado
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sexo", options=["male", "female"], format_func=lambda x: "Masculino" if x == "male" else "Femenino")
        bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Número de hijos / dependientes", min_value=0, max_value=10, value=0, step=1)
        
    with col2:
        smoker = st.selectbox("¿Es fumador?", options=["no", "yes"], format_func=lambda x: "No" if x == "no" else "Sí")
        region = st.selectbox("Región de residencia", options=["southeast", "southwest", "northwest", "northeast"])
        charges = st.number_input("Cargos Médicos Estimados ($)", min_value=0.0, max_value=100000.0, value=5000.0, step=500.0)

    st.markdown("---")
    
    # Botón para realizar el cálculo
    if st.button("📊 Evaluar Riesgo del Cliente", type="primary"):
        # Construir el diccionario con las entradas del usuario
        datos_cliente = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': charges
        }
        
        # Convertir a DataFrame (el formato que espera el ColumnTransformer del pipeline)
        df_cliente = pd.DataFrame([datos_cliente])
        
        # Realizar la predicción utilizando el modelo cargado
        prediccion_numerica = pipeline_svm.predict(df_cliente)[0]
        
        # Formatear el resultado visual según el nivel de riesgo
        st.subheader("Resultado del Análisis:")
        
        if prediccion_numerica == 0:
            st.success("🟢 **Bajo Riesgo (Cargos Médicos Mínimos)**")
            st.info("El perfil del cliente denota un comportamiento saludable o cargos financieros estables para la aseguradora.")
        elif prediccion_numerica == 1:
            st.warning("🟡 **Riesgo Medio (Cargos Médicos Moderados)**")
            st.info("El asegurado presenta métricas estándar o moderadas. Requiere un seguimiento regular en la póliza.")
        elif prediccion_numerica == 2:
            st.error("🔴 **Alto Riesgo (Cargos Médicos Elevados / Crónicos)**")
            st.info("Atención: El perfil financiero o clínico representa una alta siniestralidad potencial (cargos médicos elevados).")
