# ==========================================
# app.py - Ejecución y Predicción Actuarial
# ==========================================
import os
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def cargar_modelos():
    """Carga los modelos entrenados previamente desde la carpeta models."""
    ruta_svm = 'models/svm_riesgo_actuarial.pkl'
    
    if not os.path.exists(ruta_svm):
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en '{ruta_svm}'. "
            "Asegúrate de ejecutar primero el script de entrenamiento para generar los archivos .pkl."
        )
    
    # Cargamos el pipeline de SVM que ya incluye el preprocesamiento (ColumnTransformer)
    modelo_svm = joblib.load(ruta_svm)
    return modelo_svm

def predecir_riesgo_cliente(datos_cliente, modelo):
    """
    Recibe un diccionario con los datos del cliente, los convierte a DataFrame
    y utiliza el pipeline cargado para predecir el nivel de riesgo.
    """
    # Mapeo inverso de las etiquetas numéricas a texto descriptivo
    mapeo_riesgo = {
        0: "Bajo Riesgo (Cargos Médicos Mínimos)",
        1: "Riesgo Medio (Cargos Médicos Moderados)",
        2: "Alto Riesgo (Cargos Médicos Elevados / Crónicos)"
    }
    
    # Convertir el diccionario de entrada en un DataFrame de una sola fila
    df_cliente = pd.DataFrame([datos_cliente])
    
    # Asegurar el formato string y minúsculas para las variables categóricas
    for col in ['sex', 'smoker', 'region']:
        df_cliente[col] = df_cliente[col].astype(str).str.strip().str.lower()
    
    # El pipeline de SVM aplica automáticamente el StandardScaler y OneHotEncoder antes de clasificar
    prediccion_numerica = modelo.predict(df_cliente)[0]
    
    return mapeo_riesgo[prediccion_numerica]

# ==========================================
# Ejemplo de Uso del Script en Producción
# ==========================================
if __name__ == "__main__":
    print("--- Inicializando Sistema de Evaluación de Riesgo Actuarial ---")
    
    try:
        # 1. Cargar el modelo guardado
        pipeline_produccion = cargar_modelos()
        print("Modelos cargados exitosamente de la carpeta 'models/'.\n")
        
        # 2. Definir un caso de prueba para evaluación (Simulando una petición entrante)
        # Puedes cambiar estos valores para probar diferentes perfiles de clientes
        nuevo_cliente = {
            'age': 45,
            'sex': 'female',
            'bmi': 32.5,
            'children': 2,
            'smoker': 'yes',      # Cambia a 'no' para ver cómo se reduce el riesgo
            'region': 'southeast',
            'charges': 41000.0    # El estimador financiero o cargo base estimado
        }
        
        print("Datos del cliente a evaluar:")
        for clave, valor in nuevo_cliente.items():
            print(f"  - {clave.capitalize()}: {valor}")
            
        # 3. Ejecutar la predicción
        resultado = predecir_riesgo_cliente(nuevo_cliente, pipeline_produccion)
        
        print("\n==============================================")
        print(f"RESULTADO DEL ANÁLISIS: {resultado}")
        print("==============================================")
        
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema durante la ejecución: {e}")
