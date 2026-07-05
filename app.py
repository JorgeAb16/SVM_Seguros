import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Actuarial | SVM & K-means",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

KMEANS_PATH = MODEL_DIR / "kmeans_riesgo_actuarial.pkl"
SVM_PATH = MODEL_DIR / "svm_riesgo_actuarial.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"
CSV_CLUSTERS_PATH = OUTPUT_DIR / "insurance_con_clusters.csv"
SVM_RESULTS_PATH = OUTPUT_DIR / "svm_resultados_kernels.csv"

# ----------------------------------------------------------------------------
# Estilos personalizados
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f6f8fb;
    }
    .app-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f2545 0%, #1d4e89 55%, #2e86ab 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(15, 37, 69, 0.25);
    }
    .app-header h1 {
        margin-bottom: 0.2rem;
        font-size: 1.9rem;
    }
    .app-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .risk-card {
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    .risk-bajo { background: linear-gradient(135deg, #1e8e5a, #43b581); }
    .risk-medio { background: linear-gradient(135deg, #c98a13, #f0a500); }
    .risk-alto { background: linear-gradient(135deg, #b3261e, #e04b3f); }
    .metric-box {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eaeef3;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f2545;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f3f8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Riesgo Actuarial — K-means & SVM</h1>
        <p>IS-701 · Inteligencia Artificial · Campus Comayagua · Segmentación no supervisada y clasificación de riesgo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Carga de modelos y datos (cacheado)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelo_kmeans():
    if not KMEANS_PATH.exists():
        return None
    return joblib.load(KMEANS_PATH)


@st.cache_resource(show_spinner=False)
def cargar_modelos_svm():
    if not SVM_PATH.exists():
        return None
    return joblib.load(SVM_PATH)


@st.cache_data(show_spinner=False)
def cargar_metadata():
    if not META_PATH.exists():
        return None
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_csv_clusters():
    if not CSV_CLUSTERS_PATH.exists():
        return None
    return pd.read_csv(CSV_CLUSTERS_PATH)


@st.cache_data(show_spinner=False)
def cargar_csv_svm():
    if not SVM_RESULTS_PATH.exists():
        return None
    return pd.read_csv(SVM_RESULTS_PATH)


modelo_completo = cargar_modelo_kmeans()
svm_models = cargar_modelos_svm()
metadata = cargar_metadata()
df_clusters = cargar_csv_clusters()
df_svm = cargar_csv_svm()

if metadata is not None:
    mapa_riesgo = {int(k): v for k, v in metadata.get("mapa_riesgo", {}).items()}
else:
    mapa_riesgo = {0: "Bajo", 1: "Medio", 2: "Alto"}

EXPLICACIONES = {
    "Bajo": "Cliente agrupado con perfiles de menor costo médico promedio.",
    "Medio": "Cliente agrupado con perfiles de costo y factores de riesgo intermedios.",
    "Alto": "Cliente agrupado con perfiles de mayor costo médico promedio y/o factores de riesgo relevantes.",
}

RISK_CSS_CLASS = {"Bajo": "risk-bajo", "Medio": "risk-medio", "Alto": "risk-alto"}
RISK_EMOJI = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}

# ----------------------------------------------------------------------------
# Aviso si faltan archivos
# ----------------------------------------------------------------------------
archivos_faltantes = [
    str(p) for p in [KMEANS_PATH, META_PATH] if not p.exists()
]
if archivos_faltantes:
    st.warning(
        "⚠️ No se encontraron algunos archivos necesarios: "
        + ", ".join(archivos_faltantes)
        + ". Copia las carpetas `models/` y `outputs/` generadas por el notebook "
        "en la misma ubicación que `app.py`."
    )

# ----------------------------------------------------------------------------
# Sidebar - formulario de cliente nuevo
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Datos del cliente")
    st.caption("Completa la información para estimar su nivel de riesgo actuarial.")

    age = st.slider("Edad", min_value=18, max_value=100, value=45)
    sex = st.selectbox("Sexo", ["male", "female"], format_func=lambda x: "Masculino" if x == "male" else "Femenino")
    bmi = st.slider("BMI (índice de masa corporal)", min_value=10.0, max_value=60.0, value=31.2, step=0.1)
    children = st.slider("Número de hijos", min_value=0, max_value=10, value=2)
    smoker = st.selectbox("¿Fumador?", ["yes", "no"], format_func=lambda x: "Sí" if x == "yes" else "No")
    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
    charges = st.number_input("Cargos médicos estimados ($)", min_value=0.0, value=28000.0, step=500.0)

    evaluar = st.button("🔍 Evaluar riesgo", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Tabs principales
# ----------------------------------------------------------------------------
tab_eval, tab_clusters, tab_svm, tab_datos = st.tabs(
    ["🧮 Evaluar cliente", "📊 Segmentación (K-means)", "🧩 Comparación SVM", "📁 Datos"]
)

# --- Tab 1: Evaluación de cliente -------------------------------------------
with tab_eval:
    st.subheader("Resultado de la evaluación")

    if modelo_completo is None:
        st.error(
            "No se pudo cargar `models/kmeans_riesgo_actuarial.pkl`. "
            "Verifica que el archivo exista en esa ruta relativa a `app.py`."
        )
    else:
        if evaluar:
            cliente = pd.DataFrame([{
                "age": age,
                "sex": str(sex).lower(),
                "bmi": bmi,
                "children": children,
                "smoker": str(smoker).lower(),
                "region": str(region).lower(),
                "charges": charges,
            }])

            cluster = int(modelo_completo.predict(cliente)[0])
            riesgo = mapa_riesgo.get(cluster, "Desconocido")
            explicacion = EXPLICACIONES.get(riesgo, "Sin información adicional para este cluster.")

            col1, col2 = st.columns([1, 2])
            with col1:
                css_class = RISK_CSS_CLASS.get(riesgo, "risk-medio")
                emoji = RISK_EMOJI.get(riesgo, "⚪")
                st.markdown(
                    f"""
                    <div class="risk-card {css_class}">
                        <div style="font-size:2.2rem;">{emoji}</div>
                        <div style="font-size:1.5rem; margin-top:0.3rem;">Riesgo {riesgo}</div>
                        <div style="font-size:0.9rem; opacity:0.9; margin-top:0.4rem;">Cluster #{cluster}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("**Explicación del modelo**")
                st.info(explicacion)
                st.markdown("**Datos evaluados**")
                st.dataframe(cliente, use_container_width=True, hide_index=True)
        else:
            st.info("Completa el formulario en la barra lateral y presiona **Evaluar riesgo**.")

    if metadata is not None:
        with st.expander("ℹ️ Metadatos del modelo"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters (K)", metadata.get("n_clusters", "—"))
            c2.metric("Silhouette score", metadata.get("silhouette_score", "—"))
            c3.metric("Kernels SVM entrenados", len(metadata.get("svm_kernels", [])))
            st.json(metadata)

# --- Tab 2: Segmentación K-means ---------------------------------------------
with tab_clusters:
    st.subheader("Perfil de los clusters entrenados")

    if df_clusters is None:
        st.warning(
            "No se encontró `outputs/insurance_con_clusters.csv`. "
            "Copia ese archivo generado por el notebook para ver esta sección."
        )
    else:
        resumen = df_clusters.groupby("riesgo_actuarial").agg(
            cantidad_clientes=("riesgo_actuarial", "count"),
            edad_promedio=("age", "mean"),
            bmi_promedio=("bmi", "mean"),
            cargos_promedio=("charges", "mean"),
            porcentaje_fumadores=("smoker", lambda x: (x == "yes").mean() * 100),
        ).round(2)

        orden = [r for r in ["Bajo", "Medio", "Alto"] if r in resumen.index]
        resumen = resumen.reindex(orden)

        cols = st.columns(len(resumen))
        for col, (riesgo, fila) in zip(cols, resumen.iterrows()):
            css_class = RISK_CSS_CLASS.get(riesgo, "risk-medio")
            emoji = RISK_EMOJI.get(riesgo, "⚪")
            with col:
                st.markdown(
                    f"""
                    <div class="risk-card {css_class}">
                        <div style="font-size:1.6rem;">{emoji} {riesgo}</div>
                        <div style="font-size:0.85rem; margin-top:0.5rem;">
                            Clientes: {int(fila['cantidad_clientes'])}<br>
                            Edad prom.: {fila['edad_promedio']}<br>
                            BMI prom.: {fila['bmi_promedio']}<br>
                            Cargos prom.: ${fila['cargos_promedio']:,.0f}<br>
                            % Fumadores: {fila['porcentaje_fumadores']}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Distribución de cargos médicos por nivel de riesgo")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(
            data=df_clusters, x="riesgo_actuarial", y="charges",
            order=orden, hue="riesgo_actuarial", palette="viridis", ax=ax, legend=False,
        )
        ax.set_xlabel("Riesgo actuarial")
        ax.set_ylabel("Cargos médicos ($)")
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Composición de fumadores por nivel de riesgo")
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        sns.countplot(
            data=df_clusters, x="riesgo_actuarial", hue="smoker",
            order=orden, palette="rocket", ax=ax2,
        )
        ax2.set_xlabel("Riesgo actuarial")
        ax2.set_ylabel("Cantidad de clientes")
        st.pyplot(fig2, use_container_width=True)

# --- Tab 3: Comparación de kernels SVM ---------------------------------------
with tab_svm:
    st.subheader("Comparación de kernels SVM (entrenados sobre componentes PCA)")

    if df_svm is None:
        st.warning(
            "No se encontró `outputs/svm_resultados_kernels.csv`. "
            "Copia ese archivo generado por el notebook para ver esta sección."
        )
    else:
        st.dataframe(
            df_svm.style.background_gradient(cmap="Blues", subset=["accuracy", "precision_macro"]),
            use_container_width=True, hide_index=True,
        )

        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        df_melt = df_svm.melt(id_vars="kernel", value_vars=["accuracy", "precision_macro"],
                               var_name="métrica", value_name="valor")
        sns.barplot(data=df_melt, x="kernel", y="valor", hue="métrica", palette="mako", ax=ax3)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Valor")
        ax3.set_xlabel("Kernel")
        st.pyplot(fig3, use_container_width=True)

        st.caption(
            "Nota: estos modelos SVM fueron entrenados sobre 2 componentes principales (PCA) "
            "únicamente con fines de visualización de fronteras de decisión en clase; "
            "la predicción de riesgo para clientes nuevos en esta app usa el pipeline de K-means."
        )

    if svm_models is not None:
        st.caption(f"Kernels disponibles en el archivo `.pkl`: {', '.join(svm_models.keys())}")

# --- Tab 4: Datos crudos ------------------------------------------------------
with tab_datos:
    st.subheader("Datos con clusters asignados")
    if df_clusters is None:
        st.warning("No se encontró `outputs/insurance_con_clusters.csv`.")
    else:
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Descargar CSV",
            data=df_clusters.to_csv(index=False).encode("utf-8"),
            file_name="insurance_con_clusters.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Proyecto académico · IS-701 Inteligencia Artificial · Campus Comayagua · UNAH")
