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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        background-attachment: fixed;
    }
    
    /* Header principal */
    .app-header {
        padding: 2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3a7bd5 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -20%;
        width: 150%;
        height: 150%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    }
    
    .app-header h1 {
        margin-bottom: 0.3rem;
        font-size: 2rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .app-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Tarjetas de riesgo */
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .risk-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.3);
    }
    
    .risk-bajo { 
        background: linear-gradient(135deg, #0b8a5e 0%, #15b87e 100%);
    }
    
    .risk-medio { 
        background: linear-gradient(135deg, #d4840a 0%, #f0a500 100%);
    }
    
    .risk-alto { 
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
    }
    
    /* Sidebar - Solo afecta a la sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #1e2d42 50%, #243447 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] * {
        color: #e8ecf1 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #2a5298 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(42, 82, 152, 0.6) !important;
    }
    
    /* Estilos para pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f2f6;
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: #2c3e50;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 60, 114, 0.08);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* CORRECCIÓN: Métricas con fondo blanco y texto oscuro */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e1e5eb;
    }
    
    div[data-testid="stMetric"] label {
        color: #5a6c7d !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1e3c72 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #5a6c7d !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    
    /* Asegurar que todo el contenido principal tenga texto oscuro */
    .main .stMarkdown,
    .main p,
    .main span,
    .main div {
        color: #2c3e50;
    }
    
    /* Pero mantener blanco en elementos específicos */
    .app-header,
    .app-header *,
    .risk-card,
    .risk-card * {
        color: white !important;
    }
    
    /* Info, warning, error boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a2332 0%, #1e2d42 100%);
        border-radius: 12px;
        margin-top: 2rem;
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    .footer,
    .footer * {
        color: #a0aec0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Plataforma de Riesgo Actuarial</h1>
        <p>Análisis con K-means & SVM · IS-701 Inteligencia Artificial · UNAH Campus Comayagua</p>
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
    "Bajo": "Perfil con menor costo médico promedio y factores de riesgo controlados.",
    "Medio": "Perfil con costo y factores de riesgo intermedios. Se recomienda seguimiento.",
    "Alto": "Perfil con mayor costo médico promedio y factores de riesgo significativos. Requiere atención prioritaria.",
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
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Datos del Cliente")
    st.caption("Complete la información para estimar el nivel de riesgo.")
    
    age = st.slider("🎂 Edad", min_value=18, max_value=100, value=45)
    sex = st.selectbox("⚧ Sexo", ["male", "female"], 
                      format_func=lambda x: "Masculino" if x == "male" else "Femenino")
    bmi = st.slider("⚖️ BMI", min_value=10.0, max_value=60.0, value=31.2, step=0.1)
    children = st.slider("👶 Hijos", min_value=0, max_value=10, value=2)
    smoker = st.selectbox("🚬 Fumador", ["yes", "no"], 
                         format_func=lambda x: "Sí" if x == "yes" else "No")
    region = st.selectbox("🗺️ Región", ["southeast", "southwest", "northeast", "northwest"],
                        format_func=lambda x: {
                            "southeast": "Sureste", "southwest": "Suroeste",
                            "northeast": "Noreste", "northwest": "Noroeste"
                        }[x])
    charges = st.number_input("💵 Cargos médicos ($)", min_value=0.0, value=28000.0, step=500.0)
    
    st.markdown("---")
    evaluar = st.button("🔍 Evaluar Riesgo", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Tabs principales
# ----------------------------------------------------------------------------
tab_eval, tab_clusters, tab_svm, tab_datos = st.tabs(
    ["🧮 Evaluar Cliente", "📊 Segmentación K-means", "🧩 Comparación SVM", "📁 Datos"]
)

# --- Tab 1: Evaluación de cliente -------------------------------------------
with tab_eval:
    if modelo_completo is None:
        st.error("❌ No se pudo cargar el modelo K-means. Verifique que el archivo exista.")
    else:
        if evaluar:
            cliente = pd.DataFrame([{
                "age": age, "sex": str(sex).lower(), "bmi": bmi,
                "children": children, "smoker": str(smoker).lower(),
                "region": str(region).lower(), "charges": charges,
            }])

            cluster = int(modelo_completo.predict(cliente)[0])
            riesgo = mapa_riesgo.get(cluster, "Desconocido")
            explicacion = EXPLICACIONES.get(riesgo, "")

            col1, col2 = st.columns([1, 2])
            with col1:
                css_class = RISK_CSS_CLASS.get(riesgo, "risk-medio")
                emoji = RISK_EMOJI.get(riesgo, "⚪")
                st.markdown(
                    f"""
                    <div class="risk-card {css_class}">
                        <div style="font-size:3rem;">{emoji}</div>
                        <div style="font-size:1.5rem; margin-top:0.5rem; font-weight:700;">Riesgo {riesgo}</div>
                        <div style="font-size:0.95rem; opacity:0.9; margin-top:0.5rem;">Grupo #{cluster}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("### 📋 Resultado del Análisis")
                st.info(explicacion)
                
                st.markdown("**Datos del cliente evaluado:**")
                st.dataframe(cliente, use_container_width=True, hide_index=True)
                
                # Métricas con fondo blanco y texto oscuro
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🎂 Edad", f"{age} años")
                with col_b:
                    st.metric("⚖️ BMI", f"{bmi:.1f}")
                with col_c:
                    st.metric("💵 Cargos", f"${charges:,.2f}")
        else:
            st.info("👈 Complete el formulario y presione **Evaluar Riesgo** para ver los resultados.")

# --- Tab 2: Segmentación K-means ---------------------------------------------
with tab_clusters:
    if df_clusters is None:
        st.warning("📁 No se encontró el archivo de clusters. Verifique la carpeta `outputs/`.")
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

        # KPIs
        total = len(df_clusters)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("👥 Total Clientes", f"{total:,}")
        with k2:
            bajo_count = int(resumen.loc["Bajo", "cantidad_clientes"]) if "Bajo" in resumen.index else 0
            st.metric("🟢 Bajo Riesgo", f"{bajo_count:,}")
        with k3:
            medio_count = int(resumen.loc["Medio", "cantidad_clientes"]) if "Medio" in resumen.index else 0
            st.metric("🟡 Medio Riesgo", f"{medio_count:,}")
        with k4:
            alto_count = int(resumen.loc["Alto", "cantidad_clientes"]) if "Alto" in resumen.index else 0
            st.metric("🔴 Alto Riesgo", f"{alto_count:,}")
        
        st.markdown("---")
        
        cols = st.columns(len(resumen))
        for col, (riesgo, fila) in zip(cols, resumen.iterrows()):
            css_class = RISK_CSS_CLASS.get(riesgo, "risk-medio")
            emoji = RISK_EMOJI.get(riesgo, "⚪")
            with col:
                st.markdown(
                    f"""
                    <div class="risk-card {css_class}">
                        <div style="font-size:2rem;">{emoji}</div>
                        <div style="font-size:1.3rem; font-weight:700; margin-top:0.5rem;">{riesgo}</div>
                        <div style="font-size:0.85rem; margin-top:0.8rem; line-height:1.6;">
                            {int(fila['cantidad_clientes'])} clientes<br>
                            Edad prom: {fila['edad_promedio']:.1f}<br>
                            BMI prom: {fila['bmi_promedio']:.1f}<br>
                            Cargos prom: ${fila['cargos_promedio']:,.0f}<br>
                            Fumadores: {fila['porcentaje_fumadores']:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### 💰 Distribución de Cargos Médicos")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            sns.boxplot(
                data=df_clusters, x="riesgo_actuarial", y="charges",
                order=orden, hue="riesgo_actuarial", 
                palette=["#0b8a5e", "#f0a500", "#e74c3c"], 
                ax=ax, legend=False, linewidth=2
            )
            ax.set_xlabel("Riesgo Actuarial", fontweight='bold', color='#2c3e50')
            ax.set_ylabel("Cargos Médicos ($)", fontweight='bold', color='#2c3e50')
            ax.tick_params(colors='#2c3e50')
            ax.grid(True, alpha=0.3, linestyle='--')
            st.pyplot(fig, use_container_width=True)

        with col_g2:
            st.markdown("#### 🚬 Fumadores por Nivel de Riesgo")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor('#f8f9fa')
            ax2.set_facecolor('#f8f9fa')
            sns.countplot(
                data=df_clusters, x="riesgo_actuarial", hue="smoker",
                order=orden, palette=["#5dade2", "#ec7063"], ax=ax2
            )
            ax2.set_xlabel("Riesgo Actuarial", fontweight='bold', color='#2c3e50')
            ax2.set_ylabel("Cantidad de Clientes", fontweight='bold', color='#2c3e50')
            ax2.tick_params(colors='#2c3e50')
            ax2.legend(title="Fumador", labels=["No", "Sí"])
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
            st.pyplot(fig2, use_container_width=True)

# --- Tab 3: Comparación SVM ---------------------------------------
with tab_svm:
    if df_svm is None:
        st.warning("📁 No se encontró el archivo de resultados SVM. Verifique la carpeta `outputs/`.")
    else:
        best_model = df_svm.loc[df_svm['accuracy'].idxmax()]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🏆 Mejor Kernel", best_model['kernel'].upper())
        with c2:
            st.metric("📈 Accuracy", f"{best_model['accuracy']:.3f}")
        with c3:
            st.metric("🎯 Precisión Macro", f"{best_model['precision_macro']:.3f}")
        
        st.markdown("---")
        
        st.dataframe(
            df_svm.style.background_gradient(cmap="RdYlGn", subset=["accuracy", "precision_macro"])
            .format("{:.3f}", subset=["accuracy", "precision_macro"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### 📈 Comparativa de Kernels")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        fig3.patch.set_facecolor('#f8f9fa')
        ax3.set_facecolor('#f8f9fa')
        
        df_melt = df_svm.melt(id_vars="kernel", value_vars=["accuracy", "precision_macro"],
                               var_name="Métrica", value_name="Valor")
        
        bars = sns.barplot(data=df_melt, x="kernel", y="Valor", hue="Métrica", 
                          palette=["#2a5298", "#3a7bd5"], ax=ax3)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Valor", fontweight='bold', color='#2c3e50')
        ax3.set_xlabel("Kernel", fontweight='bold', color='#2c3e50')
        ax3.tick_params(colors='#2c3e50')
        ax3.legend(title="Métrica")
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        for p in ax3.patches:
            ax3.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
        
        st.pyplot(fig3, use_container_width=True)

        st.info(
            "💡 Los modelos SVM se entrenaron sobre 2 componentes PCA para visualización. "
            "La predicción de riesgo usa el pipeline de K-means."
        )

    if svm_models is not None:
        st.success(f"✅ Kernels cargados: {', '.join([k.upper() for k in svm_models.keys()])}")

# --- Tab 4: Datos --------------------------------------
with tab_datos:
    if df_clusters is None:
        st.warning("📁 No se encontró el archivo de datos.")
    else:
        st.metric("📊 Registros Totales", f"{len(df_clusters):,}")
        st.markdown("---")
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)
        
        st.download_button(
            "⬇️ Descargar CSV",
            data=df_clusters.to_csv(index=False).encode("utf-8"),
            file_name="insurance_con_clusters.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <strong>IS-701 Inteligencia Artificial</strong> · UNAH Campus Comayagua<br>
        <span style="opacity:0.7;">Jorge Abraham Fajardo López · 20231900189</span>
    </div>
    """,
    unsafe_allow_html=True,
)
