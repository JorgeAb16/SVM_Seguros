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
# Estilos personalizados - Interfaz profesional mejorada
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Importación de fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Header principal con efecto glassmorphism */
    .app-header {
        padding: 2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .app-header h1 {
        margin-bottom: 0.3rem;
        font-size: 2.2rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        margin: 0;
        opacity: 0.95;
        font-size: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Tarjetas de riesgo con efectos mejorados */
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .risk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(0,0,0,0.3);
    }
    
    .risk-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
        pointer-events: none;
    }
    
    .risk-bajo { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .risk-medio { 
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    
    .risk-alto { 
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    /* Cajas métricas con diseño moderno */
    .metric-box {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Sidebar mejorada */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] * {
        color: #e0e6ed !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px 12px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid #e1e4e8;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: transparent;
    }
    
    /* DataFrames y tablas */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }
    
    /* Métricas con estilo */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 2rem;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        margin-top: 2rem;
        color: #a0aec0;
    }
    
    /* Mejoras en gráficos */
    .js-plotly-plot, .plot-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }
    
    /* Animación de carga */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header mejorado con ícono animado
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Plataforma de Riesgo Actuarial</h1>
        <p>Análisis inteligente con K-means & SVM · IS-701 Inteligencia Artificial</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">📍 Campus Comayagua · 👨‍💻 Jorge Abraham Fajardo López · 🆔 20231900189</p>
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
    "Bajo": "✅ Cliente agrupado con perfiles de menor costo médico promedio y factores de riesgo controlados.",
    "Medio": "⚠️ Cliente agrupado con perfiles de costo y factores de riesgo intermedios. Se recomienda seguimiento.",
    "Alto": "🚨 Cliente agrupado con perfiles de mayor costo médico promedio y/o factores de riesgo significativos. Requiere atención prioritaria.",
}

RISK_CSS_CLASS = {"Bajo": "risk-bajo", "Medio": "risk-medio", "Alto": "risk-alto"}
RISK_EMOJI = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
RISK_ICON = {"Bajo": "✅", "Medio": "⚠️", "Alto": "🚨"}

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
# Sidebar - formulario de cliente nuevo con diseño mejorado
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Evaluación de Cliente")
    st.markdown("---")
    st.caption("Complete la información para estimar el nivel de riesgo actuarial.")
    
    with st.container():
        st.markdown("### 👤 Datos Personales")
        age = st.slider("🎂 Edad", min_value=18, max_value=100, value=45, help="Edad del asegurado")
        sex = st.selectbox("⚧ Sexo", ["male", "female"], 
                          format_func=lambda x: "👨 Masculino" if x == "male" else "👩 Femenino")
        
    with st.container():
        st.markdown("### 🏥 Indicadores de Salud")
        bmi = st.slider("⚖️ BMI (Índice de Masa Corporal)", min_value=10.0, max_value=60.0, 
                       value=31.2, step=0.1, help="Body Mass Index")
        smoker = st.selectbox("🚬 ¿Fumador?", ["yes", "no"], 
                             format_func=lambda x: "🚬 Sí" if x == "yes" else "🚭 No")
    
    with st.container():
        st.markdown("### 👨‍👩‍👧‍👦 Información Familiar")
        children = st.slider("👶 Número de hijos", min_value=0, max_value=10, value=2)
    
    with st.container():
        st.markdown("### 🌍 Ubicación y Costos")
        region = st.selectbox("🗺️ Región", ["southeast", "southwest", "northeast", "northwest"],
                            format_func=lambda x: {
                                "southeast": "🌅 Sureste",
                                "southwest": "🌄 Suroeste",
                                "northeast": "🌆 Noreste",
                                "northwest": "🏔️ Noroeste"
                            }[x])
        charges = st.number_input("💵 Cargos médicos estimados ($)", min_value=0.0, 
                                 value=28000.0, step=500.0, format="%.2f")
    
    st.markdown("---")
    evaluar = st.button("🔍 Analizar Riesgo", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("💡 Los resultados se basan en patrones identificados por el modelo K-means.")

# ----------------------------------------------------------------------------
# Tabs principales con iconos mejorados
# ----------------------------------------------------------------------------
tab_eval, tab_clusters, tab_svm, tab_datos = st.tabs(
    ["🧮 Evaluación Individual", "📊 Segmentación K-means", "🧩 Rendimiento SVM", "📁 Datos Maestros"]
)

# --- Tab 1: Evaluación de cliente -------------------------------------------
with tab_eval:
    st.markdown("### 📋 Análisis de Riesgo Individual")
    
    if modelo_completo is None:
        st.error(
            "❌ No se pudo cargar `models/kmeans_riesgo_actuarial.pkl`. "
            "Verifique que el archivo exista en la ruta correcta."
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
            icono_riesgo = RISK_ICON.get(riesgo, "⚪")

            # Resultado principal con animación
            col1, col2 = st.columns([1, 2])
            with col1:
                css_class = RISK_CSS_CLASS.get(riesgo, "risk-medio")
                emoji = RISK_EMOJI.get(riesgo, "⚪")
                st.markdown(
                    f"""
                    <div class="risk-card {css_class}">
                        <div style="font-size:3rem; margin-bottom: 0.5rem;">{emoji}</div>
                        <div style="font-size:1.8rem; font-weight: 700; margin-top:0.3rem;">{icono_riesgo} Riesgo {riesgo}</div>
                        <div style="font-size:1rem; opacity:0.95; margin-top:0.5rem;">Grupo #{cluster}</div>
                        <div style="font-size:0.85rem; opacity:0.9; margin-top:0.8rem; line-height: 1.4;">
                            {explicacion[:100]}...
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown("#### 🔍 Análisis Detallado")
                st.info(explicacion)
                
                st.markdown("#### 📊 Perfil del Cliente Evaluado")
                st.dataframe(cliente, use_container_width=True, hide_index=True)
                
                # Métricas adicionales en cards
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("🎂 Edad", f"{age} años")
                with met_col2:
                    st.metric("⚖️ BMI", f"{bmi:.1f}")
                with met_col3:
                    st.metric("💵 Cargos", f"${charges:,.2f}")
        else:
            st.info("👈 Complete el formulario en la barra lateral y presione **Analizar Riesgo** para ver los resultados.")

    if metadata is not None:
        with st.expander("🔬 Metadatos y Configuración del Modelo"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Clusters (K)", metadata.get("n_clusters", "—"))
            c2.metric("📈 Silhouette Score", f"{metadata.get('silhouette_score', 0):.3f}")
            c3.metric("🧩 Kernels SVM", len(metadata.get("svm_kernels", [])))
            c4.metric("📊 Features", len(metadata.get("features", [])))
            
            st.markdown("#### 📋 Configuración Completa")
            st.json(metadata)

# --- Tab 2: Segmentación K-means ---------------------------------------------
with tab_clusters:
    st.markdown("### 📊 Perfil de Segmentación por Clusters")
    
    if df_clusters is None:
        st.warning(
            "📁 No se encontró `outputs/insurance_con_clusters.csv`. "
            "Copie ese archivo generado por el notebook para visualizar esta sección."
        )
    else:
        # KPIs principales
        total_clientes = len(df_clusters)
        riesgo_counts = df_clusters["riesgo_actuarial"].value_counts()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("👥 Total Clientes", f"{total_clientes:,}")
        with kpi2:
            st.metric("🟢 Bajo Riesgo", f"{riesgo_counts.get('Bajo', 0):,}")
        with kpi3:
            st.metric("🟡 Riesgo Medio", f"{riesgo_counts.get('Medio', 0):,}")
        with kpi4:
            st.metric("🔴 Alto Riesgo", f"{riesgo_counts.get('Alto', 0):,}")
        
        st.markdown("---")
        st.markdown("#### 🎯 Resumen por Nivel de Riesgo")
        
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
                        <div style="font-size:2rem; margin-bottom:0.5rem;">{emoji}</div>
                        <div style="font-size:1.3rem; font-weight: 700;">{riesgo}</div>
                        <div style="font-size:0.9rem; margin-top:1rem; line-height: 1.6;">
                            👥 {int(fila['cantidad_clientes'])} clientes<br>
                            🎂 {fila['edad_promedio']:.1f} años prom.<br>
                            ⚖️ BMI {fila['bmi_promedio']:.1f}<br>
                            💵 ${fila['cargos_promedio']:,.0f} prom.<br>
                            🚬 {fila['porcentaje_fumadores']:.1f}% fumadores
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        
        # Gráficos mejorados
        graf_col1, graf_col2 = st.columns(2)
        
        with graf_col1:
            st.markdown("#### 💰 Distribución de Cargos Médicos")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            
            sns.boxplot(
                data=df_clusters, x="riesgo_actuarial", y="charges",
                order=orden, hue="riesgo_actuarial", palette=["#11998e", "#f2994a", "#eb3349"], 
                ax=ax, legend=False, linewidth=2
            )
            ax.set_xlabel("Nivel de Riesgo Actuarial", fontweight='bold')
            ax.set_ylabel("Cargos Médicos ($)", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            st.pyplot(fig, use_container_width=True)

        with graf_col2:
            st.markdown("#### 🚬 Impacto del Tabaquismo")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor('#f8f9fa')
            ax2.set_facecolor('#f8f9fa')
            
            sns.countplot(
                data=df_clusters, x="riesgo_actuarial", hue="smoker",
                order=orden, palette=["#74b9ff", "#ff7675"], ax=ax2
            )
            ax2.set_xlabel("Nivel de Riesgo Actuarial", fontweight='bold')
            ax2.set_ylabel("Cantidad de Clientes", fontweight='bold')
            ax2.legend(title="Fumador", labels=["No", "Sí"])
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
            st.pyplot(fig2, use_container_width=True)

# --- Tab 3: Comparación de kernels SVM ---------------------------------------
with tab_svm:
    st.markdown("### 🧩 Rendimiento de Modelos SVM")
    st.caption("Comparativa de kernels entrenados sobre componentes principales (PCA)")

    if df_svm is None:
        st.warning(
            "📁 No se encontró `outputs/svm_resultados_kernels.csv`. "
            "Copie ese archivo generado por el notebook para ver esta sección."
        )
    else:
        # Métricas destacadas
        best_model = df_svm.loc[df_svm['accuracy'].idxmax()]
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("🏆 Mejor Kernel", best_model['kernel'].upper())
        with metric_col2:
            st.metric("📈 Mejor Accuracy", f"{best_model['accuracy']:.3f}")
        with metric_col3:
            st.metric("🎯 Mejor Precisión", f"{best_model['precision_macro']:.3f}")
        
        st.markdown("---")
        st.markdown("#### 📊 Resultados por Kernel")
        
        st.dataframe(
            df_svm.style.background_gradient(cmap="RdYlGn", subset=["accuracy", "precision_macro"])
            .format("{:.3f}", subset=["accuracy", "precision_macro"]),
            use_container_width=True, hide_index=True,
        )

        # Gráfico comparativo mejorado
        st.markdown("#### 📈 Comparativa Visual de Kernels")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        fig3.patch.set_facecolor('#f8f9fa')
        ax3.set_facecolor('#f8f9fa')
        
        df_melt = df_svm.melt(id_vars="kernel", value_vars=["accuracy", "precision_macro"],
                               var_name="Métrica", value_name="Valor")
        
        bars = sns.barplot(data=df_melt, x="kernel", y="Valor", hue="Métrica", 
                          palette=["#667eea", "#764ba2"], ax=ax3)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Valor de Métrica", fontweight='bold')
        ax3.set_xlabel("Kernel SVM", fontweight='bold')
        ax3.legend(title="Métrica de Evaluación")
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Agregar valores en las barras
        for p in ax3.patches:
            ax3.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        st.pyplot(fig3, use_container_width=True)

        st.info(
            "💡 **Nota técnica:** Estos modelos SVM fueron entrenados sobre 2 componentes principales (PCA) "
            "únicamente con fines de visualización de fronteras de decisión. "
            "La predicción de riesgo para clientes nuevos utiliza el pipeline de K-means."
        )

    if svm_models is not None:
        st.success(f"✅ Kernels disponibles: {', '.join([k.upper() for k in svm_models.keys()])}")

# --- Tab 4: Datos crudos ------------------------------------------------------
with tab_datos:
    st.markdown("### 📁 Dataset Completo con Clusters")
    
    if df_clusters is None:
        st.warning("📁 No se encontró `outputs/insurance_con_clusters.csv`.")
    else:
        # Estadísticas rápidas
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("📊 Registros Totales", f"{len(df_clusters):,}")
        with stat_col2:
            st.metric("📋 Columnas", len(df_clusters.columns))
        with stat_col3:
            st.metric("💾 Tamaño Estimado", f"{df_clusters.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown("---")
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)
        
        col_down1, col_down2 = st.columns([1, 3])
        with col_down1:
            st.download_button(
                "⬇️ Descargar CSV Completo",
                data=df_clusters.to_csv(index=False).encode("utf-8"),
                file_name="insurance_con_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_down2:
            st.caption("📥 Descargue el dataset completo con las asignaciones de clusters para análisis externos.")

# Footer mejorado
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            🤖 IS-701 Inteligencia Artificial
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            Universidad Nacional Autónoma de Honduras · Campus Comayagua
        </div>
        <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
            Desarrollado con Streamlit · Modelos: K-means & SVM · © 2024
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
