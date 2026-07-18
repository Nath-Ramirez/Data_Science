"""
Aplicación interactiva de análisis de datos sintéticos de COVID-19
====================================================================
Genera un dataset sintético (10.000 registros / 8 columnas con tipos
de datos variados), calcula estadística cuantitativa y cualitativa,
y permite explorar los datos con gráficas dinámicas de Plotly
totalmente personalizables por el usuario.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Análisis Interactivo COVID-19 (Datos Sintéticos)",
    page_icon="🦠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1) GENERACIÓN DE DATOS SINTÉTICOS
# ---------------------------------------------------------------------------
DEPARTAMENTOS = [
    "Antioquia", "Cundinamarca", "Valle del Cauca", "Atlántico",
    "Santander", "Bolívar", "Nariño", "Córdoba", "Risaralda", "Tolima",
]
GENEROS = ["Masculino", "Femenino", "Otro"]
SEVERIDADES = ["Leve", "Moderado", "Grave", "Crítico"]
SEVERIDAD_ORDEN = {"Leve": 0, "Moderado": 1, "Grave": 2, "Crítico": 3}


@st.cache_data(show_spinner=True)
def generar_datos_sinteticos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Simula un dataset de pacientes COVID-19 con 8 columnas de tipos distintos:
    int, float, category (x3), datetime, bool y category ordinal (severidad).
    """
    rng = np.random.default_rng(semilla)

    # 1. id_paciente -> int
    id_paciente = np.arange(1, n_registros + 1)

    # 2. edad -> int (distribución realista, más peso en adultos/mayores)
    edad = rng.gamma(shape=4.5, scale=11, size=n_registros).astype(int)
    edad = np.clip(edad, 0, 99)

    # 3. genero -> category
    genero = rng.choice(GENEROS, size=n_registros, p=[0.48, 0.49, 0.03])

    # 4. departamento -> category
    departamento = rng.choice(DEPARTAMENTOS, size=n_registros)

    # 5. fecha_diagnostico -> datetime (últimos ~2 años, con estacionalidad simple)
    fecha_inicio = datetime(2024, 1, 1)
    dias_rango = 730
    offsets = rng.integers(0, dias_rango, size=n_registros)
    fecha_diagnostico = pd.to_datetime(fecha_inicio) + pd.to_timedelta(offsets, unit="D")

    # 6. severidad -> category ordinal, correlacionada levemente con la edad
    prob_base = np.vstack([
        np.clip(0.55 - edad / 400, 0.15, 0.7),   # Leve
        np.full(n_registros, 0.30),              # Moderado
        np.clip(0.10 + edad / 500, 0.05, 0.30),  # Grave
        np.clip(0.05 + edad / 700, 0.02, 0.20),  # Crítico
    ]).T
    prob_base = prob_base / prob_base.sum(axis=1, keepdims=True)
    severidad = np.array([
        rng.choice(SEVERIDADES, p=prob_base[i]) for i in range(n_registros)
    ])

    # 7. dias_hospitalizacion -> float, 0 para casos leves, mayor para severos
    base_dias = np.select(
        [severidad == "Leve", severidad == "Moderado",
         severidad == "Grave", severidad == "Crítico"],
        [rng.exponential(0.5, n_registros),
         rng.exponential(3, n_registros),
         rng.exponential(8, n_registros) + 3,
         rng.exponential(14, n_registros) + 7],
    )
    dias_hospitalizacion = np.round(np.clip(base_dias, 0, 60), 1)

    # 8. vacunado -> bool
    prob_vacunado = np.clip(0.9 - edad / 500, 0.4, 0.95)
    vacunado = rng.random(n_registros) < prob_vacunado

    df = pd.DataFrame({
        "id_paciente": id_paciente,
        "edad": edad,
        "genero": pd.Categorical(genero, categories=GENEROS),
        "departamento": pd.Categorical(departamento, categories=DEPARTAMENTOS),
        "fecha_diagnostico": fecha_diagnostico,
        "severidad": pd.Categorical(severidad, categories=SEVERIDADES, ordered=True),
        "dias_hospitalizacion": dias_hospitalizacion,
        "vacunado": vacunado,
    })
    return df


# ---------------------------------------------------------------------------
# BARRA LATERAL: Controles de simulación y filtros
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración de simulación")

n_registros = st.sidebar.slider(
    "Número de registros a simular", min_value=1000, max_value=20000,
    value=10000, step=500,
)
semilla = st.sidebar.number_input("Semilla aleatoria (reproducibilidad)", value=42, step=1)

df_full = generar_datos_sinteticos(n_registros, semilla)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filtros")

deptos_sel = st.sidebar.multiselect(
    "Departamento", options=DEPARTAMENTOS, default=DEPARTAMENTOS
)
generos_sel = st.sidebar.multiselect("Género", options=GENEROS, default=GENEROS)
severidad_sel = st.sidebar.multiselect(
    "Severidad", options=SEVERIDADES, default=SEVERIDADES
)
edad_min, edad_max = st.sidebar.slider(
    "Rango de edad", 0, 99, (0, 99)
)
fecha_min = df_full["fecha_diagnostico"].min()
fecha_max = df_full["fecha_diagnostico"].max()
rango_fechas = st.sidebar.date_input(
    "Rango de fecha de diagnóstico",
    value=(fecha_min.date(), fecha_max.date()),
    min_value=fecha_min.date(), max_value=fecha_max.date(),
)
solo_vacunados = st.sidebar.selectbox(
    "Estado de vacunación", ["Todos", "Solo vacunados", "Solo no vacunados"]
)

# Aplicar filtros
df = df_full[
    df_full["departamento"].isin(deptos_sel)
    & df_full["genero"].isin(generos_sel)
    & df_full["severidad"].isin(severidad_sel)
    & df_full["edad"].between(edad_min, edad_max)
].copy()

if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    f_ini, f_fin = rango_fechas
    df = df[
        (df["fecha_diagnostico"] >= pd.Timestamp(f_ini))
        & (df["fecha_diagnostico"] <= pd.Timestamp(f_fin))
    ]

if solo_vacunados == "Solo vacunados":
    df = df[df["vacunado"]]
elif solo_vacunados == "Solo no vacunados":
    df = df[~df["vacunado"]]

st.sidebar.markdown(f"**Registros tras filtros:** {len(df):,}")

# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.title("🦠 Panel Interactivo de Análisis de Datos Sintéticos COVID-19")
st.caption(
    "Todos los datos son **completamente simulados** dentro de la plataforma "
    "con fines demostrativos y educativos. No representan pacientes reales."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total registros (filtrados)", f"{len(df):,}")
col_b.metric("Edad promedio", f"{df['edad'].mean():.1f} años" if len(df) else "N/A")
col_c.metric(
    "Días hosp. promedio",
    f"{df['dias_hospitalizacion'].mean():.1f}" if len(df) else "N/A",
)
tasa_vac = df["vacunado"].mean() * 100 if len(df) else 0
col_d.metric("Tasa de vacunación", f"{tasa_vac:.1f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# TABS PRINCIPALES
# ---------------------------------------------------------------------------
tab_datos, tab_cuant, tab_cual, tab_graf = st.tabs(
    ["📄 Datos", "🔢 Estadística Cuantitativa", "🏷️ Estadística Cualitativa", "📊 Análisis Gráfico Dinámico"]
)

# ------------------------- TAB 1: DATOS -----------------------------------
with tab_datos:
    st.subheader("Vista previa del dataset simulado")
    st.write(
        "Esquema: `id_paciente` (int) · `edad` (int) · `genero` (categórica) · "
        "`departamento` (categórica) · `fecha_diagnostico` (datetime) · "
        "`severidad` (categórica ordinal) · `dias_hospitalizacion` (float) · "
        "`vacunado` (bool)"
    )
    n_filas = st.slider("Filas a mostrar", 5, 200, 20)
    st.dataframe(df.head(n_filas), use_container_width=True)

    st.markdown("**Tipos de datos por columna**")
    tipos_df = pd.DataFrame({
        "columna": df_full.columns,
        "tipo_dato": [str(t) for t in df_full.dtypes],
    })
    st.dataframe(tipos_df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)", data=csv,
        file_name="covid_datos_sinteticos.csv", mime="text/csv",
    )

# ------------------------- TAB 2: CUANTITATIVA -----------------------------
with tab_cuant:
    st.subheader("Estadística descriptiva - Variables cuantitativas")
    numericas = ["edad", "dias_hospitalizacion"]

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        resumen = df[numericas].describe().T
        resumen["mediana"] = df[numericas].median()
        resumen["varianza"] = df[numericas].var()
        resumen["asimetria"] = df[numericas].skew()
        resumen["curtosis"] = df[numericas].kurt()
        st.dataframe(resumen.round(2), use_container_width=True)

        st.markdown("#### Matriz de correlación")
        corr = df[numericas + ["vacunado"]].assign(vacunado=df["vacunado"].astype(int)).corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Distribución con umbral personalizable")
        var_num = st.selectbox("Variable numérica", numericas, key="var_num_cuant")
        umbral = st.slider(
            f"Umbral para '{var_num}'",
            float(df[var_num].min()), float(df[var_num].max()),
            float(df[var_num].median()),
        )
        pct_sobre_umbral = (df[var_num] > umbral).mean() * 100
        st.info(f"**{pct_sobre_umbral:.1f}%** de los registros superan el umbral de {umbral:.1f}")

        fig_hist = px.histogram(df, x=var_num, nbins=40, marginal="box",
                                 title=f"Distribución de {var_num}")
        fig_hist.add_vline(x=umbral, line_dash="dash", line_color="red",
                            annotation_text="Umbral")
        st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------- TAB 3: CUALITATIVA ------------------------------
with tab_cual:
    st.subheader("Estadística descriptiva - Variables cualitativas")
    categoricas = ["genero", "departamento", "severidad", "vacunado"]

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        var_cat = st.selectbox("Variable categórica", categoricas, key="var_cat")
        tabla_frec = (
            df[var_cat].value_counts(dropna=False)
            .rename("frecuencia").to_frame()
        )
        tabla_frec["porcentaje (%)"] = (
            tabla_frec["frecuencia"] / tabla_frec["frecuencia"].sum() * 100
        ).round(2)
        st.dataframe(tabla_frec, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                tabla_frec.reset_index(), x=var_cat, y="frecuencia",
                color=var_cat, title=f"Frecuencia de {var_cat}",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(
                tabla_frec.reset_index(), names=var_cat, values="frecuencia",
                title=f"Proporción de {var_cat}", hole=0.35,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Tabla cruzada (contingencia)")
        var_cat2 = st.selectbox(
            "Cruzar con otra variable categórica", categoricas,
            index=1 if var_cat != categoricas[1] else 2, key="var_cat2",
        )
        if var_cat2 != var_cat:
            tabla_cruzada = pd.crosstab(df[var_cat], df[var_cat2])
            st.dataframe(tabla_cruzada, use_container_width=True)
            fig_stack = px.bar(
                tabla_cruzada.reset_index(), x=var_cat, y=tabla_cruzada.columns.tolist(),
                title=f"{var_cat} vs {var_cat2}", barmode="stack",
            )
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.caption("Selecciona una variable distinta para la tabla cruzada.")

# ------------------------- TAB 4: GRÁFICO DINÁMICO --------------------------
with tab_graf:
    st.subheader("Explorador gráfico dinámico")

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        todas_columnas = ["edad", "dias_hospitalizacion", "genero", "departamento",
                           "severidad", "vacunado", "fecha_diagnostico"]

        c1, c2, c3 = st.columns(3)
        with c1:
            tipo_grafico = st.selectbox(
                "Tipo de gráfica",
                ["Histograma", "Dispersión (Scatter)", "Caja (Box)", "Barras",
                 "Serie temporal", "Violín"],
            )
        with c2:
            var_x = st.selectbox("Variable eje X", todas_columnas, index=0)
        with c3:
            var_color = st.selectbox(
                "Colorear / agrupar por", ["(ninguna)"] + todas_columnas, index=0
            )
        color_arg = None if var_color == "(ninguna)" else var_color

        var_y = None
        if tipo_grafico in ("Dispersión (Scatter)", "Caja (Box)", "Violín", "Barras"):
            opciones_y = ["(ninguna)"] + todas_columnas
            var_y = st.selectbox("Variable eje Y", opciones_y, index=0)
            var_y = None if var_y == "(ninguna)" else var_y

        mostrar_umbral = st.checkbox("Agregar línea de umbral personalizada", value=False)
        umbral_val = None
        if mostrar_umbral:
            columna_umbral = var_y if var_y in ("edad", "dias_hospitalizacion") else (
                var_x if var_x in ("edad", "dias_hospitalizacion") else None
            )
            if columna_umbral:
                umbral_val = st.slider(
                    f"Valor del umbral ({columna_umbral})",
                    float(df[columna_umbral].min()), float(df[columna_umbral].max()),
                    float(df[columna_umbral].median()),
                )
            else:
                st.caption("El umbral solo aplica a variables numéricas (edad / días hospitalización).")

        fig = None
        try:
            if tipo_grafico == "Histograma":
                bins = st.slider("Número de bins", 5, 100, 30)
                fig = px.histogram(df, x=var_x, color=color_arg, nbins=bins,
                                    title=f"Histograma de {var_x}", barmode="overlay",
                                    opacity=0.75)
            elif tipo_grafico == "Dispersión (Scatter)":
                if var_y:
                    fig = px.scatter(df, x=var_x, y=var_y, color=color_arg,
                                      opacity=0.6, title=f"{var_x} vs {var_y}")
                else:
                    st.warning("Selecciona una variable Y para el scatter.")
            elif tipo_grafico == "Caja (Box)":
                fig = px.box(df, x=var_x, y=var_y, color=color_arg,
                             title=f"Boxplot de {var_y or var_x}")
            elif tipo_grafico == "Violín":
                fig = px.violin(df, x=var_x, y=var_y, color=color_arg, box=True,
                                 title=f"Violín de {var_y or var_x}")
            elif tipo_grafico == "Barras":
                if var_y and pd.api.types.is_numeric_dtype(df[var_y]):
                    agg_df = df.groupby(var_x, observed=True)[var_y].mean().reset_index()
                    fig = px.bar(agg_df, x=var_x, y=var_y, color=color_arg if color_arg in agg_df.columns else None,
                                 title=f"Promedio de {var_y} por {var_x}")
                else:
                    conteo = df[var_x].value_counts().reset_index()
                    conteo.columns = [var_x, "conteo"]
                    fig = px.bar(conteo, x=var_x, y="conteo", title=f"Conteo por {var_x}")
            elif tipo_grafico == "Serie temporal":
                freq = st.selectbox("Agregación temporal", ["D", "W", "M"], index=2,
                                     format_func=lambda f: {"D": "Diaria", "W": "Semanal", "M": "Mensual"}[f])
                serie = (
                    df.set_index("fecha_diagnostico")
                    .resample(freq).size().reset_index(name="casos")
                )
                fig = px.line(serie, x="fecha_diagnostico", y="casos", markers=True,
                              title="Evolución de casos en el tiempo")

            if fig is not None:
                if umbral_val is not None:
                    fig.add_hline(y=umbral_val, line_dash="dash", line_color="red",
                                  annotation_text="Umbral") if tipo_grafico != "Histograma" else \
                    fig.add_vline(x=umbral_val, line_dash="dash", line_color="red",
                                  annotation_text="Umbral")
                fig.update_layout(height=550)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No fue posible generar la gráfica con esta combinación de variables: {e}")

st.markdown("---")
st.caption(
    "Aplicación construida con Streamlit + Plotly · Datos 100% sintéticos generados "
    "en tiempo real dentro de la plataforma."
)
