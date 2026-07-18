"""
Dashboard interactivo de análisis de datos sintéticos de Fórmula 1
====================================================================
Proyecto académico - Ciencia de Datos - EAFIT 2026
Autora: Nathaly Ramírez | Julio de 2026

Genera un dataset sintético de resultados de carreras de F1 (pilotos,
equipos, circuitos, clima, incidentes/accidentes, puntos, etc.),
calcula estadística cuantitativa y cualitativa, construye al menos
una serie de tiempo, y permite exploración gráfica dinámica con
Plotly totalmente interactiva. El acceso al panel está protegido
con una clave de operación.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard F1 - Ciencia de Datos EAFIT",
    page_icon="🏎️",
    layout="wide",
)

CLAVE_ACCESO = "4650"

# ---------------------------------------------------------------------------
# PANEL IZQUIERDO PERSONALIZADO (siempre visible)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h2 style="margin-bottom:0;">🏎️ EAFIT 2026</h2>
            <p style="font-size:16px; margin:2px 0;">Ciencia de Datos</p>
            <p style="font-size:15px; margin:2px 0;"><b>Nathaly Ramírez</b></p>
            <p style="font-size:13px; margin:2px 0; color:gray;">Julio de 2026</p>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# CONTROL DE ACCESO POR CLAVE
# ---------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔒 Acceso al Dashboard de Fórmula 1")
    st.write("Ingresa la clave de operación para continuar.")
    clave_ingresada = st.text_input("Clave de acceso", type="password")
    if st.button("Ingresar"):
        if clave_ingresada == CLAVE_ACCESO:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Clave incorrecta. Intenta nuevamente.")
    st.stop()

# ---------------------------------------------------------------------------
# 1) GENERACIÓN DE DATOS SINTÉTICOS DE FÓRMULA 1
# ---------------------------------------------------------------------------
CIRCUITOS = {
    "Monza": {"pais": "Italia", "tipo": "Alta velocidad", "riesgo_base": 0.10},
    "Silverstone": {"pais": "Reino Unido", "tipo": "Mixto", "riesgo_base": 0.10},
    "Spa-Francorchamps": {"pais": "Bélgica", "tipo": "Mixto", "riesgo_base": 0.13},
    "Interlagos": {"pais": "Brasil", "tipo": "Mixto", "riesgo_base": 0.12},
    "Suzuka": {"pais": "Japón", "tipo": "Técnico", "riesgo_base": 0.11},
    "Circuito de Mónaco": {"pais": "Mónaco", "tipo": "Calle", "riesgo_base": 0.22},
    "Circuit of the Americas": {"pais": "Estados Unidos", "tipo": "Mixto", "riesgo_base": 0.10},
    "Bahrain International Circuit": {"pais": "Baréin", "tipo": "Técnico", "riesgo_base": 0.09},
    "Albert Park": {"pais": "Australia", "tipo": "Semi-calle", "riesgo_base": 0.15},
    "Red Bull Ring": {"pais": "Austria", "tipo": "Alta velocidad", "riesgo_base": 0.09},
    "Hungaroring": {"pais": "Hungría", "tipo": "Técnico", "riesgo_base": 0.10},
    "Circuit Zandvoort": {"pais": "Países Bajos", "tipo": "Técnico", "riesgo_base": 0.12},
    "Marina Bay": {"pais": "Singapur", "tipo": "Calle", "riesgo_base": 0.24},
    "Jeddah Corniche Circuit": {"pais": "Arabia Saudita", "tipo": "Calle", "riesgo_base": 0.21},
    "Losail International Circuit": {"pais": "Catar", "tipo": "Alta velocidad", "riesgo_base": 0.09},
    "Circuit Gilles Villeneuve": {"pais": "Canadá", "tipo": "Semi-calle", "riesgo_base": 0.16},
}

EQUIPOS = [
    "Apex Racing", "Velocity GP", "Thunder Motorsport", "Nova Racing Team",
    "Falcon Grand Prix", "Titan Racing", "Zenith F1 Team", "Meridian Motorsport",
    "Vortex Racing", "Horizon GP",
]

PILOTOS = [
    "Mateo Correa", "Lucas Ferreira", "Adrián Solís", "Kenji Watanabe",
    "Erik Lindqvist", "Marco Bellini", "Diego Ibáñez", "Noah Fischer",
    "Rafael Tavares", "Tomás Herrera", "Liam O'Connell", "Hugo Duarte",
    "Sebastian Kruger", "Andrés Molina", "Felix Bergman", "Ivo Santos",
    "Nikolai Petrov", "Bruno Castellani", "Owen Mitchell", "Julian Vidal",
    "Marcus Aldridge", "Théo Lambert",
]

CLIMAS = ["Seco", "Nublado", "Lluvia ligera", "Lluvia intensa"]
CLIMA_RIESGO = {"Seco": 0.0, "Nublado": 0.02, "Lluvia ligera": 0.10, "Lluvia intensa": 0.22}

PUNTOS_POSICION = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


@st.cache_data(show_spinner=True)
def generar_datos_f1(n_temporadas: int, rondas_por_temporada: int, semilla: int) -> pd.DataFrame:
    """Simula resultados de carreras de F1: relación piloto-circuito,
    incidentes/accidentes, clima, puntos y evolución temporal."""
    rng = np.random.default_rng(semilla)

    # Atributos intrínsecos fijos por piloto (habilidad y "agresividad")
    n_pilotos = len(PILOTOS)
    habilidad = rng.normal(loc=0, scale=1, size=n_pilotos)
    habilidad = (habilidad - habilidad.min()) / (habilidad.max() - habilidad.min())  # 0-1
    agresividad = rng.uniform(0.05, 0.35, size=n_pilotos)  # tendencia a incidentes
    equipo_piloto = rng.choice(EQUIPOS, size=n_pilotos)

    circuitos_nombres = list(CIRCUITOS.keys())
    registros = []
    id_carrera = 1
    anio_inicio = 2026 - n_temporadas

    for t in range(n_temporadas):
        temporada = anio_inicio + t
        circuitos_temporada = rng.choice(
            circuitos_nombres, size=rondas_por_temporada, replace=True
        )
        for ronda in range(1, rondas_por_temporada + 1):
            circuito = circuitos_temporada[ronda - 1]
            info_circuito = CIRCUITOS[circuito]
            clima = rng.choice(CLIMAS, p=[0.55, 0.25, 0.15, 0.05])
            fecha = pd.Timestamp(year=temporada, month=1, day=1) + pd.Timedelta(
                days=int(ronda * 14 + rng.integers(0, 5))
            )

            # posiciones de salida en base a habilidad + ruido
            ruido_clasif = rng.normal(0, 0.15, size=n_pilotos)
            score_clasif = habilidad + ruido_clasif
            orden_salida = np.argsort(-score_clasif)  # índices de pilotos, mejor primero

            resultados_ronda = []
            for pos_salida, idx_piloto in enumerate(orden_salida, start=1):
                piloto = PILOTOS[idx_piloto]
                equipo = equipo_piloto[idx_piloto]

                prob_incidente = np.clip(
                    info_circuito["riesgo_base"] + CLIMA_RIESGO[clima] + agresividad[idx_piloto] * 0.3,
                    0.01, 0.8,
                )
                incidentes = rng.binomial(2, prob_incidente)  # 0,1 o 2 incidentes en la carrera
                dnf = rng.random() < (prob_incidente * 0.35)  # abandono por accidente grave

                ruido_carrera = rng.normal(0, 0.2)
                score_final = habilidad[idx_piloto] + ruido_carrera - incidentes * 0.08
                paradas_boxes = int(rng.integers(1, 4)) + (1 if clima != "Seco" else 0)
                tiempo_vuelta = round(rng.normal(90 - habilidad[idx_piloto] * 8, 1.8), 3)
                if clima == "Lluvia intensa":
                    tiempo_vuelta += rng.uniform(4, 9)
                elif clima == "Lluvia ligera":
                    tiempo_vuelta += rng.uniform(1, 3)

                resultados_ronda.append({
                    "id_carrera": id_carrera,
                    "temporada": temporada,
                    "ronda": ronda,
                    "circuito": circuito,
                    "tipo_circuito": info_circuito["tipo"],
                    "pais": info_circuito["pais"],
                    "fecha": fecha,
                    "clima": clima,
                    "piloto": piloto,
                    "equipo": equipo,
                    "posicion_salida": pos_salida,
                    "score_final": score_final,
                    "incidentes": int(incidentes),
                    "dnf": bool(dnf),
                    "paradas_boxes": paradas_boxes,
                    "tiempo_vuelta_promedio_seg": round(tiempo_vuelta, 2),
                })

            # Asignar posición de llegada según score_final (los DNF van al final)
            df_ronda = pd.DataFrame(resultados_ronda)
            df_ronda["orden_llegada"] = df_ronda["dnf"].astype(int) * 1000 - df_ronda["score_final"]
            df_ronda = df_ronda.sort_values("orden_llegada").reset_index(drop=True)
            df_ronda["posicion_llegada"] = np.where(
                df_ronda["dnf"], np.nan, np.arange(1, len(df_ronda) + 1)
            )
            df_ronda["puntos"] = df_ronda["posicion_llegada"].map(PUNTOS_POSICION).fillna(0)
            df_ronda = df_ronda.drop(columns=["orden_llegada", "score_final"])

            registros.append(df_ronda)
            id_carrera += 1

    df_final = pd.concat(registros, ignore_index=True)
    df_final["accidente"] = df_final["incidentes"] > 0
    return df_final


# ---------------------------------------------------------------------------
# BARRA LATERAL: Controles de simulación y filtros
# ---------------------------------------------------------------------------
st.sidebar.subheader("⚙️ Configuración de simulación")
n_temporadas = st.sidebar.slider("Número de temporadas simuladas", 3, 16, 12)
rondas_por_temporada = st.sidebar.slider("Carreras por temporada", 10, 24, 20)
semilla = st.sidebar.number_input("Semilla aleatoria", value=7, step=1)

df_full = generar_datos_f1(n_temporadas, rondas_por_temporada, semilla)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filtros")

temporadas_sel = st.sidebar.multiselect(
    "Temporada", sorted(df_full["temporada"].unique()),
    default=sorted(df_full["temporada"].unique()),
)
circuitos_sel = st.sidebar.multiselect(
    "Circuito", sorted(df_full["circuito"].unique()),
    default=sorted(df_full["circuito"].unique()),
)
pilotos_sel = st.sidebar.multiselect(
    "Piloto", sorted(df_full["piloto"].unique()),
    default=sorted(df_full["piloto"].unique()),
)
clima_sel = st.sidebar.multiselect(
    "Clima", CLIMAS, default=CLIMAS,
)

df = df_full[
    df_full["temporada"].isin(temporadas_sel)
    & df_full["circuito"].isin(circuitos_sel)
    & df_full["piloto"].isin(pilotos_sel)
    & df_full["clima"].isin(clima_sel)
].copy()

st.sidebar.markdown(f"**Registros tras filtros:** {len(df):,}")
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Cerrar sesión"):
    st.session_state.autenticado = False
    st.rerun()

# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.title("🏎️ Dashboard Interactivo - Análisis de Datos Sintéticos de Fórmula 1")
st.caption(
    "Todos los datos (pilotos, equipos, resultados e incidentes) son **completamente "
    "simulados** dentro de la plataforma con fines académicos. No representan personas "
    "ni eventos reales."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Registros (filtrados)", f"{len(df):,}")
col_b.metric("Carreras únicas", f"{df['id_carrera'].nunique():,}")
prom_accidentes = df["incidentes"].mean() if len(df) else 0
col_c.metric("Promedio de incidentes/registro", f"{prom_accidentes:.2f}")
tasa_dnf = df["dnf"].mean() * 100 if len(df) else 0
col_d.metric("Tasa de abandono (DNF)", f"{tasa_dnf:.1f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# TABS PRINCIPALES
# ---------------------------------------------------------------------------
tab_datos, tab_cuant, tab_cual, tab_series, tab_graf = st.tabs(
    ["📄 Datos", "🔢 Estadística Cuantitativa", "🏷️ Estadística Cualitativa",
     "⏱️ Serie de Tiempo", "📊 Análisis Gráfico Dinámico"]
)

# ------------------------- TAB 1: DATOS -----------------------------------
with tab_datos:
    st.subheader("Vista previa del dataset simulado de Fórmula 1")
    st.write(
        "Cada fila representa el resultado de **un piloto en una carrera**. Columnas: "
        "`id_carrera`, `temporada`, `ronda`, `circuito`, `tipo_circuito`, `pais`, `fecha`, "
        "`clima`, `piloto`, `equipo`, `posicion_salida`, `posicion_llegada`, `incidentes`, "
        "`accidente`, `dnf`, `paradas_boxes`, `tiempo_vuelta_promedio_seg`, `puntos`."
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
        file_name="f1_datos_sinteticos.csv", mime="text/csv",
    )

# ------------------------- TAB 2: CUANTITATIVA -----------------------------
with tab_cuant:
    st.subheader("Estadística descriptiva - Variables cuantitativas")
    numericas = ["posicion_salida", "posicion_llegada", "incidentes",
                 "paradas_boxes", "tiempo_vuelta_promedio_seg", "puntos"]

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        resumen = df[numericas].describe().T
        resumen["mediana"] = df[numericas].median()
        resumen["varianza"] = df[numericas].var()
        st.dataframe(resumen.round(2), use_container_width=True)

        st.markdown("#### Matriz de correlación")
        corr = df[numericas + ["dnf"]].assign(dnf=df["dnf"].astype(int)).corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Distribución con umbral personalizable")
        var_num = st.selectbox("Variable numérica", numericas, key="var_num_cuant")
        col_valida = df[var_num].dropna()
        umbral = st.slider(
            f"Umbral para '{var_num}'",
            float(col_valida.min()), float(col_valida.max()), float(col_valida.median()),
        )
        pct_sobre_umbral = (col_valida > umbral).mean() * 100
        st.info(f"**{pct_sobre_umbral:.1f}%** de los registros superan el umbral de {umbral:.1f}")

        fig_hist = px.histogram(df, x=var_num, nbins=40, marginal="box",
                                 title=f"Distribución de {var_num}")
        fig_hist.add_vline(x=umbral, line_dash="dash", line_color="red", annotation_text="Umbral")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Promedio de accidentes por piloto y por circuito")
        col1, col2 = st.columns(2)
        with col1:
            acc_piloto = (
                df.groupby("piloto")["incidentes"].mean().sort_values(ascending=False).head(15)
            )
            fig_acc_p = px.bar(
                acc_piloto.reset_index(), x="piloto", y="incidentes",
                title="Top 15 pilotos - Promedio de incidentes por carrera",
            )
            st.plotly_chart(fig_acc_p, use_container_width=True)
        with col2:
            acc_circ = (
                df.groupby("circuito")["incidentes"].mean().sort_values(ascending=False)
            )
            fig_acc_c = px.bar(
                acc_circ.reset_index(), x="circuito", y="incidentes",
                title="Promedio de incidentes por circuito",
            )
            st.plotly_chart(fig_acc_c, use_container_width=True)

# ------------------------- TAB 3: CUALITATIVA ------------------------------
with tab_cual:
    st.subheader("Estadística descriptiva - Variables cualitativas y relación piloto-circuito")
    categoricas = ["equipo", "circuito", "clima", "tipo_circuito", "pais"]

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        var_cat = st.selectbox("Variable categórica", categoricas, key="var_cat")
        tabla_frec = df[var_cat].value_counts(dropna=False).rename("frecuencia").to_frame()
        tabla_frec["porcentaje (%)"] = (
            tabla_frec["frecuencia"] / tabla_frec["frecuencia"].sum() * 100
        ).round(2)
        st.dataframe(tabla_frec, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(tabla_frec.reset_index(), x=var_cat, y="frecuencia",
                              color=var_cat, title=f"Frecuencia de {var_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(tabla_frec.reset_index(), names=var_cat, values="frecuencia",
                              title=f"Proporción de {var_cat}", hole=0.35)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Relación Piloto vs Circuito (heatmap de puntos promedio)")
        pilotos_top = st.slider("Cantidad de pilotos a incluir (top por puntos totales)", 5, 22, 12)
        top_pilotos_lista = (
            df.groupby("piloto")["puntos"].sum().sort_values(ascending=False).head(pilotos_top).index
        )
        df_top = df[df["piloto"].isin(top_pilotos_lista)]
        tabla_piloto_circuito = df_top.pivot_table(
            index="piloto", columns="circuito", values="puntos", aggfunc="mean"
        ).fillna(0)
        fig_heat = px.imshow(
            tabla_piloto_circuito, aspect="auto", color_continuous_scale="Viridis",
            title="Puntos promedio por piloto y circuito",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Tabla cruzada de incidentes: Circuito vs Clima")
        tabla_cruzada = df.pivot_table(
            index="circuito", columns="clima", values="incidentes", aggfunc="mean"
        ).fillna(0).round(2)
        st.dataframe(tabla_cruzada, use_container_width=True)

# ------------------------- TAB 4: SERIE DE TIEMPO --------------------------
with tab_series:
    st.subheader("Evolución temporal")

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        modo_serie = st.radio(
            "Selecciona la serie de tiempo a visualizar",
            ["Promedio de incidentes por temporada", "Puntos acumulados de un piloto",
             "Tiempo de vuelta promedio por temporada"],
            horizontal=False,
        )

        if modo_serie == "Promedio de incidentes por temporada":
            serie = df.groupby("temporada")["incidentes"].mean().reset_index()
            fig_serie = px.line(serie, x="temporada", y="incidentes", markers=True,
                                 title="Promedio de incidentes por temporada")
            umbral_serie = st.slider(
                "Línea de umbral de incidentes", 0.0, float(serie["incidentes"].max() + 0.5),
                float(serie["incidentes"].mean()),
            )
            fig_serie.add_hline(y=umbral_serie, line_dash="dash", line_color="red",
                                 annotation_text="Umbral")
            st.plotly_chart(fig_serie, use_container_width=True)

        elif modo_serie == "Puntos acumulados de un piloto":
            piloto_elegido = st.selectbox("Selecciona un piloto", sorted(df["piloto"].unique()))
            df_piloto = df[df["piloto"] == piloto_elegido].sort_values("fecha")
            df_piloto["puntos_acumulados"] = df_piloto["puntos"].cumsum()
            fig_serie = px.line(df_piloto, x="fecha", y="puntos_acumulados", markers=True,
                                 title=f"Puntos acumulados en el tiempo - {piloto_elegido}")
            st.plotly_chart(fig_serie, use_container_width=True)

        else:
            serie = df.groupby("temporada")["tiempo_vuelta_promedio_seg"].mean().reset_index()
            fig_serie = px.line(serie, x="temporada", y="tiempo_vuelta_promedio_seg", markers=True,
                                 title="Tiempo de vuelta promedio por temporada (segundos)")
            st.plotly_chart(fig_serie, use_container_width=True)

# ------------------------- TAB 5: GRÁFICO DINÁMICO --------------------------
with tab_graf:
    st.subheader("Explorador gráfico dinámico")

    if len(df) == 0:
        st.warning("No hay datos con los filtros actuales.")
    else:
        todas_columnas = [
            "posicion_salida", "posicion_llegada", "incidentes", "paradas_boxes",
            "tiempo_vuelta_promedio_seg", "puntos", "equipo", "circuito", "clima",
            "tipo_circuito", "pais", "piloto", "fecha", "temporada",
        ]

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
            var_color = st.selectbox("Colorear / agrupar por", ["(ninguna)"] + todas_columnas, index=0)
        color_arg = None if var_color == "(ninguna)" else var_color

        var_y = None
        if tipo_grafico in ("Dispersión (Scatter)", "Caja (Box)", "Violín", "Barras"):
            opciones_y = ["(ninguna)"] + todas_columnas
            var_y = st.selectbox("Variable eje Y", opciones_y, index=0)
            var_y = None if var_y == "(ninguna)" else var_y

        mostrar_umbral = st.checkbox("Agregar línea de umbral personalizada", value=False)
        umbral_val = None
        numericas_umbral = ["posicion_salida", "posicion_llegada", "incidentes",
                             "paradas_boxes", "tiempo_vuelta_promedio_seg", "puntos"]
        if mostrar_umbral:
            columna_umbral = var_y if var_y in numericas_umbral else (
                var_x if var_x in numericas_umbral else None
            )
            if columna_umbral:
                col_valida = df[columna_umbral].dropna()
                umbral_val = st.slider(
                    f"Valor del umbral ({columna_umbral})",
                    float(col_valida.min()), float(col_valida.max()), float(col_valida.median()),
                )
            else:
                st.caption("El umbral solo aplica a variables numéricas.")

        fig = None
        try:
            if tipo_grafico == "Histograma":
                bins = st.slider("Número de bins", 5, 100, 30)
                fig = px.histogram(df, x=var_x, color=color_arg, nbins=bins,
                                    title=f"Histograma de {var_x}", barmode="overlay", opacity=0.75)
            elif tipo_grafico == "Dispersión (Scatter)":
                if var_y:
                    fig = px.scatter(df, x=var_x, y=var_y, color=color_arg, opacity=0.6,
                                      title=f"{var_x} vs {var_y}")
                else:
                    st.warning("Selecciona una variable Y para el scatter.")
            elif tipo_grafico == "Caja (Box)":
                fig = px.box(df, x=var_x, y=var_y, color=color_arg, title=f"Boxplot de {var_y or var_x}")
            elif tipo_grafico == "Violín":
                fig = px.violin(df, x=var_x, y=var_y, color=color_arg, box=True,
                                 title=f"Violín de {var_y or var_x}")
            elif tipo_grafico == "Barras":
                if var_y and pd.api.types.is_numeric_dtype(df[var_y]):
                    agg_df = df.groupby(var_x, observed=True)[var_y].mean().reset_index()
                    fig = px.bar(agg_df, x=var_x, y=var_y,
                                 color=color_arg if color_arg in agg_df.columns else None,
                                 title=f"Promedio de {var_y} por {var_x}")
                else:
                    conteo = df[var_x].value_counts().reset_index()
                    conteo.columns = [var_x, "conteo"]
                    fig = px.bar(conteo, x=var_x, y="conteo", title=f"Conteo por {var_x}")
            elif tipo_grafico == "Serie temporal":
                freq_opcion = st.selectbox("Agregación temporal", ["Por ronda", "Por temporada"])
                if freq_opcion == "Por ronda":
                    serie = df.groupby("fecha").size().reset_index(name="registros")
                    fig = px.line(serie, x="fecha", y="registros", markers=True,
                                  title="Registros en el tiempo (por fecha de carrera)")
                else:
                    serie = df.groupby("temporada").size().reset_index(name="registros")
                    fig = px.line(serie, x="temporada", y="registros", markers=True,
                                  title="Registros por temporada")

            if fig is not None:
                if umbral_val is not None:
                    if tipo_grafico == "Histograma":
                        fig.add_vline(x=umbral_val, line_dash="dash", line_color="red",
                                      annotation_text="Umbral")
                    else:
                        fig.add_hline(y=umbral_val, line_dash="dash", line_color="red",
                                      annotation_text="Umbral")
                fig.update_layout(height=550)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No fue posible generar la gráfica con esta combinación de variables: {e}")

st.markdown("---")
st.caption(
    "Aplicación construida con Streamlit + Plotly · Datos 100% sintéticos generados "
    "en tiempo real dentro de la plataforma · EAFIT 2026 - Ciencia de Datos."
)
