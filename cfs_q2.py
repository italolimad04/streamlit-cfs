import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from st_aggrid import (
    AgGrid, GridOptionsBuilder, AgGridTheme,
    DataReturnMode, GridUpdateMode
)

from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv
import requests
import time
from typing import Tuple, Dict, Any

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
load_dotenv()

local_tz = timezone("America/Sao_Paulo")
utc_tz = timezone("UTC")

TTL = 500

# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=TTL)
def fetch_data(first_day: str, last_day: str) -> Tuple[pd.DataFrame, float]:
    response = requests.get(
        url=(
            "https://new-api.urbis.cc/communication/fidelized-clients-by-quarter"
            f"?initialDate={first_day}&finalDate={last_day}"
        )
    ).json()

    fidelizedClientsData = response["data"]["fidelizedClientsData"]
    df = pd.DataFrame(data=fidelizedClientsData)
    df = clean_data(df)
    return df, time.time()


def invalidate_cache():
    fetch_data.clear()

# =========================
# CONSTANTES (MANTER PADRÃO ATUAL)
# =========================
# Definindo Valores (baseline histórico)
total_cfs_fim_2024 = 5749
total_cfs_q1_2025 = 980
total_cfs_q2_2025 = 1063
total_cfs_q3_2025 = 1920
total_cfs_q4_2025 = 2416
total_cfs_fim_2025 = 12128

meta_anual = 50000
meta_dentro_do_ano = 50000 - total_cfs_fim_2025


# =========================
# QUARTERS / TIME
# =========================
def get_current_quarter_local(now_local: datetime) -> Tuple[int, int]:
    year = now_local.year
    month = now_local.month
    quarter = (month - 1) // 3 + 1
    return year, quarter


def quarter_start_end_utc(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """
    Retorna início/fim do trimestre em UTC.
    Boundaries em LOCAL (calendário local) -> converte pra UTC.
    Intervalo: [start, end)
    """
    if quarter == 1:
        start_local = datetime(year, 1, 1)
        end_local = datetime(year, 4, 1)
    elif quarter == 2:
        start_local = datetime(year, 4, 1)
        end_local = datetime(year, 7, 1)
    elif quarter == 3:
        start_local = datetime(year, 7, 1)
        end_local = datetime(year, 10, 1)
    elif quarter == 4:
        start_local = datetime(year, 10, 1)
        end_local = datetime(year + 1, 1, 1)
    else:
        raise ValueError("quarter deve ser 1..4")

    start_utc = local_tz.localize(start_local).astimezone(utc_tz)
    end_utc = local_tz.localize(end_local).astimezone(utc_tz)
    return start_utc, end_utc


def quarter_label_from_local(dt_utc: pd.Timestamp) -> str:
    """
    Define trimestre a partir do horário LOCAL (evita deslocamento por timezone).
    dt_utc deve ser tz-aware em UTC.
    """
    if pd.isna(dt_utc):
        return "Sem Data"

    dt_local = dt_utc.tz_convert(local_tz)
    year = dt_local.year
    month = dt_local.month
    q = (month - 1) // 3 + 1
    return f"Q{q} {str(year)[-2:]}"


def parse_quarter_label(lbl: str) -> Tuple[int, int]:
    """
    Converte 'Q1 25' -> (2025, 1) para ordenar corretamente.
    """
    try:
        parts = lbl.strip().split()
        q_part = parts[0].upper().replace("Q", "")
        y_part = parts[1]
        q = int(q_part)
        yy = int(y_part)
        year = 2000 + yy if yy < 100 else yy
        return year, q
    except Exception:
        return (9999, 9)


def calcular_semana_fiscal(data_utc: datetime, start_date_utc: datetime) -> int:
    delta = data_utc - start_date_utc
    return delta.days // 7 + 1


# =========================
# DATA PARSE (TIMEZONE SAFE)
# =========================
def parse_to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Corrige timestamp sem deslocar datas:
    - Se vier com tz (Z/offset): converte para UTC
    - Se vier naive: assume LOCAL e converte para UTC
    """
    s = series.copy()
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # se já tem tz, converte
    try:
        tzinfo = dt.dt.tz
    except Exception:
        tzinfo = None

    if tzinfo is not None:
        return dt.dt.tz_convert(utc_tz)

    # naive -> assume local e converte
    dt_local = dt.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
    return dt_local.dt.tz_convert(utc_tz)


# =========================
# DATA CLEANING
# =========================
def normalizar_valor(valor: Any) -> float:
    if isinstance(valor, str) and "," in valor:
        try:
            return float(valor.replace(".", "").replace(",", "."))
        except Exception:
            return 0.0
    try:
        return float(valor)
    except Exception:
        return 0.0


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Data
    if "Data" in df.columns:
        df["Data"] = parse_to_utc_datetime(df["Data"])

    # Valor Economizado
    if "Valor Economizado" in df.columns:
        df["Valor Economizado"] = df["Valor Economizado"].apply(normalizar_valor).round(2)

    # Ajustes de Canal
    if set(["Canal", "Pesquisa"]).issubset(df.columns):
        mask_drog = (
            (df["Canal"] == "email")
            & (df["Pesquisa"] != "campanha-voucher-$400-part2---2025-clube-unimed-jp")
            & (df["Pesquisa"] != "campanha-voucher-$400---2025-clube-unimed-jp")
            & (df["Pesquisa"] != "limone-")
        )
        df.loc[mask_drog, "Canal"] = "Drogarias"

        mask_unimed = df["Pesquisa"].isin([
            "campanha-voucher-$400---2025-clube-unimed-jp",
            "campanha-voucher-$400-part2---2025-clube-unimed-jp",
            "limone-"
        ])
        df.loc[mask_unimed, "Canal"] = "Ação Unimed"

        mask_email_ind = (
            df["Pesquisa"].astype(str).str.contains("Muito relevante", case=True, na=False)
            | df["Pesquisa"].astype(str).str.contains("Relevante", case=True, na=False)
        )
        df.loc[mask_email_ind, "Canal"] = "E-mail Individual"

        df.loc[df["Canal"] == "club", "Canal"] = "Clube"

    # Parceiro / Clube / Satisfação
    if "Parceiro" in df.columns:
        df.loc[df["Parceiro"] == "Farmácias Pague Menos", "Parceiro"] = "Pague Menos"

    if "Clube" in df.columns:
        df.loc[df["Clube"] == "Clínica SiM+", "Clube"] = "Clínica SiM"
        df.loc[df["Clube"] == "Club de Vantagens | Sócio Vozão", "Clube"] = "Sócio Vozão"
        df.loc[df["Clube"] == "MOV Fibra", "Clube"] = "Mov Telecom"
        df.loc[df["Clube"] == "Clube O Povo", "Clube"] = "O Povo"

    if "Satisfação" in df.columns:
        df.loc[df["Satisfação"] == "Muito relevante", "Satisfação"] = "Muito Relevante"
        df.loc[df["Satisfação"] == "Muito_relevante", "Satisfação"] = "Muito Relevante"

    # Tratando Parceiro
    if "Parceiro" in df.columns:
        df.loc[df["Parceiro"] == "Cantinho do Frango - Aldeota", "Parceiro"] = "Cantinho do Frango"
        df.loc[df["Parceiro"] == "PAGUE MENOS", "Parceiro"] = "Pague Menos"
        df.loc[df["Parceiro"] == "Farmácias", "Parceiro"] = "Pague Menos"
        df.loc[df["Parceiro"] == "CENTAURO", "Parceiro"] = "Centauro"
        df.loc[df["Parceiro"] == "CASAS BAHIA", "Parceiro"] = "Casas Bahia"
        df["Parceiro"] = df["Parceiro"].astype(str).str.strip().str.title()

    return df


# =========================
# UI: TÍTULO + CONTROLES
# =========================
st.title("Painel de Clientes Fidelizados - Dashboard Geral")

with st.sidebar:
    st.subheader("Parâmetros")
    st.caption("O endpoint retornará todos os dados dentro do intervalo informado.")

    default_start = datetime(2024, 1, 1).date()
    default_end = (datetime.now(local_tz).date() + timedelta(days=1))

    initialDate = st.date_input("Data inicial", value=default_start)
    finalDate = st.date_input("Data final", value=default_end)

    st.markdown("---")
    st.subheader("Trimestre em foco (automático)")
    st.caption("Sempre usa o **trimestre atual** com base no horário local.")

    now_local = datetime.now(local_tz)
    focus_year, focus_quarter = get_current_quarter_local(now_local)
    st.write(f"**Trimestre atual:** Q{focus_quarter} {str(focus_year)[-2:]}")

    if st.button("🔄 Recarregar dados"):
        invalidate_cache()

firstDayOfRange = str(initialDate)
lastDayOfRange = str(finalDate)


df_fidelized_clients_by_survey, last_updated = fetch_data(firstDayOfRange, lastDayOfRange)

data = pd.DataFrame()
data = pd.concat([data, df_fidelized_clients_by_survey], ignore_index=True)

# =========================
# TRIMESTRE FOCO (atual automático)
# =========================
q_start_utc, q_end_utc = quarter_start_end_utc(focus_year, focus_quarter)

data_non_null = data.dropna(subset=["Data"]).copy() if "Data" in data.columns else data.copy()
if "Data" in data_non_null.columns:
    data_non_null["QuarterLabel"] = data_non_null["Data"].apply(quarter_label_from_local)
else:
    data_non_null["QuarterLabel"] = "Sem Data"

data_focus_quarter = data_non_null[
    (data_non_null["Data"] >= q_start_utc) & (data_non_null["Data"] < q_end_utc)
].copy()

focus_label = f"Q{focus_quarter} {str(focus_year)[-2:]}"

# =========================
# CONSTANTES APLICADAS (MANTER PADRÃO)
# =========================
total_cfs_quarter_current = int(data_focus_quarter.shape[0])
print("Total CFs Quarter:", total_cfs_quarter_current)

# === RECORTES CORRETOS (para não duplicar baseline) ===
now_local = datetime.now(local_tz)
focus_year, focus_quarter = get_current_quarter_local(now_local)

q_start_utc, q_end_utc = quarter_start_end_utc(focus_year, focus_quarter)
data_focus_quarter = data_non_null[
    (data_non_null["Data"] >= q_start_utc) & (data_non_null["Data"] < q_end_utc)
].copy()

# Ano atual (YTD) em UTC usando bordas do horário LOCAL
year_start_utc = local_tz.localize(datetime(focus_year, 1, 1)).astimezone(utc_tz)
year_end_utc = local_tz.localize(datetime(focus_year + 1, 1, 1)).astimezone(utc_tz)

data_year = data_non_null[
    (data_non_null["Data"] >= pd.Timestamp(year_start_utc))
    & (data_non_null["Data"] < pd.Timestamp(year_end_utc))
].copy()

focus_label = f"Q{focus_quarter} {str(focus_year)[-2:]}"

total_cfs_year_current = int(data_year.shape[0])               # ano atual (ex: 2026 YTD)


# Total fidelizados (baseline + trimestre atual do dataset)
total_fidelizados = (
    int(total_cfs_year_current)
    + total_cfs_q1_2025 + total_cfs_q2_2025 + total_cfs_q3_2025 + total_cfs_q4_2025
    + total_cfs_fim_2024
)

total_cfs_2026 = (total_fidelizados - total_cfs_fim_2025)

# Dicionário oficial de trimestres (baseline + atual)
total_cfs_quarter: Dict[str, int] = {
    "Q1 24": 796,
    "Q2 24": 531,
    "Q3 24": 853,
    "Q4 24": 718,
    "Q1 25": 980,
    "Q2 25": 1063,
    "Q3 25": 1920,
    "Q4 25": 2416,
    "Q1 26": total_cfs_quarter_current,  # mantém seu padrão atual (se quiser trocar dinamicamente por focus_label, me diga)
}

# Ordenação correta
quarter_labels_sorted = sorted(list(total_cfs_quarter.keys()), key=parse_quarter_label)
total_cfs_quarter_dict: Dict[str, int] = {lbl: int(total_cfs_quarter[lbl]) for lbl in quarter_labels_sorted}

# =========================
# SEMANAS + AGG (trimestre foco)
# =========================
start_date_quarter = q_start_utc
data_atual_utc = datetime.now(local_tz).astimezone(utc_tz)

df_focus_for_week = data_focus_quarter.copy()
if "Data" in df_focus_for_week.columns:
    df_focus_for_week["Semana"] = df_focus_for_week["Data"].apply(
        lambda x: calcular_semana_fiscal(x.to_pydatetime(), start_date_quarter)
        if pd.notna(x) else 1
    )
else:
    df_focus_for_week["Semana"] = 1

data_max = df_focus_for_week["Data"].max() if "Data" in df_focus_for_week.columns and not df_focus_for_week.empty else pd.Timestamp(start_date_quarter)

if pd.notna(data_max) and data_max < pd.Timestamp(data_atual_utc):
    additional_weeks = pd.date_range(
        start=(data_max + pd.Timedelta(days=7)),
        end=pd.Timestamp(data_atual_utc),
        freq="7D",
        tz=utc_tz
    )
    additional_data = pd.DataFrame({
        "Data": additional_weeks,
        "Semana": additional_weeks.map(lambda x: calcular_semana_fiscal(x.to_pydatetime(), start_date_quarter))
    })
    df_focus_for_week = pd.concat([df_focus_for_week, additional_data], ignore_index=True)

max_week = int(df_focus_for_week["Semana"].max()) if not df_focus_for_week.empty else 1
all_weeks = pd.DataFrame({"Semana": list(range(1, max_week + 1))})

agg_data = df_focus_for_week.groupby("Semana").size().reset_index(name="Novos CFs")
agg_data = all_weeks.merge(agg_data, on="Semana", how="left").fillna(0)

agg_data["Semana"] = agg_data["Semana"].astype(int)
agg_data["Novos CFs"] = agg_data["Novos CFs"].astype(int)
agg_data["Total CFs"] = agg_data["Novos CFs"].cumsum()

meta_cfs_tri = 5000
agg_data["Meta Trimestre"] = meta_cfs_tri

agg_data["Data_Inicio_Semana"] = agg_data["Semana"].apply(
    lambda x: start_date_quarter + timedelta(weeks=x - 1)
)
agg_data["Data_Inicio_Semana"] = pd.to_datetime(agg_data["Data_Inicio_Semana"], errors="coerce", utc=True)

agg_data = agg_data[["Data_Inicio_Semana", "Semana", "Novos CFs", "Total CFs", "Meta Trimestre"]]

semana_atual = calcular_semana_fiscal(data_atual_utc, start_date_quarter)
semana_anterior = semana_atual - 1

try:
    resultados_semana_atual = int(agg_data[agg_data["Semana"] == semana_atual]["Novos CFs"].values[0])
except Exception:
    resultados_semana_atual = 0

try:
    resultados_semana_anterior = int(agg_data[agg_data["Semana"] == semana_anterior]["Novos CFs"].values[0])
except Exception:
    resultados_semana_anterior = 0

# =========================
# KPIs (mantendo padrão constantes)
# =========================
meta_dentro_do_ano = 20000 - total_cfs_fim_2025
faltam_para_meta = meta_anual - total_fidelizados
if faltam_para_meta < 0:
    faltam_para_meta = 0

percentual_meta_tri = (total_cfs_quarter_current / meta_cfs_tri) * 100 if meta_cfs_tri else 0

porcentagem_meta_anual = (total_cfs_2026 / meta_dentro_do_ano) * 100 if meta_dentro_do_ano else 0
restante_meta_anual = 100 - porcentagem_meta_anual
if restante_meta_anual < 0:
    restante_meta_anual = 0

# =========================
# FIGURAS
# =========================
fig_total = go.Figure()

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_fidelizados,
    title={"text": f"<span style='color:#1B0A63;'>Clientes Fidelizados</span> <br><span style='font-size:0.9em;color:#19C78A'>Meta 2026: {meta_anual:.0f} CFs</span>"},
    domain={'row': 0, 'column': 0},
    number={"font": {"size": 70, "color": "#1B0A63"}, "valueformat": "d"},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=faltam_para_meta,
    title={"text": f"<span style='color:#1B0A63;'>Restam para a Meta Anual</span> "},
    domain={'row': 0, 'column': 1},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_quarter_current,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs no Trimestre Atual ({focus_label})</span> <br><span style='font-size:0.9em;color:#19C78A'>Meta: 5000. {percentual_meta_tri:.2f}% atingido </span>"},
    domain={'row': 0, 'column': 2},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_2026,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs em 2026</span>"},
    domain={'row': 0, 'column': 3},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

fig_total.add_trace(go.Indicator(
    mode="number",
    value=int(resultados_semana_anterior),
    title={"text": "<span style='color:#1B0A63;'>Novos CFs (última semana)</span>"},
    domain={'row': 0, 'column': 4},
    number={"font": {"size": 70, "color": "#1B0A63"}}
))

fig_total.update_layout(
    margin=dict(t=0, b=0),
    grid={'rows': 1, 'columns': 5, 'pattern': "independent"},
    template={'data': {'indicator': [{'mode': "number+delta"}]}}
)

# fig (Total vs meta) — trimestre atual
fig = go.Figure(data=[
    go.Bar(
        name="<b>Total CFs<b>",
        x=[f"Clientes Fidelizados ({focus_label})"],
        y=[total_cfs_quarter_current],
        marker_color="#FFA726",
        text=[f"<b>{total_cfs_quarter_current}"],
        textposition="auto",
        textfont=dict(size=20, color="black", family="Roboto"),
    ),
    go.Bar(
        name="<b>Meta<b>",
        x=[f"Clientes Fidelizados ({focus_label})"],
        y=[meta_cfs_tri],
        marker_color="#42A5F5",
        text=[f"<b>{meta_cfs_tri}"],
        textposition="auto",
        textfont=dict(size=20, color="black", family="Roboto"),
    )
])

fig.update_layout(
    title={'text': f'Resultado Obtido (Trimestre Atual - {focus_label})', 'x': 0.5, 'xanchor': 'center',
           'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    barmode='group',
    width=800,
    height=600,
    font=dict(size=16, color='black', family='Roboto'),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# fig2 (Top 20 parceiros) — trimestre atual
parceiro_counts = data_focus_quarter["Parceiro"].value_counts().nlargest(20) if "Parceiro" in data_focus_quarter.columns else pd.Series(dtype=int)

fig2 = go.Figure(data=[
    go.Bar(
        name="<b>Clientes por Parceiro</b>",
        x=parceiro_counts.index,
        y=parceiro_counts.values,
        marker_color="#19C78A",
        text=parceiro_counts.values,
        textposition="auto",
        textfont=dict(size=20, color="black", family="Roboto"),
    )
])

fig2.update_layout(
    title={'text': f'CFs por Parceiro - Top 20 ({focus_label})', 'x': 0.6, 'xanchor': 'center',
           'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    height=750,
    font=dict(size=25, color='black', family='Roboto'),
    plot_bgcolor='white',
    bargap=0.05,
    paper_bgcolor='white',
    xaxis=dict(tickangle=-45, title='Parceiro', tickfont=dict(size=15, color='black', family='Roboto')),
    yaxis=dict(title='Clientes Fidelizados', tickfont=dict(size=25, color='black', family='Roboto'), dtick=100),
)

# fig3 (Canal) — trimestre atual
canal_counts = data_focus_quarter["Canal"].value_counts() if "Canal" in data_focus_quarter.columns else pd.Series(dtype=int)

fig3 = go.Figure(data=[
    go.Bar(
        name="<b>CFs por Canal</b>",
        x=canal_counts.index,
        y=canal_counts.values,
        marker_color="#19C78A",
        text=canal_counts.values,
        textposition="auto",
        textfont=dict(size=20, color="black", family="Roboto"),
        width=0.5
    )
])

fig3.update_layout(
    title={'text': f'CFs por Canal ({focus_label})', 'x': 0.5, 'xanchor': 'center',
           'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    width=800,
    height=600,
    font=dict(size=16, color='black', family='Roboto'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(title='Canal de Pesquisa', tickfont=dict(size=16, color='black', family='Roboto')),
    yaxis=dict(title='Clientes Fidelizados', tickfont=dict(size=16, color='black', family='Roboto')),
)

# fig4 (Clube top 20) — trimestre atual
clube_counts = (
    data_focus_quarter["Clube"].value_counts()
    if "Clube" in data_focus_quarter.columns
    else pd.Series(dtype=int)
)

top_n = 20
top_clubes = clube_counts.nlargest(top_n)

# --- detecta outlier pra não "amassar" barras
vals = top_clubes.values.astype(float)
max_v = float(vals.max()) if len(vals) else 0.0
p90 = float(np.percentile(vals, 90)) if len(vals) >= 2 else max_v
usar_log = (p90 > 0) and (max_v / p90 >= 4)  # ajuste fino: 4x o p90

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=top_clubes.values,
    y=top_clubes.index,
    orientation="h",
    text=top_clubes.values,
    textposition="outside",     # fora pra não sumir em barras pequenas
    cliponaxis=False,           # evita cortar texto na borda
    marker=dict(color=px.colors.qualitative.Plotly, line=dict(color="white", width=1)),
    hovertemplate="<b>%{y}</b><br>CFs: %{x}<extra></extra>",
))

fig4.update_layout(
    title=dict(
        text=f"CFs por Clube - Top {top_n} ({focus_label})",
        x=0.5,
        xanchor="center",
        font=dict(size=22, family="Roboto", color="black"),
    ),
    xaxis=dict(
        title="Quantidade de CFs",
        tickfont=dict(size=14),
        gridcolor="rgba(200,200,200,0.3)",
        type="log" if usar_log else "linear",
        rangemode="tozero",
    ),
    yaxis=dict(title="Clube", tickfont=dict(size=13), automargin=True),
    font=dict(size=13, family="Roboto"),
    height=800,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=80, b=40, l=200, r=40),  # mais espaço pros nomes
)

fig4.update_yaxes(autorange="reversed")

# fig5 (Novos vs Total) — trimestre atual
fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=agg_data["Semana"],
    y=agg_data["Total CFs"],
    mode="lines+markers+text",
    name="Total Trimestre",
    text=agg_data["Total CFs"],
    textposition="top center",
    line=dict(color="blue", width=2),
    marker=dict(size=10, symbol="diamond", color="blue"),
    legendgroup=1
))

fig5.add_trace(go.Bar(
    x=agg_data["Semana"],
    y=agg_data["Novos CFs"],
    name="Novos CFs",
    text=agg_data["Novos CFs"],
    textposition="outside",
    marker_color="orange",
    legendgroup=2
))

fig5.add_trace(go.Scatter(
    x=agg_data["Semana"],
    y=agg_data["Meta Trimestre"],
    mode="lines",
    name="Meta Trimestre",
    line=dict(color="red", dash="dash"),
    marker=dict(size=0)
))

fig5.update_layout(
    title=f"Novos CFs Vs Total ({focus_label})",
    xaxis_title="Semana",
    yaxis_title="Número de CFs",
    barmode="group",
    template="plotly_white",
    bargroupgap=0.1,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=60, b=40),
    height=600,
)

for trace in fig5.data:
    if trace.name == "Total Trimestre":
        trace.textposition = "top center"
    elif trace.name == "Novos CFs":
        trace.textposition = "outside"

# =========================
# FUNÇÕES (tabela interativa + export CSV)
# =========================
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def criar_tabela_interativa(df: pd.DataFrame, export_file_name: str, key_suffix: str) -> None:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filterable=True, sortable=True, editable=False, resizable=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_side_bar(filters_panel=True, columns_panel=True)
    gb.configure_selection("single", use_checkbox=True)

    for col in df.columns:
        gb.configure_column(col, filter="agTextColumnFilter", sortable=True)

    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=600,
        width="100%",
        theme=AgGridTheme.STREAMLIT
    )

    df_filtrado = pd.DataFrame(grid_response["data"])
    st.markdown(f"🔍 **Linhas após filtro:** {len(df_filtrado)}")

    if len(df_filtrado) > 0:
        st.download_button(
            label="📥 Baixar dados filtrados (CSV)",
            data=dataframe_to_csv_bytes(df_filtrado),
            file_name=export_file_name,
            mime="text/csv",
            key=f"download_filtered_{export_file_name}_{key_suffix}",
        )
    else:
        st.info("Nenhuma linha para exportar — ajuste os filtros acima.")


def criar_grafico_trimestral(total_cfs_quarter_dict: Dict[str, int]) -> go.Figure:
    trimestres = list(total_cfs_quarter_dict.keys())
    resultados = list(total_cfs_quarter_dict.values())
    meta_trimestre = 722

    fig_tri = go.Figure()

    fig_tri.add_trace(go.Bar(
        x=trimestres,
        y=resultados,
        name="Resultados",
        marker_color="#3AB78B",
        text=[f"<b>{v}</b>" for v in resultados],
        textposition="inside",
        insidetextfont=dict(size=16, color="black")
    ))

    fig_tri.add_trace(go.Scatter(
        x=trimestres,
        y=[meta_trimestre] * len(trimestres),
        mode="lines",
        name="Média",
        line=dict(color="red", dash="dash", width=2),
        hovertemplate="Meta: %{y}<extra></extra>"
    ))

    fig_tri.update_layout(
        title="Comparação de Resultados por Trimestre",
        xaxis_title="Trimestre",
        yaxis_title="Número de Clientes Fidelizados",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5)
    )

    return fig_tri


# =========================
# EXPORT CSV (trimestre foco)
# =========================
def render_export_focus_csv(df_focus: pd.DataFrame, key_suffix: str = "") -> None:
    st.subheader("Export do trimestre atual")
    if df_focus is None or df_focus.empty:
        st.info("Sem dados no trimestre atual para exportar.")
        return

    st.download_button(
        label="📥 Baixar CSV do trimestre atual",
        data=dataframe_to_csv_bytes(df_focus),
        file_name=f"clientes_fidelizados_{focus_label.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_focus_csv_{focus_label.replace(' ', '_')}_{key_suffix}",
    )


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Resumo (Geral)",
        "Resultados por Trimestre (Geral)",
        "Trimestre Atual",
        "CFs por Clube (Tri Atual)",
        "CFs por Parceiros (Tri Atual)",
        "CFs por Canal (Tri Atual)",
        "Valores Economizados (Tri Atual)",
        "Tabela Interativa (Tri Atual)",
        "Tabela Interativa (Geral)",
    ]
)

with tab1:
    st.plotly_chart(fig_total, use_container_width=True)
    st.caption(f"Última atualização: {datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    fig_total_trimestres = criar_grafico_trimestral(total_cfs_quarter_dict)
    st.plotly_chart(fig_total_trimestres, use_container_width=True)

with tab3:
    render_export_focus_csv(data_focus_quarter, key_suffix="tab3")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    render_export_focus_csv(data_focus_quarter, key_suffix="tab4")
    st.markdown("---")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    render_export_focus_csv(data_focus_quarter, key_suffix="tab5")
    st.markdown("---")
    st.plotly_chart(fig2, use_container_width=True)

with tab6:
    render_export_focus_csv(data_focus_quarter, key_suffix="tab6")
    st.markdown("---")
    st.plotly_chart(fig3, use_container_width=True)

with tab8:
    render_export_focus_csv(data_focus_quarter, key_suffix="tab8")
    st.markdown("---")
    criar_tabela_interativa(
        data_focus_quarter,
        export_file_name=f"clientes_fidelizados_filtrado_{focus_label.replace(' ', '_')}.csv",
        key_suffix="tab8"
    )

with tab9:
    criar_tabela_interativa(
        data_non_null,
        export_file_name="clientes_fidelizados_filtrado_geral.csv",
        key_suffix="tab9"
    )