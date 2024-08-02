import pandas as pd
import requests as reqs
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme, DataReturnMode, GridUpdateMode
from datetime import datetime, timedelta
import gspread as gs
from google.oauth2.service_account import Credentials
import os
import json
from dotenv import load_dotenv

import logging

# Configurar o logger
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Opções
st.set_page_config(layout="wide")

st.title("Painel de Clientes Fidelizados no Q3")

load_dotenv()
#Configuração do Google Cloud - Sheets API

def carregar_dados_do_google_sheets():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds_json = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gs.authorize(credentials)
    sheet = client.open_by_key(st.secrets["SHEET_KEY"]).sheet1
    data = sheet.get_all_values()  # Tenta obter todos os valores como strings
    headers = data.pop(0)  # Remove o cabeçalho da lista de dados
    return pd.DataFrame(data, columns=headers, dtype=str)

# Carregar os dados
data = carregar_dados_do_google_sheets()

data_estaticos = data.copy()

# Conversões e cálculos
data['Data'] = pd.to_datetime(data['Data'], errors='coerce')

def carregar_dados_do_google_sheets():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds_json = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gs.authorize(credentials)
    sheet = client.open_by_key(st.secrets["SHEET_KEY"]).sheet1
    data = sheet.get_all_values()  # Tenta obter todos os valores como strings
    headers = data.pop(0)  # Remove o cabeçalho da lista de dados
    return pd.DataFrame(data, columns=headers, dtype=str)

# Carregar os dados
data = carregar_dados_do_google_sheets()

# Conversões e cálculos
data['Data'] = pd.to_datetime(data['Data'], errors='coerce')

logger.info(data.shape)

logger.info(data['Data'].tail())
# Função para calcular a semana fiscal
def calcular_semana_fiscal(data, start_date):
    delta = data - start_date
    return delta.days // 7 + 1

# Definir a data de início do terceiro trimestre
start_date_q3 = datetime.strptime('2024-06-27', '%Y-%m-%d')

# Calcular a semana fiscal para cada registro
data['Semana'] = data['Data'].apply(lambda x: calcular_semana_fiscal(x, start_date_q3))

# Verificar a data de hoje e a data máxima nos dados
data_atual = datetime.today()
data_max = data['Data'].max()
logger.info(f'Data máxima nos dados: {data_max}')

# Adicionar registros para semanas até a data atual
if data_max < data_atual:
    additional_weeks = pd.date_range(start=data_max + timedelta(days=7), end=data_atual, freq='7D')
    additional_data = pd.DataFrame({'Data': additional_weeks, 'Semana': additional_weeks.map(lambda x: calcular_semana_fiscal(x, start_date_q3))})
    data = pd.concat([data, additional_data], ignore_index=True)

all_weeks = pd.DataFrame({'Semana': range(1, int(data['Semana'].max()) + 1)})

# Agrupar os dados por semana e mesclar com todas as semanas
agg_data = data.groupby('Semana').size().reset_index(name='Novos CFs')
agg_data = all_weeks.merge(agg_data, on='Semana', how='left').fillna(0)

# Garantir que as colunas sejam inteiros
agg_data['Semana'] = agg_data['Semana'].astype(int)
agg_data['Novos CFs'] = agg_data['Novos CFs'].astype(int)

# Adicionar colunas adicionais
agg_data['Total CFs'] = agg_data['Novos CFs'].cumsum()
agg_data['Meta Q3'] = 400
agg_data['Total_Geral'] = agg_data['Total CFs'] + 4178  # Ajuste conforme necessário

# Adicionar coluna 'Data_Inicio_Semana'
agg_data['Data_Inicio_Semana'] = agg_data['Semana'].apply(lambda x: start_date_q3 + timedelta(weeks=x-1))

# Verificar e converter 'Data_Inicio_Semana' para datetime
agg_data['Data_Inicio_Semana'] = pd.to_datetime(agg_data['Data_Inicio_Semana'], errors='coerce')

# Reorganizar as colunas
agg_data = agg_data[['Data_Inicio_Semana', 'Semana', 'Novos CFs', 'Total CFs', 'Meta Q3', 'Total_Geral']]

# Identificar a semana atual e a semana anterior
semana_atual = calcular_semana_fiscal(data_atual, start_date_q3)

logger.info(agg_data['Semana'].head())
logger.info(f'Semana Atual: {semana_atual}')

semana_anterior = semana_atual - 1
logger.info(f'Semana Anterior: {semana_anterior}')

# Pegar os resultados para esta semana e a anterior com verificações de erro
try:
    resultados_semana_atual = agg_data[agg_data['Semana'] == semana_atual]['Novos CFs'].values[0]
except IndexError:
    resultados_semana_atual = 0  # Definir um valor padrão ou tomar outra ação

try:
    resultados_semana_anterior = agg_data[agg_data['Semana'] == semana_anterior]['Novos CFs'].values[0]
except IndexError:
    resultados_semana_anterior = 0  # Definir um valor padrão ou tomar outra ação

logger.info(f'Resultados Semana Anterior: {resultados_semana_anterior}')
logger.info(f'Resultados Semana Atual: {resultados_semana_atual}')

## Tratando dados
# Corrigir os valores na coluna 'Parceiro' usando .loc para evitar warnings
data.loc[data['Parceiro'] == 'Cantinho do Frango - Aldeota', 'Parceiro'] = 'Cantinho do Frango'
data.loc[data['Parceiro'] == 'PAGUE MENOS', 'Parceiro'] = 'Pague Menos'
data.loc[data['Parceiro'] == 'Farmácias', 'Parceiro'] = 'Pague Menos'
data.loc[data['Parceiro'] == 'CENTAURO', 'Parceiro'] = 'Centauro'
data.loc[data['Parceiro'] == 'CASAS BAHIA', 'Parceiro'] = 'Casas Bahia'

data['Parceiro'] = data['Parceiro'].str.strip().str.title()

# Calcular o número total de CFs
total_cfs = data.shape[0]
meta_cfs = 400

# Criar a visualização
fig = go.Figure(data=[
    go.Bar(name='<b>Total CFs<b>', x=['Clientes Fidelizados'], y=[total_cfs], marker_color='#FFA726', text=[f'<b>{total_cfs}'], textposition='auto', textfont=dict(size=20, color='black', family='Roboto')),
    go.Bar(name='<b>Meta<b>', x=['Clientes Fidelizados'], y=[meta_cfs], marker_color='#42A5F5', text=[f'<b>{meta_cfs}'], textposition='auto', textfont=dict(size=20, color='black', family='Roboto'))
])

# Atualizar layout do gráfico
fig.update_layout(
    title={'text': 'Resultado Obtido vs Meta', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    barmode='group',
    width=800,
    height=600,
    font=dict(size=16, color='black', family='Roboto'),
    plot_bgcolor='white',
    legend=dict(
        font=dict(size=20)
    ),
    xaxis=dict(
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=16, color='black', family='Roboto')
    ),
    yaxis=dict(
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=16, color='black', family='Roboto')
    ),
    paper_bgcolor='white'
)


# Criar a visualização para Número de Clientes por Parceiro
parceiro_counts = data['Parceiro'].value_counts()

fig2 = go.Figure(data=[
    go.Bar(name='<b>Clientes por Parceiro</b>', x=parceiro_counts.index, y=parceiro_counts.values, marker_color='#19C78A', text=parceiro_counts.values, textposition='auto', textfont=dict(size=20, color='black', family='Roboto'))
])

fig2.update_layout(
    title={'text': 'CFs por Parceiro', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    height=600,
    font=dict(size=22, color='black', family='Roboto'),
    plot_bgcolor='white',
    legend=dict(
        font=dict(size=20)
    ),
    xaxis=dict(
        tickangle=-45,
        title='Parceiro',
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=18, color='black', family='Roboto')
    ),
    yaxis=dict(
        title='Clientes Fidelizados',
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=18, color='black', family='Roboto'),
        dtick=10
    ),
    bargap=0.1,  # Diminuir o espaçamento entre as barras
    paper_bgcolor='white'
)

# Criar a visualização para Comparação entre Canais de Pesquisa
canal_counts = data['Canal'].value_counts()

fig3 = go.Figure(data=[
    go.Bar(name='<b>CFs por Canal</b>', x=canal_counts.index, y=canal_counts.values, marker_color='#19C78A', text=canal_counts.values, textposition='auto', textfont=dict(size=20, color='black', family='Roboto'), width=0.5)
])

fig3.update_layout(
    title={'text': 'CFs por Canal', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
    width=800,
    height=600,
    font=dict(size=16, color='black', family='Roboto'),
    plot_bgcolor='white',
    legend=dict(
        font=dict(size=20)
    ),
    xaxis=dict(
        title='Canal de Pesquisa',
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=16, color='black', family='Roboto')
    ),
    yaxis=dict(
        title='Clientes Fidelizados',
        titlefont=dict(size=20, color='black', family='Roboto'),
        tickfont=dict(size=16, color='black', family='Roboto')
    ),
    paper_bgcolor='white'
)

# Calcular a contagem de CFs por Clube
clube_counts = data['Clube'].value_counts()

# Manter os 5 principais clubes e agrupar os restantes em "Outros"
top_clubes = clube_counts.nlargest(4)
outros_clubes = clube_counts.iloc[4:].sum()
top_clubes['Outros'] = outros_clubes

# Definir cores específicas para os principais clubes
colors = {
    'Clínica SiM': '#8b45dc',
    'Sócio Vozão': '#222',
    'Mov': '#EDFF00',
    'O Povo': '#F3951A',
    'Outros': 'grey'
}

# Para os outros clubes, usaremos uma paleta de cores suave
default_colors = ['#636EFA', '#EF553B', '#00CC96', '#FFA15A']

# Garantir que todas as cores estejam na mesma lista
final_colors = [colors.get(label, default_colors[i % len(default_colors)]) for i, label in enumerate(top_clubes.index)]

# Criar a visualização de pizza para CFs por Clube
fig4 = go.Figure(data=[
    go.Pie(
        labels=[f'<b>{val}</b>' for val in top_clubes.index],
        values=top_clubes.values, 
        textinfo='label+percent', 
        insidetextorientation='radial',
        text=[f'<b>{val}</b>' for val in top_clubes.values],
        hoverinfo='label+value+percent',
        marker=dict(
            colors=final_colors,
            line=dict(color='#FFFFFF', width=2)
        )
    )
])

fig4.update_layout(
    title={
        'text': 'CFs por Clube', 
        'x': 0.5, 
        'xanchor': 'center', 
        'font': {
            'size': 24, 
            'color': 'black', 
            'family': 'Roboto'
        }
    },
    font=dict(size=16, color='black', family='Roboto'),
    paper_bgcolor='white',
    showlegend=True,
    legend=dict(
        font=dict(size=16),
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    )
)

# Criar a visualização
fig5 = go.Figure()

# Adicionar trace para Total Q3
fig5.add_trace(go.Scatter(
    x=agg_data['Semana'], y=agg_data['Total CFs'], mode='lines+markers+text',
    name='Total Q3', text=agg_data['Total CFs'], textposition='top center',
    line=dict(color='blue', width=2),
    marker=dict(size=10, symbol='diamond', color='blue'),
    offsetgroup=1
))

# Adicionar trace para Novos CFs
fig5.add_trace(go.Bar(
    x=agg_data['Semana'], y=agg_data['Novos CFs'], name='Novos CFs', text=agg_data['Novos CFs'],
    textposition='outside', marker_color='orange',
    offsetgroup=2
))

# Adicionar trace para Meta Q3
fig5.add_trace(go.Scatter(
    x=agg_data['Semana'], y=agg_data['Meta Q3'], mode='lines',
    name='Meta Q3', line=dict(color='red', dash='dash'), marker=dict(size=0)
))

# Atualizar layout do gráfico
fig5.update_layout(
    title='Novos CFs e Total Q3',
    xaxis_title='Semana',
    yaxis_title='Número de CFs',
    barmode='group',
    template='plotly_white',
    bargroupgap=0.1,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    height=600,
    xaxis=dict(
        tickangle=-45,
        tickmode='array',
        tickvals=agg_data['Data_Inicio_Semana'],
        ticktext=agg_data['Data_Inicio_Semana']
    )
)

# Ajustar a posição do texto nos traces individuais
for trace in fig5.data:
    if trace.name == 'Total Q3':
        trace.textposition = 'top center'
    elif trace.name == 'Novos CFs':
        trace.textposition = 'outside'

# Copiar os dados
data_aux = data.copy()
# Remover pontos (separador de milhares) e substituir vírgulas por pontos (separador decimal)
data_aux['Valor Economizado'] = data_aux['Valor Economizado'].apply(lambda x: str(x).replace('.', ''))
data_aux['Valor Economizado'] = data_aux['Valor Economizado'].apply(lambda x: str(x).replace(',', '.'))

# Converter valores não convertíveis para NaN
data_aux['Valor Economizado'] = pd.to_numeric(data_aux['Valor Economizado'], errors='coerce')

# Filtrar valores diferentes de 0.00 (opcional, dependendo da lógica desejada)

# Converter a coluna para float
data_aux['Valor Economizado'] = data_aux['Valor Economizado'].astype(float)

data_aux = data_aux[data_aux['Valor Economizado'] != 0.00]

# Verificar se há dados após o processamento
if data_aux.empty:
    logger.warning("Nenhum dado disponível após o processamento.")

fig6 = px.scatter(
    data_frame=data_aux,
    x='Satisfação',
    y='Valor Economizado',
    color='Satisfação',
    title='Distribuição dos Valores Economizados e Nível de Satisfação',
    labels={
        'Valor Economizado': 'Valor Economizado (R$)',
        'Satisfação': 'Nível de Satisfação'
    },
    template='plotly_white'
)

# Atualizar layout do gráfico
fig6.update_layout(
    height=600,
    xaxis_title='Nível de Satisfação',
    yaxis_title='Valor Economizado (R$)',
    legend_title='Nível de Satisfação',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Criar o gráfico de violino para a distribuição dos valores economizados por nível de satisfação
fig7 = px.violin(
    data_frame=data_aux,
    y='Valor Economizado',
    x='Satisfação',
    color='Satisfação',
    box=True,  # Adicionar box plot dentro do gráfico de violino
    points="all",  # Mostrar todos os pontos
    title='Distribuição dos Valores Economizados por Nível de Satisfação',
    labels={
        'Valor Economizado': 'Valor Economizado (R$)',
        'Satisfação': 'Nível de Satisfação'
    },
    template='plotly_white'
)

# Atualizar layout do gráfico
fig7.update_layout(
    height=600,
    xaxis_title='Nível de Satisfação',
    yaxis_title='Valor Economizado (R$)',
    legend_title='Nível de Satisfação',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Filtrar os dados para cada nível de satisfação
data_relevante = data_aux[data_aux['Satisfação'] == 'Relevante']
data_muito_relevante = data_aux[data_aux['Satisfação'] == 'Muito Relevante']

estatisticas_relevante = data_relevante['Valor Economizado'].describe(percentiles=[.25, .5, .75]).to_dict()
estatisticas_relevante['mean'] = data_relevante['Valor Economizado'].mean()
estatisticas_relevante['count'] = data_relevante['Valor Economizado'].count()

estatisticas_muito_relevante = data_muito_relevante['Valor Economizado'].describe(percentiles=[.25, .5, .75]).to_dict()
estatisticas_muito_relevante['mean'] = data_muito_relevante['Valor Economizado'].mean()
estatisticas_muito_relevante['count'] = data_muito_relevante['Valor Economizado'].count()

# Criar gráficos de violino
fig_relevante = px.violin(
    data_frame=data_relevante,
    y='Valor Economizado',
    x='Satisfação',
    color='Satisfação',
    box=True,
    points="all",
    title='Valores Economizados X Relevância',
    labels={
        'Valor Economizado': 'Valor Economizado (R$)',
        'Satisfação': 'Nível de Satisfação'
    },
    template='plotly_white'
)

fig_muito_relevante = px.violin(
    data_frame=data_muito_relevante,
    y='Valor Economizado',
    x='Satisfação',
    color='Satisfação',
    box=True,
    points="all",
    title='Valores Economizados X Relevância',
    labels={
        'Valor Economizado': 'Valor Economizado (R$)',
        'Satisfação': 'Nível de Satisfação'
    },
    template='plotly_white',
    color_discrete_map={'Muito Relevante': '#19C78A'}
)

# Ajustar layout e fontes dos gráficos
def ajustar_layout(fig):
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title_font=dict(size=18, family='Roboto'),
        yaxis_title_font=dict(size=18, family='Roboto'),
        title_font=dict(size=22, family='Roboto'),
        legend_font=dict(size=16, family='Roboto'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_yaxes(rangemode="tozero")  # Ajustar o eixo y para iniciar em zero
    fig.update_traces(marker=dict(size=10), selector=dict(type='violin'))


fig_relevante.update_yaxes(range=[0, data_relevante['Valor Economizado'].max() * 1.1])  # Ajustar o eixo y para iniciar em zero
fig_muito_relevante.update_yaxes(range=[0, data_muito_relevante['Valor Economizado'].max() * 1.1])

ajustar_layout(fig_relevante)
ajustar_layout(fig_muito_relevante)

data['Valor Economizado'] = data['Valor Economizado'].str.replace(',', '.').astype(float).round(2)


def criar_tabela_interativa(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filterable=True, sortable=True, editable=False, resizable=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_side_bar(filters_panel=True, columns_panel=True)
    gb.configure_selection('single', use_checkbox=True)
    
    # Configurar todas as colunas para serem filtráveis e ordenáveis
    for col in df.columns:
        gb.configure_column(col, filter='agTextColumnFilter', sortable=True)
    
    grid_options = gb.build()
    
    AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=600,
        width='100%',
        theme=AgGridTheme.STREAMLIT
    )

def criar_histograma(df):
    bins = [0, 20, 50, 150, 400, float('inf')]
    labels = ['Até 20 reais', 'Entre 20 e 50 reais', 'Entre 50 e 150 reais', 'Entre 150 e 400 reais', 'Mais de 400 reais']
    df['Faixa de Valor'] = pd.cut(df['Valor Economizado'], bins=bins, labels=labels, include_lowest=True)

    hist_data = df['Faixa de Valor'].value_counts().reset_index()
    hist_data.columns = ['Faixa de Valor', 'Contagem']
    hist_data = hist_data.sort_values(by='Contagem', ascending=False)

    fig = px.bar(
        hist_data,
        x='Faixa de Valor',
        y='Contagem',
        title='Valores Economizados por Faixa',
        labels={'Faixa de Valor': 'Faixa de Valor Economizado (R$)', 'Contagem': 'Quantidade'},
        text='Contagem',
        template='plotly_white'
    )
    
    fig.update_layout(
        height=600,
        font=dict(size=14, family='Roboto', color='black'),
        xaxis=dict(
            title=dict(text='<b>Faixa de Valor Economizado (R$)</b>', font=dict(size=18, family='Roboto', color='black')),
            tickfont=dict(size=14, family='Roboto', color='black')
        ),
        yaxis=dict(
            title=dict(text='<b>Contagem</b>', font=dict(size=18, family='Roboto', color='black')),
            tickfont=dict(size=14, family='Roboto', color='black')
        )
    )
    
    fig.update_traces(
        textposition='outside',
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=18, family='Roboto', color='black')
    )
    
    return fig

#Função para criar o gráfico de comparação entre Q2 e Q3
def criar_comparacao_q2_Q3(df_q2, df_Q3):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_q2['Semana'],
        y=df_q2['Total CFs'],
        mode='lines+markers+text',
        name='CFs Q2',
        line=dict(color='#19C78A'),
        marker=dict(color='#19C78A'),
        text=df_q2['Novos CFs'],
        textposition='top center'
    ))

    fig.add_trace(go.Scatter(
        x=df_Q3['Semana'],
        y=df_Q3['Total CFs'],
        mode='lines+markers+text',
        name='  CFs Q3',
        line=dict(color='red'),
        marker=dict(color='red'),
        text=df_Q3['Novos CFs'],
        textposition='top center'
    ))

    fig.update_layout(
        title='CFs q2 X Q3 por Semana',
        xaxis_title='Semana',
        yaxis_title='Número de CFs',
        height=600,
        font=dict(size=14, family='Roboto', color='black'),
        xaxis=dict(
            title=dict(text='<b>Semana</b>', font=dict(size=18, family='Roboto', color='black')),
            tickfont=dict(size=14, family='Roboto', color='black'),
            tickmode='linear'
        ),
        yaxis=dict(
            title=dict(text='<b>Número de CFs</b>', font=dict(size=18, family='Roboto', color='black')),
            tickfont=dict(size=14, family='Roboto', color='black')
        )
    )

    return fig


# Quantidade geral de clientes fidelizados
total_fidelizados = data_estaticos.shape[0] + 4178
logger.info(data_estaticos.shape[0])
logger.info('total_fidelizados')
logger.info(total_fidelizados)

meta_anual = 5000
total_cfs_q3 = data_estaticos.shape[0]
total_cfs_2024 = total_fidelizados - 2851  # Total de clientes fidelizados em 2024

# Calcular a diferença entre as semanas
diferenca_semanal = resultados_semana_atual - resultados_semana_anterior

# Calcular quantos faltam para a meta
faltam_para_meta = meta_anual - total_fidelizados

# Calcular porcentagens
porcentagem_meta_anual = (total_fidelizados / meta_anual) * 100
porcentagem_meta_q3 = (total_cfs_q3 / meta_cfs) * 100
porcentagem_meta_2024 = (total_cfs_2024 / meta_anual) * 100
restante_meta_anual = 100 - porcentagem_meta_anual

# Criar painel informativo
fig_total = go.Figure()

# Adicionar total fidelizados
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_fidelizados,
    title={"text": f"<span style='color:#1B0A63;'>Clientes Fidelizados</span><br><span style='font-size:0.9em;color:#19C78A'>{porcentagem_meta_anual:.2f}% da meta anual</span>"},
    domain={'row': 0, 'column': 0},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar faltam para meta
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=faltam_para_meta,
    title={"text": f"<span style='color:#1B0A63;'>Faltam para a Meta Anual</span><br><span style='font-size:0.9em;color:#19C78A'>{restante_meta_anual:.2f}% da meta em aberto</span>"},
    domain={'row': 0, 'column': 1},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total CFs Q3
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_q3,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs no Q3</span><br><span style='font-size:0.9em;color:#19C78A'>{porcentagem_meta_q3:.2f}% da meta do trimestre</span>"},
    domain={'row': 0, 'column': 2},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total de clientes fidelizados em 2024
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_2024,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs em 2024</span><br>"},
    domain={'row': 0, 'column': 3},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total de clientes fidelizados na semana
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=diferenca_semanal,
    title={"text": f"<span style='color:#1B0A63;'>Novos CFs</span><br><span style='font-size:0.9em;color:#19C78A'>em relação à semana anterior</span>"},
    domain={'row': 0, 'column': 4},
    number={"font": {"size": 70, "color": "#1B0A63"}}
))

# Atualizar layout
fig_total.update_layout(
    margin=dict(t=0, b=0),
    grid={'rows': 1, 'columns': 5, 'pattern': "independent"},
    template={'data': {'indicator': [{
        'title': {'text': "<b>Dias</b>"},
        'mode': "number+delta+gauge"}]}}
)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Resumo", "Resultados Q3", "CFs por Clube", "CFs por Parceiros", "CFs por Canal", "Valores Economizados", "Tabela Interativa"]
)

with tab1:
    st.plotly_chart(fig_total, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig4, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig2, use_container_width=True)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig3, use_container_width=True)

with tab6:
   col1, col2, col3 = st.columns(3)
   with col1:
    st.plotly_chart(fig_relevante, use_container_width=True)
    st.subheader('Estatísticas - Relevante')
    st.write(f"Quantidade de respostas: {estatisticas_relevante['count']}")
    st.write(f"Valor Mínimo: R$ {estatisticas_relevante['min']:.2f}")
    st.write(f"Quartil 1: R$ {estatisticas_relevante['25%']:.2f}")
    st.write(f"Mediana: R$ {estatisticas_relevante['50%']:.2f}")
    st.write(f"Quartil 3: R$ {estatisticas_relevante['75%']:.2f}")
    st.write(f"Valor Máximo: R$ {estatisticas_relevante['max']:.2f}")
    st.write(f"Média: R$ {estatisticas_relevante['mean']:.2f}")
   with col2:
    st.plotly_chart(fig_muito_relevante, use_container_width=True)
    st.subheader('Estatísticas - Muito Relevante')
    st.write(f"Quantidade de respostas: {estatisticas_muito_relevante['count']}")
    st.write(f"Valor Mínimo: R$ {estatisticas_muito_relevante['min']:.2f}")
    st.write(f"Quartil 1: R$ {estatisticas_muito_relevante['25%']:.2f}")
    st.write(f"Mediana: R$ {estatisticas_muito_relevante['50%']:.2f}")
    st.write(f"Quartil 3: R$ {estatisticas_muito_relevante['75%']:.2f}")
    st.write(f"Valor Máximo: R$ {estatisticas_muito_relevante['max']:.2f}")
    st.write(f"Média: R$ {estatisticas_muito_relevante['mean']:.2f}")    
   with col3:
    st.plotly_chart(criar_histograma(data), use_container_width=True) 

with tab7:
    criar_tabela_interativa(data)


#     st.header("Comparação de CFs entre Q2 e Q3")
#     comparacao_fig = criar_comparacao_q2_Q3(data_q2, data_Q3)
#     st.plotly_chart(comparacao_fig, use_container_width=True)