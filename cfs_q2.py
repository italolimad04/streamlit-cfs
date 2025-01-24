import pandas as pd
pd.options.mode.chained_assignment = None  
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
from pytz import timezone
import requests
import time

# import logging

# # Configurar o logger
# logging.basicConfig(
#     format='%(asctime)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("app.log"),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# Opções
st.set_page_config(layout="wide")

st.title("Painel de Clientes Fidelizados no Q1 2025")

load_dotenv()

# def carregar_dados_do_google_sheets():
#     scopes = [
#         "https://www.googleapis.com/auth/spreadsheets.readonly",
#         "https://www.googleapis.com/auth/drive.readonly"
#     ]
#     creds_json = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
#     creds_dict = json.loads(creds_json)
#     credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
#     client = gs.authorize(credentials)
#     sheet = client.open_by_key(st.secrets["SHEET_KEY"]).sheet1
#     data = sheet.get_all_values()  # Tenta obter todos os valores como strings
#     headers = data.pop(0)  # Remove o cabeçalho da lista de dados
#     return pd.DataFrame(data, columns=headers, dtype=str)

# # Carregar os dados
# data = carregar_dados_do_google_sheets()

# load_dotenv()
# token = os.getenv('TOKEN_ADMIN')
# headers = {
#     'Authorization': f'Bearer {token}',
#     'Content-Type': 'application/json'
# }

TTL = 500
def invalidate_cache():
  fetch_data.clear()

firstDayOfQuarter = '2025-01-01'
lastDayOfQuarter = '2025-03-30'

@st.cache_data(ttl=TTL) 
def fetch_data():
    response = requests.get(url=f"https://new-api.urbis.cc/communication/fidelized-clients-by-quarter?initialDate={firstDayOfQuarter}&finalDate={lastDayOfQuarter}").json()
    fidelizedClientsData = response['data']['fidelizedClientsData']

    print('fidelizedClientsData: ', fidelizedClientsData)


    df_fidelized_clients_by_survey = pd.DataFrame(data=fidelizedClientsData)

    df_fidelized_clients_by_survey['Valor Economizado'] = df_fidelized_clients_by_survey['Valor Economizado'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

    # Converter para float e tratar valores inválidos como NaN
    df_fidelized_clients_by_survey['Valor Economizado'] = pd.to_numeric(df_fidelized_clients_by_survey['Valor Economizado'], errors='coerce')

    # Preencher NaN com 0
    df_fidelized_clients_by_survey['Valor Economizado'] = df_fidelized_clients_by_survey['Valor Economizado'].fillna(0)

    # Arredondar para duas casas decimais
    df_fidelized_clients_by_survey['Valor Economizado'] = df_fidelized_clients_by_survey['Valor Economizado'].round(2)

    print(df_fidelized_clients_by_survey['Valor Economizado'])

    return df_fidelized_clients_by_survey, time.time()

df_fidelized_clients_by_survey, last_updated = fetch_data()


data = pd.DataFrame()
data = pd.concat([data, df_fidelized_clients_by_survey])

data['Data'] = pd.to_datetime(data['Data'], errors='coerce')

print(data.columns)

# Adequando valor dos dados no dataframe consolidado
data['Canal'].loc[data['Canal'] == 'email'] = 'Drogarias'
data.loc[data['Pesquisa'].str.contains('Muito relevante', case=True) | data['Pesquisa'].str.contains('Relevante', case=True), 'Canal'] = 'E-mail Individual'
data['Canal'].loc[data['Canal'] == 'club'] = 'Clube'

print(data['Canal'].value_counts())

data['Parceiro'].loc[data['Parceiro'] == 'Farmácias Pague Menos'] = 'Pague Menos'
data['Clube'].loc[data['Clube'] == 'Clínica SiM+'] = 'Clínica SiM'
data['Clube'].loc[data['Clube'] == 'Club de Vantagens | Sócio Vozão'] = 'Sócio Vozão'
data['Clube'].loc[data['Clube'] == 'MOV Fibra'] = 'Mov Telecom'
data['Clube'].loc[data['Clube'] == 'Clube O Povo'] = 'O Povo'
data['Satisfação'].loc[data['Satisfação'] == 'Muito relevante'] = 'Muito Relevante'

local_tz = timezone('America/Sao_Paulo')  # Ajuste conforme necessário
utc_tz = timezone('UTC')

# Função para calcular a semana fiscal
def calcular_semana_fiscal(data, start_date):
    delta = data - start_date
    return delta.days // 7 + 1

start_date_quarter = datetime(2025, 1, 1, tzinfo=local_tz)
data['Data'] = pd.to_datetime(data['Data']).dt.tz_localize(local_tz).dt.tz_convert(utc_tz)
data['Semana'] = data['Data'].apply(lambda x: calcular_semana_fiscal(x, start_date_quarter))
data_atual = datetime.now(local_tz).astimezone(utc_tz)
data_max = data['Data'].max()

if data_max < data_atual:
    additional_weeks = pd.date_range(start=data_max + timedelta(days=7), end=data_atual, freq='7D', tz=utc_tz)
    additional_data = pd.DataFrame({'Data': additional_weeks, 'Semana': additional_weeks.map(lambda x: calcular_semana_fiscal(x, start_date_quarter))})
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
agg_data['Meta Trimestre'] = 1000
agg_data['Total_Geral'] = agg_data['Total CFs'] + 5749  # Ajuste conforme necessário

# Adicionar coluna 'Data_Inicio_Semana'
agg_data['Data_Inicio_Semana'] = agg_data['Semana'].apply(lambda x: start_date_quarter + timedelta(weeks=x-1))

# Verificar e converter 'Data_Inicio_Semana' para datetime
agg_data['Data_Inicio_Semana'] = pd.to_datetime(agg_data['Data_Inicio_Semana'], errors='coerce')

# Reorganizar as colunas
agg_data = agg_data[['Data_Inicio_Semana', 'Semana', 'Novos CFs', 'Total CFs', 'Meta Trimestre', 'Total_Geral']]

# Identificar a semana atual e a semana anterior
semana_atual = calcular_semana_fiscal(data_atual, start_date_quarter)

# logger.info(agg_data['Semana'].head())
# logger.info(f'Semana Atual: {semana_atual}')

semana_anterior = semana_atual - 1
# logger.info(f'Semana Anterior: {semana_anterior}')
# Pegar os resultados para esta semana e a anterior com verificações de erro
try:
    resultados_semana_atual = agg_data[agg_data['Semana'] == semana_atual]['Novos CFs'].values[0]
except IndexError:
    resultados_semana_atual = 0  # Definir um valor padrão ou tomar outra ação

try:
    resultados_semana_anterior = agg_data[agg_data['Semana'] == semana_anterior]['Novos CFs'].values[0]
except IndexError:
    resultados_semana_anterior = 0  # Definir um valor padrão ou tomar outra ação

# logger.info(f'Resultados Semana Anterior: {resultados_semana_anterior}')
# logger.info(f'Resultados Semana Atual: {resultados_semana_atual}')

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
meta_cfs_tri = 1000

# Criar a visualização
fig = go.Figure(data=[
    go.Bar(name='<b>Total CFs<b>', x=['Clientes Fidelizados'], y=[total_cfs], marker_color='#FFA726', text=[f'<b>{total_cfs}'], textposition='auto', textfont=dict(size=20, color='black', family='Roboto')),
     go.Bar(name='<b>Meta<b>', x=['Clientes Fidelizados'], y=[meta_cfs_tri], marker_color='#42A5F5', text=[f'<b>{meta_cfs_tri}'], textposition='auto', textfont=dict(size=20, color='black', family='Roboto'))
])

# Atualizar layout do gráfico
fig.update_layout(
    title={'text': 'Resultado Obtido', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24, 'color': 'black', 'family': 'Roboto'}},
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
        dtick=50
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

top_clubes = clube_counts.nlargest(6)

colors = {
    'Clínica SiM': '#6A0DAD',  # Roxo vibrante
    'Sócio Vozão': '#333333',  # Preto suave
    'Mov Telecom': '#FFD700',  # Amarelo dourado
    'O Povo': '#FF4500',       # Laranja vibrante
    'Clube Odontoart': '#1E90FF',  # Azul forte
    'Tigrão de Vantagens': '#FF6347'  # Vermelho suave
}

# Garantir que todas as cores estejam na mesma lista
final_colors = [colors.get(label, '#B0C4DE') for label in top_clubes.index]

# Criar a visualização de pizza para CFs por Clube
fig4 = go.Figure(data=[
    go.Pie(
        labels=top_clubes.index,
        values=top_clubes.values,
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+value+percent',
        marker=dict(
            colors=final_colors,
            line=dict(color='#FFFFFF', width=1)
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
    font=dict(size=12, color='black', family='Roboto'),
    paper_bgcolor='white',
    margin=dict(t=100, b=100, l=50, r=50),  # Ajustando as margens
    showlegend=True,
    legend=dict(
        font=dict(size=14),
        orientation="h",
        yanchor="bottom",
        y=-0.4,  # Diminuindo para dar espaço ao gráfico
        xanchor="center",
        x=0.5
    )
)

# Criar a visualização
fig5 = go.Figure()

# logger.info('Semanas')
# logger.info(agg_data['Semana'])

# logger.info("agg_data['Total CFs']")
# logger.info(agg_data['Total CFs'])


# Adicionar trace para Total Trimestre
fig5.add_trace(go.Scatter(
    x=agg_data['Semana'], y=agg_data['Total CFs'], mode='lines+markers+text',
    name='Total Trimestre', text=agg_data['Total CFs'], textposition='top center',
    line=dict(color='blue', width=2),
    marker=dict(size=10, symbol='diamond', color='blue'),
    legendgroup=1
))

# Adicionar trace para Novos CFs
fig5.add_trace(go.Bar(
    x=agg_data['Semana'], y=agg_data['Novos CFs'], name='Novos CFs', text=agg_data['Novos CFs'],
    textposition='outside', marker_color='orange',
    legendgroup=2
))

# Adicionar trace para Meta Trimestre
fig5.add_trace(go.Scatter(
    x=agg_data['Semana'], y=agg_data['Meta Trimestre'], mode='lines',
    name='Meta Trimestre', line=dict(color='red', dash='dash'), marker=dict(size=0)
))

# Atualizar layout do gráfico
fig5.update_layout(
    title='Novos CFs Vs Total',
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
    if trace.name == 'Total Trimestre':
        trace.textposition = 'top center'
    elif trace.name == 'Novos CFs':
        trace.textposition = 'outside'

# Copiar os dados
data_aux = data.copy()

# Remover pontos (separador de milhares) e substituir vírgulas por pontos (separador decimal)
# data_aux['Valor Economizado'] = data_aux['Valor Economizado'].apply(lambda x: str(x).replace('.', ''))
# data_aux['Valor Economizado'] = data_aux['Valor Economizado'].apply(lambda x: str(x).replace(',', '.'))

# # Converter valores não convertíveis para NaN
# data_aux['Valor Economizado'] = pd.to_numeric(data_aux['Valor Economizado'], errors='coerce')

# Filtrar valores diferentes de 0.00 (opcional, dependendo da lógica desejada)

# Converter a coluna para float
#data_aux['Valor Economizado'] = data_aux['Valor Economizado'].astype(float)

data_aux = data_aux[data_aux['Valor Economizado'] != 0.00]

# Verificar se há dados após o processamento
#if data_aux.empty:
    #logger.warning("Nenhum dado disponível após o processamento.")

print('data_aux: ', data_aux.tail())
print('data_aux: ', data_aux['Valor Economizado'])

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

print("satisfação")
print(data_aux.columns)

# Filtrar os dados para cada nível de satisfação
data_relevante = data_aux[data_aux['Satisfação'].isin(['Relevante'])]
data_muito_relevante = data_aux[data_aux['Satisfação'].isin (['Muito Relevante', 'Muito relevante'])]

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

#data['Valor Economizado'] = data['Valor Economizado'].str.replace(',', '.').astype(float).round(2)


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

#Função para criar o gráfico de comparação entre Q2 e Trimestre
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
        name='  CFs Trimestre',
        line=dict(color='red'),
        marker=dict(color='red'),
        text=df_Q3['Novos CFs'],
        textposition='top center'
    ))

    fig.update_layout(
        title='CFs q2 X Trimestre por Semana',
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


# Definindo Valores
total_cfs_fim_2024=5749


meta_anual = 20000
meta_dentro_do_ano= 20000 - total_cfs_fim_2024
total_cfs_quarter = data.shape[0]

total_fidelizados = data.shape[0] + total_cfs_fim_2024 # Ajustar este número 
total_cfs_2025 = total_fidelizados - total_cfs_fim_2024  # Total de clientes fidelizados em 2023

# Calcular a diferença entre as semanas
diferenca_semanal = resultados_semana_atual - resultados_semana_anterior


faltam_para_meta = meta_anual - total_fidelizados
if faltam_para_meta < 0:
    faltam_para_meta = 0

# Calcular porcentagens
porcentagem_meta_anual = (total_cfs_2025/ (meta_anual - total_cfs_fim_2024)) * 100
porcentagem_meta_quarter = (total_cfs_quarter / meta_cfs_tri) * 100
porcentagem_meta_2024 = (total_cfs_2025 / meta_anual) * 100
restante_meta_anual = 100 - porcentagem_meta_anual
resultado_q4_2024=718
comparacao_trimestre_anterior = (total_cfs_quarter/resultado_q4_2024)*100
if restante_meta_anual < 0:
    restante_meta_anual = 0

fig_total = go.Figure()

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_fidelizados,
    title={"text": f"<span style='color:#1B0A63;'>Clientes Fidelizados</span> <br><span style='font-size:0.9em;color:#19C78A'>Meta: {meta_anual:.0f} CFs acumulados ao final de 2025</span>"},
    domain={'row': 0, 'column': 0},
    number={"font": {"size": 70, "color": "#1B0A63"}, "valueformat": "d"},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=faltam_para_meta,
    title={"text": f"<span style='color:#1B0A63;'>Restam para a Meta Anual <br><span style='font-size:0.9em;color:#19C78A'>Meta: {meta_dentro_do_ano:.0f} novos CFs em 2025</span>"},
    domain={'row': 0, 'column': 1},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total CFs Trimestre
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_quarter,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs no Q1 2025 <br><span style='font-size:0.9em;color:#19C78A'>Meta: {total_cfs_quarter:.0f} novos CFs no Q1 25</span>"},
    domain={'row': 0, 'column': 2},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total de clientes fidelizados em 2024
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=total_cfs_2025,
    title={"text": f"<span style='color:#1B0A63;'>Total CFs em 2025</span><br> <span style='font-size:0.9em;color:#19C78A'>{comparacao_trimestre_anterior:.2f}% vs. Q4 24</span>"},
    domain={'row': 0, 'column': 3},
    number={"font": {"size": 70, "color": "#1B0A63"}},
    delta={'position': "bottom", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
))

# Adicionar total de clientes fidelizados na semana
fig_total.add_trace(go.Indicator(
    mode="number+delta",
    value=resultados_semana_anterior,
    title={"text": f"<span style='color:#1B0A63;'>Novos CFs</span>"},
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

def criar_grafico_trimestral(total_cfs_quarter):
    # Extrair os trimestres e os valores
    trimestres = list(total_cfs_quarter.keys())
    resultados = list(total_cfs_quarter.values())
    meta_trimestre = 722  # Meta para cada trimestre

    # Criar a figura
    fig = go.Figure()

    # Adicionar barras para os resultados trimestrais
    fig.add_trace(go.Bar(
        x=trimestres,
        y=resultados,
        name="Resultados",
        marker_color="#3AB78B",
        text=resultados,
        textposition="outside"
    ))

    # Adicionar linha de meta
    fig.add_trace(go.Scatter(
        x=trimestres,
        y=[meta_trimestre] * len(trimestres),
        mode="lines",
        name="Média",
        line=dict(color="red", dash="dash", width=2),
        hovertemplate="Meta: %{y}<extra></extra>"
    ))

    # Configurar layout
    fig.update_layout(
        title="Comparação de Resultados por Trimestre - 2025",
        xaxis_title="Trimestre",
        yaxis_title="Número de Clientes Fidelizados",
        template="plotly_white",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig

# Exemplo de uso na aba tab2
total_cfs_quarter = {
    "Q1 24": 796,
    "Q2 24": 531,
    "Q3 24": 853,
    "Q4 24": 718,
    "Q1 25": total_cfs_quarter
}

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Resumo",  "Resultados por Trimestre", "Resultados Q1 25", "CFs por Clube", "CFs por Parceiros", "CFs por Canal", "Valores Economizados", "Tabela Interativa"]
)

with tab1:
    print('total_cfs_quarter: ', total_cfs_quarter)
    #st.balloons()
    st.plotly_chart(fig_total, use_container_width=True)

with tab2:
    fig_total = criar_grafico_trimestral(total_cfs_quarter)
    st.plotly_chart(fig_total, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig4, use_container_width=True)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig2, use_container_width=True)

with tab6:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig3, use_container_width=True)

with tab7:
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

with tab8:
    criar_tabela_interativa(data)


#     st.header("Comparação de CFs entre Q2 e Trimestre")
#     comparacao_fig = criar_comparacao_q2_Q3(data_q2, data_Q3)
#     st.plotly_chart(comparacao_fig, use_container_width=True)