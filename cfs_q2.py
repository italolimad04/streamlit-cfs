import pandas as pd
import requests as reqs
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme, DataReturnMode, GridUpdateMode


# Opções
st.set_page_config(layout="wide")

st.title("Painel de Clientes Fidelizados no Q2")

data = pd.read_excel(
    "Dados/Resultados CFs Q2.xlsx",
    dtype=str,
)

print(data.head())

# Dados fornecidos
data_per_week = {
    'Semana': [
        '26/04/2024', '03/05/2024', '10/05/2024', '17/05/2024', '24/05/2024',
        '31/05/2024', '07/06/2024', '14/06/2024', '21/06/2024', '28/06/2024'
    ],
    'Novos CFs': [5, 172, 16, 6, 70, 122, 65, 64, 9, 4],
    'Total CFs': [5, 177, 191, 197, 267, 389, 454, 518, 527, 531],
    'Meta Q2': [900]*10,
    'Total Geral': [3650, 3822, 3838, 3844, 3914, 4036, 4101, 4165, 4174, 4178]
}

data_per_week_q1 = {
    "Semana": [
        "02/02/2024", "09/02/2024", "16/02/2024", "23/02/2024", "01/03/2024",
        "08/03/2024", "15/03/2024", "22/03/2024", "28/03/2024", "02/04/2024",
        "05/04/2024"
    ],
    "Novos CFs": [6, 3, 0, 125, 34, 96, 218, 110, 15, 6, 2],
    "Total CFs": [149, 152, 152, 277, 311, 407, 625, 771, 786, 792, 794]
}

# Criar DataFrame
data_per_week = pd.DataFrame(data_per_week)
# Converter a coluna 'Semana' para datetime
data_per_week['Semana'] = pd.to_datetime(data_per_week['Semana'], format='%d/%m/%Y')

# Criar dataframe do Q2

data_q1 = pd.DataFrame(data_per_week_q1)
data_q1['Semana'] = pd.to_datetime(data_q1['Semana'], format='%d/%m/%Y')
data_q1 = data_q1.sort_values(by='Semana').reset_index(drop=True)
data_q1['Semana'] = data_q1.index + 1


data_q2 = pd.DataFrame(data_per_week)
data_q2['Semana'] = pd.to_datetime(data_q2['Semana'], format='%d/%m/%Y')
data_q2 = data_q2.sort_values(by='Semana').reset_index(drop=True)
data_q2['Semana'] = data_q2.index + 1


## Tratando dados
data['Parceiro'].loc[data['Parceiro'] == 'Cantinho do Frango - Aldeota'] = 'Cantinho do Frango'
data['Parceiro'].loc[data['Parceiro'] == 'PAGUE MENOS'] = 'Pague Menos'
data['Parceiro'].loc[data['Parceiro'] == 'Farmácias'] = 'Pague Menos'
data['Parceiro'].loc[data['Parceiro'] == 'CENTAURO'] = 'Centauro'
data['Parceiro'].loc[data['Parceiro'] == 'CASAS BAHIA'] = 'Casas Bahia'

# Calcular o número total de CFs
total_cfs = data.shape[0]
meta_cfs = 900

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
        tickfont=dict(size=18, color='black', family='Roboto')
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

# Adicionar trace para Total Q2
fig5.add_trace(go.Scatter(
    x=data_per_week['Semana'], y=data_per_week['Total CFs'], mode='lines+markers+text',
    name='Total Q2', text=data_per_week['Total CFs'], textposition='top center',
    line=dict(color='blue', width=2),
    marker=dict(size=10, symbol='diamond', color='blue'),
    offsetgroup=1
))

# Adicionar trace para Novos CFs
fig5.add_trace(go.Bar(
    x=data_per_week['Semana'], y=data_per_week['Novos CFs'], name='Novos CFs', text=data_per_week['Novos CFs'],
    textposition='outside', marker_color='orange',
    offsetgroup=2
))

# Adicionar trace para Meta Q2
fig5.add_trace(go.Scatter(
    x=data_per_week['Semana'], y=data_per_week['Meta Q2'], mode='lines',
    name='Meta Q2', line=dict(color='red', dash='dash'), marker=dict(size=0)
))

# Atualizar layout do gráfico
fig5.update_layout(
    title='Novos CFs e Total Q2',
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
        tickvals=data_per_week['Semana'],
        ticktext=[d.strftime('%d/%m/%Y') for d in data_per_week['Semana']]
    )
)

# Filtrar dados onde 'Valor Economizado' está preenchido e pode ser convertido para float
data_aux = data.copy()
data_aux['Valor Economizado'] = pd.to_numeric(data_aux['Valor Economizado'], errors='coerce')
data_aux = data_aux.dropna(subset=['Valor Economizado'])

# Criar o scatter plot com valor economizado no eixo y e nível de satisfação no eixo x
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

# Calcular estatísticas descritivas
def calcular_estatisticas(df, coluna):
    estatisticas = df[coluna].describe(percentiles=[.25, .5, .75]).to_dict()
    estatisticas['mean'] = df[coluna].mean()
    estatisticas['count'] = df[coluna].count()
    return estatisticas

estatisticas_relevante = calcular_estatisticas(data_relevante, 'Valor Economizado')
estatisticas_muito_relevante = calcular_estatisticas(data_muito_relevante, 'Valor Economizado')

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
    fig.update_traces(marker=dict(size=10), selector=dict(type='violin'))

ajustar_layout(fig_relevante)
ajustar_layout(fig_muito_relevante)

data['Valor Economizado'] = data['Valor Economizado'].replace('-', '0,00').str.replace(',', '.').astype(float).round(2)

def criar_tabela_interativa(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filterable=True, sortable=True, editable=False, resizable=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
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

# Função para criar o gráfico de comparação entre Q1 e Q2
def criar_comparacao_q1_q2(df_q1, df_q2):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_q1['Semana'],
        y=df_q1['Total CFs'],
        mode='lines+markers+text',
        name='CFs Q1',
        line=dict(color='#19C78A'),
        marker=dict(color='#19C78A'),
        text=df_q1['Novos CFs'],
        textposition='top center'
    ))

    fig.add_trace(go.Scatter(
        x=df_q2['Semana'],
        y=df_q2['Total CFs'],
        mode='lines+markers+text',
        name='  CFs Q2',
        line=dict(color='red'),
        marker=dict(color='red'),
        text=df_q2['Novos CFs'],
        textposition='top center'
    ))

    fig.update_layout(
        title='CFs Q1 X Q2 por Semana',
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

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Sumarização Resultados", "CFs por Clube", "CFs por Parceiros", "CFs por Canal", "Valores Economizados", "Tabela Interativa", "Q1XQ2"]
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig4, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig2, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig3, use_container_width=True)

with tab5:
   col1, col2, col3 = st.columns(3)
   with col1:
    st.plotly_chart(fig_relevante, use_container_width=True)
    st.subheader('Estatísticas - Relevante')
    st.write(f"Quantidade de respostas: {estatisticas_relevante['count']}")
    st.write(f"Valor Mínimo: R$ {estatisticas_relevante['min']:.2f}")
    st.write(f"Q1: R$ {estatisticas_relevante['25%']:.2f}")
    st.write(f"Mediana: R$ {estatisticas_relevante['50%']:.2f}")
    st.write(f"Q3: R$ {estatisticas_relevante['75%']:.2f}")
    st.write(f"Valor Máximo: R$ {estatisticas_relevante['max']:.2f}")
    st.write(f"Média: R$ {estatisticas_relevante['mean']:.2f}")
   with col2:
    st.plotly_chart(fig_muito_relevante, use_container_width=True)
    st.subheader('Estatísticas - Muito Relevante')
    st.write(f"Quantidade de respostas: {estatisticas_muito_relevante['count']}")
    st.write(f"Valor Mínimo: R$ {estatisticas_muito_relevante['min']:.2f}")
    st.write(f"Q1: R$ {estatisticas_muito_relevante['25%']:.2f}")
    st.write(f"Mediana: R$ {estatisticas_muito_relevante['50%']:.2f}")
    st.write(f"Q3: R$ {estatisticas_muito_relevante['75%']:.2f}")
    st.write(f"Valor Máximo: R$ {estatisticas_muito_relevante['max']:.2f}")
    st.write(f"Média: R$ {estatisticas_muito_relevante['mean']:.2f}")    
   with col3:
    st.plotly_chart(criar_histograma(data), use_container_width=True) 

with tab6:
    criar_tabela_interativa(data)

with tab7:
    st.header("Comparação de CFs entre Q1 e Q2")
    comparacao_fig = criar_comparacao_q1_q2(data_q1, data_q2)
    st.plotly_chart(comparacao_fig, use_container_width=True)
    