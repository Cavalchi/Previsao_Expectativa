import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np


def country_to_continent():
    return {
        'Asia':'Asia',
        'Africa':'Africa',
        'Oceania' :'Oceania',
        'Northern America': 'Northern America',
        'South America': 'South America',
        'Latin America and the Caribbean':'Latin America and the Caribbean',
        'Europe':'Europe',
        'Brazil': 'Brazil'
    }


df = pd.read_csv('life-expectancy.csv')
df = df[df['Year'] >= 1900]


df['Continente'] = df['Entity'].map(country_to_continent())
df = df.dropna(subset=['Continente'])
df = df.rename(columns={'Year': 'Ano'})
df_continent = df.groupby(['Continente', 'Ano'])['Period life expectancy at birth - Sex: total - Age: 0'].mean().reset_index()

fig = px.line(
    df_continent[df_continent['Ano'] <= 2023],
    x='Ano',
    y='Period life expectancy at birth - Sex: total - Age: 0',
    color='Continente',
    hover_data=['Continente'],
    title="Expectativa de vida média por continente (1900-2023)",
    markers=True
)

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(df_continent['Ano'].min(), 2024, 10)
    ),
    width=1200,
    height=800
)

fig.show()

predictions = []
for continente in df_continent['Continente'].unique():
    continente_data = df_continent[df_continent['Continente'] == continente]
    X = continente_data['Ano'].values.reshape(-1, 1)
    y = continente_data['Period life expectancy at birth - Sex: total - Age: 0'].values

    if len(X) > 1:  
        model = LinearRegression()
        model.fit(X, y)

        future_years = np.arange(2024, 2035).reshape(-1, 1)
        future_predictions = model.predict(future_years)

        for year, pred in zip(range(2024, 2035), future_predictions):
            predictions.append([continente, year, pred])


predictions_df = pd.DataFrame(
    predictions, columns=['Continente', 'Ano', 'Previsão de Expectativa de Vida']
)


fig2 = px.line(
    predictions_df,
    x='Ano',
    y='Previsão de Expectativa de Vida',
    color='Continente',
    hover_data=['Continente'],
    markers=True,
    title="Previsões de expectativa de vida média por continente (2024-2034)"
)

fig2.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(2024, 2035, 1)
    ),
    width=1200,
    height=800
)

fig2.show()
