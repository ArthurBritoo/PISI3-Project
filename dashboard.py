import os
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html

# Load all CSV files from the data folder
data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

df_list = []
for file in files:
    df = pd.read_csv(os.path.join(data_folder, file))
    df['year'] = file.split('_')[1].split('.')[0]  # Extract year from filename
    df_list.append(df)

# Concatenate all dataframes
data = pd.concat(df_list, ignore_index=True)

# Example: Assume columns 'Valor', 'Ano', 'Bairro' exist. Adjust as needed.
# If columns differ, update below accordingly.

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('ITBI Dashboard'),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in sorted(data['year'].unique())],
        value=sorted(data['year'].unique())[0],
        clearable=False
    ),
    dcc.Graph(id='valor-by-bairro'),
])

@app.callback(
    dash.dependencies.Output('valor-by-bairro', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_graph(selected_year):
    filtered = data[data['year'] == selected_year]
    fig = px.bar(filtered, x='Bairro', y='Valor', title=f'Valor por Bairro - {selected_year}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
