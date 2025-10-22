import os
import pandas as pd
import plotly.express as px

# Load and process data from all CSV files
data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

df_list = []
for file in files:
    df = pd.read_csv(os.path.join(data_folder, file), sep=';', decimal=',')
    df['year'] = file.split('_')[1].split('.')[0]  # Extract year from filename
    df_list.append(df)

# Concatenate all dataframes
data = pd.concat(df_list, ignore_index=True)

# Convert and clean the value column
data['valor_avaliacao'] = data['valor_avaliacao'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

# Group by neighborhood and find the most expensive property
most_expensive = data.groupby('bairro')['valor_avaliacao'].max().reset_index()
most_expensive = most_expensive.sort_values('valor_avaliacao', ascending=True)

# Create the bar plot using plotly
fig = px.bar(most_expensive, 
             x='valor_avaliacao', 
             y='bairro',
             orientation='h',  # horizontal bars
             title='Most Expensive Properties by Neighborhood',
             labels={'valor_avaliacao': 'Property Value (R$)',
                    'bairro': 'Neighborhood'})

# Update layout for better visualization
fig.update_layout(
    title_x=0.5,  # Center the title
    height=800,   # Increase height to accommodate all neighborhoods
    xaxis_title="Property Value (R$)",
    yaxis_title="Neighborhood",
    font=dict(size=12)
)

# Format the x-axis to show values in millions
fig.update_layout(
    xaxis=dict(
        tickformat=",.0f",
        ticksuffix=" R$"
    )
)

# Save the plot as HTML file (interactive)
fig.write_html("charts/most_expensive_by_neighborhood.html")

# Save the plot as a static image
fig.write_image("charts/most_expensive_by_neighborhood.png")

print("Graphs have been saved in the charts folder as:")
print("1. most_expensive_by_neighborhood.html (interactive)")
print("2. most_expensive_by_neighborhood.png (static image)")
