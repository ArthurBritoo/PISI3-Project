import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t, norm
import altair as alt # Importar Altair para visualizações

st.set_page_config(page_title="ITBI Price Analysis", layout="wide") # Usar layout wide para mais espaço

st.title("Confidence Interval for Price per m² • Boa Viajem ITBI")

# Function to load data from a Parquet file with caching
@st.cache_data
def load_data():
    """
    Loads ITBI data from a Parquet file.
    Caches the data to avoid reloading on every rerun.
    """
    try:
        df = pd.read_parquet("PISI3-Project/data/itbi_2023.parquet")
        return df
    except FileNotFoundError:
        st.error("Error: 'itbi_2023.parquet' not found. Please ensure the data file is in 'PISI3-Project/data/'.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")
        return None

df = load_data()

if df is not None:
    # --- Initial Data Cleaning and Type Conversion ---
    required_columns = ['valor_avaliacao', 'area_construida', 'area_terreno', 'tipo_imovel', 'bairro']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Erro: Colunas essenciais faltando no dataset. Esperadas: {required_columns}")
        st.stop()

    # Convert columns to numeric, coercing errors to NaN
    df['valor_avaliacao'] = pd.to_numeric(df['valor_avaliacao'], errors='coerce')
    df['area_construida'] = pd.to_numeric(df['area_construida'], errors='coerce')
    df['area_terreno'] = pd.to_numeric(df['area_terreno'], errors='coerce')

    # Remove rows where essential numeric columns are NaN after conversion
    df.dropna(subset=['valor_avaliacao', 'area_construida', 'area_terreno'], inplace=True)

    # --- NEW CHECK: Stop if DataFrame is empty after initial dropna ---
    if df.empty:
        st.error("Erro: Nenhum dado válido restante após a limpeza inicial de valores ausentes/inválidos nas colunas essenciais (valor_avaliacao, area_construida, area_terreno). Por favor, verifique seus dados.")
        st.stop()

    # --- Neighborhood Filter (Applies to the entire DataFrame early on) ---
    st.sidebar.header("Global Filters")
    all_bairros = df['bairro'].unique().tolist()
    selected_bairros = st.sidebar.multiselect(
        "Select Neighborhood(s)",
        options=all_bairros,
        default=all_bairros # Select all by default
    )

    if not selected_bairros:
        st.warning("Please select at least one neighborhood for analysis.")
        st.stop()

    df = df[df['bairro'].isin(selected_bairros)].copy() # Apply neighborhood filter to the main df

    if df.empty:
        st.warning("No data remaining after applying neighborhood filter. Please adjust your selections.")
        st.stop()

    # --- Categorize Property Types ---
    df['tipo_agrupado'] = 'Outros' # Default category

    # Classify 'Terrenos' first based on area criteria
    # Ensure area_terreno is not zero to avoid division by zero in the ratio
    is_terrain_by_area = (
        (df['area_construida'] <= 10) |
        ((df['area_terreno'] > 0) & (df['area_construida'] / df['area_terreno'] <= 0.05))
    )
    df.loc[is_terrain_by_area, 'tipo_agrupado'] = 'Terreno'

    # Classify 'Apartamento' and 'Casa'
    df.loc[df['tipo_imovel'] == 'Apartamento', 'tipo_agrupado'] = 'Apartamento'
    df.loc[df['tipo_imovel'] == 'Casa', 'tipo_agrupado'] = 'Casa'

    # --- Calculate 'value_m2' based on grouped type ---
    df['value_m2'] = np.nan # Initialize with NaN

    # Mask for valid 'Apartamento' based on area_construida limits
    valid_apartment_area_mask = (
        (df['tipo_agrupado'] == 'Apartamento') &
        (df['area_construida'] >= 25) &
        (df['area_construida'] <= 350)
    )

    # Mask for 'Casa' (no specific area_construida limits yet)
    valid_casa_mask = (df['tipo_agrupado'] == 'Casa')

    # Combine masks for residential calculation
    residential_mask = (
        (valid_apartment_area_mask | valid_casa_mask) &
        (df['area_construida'] > 0) &
        (df['valor_avaliacao'].notna())
    )
    df.loc[residential_mask, 'value_m2'] = df['valor_avaliacao'] / df['area_construida']

    # For Terrenos: valor_avaliacao / area_terreno
    terrain_mask = (
        (df['tipo_agrupado'] == 'Terreno') &
        (df['area_terreno'] > 0) &
        (df['valor_avaliacao'].notna())
    )
    df.loc[terrain_mask, 'value_m2'] = df['valor_avaliacao'] / df['area_terreno']
    
    st.success("Imóveis categorizados e 'value_m2' calculado por tipo, com filtragem de área para apartamentos.")

    st.subheader("Raw Data Preview (with new columns)")
    st.write(df.head())
    st.write(f"Contagem por tipo agrupado: {df['tipo_agrupado'].value_counts().to_dict()}")

    st.divider()

    # --- Streamlit UI for Filtering and Analysis ---
    st.sidebar.header("Property Type Options") # Renamed for clarity
    selected_type = st.sidebar.selectbox(
        "Select Property Type for Analysis",
        ('Todos', 'Apartamento', 'Casa', 'Terreno'),
        index=0
    )

    # Filter data based on selected type for further analysis
    if selected_type != 'Todos':
        df_filtered_by_type = df[df['tipo_agrupado'] == selected_type].copy()
    else:
        df_filtered_by_type = df[df['tipo_agrupado'].isin(['Apartamento', 'Casa', 'Terreno'])].copy()

    # --- Scatter Plot: Built Area vs. Total Price ---
    # Prepare data for scatter plot: ensure area_construida and valor_avaliacao are valid and positive
    # Now, also apply apartment-specific area filters if 'Apartamento' is selected
    df_plot_scatter = df_filtered_by_type.copy()
    
    if selected_type == 'Apartamento':
        df_plot_scatter = df_plot_scatter[
            (df_plot_scatter['area_construida'] >= 25) &
            (df_plot_scatter['area_construida'] <= 350)
        ]

    df_plot_scatter = df_plot_scatter.dropna(subset=['area_construida', 'valor_avaliacao']).copy()
    df_plot_scatter = df_plot_scatter[
        (df_plot_scatter['area_construida'] > 0) & 
        (df_plot_scatter['valor_avaliacao'] > 0)
    ]

    if not df_plot_scatter.empty:
        st.subheader(f"Scatter Plot: Built Area vs. Total Price for {selected_type}s in {', '.join(selected_bairros)}")
        scatter_chart = alt.Chart(df_plot_scatter).mark_point().encode(
            x=alt.X('area_construida', title='Built Area (m²)'),
            y=alt.Y('valor_avaliacao', title='Total Price (R$)'),
            tooltip=['tipo_imovel', 'bairro', 'area_construida', 'valor_avaliacao', 'value_m2'] # Added value_m2 to tooltip
        ).properties(
            title=f"Built Area vs. Total Price for {selected_type}s in {', '.join(selected_bairros)}"
        ).interactive() # Make chart interactive for zooming/panning
        st.altair_chart(scatter_chart, use_container_width=True)
        st.divider()
    else:
        st.info(f"No valid data for scatter plot (Built Area vs. Total Price) for {selected_type}s in {', '.join(selected_bairros)} after initial cleaning.")
    
    # Clean data for value_m2 analysis (this part remains as is, acting on the value_m2 column)
    data_for_analysis = df_filtered_by_type['value_m2'].replace([np.inf, -np.inf], np.nan).dropna()
    data_for_analysis = data_for_analysis[data_for_analysis > 0]

    if data_for_analysis.empty and selected_type != 'Todos': # Check for empty data *after* type selection
        st.warning(f"No valid data points for '{selected_type}' in {', '.join(selected_bairros)} after initial cleaning. Cannot proceed with analysis.")
        st.stop()
    elif data_for_analysis.empty and selected_type == 'Todos':
        st.warning(f"No valid data points for any selected property type in {', '.join(selected_bairros)} after initial cleaning. Cannot proceed with analysis.")
        st.stop()


    # --- New Section: Area Built Analysis for Apartments (only if 'Apartamento' is selected) ---
    if selected_type == 'Apartamento':
        st.subheader(f"Apartment Built Area Analysis (Before Price/m² Filtering) in {', '.join(selected_bairros)}")
        # Filter original df for apartments before value_m2 calculation, but after initial area limits
        apartment_area = df[
            (df['tipo_agrupado'] == 'Apartamento') & 
            (df['area_construida'] >= 25) & 
            (df['area_construida'] <= 350)
        ]['area_construida'].copy()
        
        apartment_area = apartment_area.dropna()[apartment_area > 0] # Clean area data

        if not apartment_area.empty:
            st.write("Summary Statistics for Apartment Built Area (m²):")
            st.write(apartment_area.describe())

            # Histogram of Built Area for Apartments
            chart_area = alt.Chart(pd.DataFrame({'area_construida': apartment_area})).mark_bar().encode(
                alt.X('area_construida', bin=alt.Bin(maxbins=50), title="Built Area (m²)"),
                alt.Y('count()', title="Number of Apartments")
            ).properties(title=f"Histogram of Apartment Built Area (After initial filtering) in {', '.join(selected_bairros)}")
            st.altair_chart(chart_area, use_container_width=True)
        else:
            st.info(f"No valid built area data found for apartments after initial area-based cleaning in {', '.join(selected_bairros)}.")
        st.divider()

    # Define default filter ranges based on selected type
    if selected_type == 'Apartamento':
        default_min, default_max = 2500.0, 12000.0
    elif selected_type == 'Casa':
        default_min, default_max = 1200.0, 8000.0
    elif selected_type == 'Terreno':
        default_min, default_max = 500.0, 5000.0 # Sugestão inicial para terrenos
    else: # Todos
        default_min, default_max = float(data_for_analysis.min()), float(data_for_analysis.max())

    st.subheader(f"Data Cleaning and Filtering for {selected_type}s in {', '.join(selected_bairros)}")

    # --- Visualization of 'value_m2' distribution ---
    st.subheader(f"Distribution of Price per m² for {selected_type}s in {', '.join(selected_bairros)}")
    chart = alt.Chart(pd.DataFrame({'value_m2': data_for_analysis})).mark_bar().encode(
        alt.X('value_m2', bin=alt.Bin(maxbins=50), title="Price per m² (R$)"),
        alt.Y('count()', title="Number of Properties")
    ).properties(title=f"Histogram of Price per m² for {selected_type}s (Before Custom Filtering) in {', '.join(selected_bairros)}")
    st.altair_chart(chart, use_container_width=True)

    st.markdown("--- ")
    st.subheader("Custom Filtering")
    st.write("You can refine the data by setting custom price ranges or adjusting outlier removal.")

    # Custom price range filter
    min_price_m2 = st.sidebar.number_input("Minimum Price per m² (R$)", min_value=0.0, value=default_min, format="%.2f")
    max_price_m2 = st.sidebar.number_input("Maximum Price per m² (R$)", min_value=0.0, value=default_max, format="%.2f")

    data_filtered_by_range = data_for_analysis[
        (data_for_analysis >= min_price_m2) & (data_for_analysis <= max_price_m2)
    ]

    # IQR Outlier removal option (now applied *after* custom range, if any)
    remove_outliers_iqr = st.sidebar.checkbox("Remove outliers using IQR (Interquartile Range)", value=True)
    iqr_factor = st.sidebar.slider("IQR Factor (for IQR outlier removal)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

    data = data_filtered_by_range.copy() # Start with data potentially filtered by custom range

    if remove_outliers_iqr and not data.empty:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        data = data[(data >= lower_bound) & (data <= upper_bound)]
        st.info(f"Outliers removed using IQR factor {iqr_factor}: Data filtered between {lower_bound:,.2f} and {upper_bound:,.2f} R$/m².")
    elif remove_outliers_iqr and data.empty:
         st.warning("No data left to apply IQR outlier removal after custom range filtering.")

    if data.empty:
        st.warning(f"No valid data points remaining after filtering to calculate confidence interval in {', '.join(selected_bairros)}.")
        st.stop()

    # --- Statistical Calculations ---
    n = len(data)
    mean = np.mean(data)
    median = np.median(data) # Calculate median
    std = np.std(data, ddof=1) 
    se = std / np.sqrt(n) # Standard Error

    st.subheader(f"Statistical Results for {selected_type}s in {', '.join(selected_bairros)} (After Filtering)")
    st.write(f"Sample size (after cleaning and filtering): **{n}**")
    st.write(f"Mean Price (R$/m²): **{mean:,.2f}**")
    st.write(f"Median Price (R$/m²): **{median:,.2f}**") # Display median
    st.write(f"Standard deviation: **{std:,.2f}**")
    st.write(f"Standard error of the mean: **{se:,.2f}**")

    # --- Confidence Interval Calculation ---
    confidence_level = 0.95
    alpha = 1 - confidence_level

    if n >= 30: # Heuristic for using Z-distribution for large samples
        # Z-score for a two-tailed interval
        z_critical = norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * se
        distribution_used = "Z-distribution"
    else: # Use Student's t-distribution for smaller samples
        # t-value for a two-tailed interval with n-1 degrees of freedom
        t_critical = t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * se
        distribution_used = "Student's t-distribution"

    lower_ci = mean - margin_of_error
    upper_ci = mean + margin_of_error

    st.markdown(f"\n**{int(confidence_level*100)}% Confidence Interval ({distribution_used}) for {selected_type}s in {', '.join(selected_bairros)}**")
    st.success(f"**{lower_ci:,.2f} R$/m²**  to  **{upper_ci:,.2f} R$/m²**")

    st.divider()

    # --- COMPARISON SECTION: Fixed Filters for Boa Viagem ---
    st.header("Comparison: Fixed Filters for Boa Viagem (Apartments)")
    st.write("This section shows results for a predefined set of filters to offer a comparison point.")

    df_bv_comparison = df.copy() # Start with a fresh copy of the main df after initial cleaning and value_m2 calculation

    # Apply the fixed filters
    df_bv_comparison = df_bv_comparison[
        (df_bv_comparison['bairro'] == 'BOA VIAGEM') &
        (df_bv_comparison['area_construida'] >= 60) &
        (df_bv_comparison['area_construida'] <= 200) &
        (df_bv_comparison['valor_avaliacao'] >= 350000) &
        (df_bv_comparison['value_m2'] >= 3500) &
        (df_bv_comparison['value_m2'] <= 15000)
    ]

    # Clean the 'value_m2' column for this comparison subset
    data_bv_comparison = df_bv_comparison['value_m2'].replace([np.inf, -np.inf], np.nan).dropna()
    data_bv_comparison = data_bv_comparison[data_bv_comparison > 0]

    # Prepare formatted strings for report_content to avoid SyntaxError
    n_bv_str = str(len(data_bv_comparison)) if not data_bv_comparison.empty else 'N/A'
    mean_bv_str = f"{np.mean(data_bv_comparison):,.2f}" if not data_bv_comparison.empty else 'N/A'
    median_bv_str = f"{np.median(data_bv_comparison):,.2f}" if not data_bv_comparison.empty else 'N/A'
    std_bv_str = f"{np.std(data_bv_comparison, ddof=1):,.2f}" if not data_bv_comparison.empty else 'N/A'
    se_bv_str = f"{np.std(data_bv_comparison, ddof=1) / np.sqrt(len(data_bv_comparison)):,.2f}" if not data_bv_comparison.empty else 'N/A'
    
    lower_ci_bv_str = 'N/A'
    upper_ci_bv_str = 'N/A'
    dist_used_bv_str = 'N/A'

    if not data_bv_comparison.empty:
        n_bv = len(data_bv_comparison)
        mean_bv = np.mean(data_bv_comparison)
        median_bv = np.median(data_bv_comparison)
        std_bv = np.std(data_bv_comparison, ddof=1)
        se_bv = std_bv / np.sqrt(n_bv)

        st.write(f"Filtered Data Points: **{n_bv}**")
        st.write(f"Mean Price (R$/m²): **{mean_bv:,.2f}**")
        st.write(f"Median Price (R$/m²): **{median_bv:,.2f}**")
        st.write(f"Standard deviation: **{std_bv:,.2f}**")
        st.write(f"Standard error: **{se_bv:,.2f}**")

        # Confidence Interval Calculation for comparison data
        confidence_level = 0.95 # Ensure confidence_level is defined for this section
        alpha = 1 - confidence_level

        if n_bv >= 30:
            z_critical_bv = norm.ppf((1 + confidence_level) / 2)
            margin_of_error_bv = z_critical_bv * se_bv
            dist_used_bv = "Z-distribution"
        else:
            t_critical_bv = t.ppf((1 + confidence_level) / 2, df=n_bv - 1)
            margin_of_error_bv = t_critical_bv * se_bv
            dist_used_bv = "Student's t-distribution"

        lower_ci_bv = mean_bv - margin_of_error_bv
        upper_ci_bv = mean_bv + margin_of_error_bv

        lower_ci_bv_str = f"{lower_ci_bv:,.2f}"
        upper_ci_bv_str = f"{upper_ci_bv:,.2f}"
        dist_used_bv_str = dist_used_bv

        st.markdown(f"\n**{int(confidence_level*100)}% Confidence Interval ({dist_used_bv})**")
        st.success(f"**{lower_ci_bv:,.2f} R$/m²**  to  **{upper_ci_bv:,.2f} R$/m²**")
    else:
        st.warning("No data points remaining for the fixed Boa Viagem comparison filters.")

    st.divider()

    # --- Report Download ---
    report_content = (
        f"Confidence Interval Report for Price per m² ({selected_type}s in {', '.join(selected_bairros)})\n"
        f"---------------------------------------------\n"
        f"Sample size (after cleaning and filtering): {n}\n"
        f"Mean Price (R$/m²): **{mean:,.2f}**\n"
        f"Median Price (R$/m²): **{median:,.2f}**\n"
        f"Standard deviation: **{std:,.2f}**\n"
        f"Standard error of the mean: {se:,.2f}\n"
        f"{int(confidence_level*100)}% Confidence Interval: {lower_ci:,.2f} to {upper_ci:,.2f} R$/m²\n"
        f"Distribution used: {distribution_used}\n"
        f"Outliers removed (IQR factor {iqr_factor}): {'Yes' if remove_outliers_iqr else 'No'}\n"
        f"Custom price range applied: {min_price_m2:,.2f} to {max_price_m2:,.2f} R$/m²\n"
        f"Neighborhoods: {', '.join(selected_bairros)}\n"
        f"\n--- Comparison Section (Fixed Filters for Boa Viagem) ---\n"
        f"Filtered Data Points: {n_bv_str}\n"
        f"Mean Price (R$/m²): {mean_bv_str}\n"
        f"Median Price (R$/m²): {median_bv_str}\n"
        f"Standard deviation: {std_bv_str}\n"
        f"Standard error: {se_bv_str}\n"
        f"{int(confidence_level*100)}% Confidence Interval: {lower_ci_bv_str} to {upper_ci_bv_str} R$/m²\n"
        f"Distribution used: {dist_used_bv_str}\n"
    )

    st.download_button(
        label="Download CI Report",
        data=report_content,
        file_name=f"m2_confidence_interval_report_{selected_type.lower()}_{'_'.join(selected_bairros)}.txt",
        mime="text/plain"
    )
else:
    st.warning("Waiting to load dataset or an error occurred. Please check the console for details.")
