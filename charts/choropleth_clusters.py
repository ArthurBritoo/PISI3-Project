"""
Create an interactive choropleth map of Recife neighborhoods showing average property price
with an overlaid scatter layer of properties colored by cluster.

Outputs:
 - charts/choropleth_clusters.html  (interactive)
 - Optionally: charts/sample_output/choropleth_clusters.png (if kaleido or orca available)

Usage:
    python charts/choropleth_clusters.py

The script looks for CSVs in the repo root: `quintoandar_recife.csv`, `vivareal_recife.csv`.
If no `cluster` column exists it will run a KMeans clustering (n_clusters from
`data/clustering_metadata.json` if present, otherwise 5).

This script is defensive: it attempts to auto-detect common column names for price,
area, lat/lon and neighborhood. If neighborhood names are not present in the
property CSV, the choropleth layer will not be rendered (the scatter layer will be).
"""

import os
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CHARTS_DIR = Path(__file__).resolve().parent
OUTPUT_HTML = CHARTS_DIR / "choropleth_clusters.html"
SAMPLE_PNG = CHARTS_DIR / "sample_output" / "choropleth_clusters.png"

# Candidate column name lists (common variations)
PRICE_CANDIDATES = ['price', 'preco', 'preço', 'valor', 'valor_m2', 'valor_metro', 'valor_m2_total']
AREA_CANDIDATES = ['area', 'area_construida', 'area_terreno', 'area_m2', 'area_total']
LAT_CANDIDATES = ['latitude', 'lat', 'y']
LON_CANDIDATES = ['longitude', 'lon', 'lng', 'x']
NEIGHBORHOOD_CANDIDATES = ['neighborhood', 'bairro', 'bairro_nome', 'EBAIRRNOME', 'EBAIRRNOMEOF', 'bairro_nome']
ID_CANDIDATES = ['id', 'imovel_id', 'codigo']

CSV_FILES = [ROOT / 'quintoandar_recife.csv', ROOT / 'vivareal_recife.csv']
GEOJSON_PATH = ROOT / 'data' / 'geodata' / 'recife_bairros.geojson'
CLUSTER_META = ROOT / 'data' / 'clustering_metadata.json'


def choose_existing_csv():
    files = [f for f in CSV_FILES if f.exists() and f.stat().st_size > 0]
    if not files:
        return None
    # prefer quintoandar if present
    for pref in CSV_FILES:
        if pref in files:
            return pref
    return files[0]


def load_properties(csv_path: Path):
    # try common encodings and separators
    for enc in (None, 'utf-8', 'latin-1'):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            if df.shape[0] == 0:
                continue
            return df
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV: {csv_path}")


def detect_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    # try substring match
    for k, orig in cols.items():
        for cand in candidates:
            if cand in k:
                return orig
    return None


def standardize_columns(df: pd.DataFrame):
    df = df.copy()
    price_col = detect_column(df, PRICE_CANDIDATES)
    area_col = detect_column(df, AREA_CANDIDATES)
    lat_col = detect_column(df, LAT_CANDIDATES)
    lon_col = detect_column(df, LON_CANDIDATES)
    nb_col = detect_column(df, NEIGHBORHOOD_CANDIDATES)
    id_col = detect_column(df, ID_CANDIDATES)

    rename_map = {}
    if price_col:
        rename_map[price_col] = 'price'
    if area_col:
        rename_map[area_col] = 'area'
    if lat_col:
        rename_map[lat_col] = 'latitude'
    if lon_col:
        rename_map[lon_col] = 'longitude'
    if nb_col:
        rename_map[nb_col] = 'neighborhood'
    if id_col:
        rename_map[id_col] = 'id'

    df = df.rename(columns=rename_map)
    return df


def compute_clusters(df: pd.DataFrame, n_clusters=5):
    # require numeric features; pick price, area, latitude, longitude when available
    features = []
    for f in ['price', 'area', 'latitude', 'longitude']:
        if f in df.columns:
            features.append(f)
    if not features:
        raise RuntimeError('No numeric features available to cluster')

    X = df[features].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(Xs)
    return clusters


def find_geojson_name_key(geojson):
    # inspect first feature properties
    props = None
    for f in geojson.get('features', []):
        props = f.get('properties', {})
        if props:
            break
    if not props:
        return None
    keys = [k for k in props.keys()]
    # prefer EBAIRRNOME (exists in provided file), fallback to common keys
    for cand in ['EBAIRRNOME', 'EBAIRRNOMEOF', 'name', 'NOME', 'bairro']:
        if cand in keys:
            return cand
    # otherwise return first string-like property
    for k, v in props.items():
        if isinstance(v, str):
            return k
    return keys[0]


def map_neighborhoods_to_geo(geojson, df_nb_values, geo_name_key):
    # Build a mapping from normalized names in geojson to the actual property value
    mapping = {}
    for feat in geojson.get('features', []):
        props = feat.get('properties', {})
        val = props.get(geo_name_key)
        if val is None:
            continue
        mapping[str(val).strip().lower()] = val
    # Attempt to create a new column feature_id matching geojson property values
    feature_ids = []
    for nb in df_nb_values:
        if pd.isna(nb):
            feature_ids.append(None)
            continue
        key = str(nb).strip().lower()
        if key in mapping:
            feature_ids.append(mapping[key])
        else:
            # try crude normalization: remove accents and punctuation
            import unicodedata, re
            key_norm = unicodedata.normalize('NFKD', key).encode('ASCII', 'ignore').decode()
            key_norm = re.sub(r'[^a-z0-9 ]', '', key_norm).strip()
            found = None
            for mk, mv in mapping.items():
                mk_norm = unicodedata.normalize('NFKD', mk).encode('ASCII', 'ignore').decode()
                mk_norm = re.sub(r'[^a-z0-9 ]', '', mk_norm).strip()
                if mk_norm == key_norm:
                    found = mv
                    break
            feature_ids.append(found)
    return feature_ids


def main():
    print('Starting choropleth + clusters generation...')

    csv_path = choose_existing_csv()
    if not csv_path:
        print('No property CSV found (quintoandar_recife.csv or vivareal_recife.csv). Aborting.')
        return

    print(f'Loading properties from: {csv_path}')
    df = load_properties(csv_path)
    if df.shape[0] == 0:
        print('CSV appears empty. Aborting.')
        return

    df = standardize_columns(df)

    # Cast numeric columns where possible
    for col in ['price', 'area', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Some datasets may have price per m2; try to detect and scale if needed
    if 'price' in df.columns and df['price'].median() < 1000:
        # suspiciously small median -> maybe price per m2; do nothing but warn
        print('Note: median price < 1000 — verify your price units (might be price per m2)')

    # Ensure lat/lon present for scatter
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print('Latitude/longitude columns not detected; the map will not include point layer.')

    # Determine clusters
    if 'cluster' not in df.columns:
        n_clusters = 5
        if CLUSTER_META.exists():
            try:
                meta = json.load(open(CLUSTER_META, 'r', encoding='utf-8'))
                n_clusters = meta.get('n_clusters', n_clusters)
                print(f"Using n_clusters={n_clusters} from clustering metadata")
            except Exception:
                pass
        try:
            df = df.reset_index(drop=True)
            clusters = compute_clusters(df, n_clusters=n_clusters)
            df['cluster'] = clusters
            print('Computed clusters via KMeans')
        except Exception as e:
            print('Failed to compute clusters:', e)
    else:
        print('Using existing cluster column in dataframe')

    # Load geojson
    if not GEOJSON_PATH.exists():
        print(f'GeoJSON not found at {GEOJSON_PATH}; choropleth disabled.')
        geojson = None
    else:
        geojson = json.load(open(GEOJSON_PATH, 'r', encoding='utf-8'))

    # Prepare per-neighborhood aggregates if neighborhood info exists
    choropleth_df = None
    if 'neighborhood' in df.columns and geojson is not None:
        nb_agg = df.groupby('neighborhood').agg(
            avg_price=('price', 'mean'),
            count_properties=('price', 'count'),
            dominant_cluster=('cluster', lambda x: int(Counter(x).most_common(1)[0][0]) if len(x) else -1)
        ).reset_index()
        # match neighborhood strings to geojson properties
        geo_name_key = find_geojson_name_key(geojson)
        print(f'Using geojson name key: {geo_name_key}')
        nb_agg['feature_id'] = map_neighborhoods_to_geo(geojson, nb_agg['neighborhood'], geo_name_key)
        choropleth_df = nb_agg.dropna(subset=['feature_id'])
        if choropleth_df.empty:
            print('No matching neighborhood names found between CSV and GeoJSON — choropleth will be skipped.')
            choropleth_df = None
    else:
        if 'neighborhood' not in df.columns:
            print('No neighborhood column in properties dataframe; skipping choropleth.')

    # Create map
    # Use mapbox style that doesn't require token
    mapbox_style = 'carto-positron'
    center = None
    if 'latitude' in df.columns and 'longitude' in df.columns:
        center = dict(lat=float(df['latitude'].mean()), lon=float(df['longitude'].mean()))

    if choropleth_df is not None:
        # choropleth + scatter
        fig = px.choropleth_mapbox(
            choropleth_df,
            geojson=geojson,
            locations='feature_id',
            featureidkey=f'properties.{geo_name_key}',
            color='avg_price',
            hover_data=['neighborhood', 'avg_price', 'count_properties', 'dominant_cluster'],
            color_continuous_scale='Viridis',
            mapbox_style=mapbox_style,
            center=center,
            zoom=11,
            opacity=0.6,
            title='Recife — Average Price by Neighborhood with Property Clusters'
        )
    else:
        # empty base map
        fig = px.scatter_mapbox(
            pd.DataFrame({'latitude': [center['lat']]}) if center else pd.DataFrame({'latitude': [-8.05389], 'longitude': [-34.8811]}),
            lat='latitude', lon='longitude', zoom=11, mapbox_style=mapbox_style,
            title='Recife — Property Clusters'
        )

    # Add scatter layer of properties if lat/lon present
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # reduce data for plotting if too many points
        plot_df = df.dropna(subset=['latitude', 'longitude']).copy()
        if plot_df.shape[0] > 5000:
            print('Large dataset detected — sampling 5000 points for visualization')
            plot_df = plot_df.sample(5000, random_state=42)

        scatter = px.scatter_mapbox(
            plot_df,
            lat='latitude',
            lon='longitude',
            color='cluster' if 'cluster' in plot_df.columns else None,
            size='area' if 'area' in plot_df.columns else None,
            hover_data=[c for c in ['id', 'price', 'area', 'neighborhood', 'cluster'] if c in plot_df.columns],
            opacity=0.8,
            zoom=11,
            mapbox_style=mapbox_style,
            center=center,
        )
        for trace in scatter.data:
            fig.add_trace(trace)

    fig.update_layout(margin={'r':0,'t':40,'l':0,'b':0}, legend=dict(title='Cluster'))

    # Save interactive HTML
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_HTML))
    print(f'Wrote interactive map to: {OUTPUT_HTML}')

    # Optionally save static PNG if kaleido installed
    try:
        SAMPLE_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(SAMPLE_PNG), scale=2)
        print(f'Wrote sample PNG to: {SAMPLE_PNG}')
    except Exception:
        print('Could not write PNG (kaleido/orca likely not installed). You can still open the HTML interactively.')


if __name__ == '__main__':
    main()
