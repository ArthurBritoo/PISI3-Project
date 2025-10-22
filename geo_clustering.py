import os
import json
import unicodedata
from typing import Dict, List, Tuple, Optional

import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from difflib import get_close_matches
import requests

IBGE_MUN_CODE_RECIFE = 2611606

# ---------------- Normalização/Mappings ----------------

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper().strip()

def _load_manual_bairro_mapping() -> Optional[Dict[str, str]]:
    """
    Opcional: data/bairro_to_bairrooficial.csv com colunas:
      bairro, bairro_oficial
    """
    try:
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, "data", "bairro_to_bairrooficial.csv")
        if os.path.exists(csv_path):
            df_map = pd.read_csv(csv_path)
            df_map = df_map.dropna(subset=["bairro", "bairro_oficial"])
            return { _normalize(r["bairro"]): str(r["bairro_oficial"]) for _, r in df_map.iterrows() }
    except Exception:
        pass
    return None

# ---------------- Carregamento: Malha oficial dos bairros de Recife ----------------

def _infer_bairro_prop_name(props: Dict) -> Optional[str]:
    # tenta detectar a chave do nome do bairro no GeoJSON
    keys = list(props.keys())
    preferred = [
        # chaves específicas da malha oficial de Recife
        "EBAIRRNOMEOF", "EBAIRRNOME",
        # variações comuns
        "bairro", "nome", "name",
        "nm_bairro", "no_bairro", "n_bairro",
        "nm_bairro_", "nm_bairro1",
        "NM_BAIRRO", "NO_BAIRRO", "BAIRRO", "NOME",
    ]
    # case-insensitive
    lower = {k.lower(): k for k in keys}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]

    # fallback: escolhe a string mais plausível (evita timestamps ISO)
    import re
    iso_dt = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    text_like = [k for k, v in props.items() if isinstance(v, str) and v and not iso_dt.match(v)]
    if text_like:
        # pega a string com maior diversidade de letras (melhor chance de ser nome)
        def score(k: str) -> int:
            v = props[k]
            return len(set([c for c in v if c.isalpha()]))
        return max(text_like, key=score)
    return None

def _load_bairros_recife_geometries() -> pd.DataFrame:
    """
    Lê malha oficial de bairros (GeoJSON) em:
      - data/geodata/recife_bairros.geojson
      - data/recife_bairros.geojson
    Retorna DataFrame com colunas: bairro_oficial, unit_id, geometry
    """
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "data", "geodata", "recife_bairros.geojson"),
        os.path.join(base_dir, "data", "recife_bairros.geojson"),
        os.path.join(base_dir, "data", "geodata", "recife_bairros.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    gj = json.load(f)
                features = gj.get("features", [])
                rows = []
                uid = 1
                for ft in features:
                    geom = ft.get("geometry")
                    props = ft.get("properties", {}) or {}
                    if not geom:
                        continue
                    name_key = _infer_bairro_prop_name(props) or "nome"
                    nome = str(props.get(name_key, "")).strip()
                    if not nome:
                        continue
                    try:
                        g = shape(geom).buffer(0)  # corrige geometria inválida
                    except Exception:
                        continue
                    rows.append({"unit_id": uid, "bairro_oficial": nome, "geometry": g})
                    uid += 1
                df = pd.DataFrame(rows)
                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame(columns=["unit_id", "bairro_oficial", "geometry"])

# ---------------- (Fallback) IBGE Subdistritos ----------------

def _ibge_get_subdistritos(mun_code: int = IBGE_MUN_CODE_RECIFE) -> List[Dict]:
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{mun_code}/subdistritos"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def _try_fetch_geojson(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[Dict]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _ibge_get_subdistrito_geojson(sub_id: int) -> Optional[Dict]:
    tries = [
        f"https://servicodados.ibge.gov.br/api/v3/malhas/subdistritos/{sub_id}?formato=application/vnd.geo+json",
        f"https://servicodados.ibge.gov.br/api/v3/malhas/subdistritos/{sub_id}",
        f"https://servicodados.ibge.gov.br/api/v2/malhas/subdistritos/{sub_id}?formato=application/vnd.geo+json",
        f"https://servicodados.ibge.gov.br/api/v2/malhas/subdistritos/{sub_id}",
    ]
    headers = {"Accept": "application/vnd.geo+json"}
    for u in tries:
        gj = _try_fetch_geojson(u, headers=headers)
        if gj:
            return gj
    return None

def _load_subdistritos_geometries(mun_code: int = IBGE_MUN_CODE_RECIFE) -> pd.DataFrame:
    subs = _ibge_get_subdistritos(mun_code)
    rows = []
    for s in subs:
        try:
            sub_id = int(s["id"])
            nome = s["nome"]
        except Exception:
            continue
        gj = _ibge_get_subdistrito_geojson(sub_id)
        if not gj:
            continue
        features = gj.get("features") or ([gj] if "geometry" in gj else [])
        if not features or not features[0].get("geometry"):
            continue
        try:
            geom = shape(features[0]["geometry"]).buffer(0)
            rows.append({"unit_id": sub_id, "bairro_oficial": nome, "geometry": geom})
        except Exception:
            continue
    return pd.DataFrame(rows)

# ---------------- Adjacência e mapeamento ----------------

def _build_adjacency(df_geo: pd.DataFrame) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    if df_geo.empty:
        return adj
    geoms = df_geo.set_index("unit_id")["geometry"].to_dict()
    ids = list(geoms.keys())
    for i, aid in enumerate(ids):
        a = geoms[aid]
        neighs: List[int] = []
        for bid in ids:
            if aid == bid:
                continue
            b = geoms[bid]
            try:
                # vizinho por fronteira em comum; usa buffer(0) para robustez
                if a.touches(b) or a.buffer(0).boundary.intersects(b.buffer(0).boundary):
                    # evita conexões por um único ponto: mede comprimento da interseção de fronteiras
                    inter = a.boundary.intersection(b.boundary)
                    if inter.length > 0.000001:  # tolerância
                        neighs.append(bid)
            except Exception:
                continue
        adj[aid] = neighs
    return adj

def _map_bairro_to_official(bairros: List[str], official_names: List[str]) -> Dict[str, str]:
    manual = _load_manual_bairro_mapping()
    mapping: Dict[str, str] = {}
    off_norm = [_normalize(n) for n in official_names]

    # ajustes comuns (exemplos; ajuste conforme precisar)
    synonyms = {
        "BOA VIAGEM": "BOA VIAGEM",
        "CASA FORTE": "CASA FORTE",
        "GRAÇAS": "GRACAS",
        "GRACAS": "GRACAS",
        "SÃO JOSÉ": "SAO JOSE",
        "SAO JOSE": "SAO JOSE",
    }

    for b in bairros:
        b_norm = _normalize(b)
        if manual and b_norm in manual:
            mapping[b] = manual[b_norm]
            continue
        if b_norm in synonyms:
            b_norm = synonyms[b_norm]
        # match exato
        if b_norm in off_norm:
            mapping[b] = official_names[off_norm.index(b_norm)]
            continue
        # fuzzy mais estrito
        match = get_close_matches(b_norm, off_norm, n=1, cutoff=0.9)
        if match:
            mapping[b] = official_names[off_norm.index(match[0])]
        else:
            # mantém o próprio nome como fallback (pode virar região própria)
            mapping[b] = b
    return mapping

def _name_by_id(uid: int, id_by_name: Dict[str, int]) -> str:
    for name, _id in id_by_name.items():
        if _id == uid:
            return name
    return str(uid)

# ---------------- Balanced merge ----------------

def _balanced_merge(
    counts_by_unit: pd.Series,
    adjacency: Dict[int, List[int]],
    id_by_name: Dict[str, int],
    min_tx_per_region: int,
) -> Dict[str, str]:
    assigned: Dict[int, str] = {}
    regions: Dict[str, List[int]] = {}

    units_sorted = counts_by_unit.sort_values(ascending=True).index.tolist()

    for unit_name in units_sorted:
        unit_id = id_by_name.get(unit_name)
        if unit_id is None or unit_id in assigned:
            continue

        region_label = f"região: {unit_name.strip().lower()}"
        current = [unit_id]
        assigned[unit_id] = region_label
        total = int(counts_by_unit.get(unit_name, 0))

        frontier = sorted(
            adjacency.get(unit_id, []),
            key=lambda nid: counts_by_unit.get(_name_by_id(nid, id_by_name), 0),
        )
        visited = set(current)

        while total < min_tx_per_region and frontier:
            nid = frontier.pop(0)
            if nid in visited or nid in assigned:
                visited.add(nid)
                continue
            visited.add(nid)
            n_name = _name_by_id(nid, id_by_name)
            total += int(counts_by_unit.get(n_name, 0))
            assigned[nid] = region_label
            current.append(nid)
            neighs = [x for x in adjacency.get(nid, []) if x not in visited and x not in assigned]
            neighs = sorted(neighs, key=lambda x: counts_by_unit.get(_name_by_id(x, id_by_name), 0))
            frontier.extend(neighs)

        regions[region_label] = current

    remaining = [uid for uid in id_by_name.values() if uid not in assigned]
    for uid in remaining:
        best_region, best_total = None, None
        for rlabel, uids in regions.items():
            t = sum(counts_by_unit.get(_name_by_id(x, id_by_name), 0) for x in uids)
            if best_total is None or t < best_total:
                best_region, best_total = rlabel, t
        if best_region is None:
            n_name = _name_by_id(uid, id_by_name)
            rlabel = f"região: {n_name.strip().lower()}"
            regions[rlabel] = [uid]
            assigned[uid] = rlabel
        else:
            regions[best_region].append(uid)
            assigned[uid] = best_region

    name_by_id = {v: k for k, v in id_by_name.items()}
    out: Dict[str, str] = {}
    for rlabel, uids in regions.items():
        for uid in uids:
            uname = name_by_id.get(uid)
            if uname:
                out[uname] = rlabel
    return out

# ---------------- Fallback (sem geografia) ----------------

def _balanced_regions_without_geo(
    df: pd.DataFrame,
    bairro_col: str,
    min_tx_per_region: int,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    counts = df[bairro_col].value_counts()
    bairros_sorted = counts.sort_values(ascending=True).index.tolist()

    regions: Dict[str, List[str]] = {}
    current_region: List[str] = []
    current_total = 0
    blocks = [(b, int(counts[b])) for b in bairros_sorted]

    for b, c in blocks:
        current_region.append(b)
        current_total += c
        if current_total >= min_tx_per_region:
            label = f"região: {current_region[0].strip().lower()}"
            regions[label] = current_region[:]
            current_region, current_total = [], 0

    if current_region:
        label = f"região: {current_region[0].strip().lower()}"
        regions[label] = current_region[:]

    bairro_to_region: Dict[str, str] = {}
    for reg, bs in regions.items():
        for b in bs:
            bairro_to_region[b] = reg

    df_out = df.copy()
    df_out["regiao"] = df_out[bairro_col].map(lambda b: bairro_to_region.get(b, f"região: {str(b).strip().lower()}"))

    regions["__source__"] = "fallback"
    return df_out, regions

# ---------------- Public API ----------------

def build_regions_for_recife(
    df: pd.DataFrame,
    bairro_col: str = "bairro",
    min_tx_per_region: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Tenta nesta ordem:
      1) Malha oficial de bairros (GeoJSON local) -> adjacência por fronteira
      2) Subdistritos IBGE (API) -> adjacência por fronteira
      3) Fallback sem geografia (balanceado por contagem)

    Retorna:
      - df com coluna 'regiao'
      - regions_dict com composição e chave __source__: {'bairros' | 'ibge' | 'fallback'}
    """
    if df.empty:
        dd = df.copy()
        dd["regiao"] = pd.NA
        return dd, {"__source__": "fallback"}

    # 1) Bairros oficiais
    bairros_geo = _load_bairros_recife_geometries()
    if not bairros_geo.empty:
        official = bairros_geo["bairro_oficial"].tolist()
        bairros = sorted(pd.Series(df[bairro_col].dropna().unique()).astype(str).tolist())
        bairro_to_off = _map_bairro_to_official(bairros, official)

        tmp = df[[bairro_col]].copy()
        tmp["unit_name"] = tmp[bairro_col].map(bairro_to_off)
        counts_by_unit = tmp["unit_name"].value_counts()

        adjacency = _build_adjacency(bairros_geo.rename(columns={"bairro_oficial": "unit_name"}))
        id_by_name = dict(zip(bairros_geo["bairro_oficial"], bairros_geo["unit_id"]))

        mapping = _balanced_merge(counts_by_unit, adjacency, id_by_name, min_tx_per_region)

        def to_region(b):
            uname = bairro_to_off.get(b, b)
            return mapping.get(uname, f"região: {str(uname).strip().lower()}")

        df_out = df.copy()
        df_out["regiao"] = df_out[bairro_col].map(to_region)

        regions_dict: Dict[str, List[str]] = {}
        for uname, region in mapping.items():
            regions_dict.setdefault(region, []).append(uname)
        regions_dict["__source__"] = "bairros"
        return df_out, regions_dict

    # 2) IBGE subdistritos
    subs_geo = _load_subdistritos_geometries(IBGE_MUN_CODE_RECIFE)
    if not subs_geo.empty:
        sub_names = subs_geo["bairro_oficial"].tolist()
        bairros = sorted(pd.Series(df[bairro_col].dropna().unique()).astype(str).tolist())
        bairro_to_sub = _map_bairro_to_official(bairros, sub_names)

        tmp = df[[bairro_col]].copy()
        tmp["unit_name"] = tmp[bairro_col].map(bairro_to_sub)
        counts_by_unit = tmp["unit_name"].value_counts()

        adjacency = _build_adjacency(subs_geo.rename(columns={"bairro_oficial": "unit_name"}))
        id_by_name = dict(zip(subs_geo["bairro_oficial"], subs_geo["unit_id"]))

        mapping = _balanced_merge(counts_by_unit, adjacency, id_by_name, min_tx_per_region)

        def to_region(b):
            uname = bairro_to_sub.get(b, b)
            return mapping.get(uname, f"região: {str(uname).strip().lower()}")

        df_out = df.copy()
        df_out["regiao"] = df_out[bairro_col].map(to_region)

        regions_dict: Dict[str, List[str]] = {}
        for uname, region in mapping.items():
            regions_dict.setdefault(region, []).append(uname)
        regions_dict["__source__"] = "ibge"
        return df_out, regions_dict

    # 3) Fallback sem geografia
    return _balanced_regions_without_geo(df, bairro_col=bairro_col, min_tx_per_region=min_tx_per_region)