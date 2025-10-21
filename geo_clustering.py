import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from difflib import get_close_matches

IBGE_MUN_CODE_RECIFE = 2611606

# ---------------- IBGE helpers ----------------

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
    # Tenta múltiplas rotas da API de malhas
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
            geom = shape(features[0]["geometry"])  # Polygon/MultiPolygon
            rows.append({"sub_id": sub_id, "sub_nome": nome, "geometry": geom})
        except Exception:
            continue
    return pd.DataFrame(rows)

# ---------------- Graph helpers ----------------

def _build_adjacency(df_geo: pd.DataFrame) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    if df_geo.empty:
        return adj
    for i, row_i in df_geo.iterrows():
        a_id = int(row_i["sub_id"])
        a_geom: BaseGeometry = row_i["geometry"]
        neighs: List[int] = []
        for j, row_j in df_geo.iterrows():
            if i == j:
                continue
            b_id = int(row_j["sub_id"])
            b_geom: BaseGeometry = row_j["geometry"]
            try:
                if a_geom.touches(b_geom) or a_geom.intersects(b_geom):
                    neighs.append(b_id)
            except Exception:
                continue
        adj[a_id] = neighs
    return adj

def _map_bairro_to_subdistrito(bairros: List[str], sub_nomes: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    sub_ref = [n.upper() for n in sub_nomes]
    for b in bairros:
        bu = str(b).strip().upper()
        match = get_close_matches(bu, sub_ref, n=1, cutoff=0.6)
        if match:
            mapping[b] = sub_nomes[sub_ref.index(match[0])]
        else:
            mapping[b] = b  # fallback
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

    # Ordena unidades por menor contagem
    units_sorted = counts_by_unit.sort_values(ascending=True).index.tolist()

    for unit_name in units_sorted:
        unit_id = id_by_name.get(unit_name)
        if unit_id is None or unit_id in assigned:
            continue

        region_label = f"região: {unit_name.strip().lower()}"
        current = [unit_id]
        assigned[unit_id] = region_label
        total = int(counts_by_unit.get(unit_name, 0))

        # fronteira ordenada por contagem crescente
        neigh_ids = adjacency.get(unit_id, [])
        frontier = sorted(
            neigh_ids,
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
            # adiciona vizinhos do novo nó
            neighs = [x for x in adjacency.get(nid, []) if x not in visited and x not in assigned]
            neighs = sorted(neighs, key=lambda x: counts_by_unit.get(_name_by_id(x, id_by_name), 0))
            frontier.extend(neighs)

        regions[region_label] = current

    # Unidades remanescentes -> anexa à região com menor soma atual
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

# ---------------- Fallback (sem IBGE) ----------------

def _balanced_regions_without_ibge(
    df: pd.DataFrame,
    bairro_col: str,
    min_tx_per_region: int,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    # Agrupa apenas por bairros, balanceando por contagem (sem adjacência)
    counts = df[bairro_col].value_counts()
    bairros_sorted = counts.sort_values(ascending=True).index.tolist()

    regions: Dict[str, List[str]] = {}
    current_region: List[str] = []
    current_total = 0
    blocks: List[Tuple[str, int]] = [(b, int(counts[b])) for b in bairros_sorted]

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

    # Mapeia bairro -> região
    bairro_to_region: Dict[str, str] = {}
    for reg, bs in regions.items():
        for b in bs:
            bairro_to_region[b] = reg

    df_out = df.copy()
    df_out["regiao"] = df_out[bairro_col].map(lambda b: bairro_to_region.get(b, f"região: {str(b).strip().lower()}"))

    return df_out, regions

# ---------------- Public API ----------------

def build_regions_for_recife(
    df: pd.DataFrame,
    bairro_col: str = "bairro",
    min_tx_per_region: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Retorna df com coluna 'regiao' construída a partir de subdistritos IBGE.
    Se a API do IBGE falhar, cai no fallback por bairros (balanceado por contagem).
    """
    if df.empty:
        df = df.copy()
        df["regiao"] = pd.NA
        return df, {}

    # 1) Tenta IBGE
    df_geo = _load_subdistritos_geometries(IBGE_MUN_CODE_RECIFE)
    if not df_geo.empty:
        sub_names = df_geo["sub_nome"].tolist()
        bairros = sorted(pd.Series(df[bairro_col].dropna().unique()).astype(str).tolist())
        bairro_to_sub = _map_bairro_to_subdistrito(bairros, sub_names)

        # contas por subdistrito (via mapeamento)
        df_tmp = df[[bairro_col]].copy()
        df_tmp["sub_nome"] = df_tmp[bairro_col].map(bairro_to_sub)
        counts_by_sub = df_tmp["sub_nome"].value_counts()

        adjacency = _build_adjacency(df_geo)
        id_by_name = dict(zip(df_geo["sub_nome"], df_geo["sub_id"]))

        sub_to_region = _balanced_merge(counts_by_sub, adjacency, id_by_name, min_tx_per_region)

        def to_region(b):
            sname = bairro_to_sub.get(b, b)
            reg = sub_to_region.get(sname)
            return reg or f"região: {str(sname).strip().lower()}"

        df_out = df.copy()
        df_out["regiao"] = df_out[bairro_col].map(to_region)

        regions_dict: Dict[str, List[str]] = {}
        for sname, region in sub_to_region.items():
            regions_dict.setdefault(region, []).append(sname)

        return df_out, regions_dict

    # 2) Fallback sem IBGE
    return _balanced_regions_without_ibge(df, bairro_col=bairro_col, min_tx_per_region=min_tx_per_region)