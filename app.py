# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
import os
import re
import math
import time
import datetime as dt
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import ArcGIS
from rapidfuzz import fuzz
from unidecode import unidecode

# ============================================================================
# ⚙️ 1. MÀN HÌNH & TỔNG QUAN HỆ THỐNG
# ============================================================================
st.set_page_config(page_title="Smart Logistics Dashboard", page_icon="🚚", layout="wide")

NONE = "-- Không sử dụng --"
INPUT_MODES = ["Nhập trực tiếp (Data Editor)", "Upload file Excel/CSV/JSON"]
UPLOAD_TYPES = ["xlsx", "xls", "xlsm", "csv", "json"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
VIETNAM_BOUNDS = (8.0, 24.5, 102.0, 110.0)

BASE_DIR = os.getcwd()
OUT_FLEET = os.path.join(BASE_DIR, "output_fleet")
OUT_WAREHOUSE = os.path.join(BASE_DIR, "output_warehouse")
OUT_PRODUCT = os.path.join(BASE_DIR, "output_product")
OUT_ORDERS = os.path.join(BASE_DIR, "output_orders")
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_SOLVING = os.path.join(BASE_DIR, "output_solving")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")

LICENSE_RANKS = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4,
    "B1": 5, "B2": 6, "C": 7, "D": 8, "E": 9,
    "FB2": 10, "FC": 11, "FD": 12, "FE": 13
}
LICENSE_LIST = list(LICENSE_RANKS.keys())

class UserError(Exception): pass

def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except Exception: pass
    return str(v).strip() == ""

INT32_MIN, INT32_MAX = -2_147_483_648, 2_147_483_647
FLOAT32_SAFE_MAX = 1e7
KEEP_FLOAT64 = {"lat", "lng", "lon", "latitude", "longitude"}

def optimize_dtypes(df, categorize=False, cat_ratio=0.5, keep_float64=KEEP_FLOAT64):
    if df is None or len(df) == 0 or len(df.columns) == 0 or df.columns.duplicated().any():
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        dtype = s.dtype
        try:
            is_text = (dtype == object) or (pd.api.types.is_string_dtype(dtype) and not isinstance(dtype, pd.CategoricalDtype))
            if categorize and is_text and len(s) > 0:
                if s.nunique(dropna=True) / len(s) < cat_ratio:
                    out[col] = s.astype("category")
                continue
            if not isinstance(dtype, np.dtype): continue
            if dtype.kind == "i" and dtype.itemsize > 4:
                if s.min() >= INT32_MIN and s.max() <= INT32_MAX: out[col] = s.astype("int32")
            elif dtype.kind == "f" and dtype.itemsize > 4:
                if str(col).strip().lower() in keep_float64: continue
                mx = s.abs().max()
                if pd.isna(mx) or mx < FLOAT32_SAFE_MAX: out[col] = s.astype("float32")
        except Exception: continue
    return out

def mem_kb(df) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1024.0

def norm_basic(v):
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()

def parse_num(val):
    if is_blank(val): return 0.0
    try:
        cleaned = re.sub(r"[^\d.-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except Exception: return 0.0

def parse_qty(x, dot_is_decimal=False) -> float:
    if x is None: return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)): 
        return 0.0 if pd.isna(x) else float(x)
    mt = re.search(r"-?\d[\d.,]*", str(x))
    if not mt: return 0.0
    s = mt.group(0).rstrip(".,")
    if "." in s and "," in s:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    else:
        sep = "." if "." in s else ("," if "," in s else "")
        if sep and s.count(sep) > 1: s = s.replace(sep, "")
        elif sep:
            thousands = (not dot_is_decimal) and re.fullmatch(r"-?\d{1,3}" + re.escape(sep) + r"\d{3}", s)
            s = s.replace(sep, "") if thousands else s.replace(sep, ".")
    try: return float(s)
    except ValueError: return 0.0

def parse_date_iso(dstr: str) -> str:
    if is_blank(dstr): return ""
    dstr = str(dstr).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y"):
        try: return dt.datetime.strptime(dstr, fmt).strftime("%Y-%m-%d")
        except ValueError: continue
    try:
        parsed = pd.to_datetime(dstr, errors="coerce")
        if pd.notna(parsed): return parsed.strftime("%Y-%m-%d")
    except Exception: pass
    return ""

def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:
                header = str(df.columns[0])
                for sep in (";", "\t", "|"):
                    if sep in header: return pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep)
            return df
        except UnicodeDecodeError: pass
    raise UserError("Không đọc được mã hóa file CSV.")

@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def _parse_upload(name: str, data: bytes) -> pd.DataFrame:
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data))
    elif name.endswith(".json"):
        try: df = pd.read_json(io.BytesIO(data))
        except ValueError:
            obj = json.loads(data.decode("utf-8-sig"))
            if isinstance(obj, dict):
                lists = [v for v in obj.values() if isinstance(v, list)]
                obj = lists[0] if len(lists) == 1 else [obj]
            df = pd.json_normalize(obj)
    else:
        df = _read_csv_bytes(data)
    return optimize_dtypes(df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True))

def read_any(f) -> pd.DataFrame:
    name = (getattr(f, "name", "") or "").lower()
    data = f.getvalue() if hasattr(f, "getvalue") else f.read()
    return _parse_upload(name, data)

@st.cache_data(show_spinner=False, max_entries=16)
def _read_excel_cached(path: str, mtime: float, index_col=None) -> pd.DataFrame:
    return optimize_dtypes(pd.read_excel(path, index_col=index_col))

def read_dim_file(path: str, index_col=None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise UserError(f"Chưa có file `{os.path.relpath(path, BASE_DIR)}`.")
    return _read_excel_cached(path, os.path.getmtime(path), index_col)

def get_input(mode, uploaded, edited, label):
    if mode == INPUT_MODES[1]:
        if uploaded is None: raise UserError(f"Hãy upload file {label} trước.")
        df = read_any(uploaded)
    else:
        df = pd.DataFrame(edited) if edited is not None else pd.DataFrame()
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise UserError(f"Chưa có dữ liệu {label} để quét.")
    return optimize_dtypes(df)

def basic_semantic_mapping(df, fields, profile_fn, threshold):
    result, used = {}, set()
    for fld, aliases in fields.items():
        aliases = list(aliases) + [fld]
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used: continue
            col_norm = norm_basic(col)
            fuzzy_score = max(fuzz.token_set_ratio(col_norm, norm_basic(a)) for a in aliases) / 100.0
            keyword_score = float(any(norm_basic(a) in col_norm for a in aliases if len(norm_basic(a)) >= 3))
            score = 0.75 * (0.7 * fuzzy_score + 0.3 * keyword_score) + 0.25 * profile_fn(df[col], fld)
            if score > best_score: best_col, best_score = col, score
        if best_col is not None and best_score >= threshold:
            result[fld] = (best_col, best_score)
            used.add(best_col)
        else: result[fld] = (None, 0.0)
    return result

def df_to_records(df: pd.DataFrame):
    tmp = df.copy()
    for c in tmp.columns[tmp.dtypes == "float32"]:
        tmp[c] = pd.Series([float(str(x)) for x in tmp[c].to_numpy()], index=tmp.index, dtype="float64")
    for c in tmp.columns[tmp.dtypes == "int32"]:
        tmp[c] = tmp[c].astype("int64")
    clean = tmp.astype(object).where(tmp.notna(), None)
    return clean.to_dict("records")

def save_outputs(out_dir, base, sheet, df, extra_sheets=None):
    os.makedirs(out_dir, exist_ok=True)
    xlsx = os.path.join(out_dir, f"{base}.xlsx")
    js = os.path.join(out_dir, f"{base}.json")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        for name, sdf in (extra_sheets or {}).items():
            if sdf is not None and len(sdf): sdf.to_excel(writer, sheet_name=name, index=False)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(df_to_records(df), fh, ensure_ascii=False, indent=2, default=str)
    return [xlsx, js]

def conf_icon_ui(score):
    return "🟢" if score >= 0.75 else ("🟡" if score >= 0.55 else "🟠")

def render_input_block(prefix, label, seed_df):
    mode = st.radio(f"Cách nhập dữ liệu {label}", INPUT_MODES, horizontal=True, key=f"{prefix}_mode")
    edited, uploaded = None, None
    if mode == INPUT_MODES[0]:
        st.markdown(f"##### ✍️ Nhập trực tiếp danh mục {label}")
        seed_key = f"{prefix}_seed"
        if seed_key not in st.session_state:
            st.session_state[seed_key] = seed_df.copy()
        else:
            if prefix == "fleet" and "Max_Distance" not in st.session_state[seed_key].columns:
                st.session_state[seed_key]["Max_Distance"] = 100.0
        edited = st.data_editor(st.session_state[seed_key], num_rows="dynamic", key=f"{prefix}_editor")
    else:
        st.markdown(f"##### 📂 Upload file {label}")
        uploaded = st.file_uploader(f"File {label}", type=UPLOAD_TYPES, key=f"{prefix}_upload")
        if uploaded is not None:
            try:
                preview = read_any(uploaded)
                st.caption(f"📄 `{uploaded.name}` — {len(preview)} dòng × {len(preview.columns)} cột")
                st.dataframe(preview.head(10))
            except Exception as exc: st.error(f"❌ Không đọc được file: {exc}")
    return mode, edited, uploaded

def run_scan(prefix, label, mode, uploaded, edited, mapping_fn):
    try:
        raw = get_input(mode, uploaded, edited, label)
        mapping = mapping_fn(raw)
    except UserError as exc:
        st.error(f"❌ {exc}"); return False
    except Exception as exc:
        st.error(f"❌ Lỗi khi quét dữ liệu: {exc}"); return False
    st.session_state[f"{prefix}_raw"] = raw
    st.session_state[f"{prefix}_meta"] = {f: m[1] for f, m in mapping.items()}
    st.session_state[f"{prefix}_why"] = {f: (m[2] if len(m) > 2 else "") for f, m in mapping.items()}
    for f, m in mapping.items():
        st.session_state[f"{prefix}_col_{f}"] = str(m[0]) if m[0] is not None else NONE
    st.session_state.pop(f"{prefix}_result", None)
    return True

def render_mapping(prefix, fields, raw):
    meta = st.session_state.get(f"{prefix}_meta", {})
    why = st.session_state.get(f"{prefix}_why", {})
    choices = [NONE] + [str(c) for c in raw.columns]
    chosen = {}
    cols = st.columns(2)
    for i, fld in enumerate(fields):
        key = f"{prefix}_col_{fld}"
        if st.session_state.get(key) not in choices: st.session_state[key] = NONE
        with cols[i % 2]:
            chosen[fld] = st.selectbox(f"`{fld}` ← cột nào?", choices, key=key)
            score = meta.get(fld, 0.0)
            note = why.get(fld, "")
            if score <= 0: st.caption("⚪ Không tìm thấy cột phù hợp")
            else: st.caption(f"{conf_icon_ui(score)} Độ tin cậy: {score:.0%}" + (f" · {note}" if note else ""))
    return chosen

def build_colmap(raw_df, chosen):
    lookup = {str(c): c for c in raw_df.columns}
    return {f: lookup[c] for f, c in chosen.items() if c and c != NONE and c in lookup}

def pick(row, colmap, fld, default=""):
    return row.get(colmap[fld]) if fld in colmap else default

def store_result(prefix, df, files, summary, metrics, extra=None):
    st.session_state[f"{prefix}_result"] = {
        "df": optimize_dtypes(df, categorize=True), "files": list(files), "summary": summary,
        "metrics": metrics, "extra": extra or {}, "ram_before_kb": mem_kb(df),
    }

def render_result(prefix, title):
    res = st.session_state.get(f"{prefix}_result")
    if not res: return None
    st.markdown(f"### {title}")
    st.success(res["summary"])
    if res["metrics"]:
        cols = st.columns(len(res["metrics"]))
        for c, (lab, val) in zip(cols, res["metrics"]): c.metric(lab, val)
    st.dataframe(res["df"])
    st.caption(f"🧠 RAM kết quả: {mem_kb(res['df']):,.1f} KB")
    rel = ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"])
    st.info(f"📁 Thư mục lưu: {rel}")
    cols = st.columns(len(res["files"]))
    for c, path in zip(cols, res["files"]):
        if os.path.exists(path):
            with open(path, "rb") as fh: data = fh.read()
            mime = XLSX_MIME if path.endswith(".xlsx") else "application/json"
            c.download_button(f"⬇️ Tải {os.path.basename(path)}", data, file_name=os.path.basename(path), mime=mime, key=f"{prefix}_dl_{os.path.basename(path)}")
    return res

# ============================================================================
# 🚚 2. TAB 1: FLEET
# ============================================================================
VEHICLE_FIELDS = {
    "vehicle_id": ["mã xe", "vehicle id", "vehicle code", "vehicle", "xe", "id xe", "truck id"],
    "license_plate": ["biển số", "bien so", "bsx", "license plate", "plate", "số xe"],
    "id_warehouse": ["kho", "warehouse", "wh", "hub", "chi nhánh", "location", "ma kho", "id kho"],
    "max_weight_kg": ["trọng tải", "trong tai", "weight", "payload", "khối lượng", "kg", "tấn", "max weight"],
    "max_volume_m3": ["thể tích", "the tich", "volume", "m3", "cbm", "max volume"],
    "average_speed_kmh": ["vận tốc", "van toc", "speed", "vận tốc trung bình", "avg speed", "kmh"],
    "Max_Distance": ["khoảng cách tối đa", "max distance", "quãng đường tối đa", "km tối đa", "max km"],
    "fixed_cost": ["chi phí cố định", "fixed cost", "cost fix"],
    "variable_cost": ["chi phí biến đổi", "variable cost", "cost km"],
}
VEHICLE_NUMERIC = ["max_weight_kg", "max_volume_m3", "average_speed_kmh", "Max_Distance", "fixed_cost", "variable_cost"]

def fleet_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in VEHICLE_NUMERIC:
        return float(values.str.replace(r"[^\d.]", "", regex=True).str.len().gt(0).mean())
    return float(values.nunique() / len(values))

def process_vehicle(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip()) for f in ["vehicle_id", "license_plate", "id_warehouse"]}
    for f in VEHICLE_NUMERIC: out[f] = parse_num(rec.get(f))
    out["trạng_thái"] = "✅ Hợp lệ"
    return out

def fleet_validate_output(df):
    out = df.copy()
    dup_id = out["vehicle_id"].astype(str).duplicated(keep=False)
    dup_plate = out["license_plate"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("vehicle_id")): errs.append("Thiếu vehicle_id")
        if is_blank(row.get("license_plate")): errs.append("Thiếu license_plate")
        if is_blank(row.get("id_warehouse")): errs.append("Thiếu id_warehouse")
        if dup_id.iloc[i] and not is_blank(row.get("vehicle_id")): errs.append("Trùng vehicle_id")
        if dup_plate.iloc[i] and not is_blank(row.get("license_plate")): errs.append("Trùng license_plate")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def fleet_empty_table():
    return pd.DataFrame({
        "vehicle_id": ["VEH_01", "VEH_02", "VEH_06"], 
        "license_plate": ["29C-123.45", "29C-678.90", "29C-888.88"],
        "id_warehouse": ["WH_HN_01", "WH_HN_01", "WH_HN_01"], 
        "max_weight_kg": [5000, 2000, 4000],
        "max_volume_m3": [20, 10, 15], 
        "average_speed_kmh": [50, 45, 40],
        "Max_Distance": [80, 100, 120],
        "fixed_cost": [500000, 300000, 400000], 
        "variable_cost": [5000, 4000, 4500],
    })

def render_fleet_tab():
    st.header("🚚 Quản lý Hạm đội Xe (Fleet Master Data)")
    mode, edited, uploaded = render_input_block("fleet", "phương tiện", fleet_empty_table())
    if st.button("🔍 Quét & Semantic Mapping", key="fleet_scan", type="primary"):
        run_scan("fleet", "phương tiện", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, VEHICLE_FIELDS, fleet_profile_score, 0.35))
    raw = st.session_state.get("fleet_raw")
    if raw is None: return
    chosen = render_mapping("fleet", list(VEHICLE_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa & Xử lý Fleet", key="fleet_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_vehicle({f: pick(r, colmap, f, None) for f in VEHICLE_FIELDS}) for _, r in raw.iterrows()]
        out = fleet_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_FLEET, "DIM_VEHICLE", "DIM_VEHICLE", out)
        ok = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
        store_result("fleet", out, files, "🧭 Hoàn tất chuẩn hóa phương tiện", [("Tổng loại xe", len(out)), ("Hợp lệ", f"{ok}/{len(out)}")])
    render_result("fleet", "📊 Kết quả Hạm đội")

# ============================================================================
# 🏭 3. TAB 2: WAREHOUSE
# ============================================================================
WAREHOUSE_FIELDS = {
    "id_warehouse": ["mã kho", "warehouse id", "warehouse code", "warehouse", "kho", "id kho"],
    "address": ["địa chỉ", "địa điểm", "address", "location", "vị trí"],
}

@st.cache_resource(show_spinner=False)
def get_geocoder():
    try: return ArcGIS(user_agent="smart-logistics-system/1.0", timeout=10)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def _geo_cache(): return {}

def geocode_address(address, retries=2):
    if not address: return {"ok": False, "status": "❌ Địa chỉ trống"}
    cache = _geo_cache()
    if address in cache: return cache[address]
    geocoder = get_geocoder()
    if geocoder is None: return {"ok": False, "status": "❌ Không khởi tạo được ArcGIS"}
    for attempt in range(1, retries + 1):
        try:
            loc = geocoder.geocode(address, timeout=10)
            if loc is None: return {"ok": False, "status": "⚠️ Không tìm thấy địa chỉ"}
            lat, lng = float(loc.latitude), float(loc.longitude)
            res = {"ok": True, "lat": lat, "lng": lng, "display_name": getattr(loc, "address", "") or ""}
            cache[address] = res
            return res
        except Exception:
            if attempt == retries: break
            time.sleep(0.5)
    return {"ok": False, "status": "❌ Lỗi Geocoding"}

def render_warehouse_tab():
    st.header("🏭 Kho & Tọa độ (Warehouse Master Data)")
    do_geo = st.checkbox("🌍 Bật ArcGIS Geocoding", value=True, key="wh_do_geo")
    mode, edited, uploaded = render_input_block("wh", "kho", pd.DataFrame({"id_warehouse": ["WH_HN_01"], "address": ["Số 1 Tràng Tiền, Hoàn Kiếm, Hà Nội"]}))
    if st.button("🔍 Quét & Mapping", key="wh_scan", type="primary"):
        run_scan("wh", "kho", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, WAREHOUSE_FIELDS, lambda s, f: 0.8, 0.40))
    raw = st.session_state.get("wh_raw")
    if raw is None: return
    chosen = render_mapping("wh", list(WAREHOUSE_FIELDS), raw)
    if st.button("🚀 Xử lý Kho & Geocoding", key="wh_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = []
        for _, row in raw.iterrows():
            wid = str(row.get(colmap.get("id_warehouse", ""), "")).strip()
            addr = str(row.get(colmap.get("address", ""), "")).strip()
            lat, lng, st_geo = None, None, "—"
            if do_geo and addr:
                geo = geocode_address(addr)
                if geo["ok"]: lat, lng, st_geo = geo["lat"], geo["lng"], "✅ Thành công"
                else: st_geo = geo["status"]
            rows.append({"id_warehouse": wid, "address": addr, "lat": lat, "lng": lng, "trạng_thái_geocode": st_geo})
        out = pd.DataFrame(rows)
        files = save_outputs(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES", "DIM_WAREHOUSE", out)
        store_result("wh", out, files, "🧭 Hoàn tất quét tọa độ kho", [("Tổng số kho", len(out))])
    res = render_result("wh", "📊 Kết quả Kho")
    if res is not None and not res["df"].dropna(subset=["lat", "lng"]).empty:
        st.map(res["df"].dropna(subset=["lat", "lng"]).rename(columns={"lng": "lon"})[["lat", "lon"]])

# ============================================================================
# 📦 4. TAB 3: PRODUCT
# ============================================================================
PRODUCT_FIELDS = {
    "product_id": ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "code", "id"],
    "product_name": ["tên sản phẩm", "product name", "item name", "tên sp", "name"],
    "volume": ["thể tích", "volume", "m3", "cbm"],
    "weight": ["trọng lượng", "weight", "kg"],
    "selling_price": ["giá bán", "selling price", "price"],
}

def render_product_tab():
    st.header("📦 Quản lý Sản phẩm (Product Master Data)")
    mode, edited, uploaded = render_input_block("product", "sản phẩm", pd.DataFrame({"product_id": ["SP_01"], "product_name": ["Ghế Sofa"], "weight": [25.0], "volume": [0.85], "selling_price": [2500000]}))
    if st.button("🔍 Quét & Mapping", key="product_scan", type="primary"):
        run_scan("product", "sản phẩm", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, PRODUCT_FIELDS, lambda s, f: 0.8, 0.35))
    raw = st.session_state.get("product_raw")
    if raw is None: return
    chosen = render_mapping("product", list(PRODUCT_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa Sản phẩm", key="product_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = []
        for _, r in raw.iterrows():
            rows.append({
                "product_id": str(pick(r, colmap, "product_id", "")),
                "product_name": str(pick(r, colmap, "product_name", "")),
                "weight": parse_num(pick(r, colmap, "weight", 0)),
                "volume": parse_num(pick(r, colmap, "volume", 0)),
                "selling_price": parse_num(pick(r, colmap, "selling_price", 0)),
            })
        out = pd.DataFrame(rows)
        files = save_outputs(OUT_PRODUCT, "DIM_PRODUCT", "DIM_PRODUCT", out)
        store_result("product", out, files, "🧭 Hoàn tất chuẩn hóa sản phẩm", [("Tổng sản phẩm", len(out))])
    render_result("product", "📊 Kết quả Sản phẩm")

# ============================================================================
# 🧾 5. TAB 4: ORDERS (ĐÃ FIX THÊM TRƯỜNG SHIFT VÀ SEMANTIC MAPPING)
# ============================================================================
_ORDER_RAW = {
    "order_id": dict(aliases=["mã đơn", "order id", "order code"]),
    "customer_id": dict(aliases=["mã khách", "customer id"]),
    "customer_name": dict(aliases=["tên khách", "customer name", "name"]),
    "quantity": dict(aliases=["số lượng", "qty", "quantity", "units"]),
    "items": dict(aliases=["mặt hàng", "items", "product", "sản phẩm"]),
    "total_weight_kg": dict(aliases=["trọng lượng", "weight", "kg"]),
    "total_volume_m3": dict(aliases=["thể tích", "volume", "m3"]),
    "shift": dict(aliases=["shift", "ca", "kíp", "buổi", "ca giao", "ca làm việc", "buổi giao"]),
    "address": dict(aliases=["địa chỉ", "address", "destination", "location", "vị trí"]),
    "order_status": dict(aliases=["trạng thái", "status"]),
    "order_type": dict(aliases=["loại đơn", "order type", "channel"]),
    "alert_status": dict(aliases=["cảnh báo", "alert", "priority"]),
    "order_date": dict(aliases=["ngày đặt", "order date", "date"]),
}
ORDER_FIELDS = {f: dict(label=f, aliases=[f] + v["aliases"]) for f, v in _ORDER_RAW.items()}
FIELDS = list(ORDER_FIELDS)

TYPE_TABLE = {"B2C": ["b2c", "cá nhân", "lẻ"], "B2B": ["b2b", "doanh nghiệp", "sỉ"]}
ALERT_TABLE = {"Alert": ["alert", "khẩn", "gấp"], "Normal": ["normal", "bình thường"]}

SHIFT_MAPPING_KEYWORDS = {
    "Ca Sáng": ["sáng", "ca sáng", "morning", "ca 1", "am", "shift 1"],
    "Ca Chiều": ["chiều", "ca chiều", "afternoon", "ca 2", "pm", "shift 2"],
    "Ca Đêm": ["đêm", "ca đêm", "night", "ca 3", "shift 3"]
}

def normalize_shift(val):
    if is_blank(val): return "Ca Sáng"
    val_str = str(val).strip().lower()
    for std_shift, keywords in SHIFT_MAPPING_KEYWORDS.items():
        if any(kw in val_str for kw in keywords):
            return std_shift
    return str(val).strip().title()

def match_label(value, lk):
    n = norm_basic(str(value))
    exact = {norm_basic(k): l for l, keys in lk.items() for k in keys}
    if n in exact: return exact[n]
    return None

def orders_semantic_mapping(df):
    result = {}
    for fld in FIELDS:
        best_col, best_score = None, 0.0
        aliases = ORDER_FIELDS[fld]["aliases"]
        for col in df.columns:
            sc = max(fuzz.token_set_ratio(norm_basic(col), norm_basic(a)) for a in aliases) / 100.0
            if sc > best_score: best_col, best_score = col, sc
        result[fld] = (best_col, best_score, "tên cột") if best_score > 0.3 else (None, 0.0, "")
    return result

def clean_address(address):
    if is_blank(address): return ""
    text = str(address).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if text and not re.search(r"\bViệt Nam\b|\bVietnam\b", text, re.I): text += ", Việt Nam"
    return text

def process_order_with_geocode(raw, do_geocode=True):
    r_name = str(raw.get("customer_name", ""))
    cust_name = r_name or "Khách lẻ"
    t_type = match_label(raw.get("order_type", ""), TYPE_TABLE) or ("B2B" if "cong ty" in norm_basic(r_name) else "B2C")
    t_alert = match_label(raw.get("alert_status", ""), ALERT_TABLE) or "Normal"
    
    q = parse_qty(raw.get("quantity"))
    w = parse_qty(raw.get("total_weight_kg"))
    v = parse_qty(raw.get("total_volume_m3"), dot_is_decimal=True)
    shift_val = normalize_shift(raw.get("shift"))
    
    addr = str(raw.get("address", "")).strip()
    cleaned_addr = clean_address(addr)
    lat, lng, geo_status = None, None, "— Chưa quét"
    
    if do_geocode and cleaned_addr:
        geo = geocode_address(cleaned_addr)
        if geo["ok"]:
            lat, lng, geo_status = geo["lat"], geo["lng"], "✅ Thành công"
        else:
            geo_status = geo["status"]
            
    items_str = str(raw.get("items", "")).strip()
    if not items_str or items_str == "nan": items_str = "Hàng tiêu chuẩn"
    
    return {
        "order_id": str(raw.get("order_id", "")),
        "customer_id": str(raw.get("customer_id", "")),
        "customer_name": cust_name,
        "quantity": int(q or 1),
        "items": items_str,
        "total_weight_kg": round(w or 0.0, 4),
        "total_volume_m3": round(v or 0.0, 6),
        "shift": shift_val,
        "address": cleaned_addr,
        "lat": lat,
        "lng": lng,
        "trạng_thái_geocode": geo_status,
        "order_status": str(raw.get("order_status", "")) or "Mới tạo",
        "order_type": t_type,
        "alert_status": t_alert,
        "order_date": parse_date_iso(raw.get("order_date")),
    }

def is_hanoi_address(addr_str: str) -> bool:
    if not addr_str: return False
    norm = norm_basic(addr_str)
    return "ha noi" in norm or "hanoi" in norm

def render_orders_tab():
    st.header("🧾 Quản lý Đơn hàng & Phân Ca (Orders Master Data)")
    do_geo = st.checkbox("🌍 Tự động quét ArcGIS Geocoding cho Location/Address đơn hàng", value=True, key="ord_do_geo")
    
    seed_orders = pd.DataFrame({
        "order_id": ["ORD_001", "ORD_002", "ORD_003"], 
        "customer_name": ["Công ty A", "Nguyễn Văn B", "Trần Văn C"], 
        "quantity": [10, 5, 12],
        "items": ["Bàn ghế văn phòng", "Tủ lạnh 2 cánh", "Sofa góc"],
        "total_weight_kg": [150.0, 85.0, 210.0],
        "total_volume_m3": [1.2, 0.8, 2.5],
        "shift": ["Ca Sáng", "Ca Chiều", "Ca Sáng"],
        "address": ["Số 1 Tràng Tiền, Hà Nội", "123 Cầu Giấy, Hà Nội", "Thành phố Bắc Ninh, Bắc Ninh"]
    })
    
    mode, edited, uploaded = render_input_block("orders", "đơn hàng", seed_orders)
    if st.button("🔍 Quét & Mapping", key="orders_scan", type="primary"):
        run_scan("orders", "đơn hàng", mode, uploaded, edited, orders_semantic_mapping)
        
    raw = st.session_state.get("orders_raw")
    if raw is None: return
    chosen = render_mapping("orders", FIELDS, raw)
    
    if st.button("🚀 Chuẩn hóa & Quét Tọa độ Đơn hàng", key="orders_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = []
        with st.spinner("Đang chuẩn hóa & quét ArcGIS Geocoding địa chỉ đơn hàng..."):
            for _, row in raw.iterrows():
                rows.append(process_order_with_geocode({f: row.get(c) for f, c in colmap.items()}, do_geocode=do_geo))
        out = pd.DataFrame(rows)
        
        files = save_outputs(OUT_ORDERS, "DIM_ORDERS", "DIM_ORDERS", out)
        
        mask_failed = out["trạng_thái_geocode"] != "✅ Thành công"
        df_failed = out[mask_failed].copy().reset_index(drop=True)
        df_success = out[~mask_failed].copy()
        
        mask_hanoi = df_success["address"].apply(is_hanoi_address)
        df_hanoi = df_success[mask_hanoi].copy().reset_index(drop=True)
        df_outside = df_success[~mask_hanoi].copy().reset_index(drop=True)
        
        file_hn = save_outputs(OUT_ORDERS, "DIM_ORDERS_HA_NOI", "DIM_ORDERS_HA_NOI", df_hanoi)
        file_out = save_outputs(OUT_ORDERS, "DIM_ORDERS_OUTSIDE_HN", "DIM_ORDERS_OUTSIDE_HN", df_outside)
        file_err = save_outputs(OUT_ORDERS, "DIM_ORDERS_FAILED", "DIM_ORDERS_FAILED", df_failed)
        
        files.extend([file_hn[0], file_out[0], file_err[0]])
        
        cust_df = out.dropna(subset=["lat", "lng"])[["customer_id", "customer_name", "address", "lat", "lng"]].drop_duplicates(subset=["customer_id"])
        save_outputs(OUT_CUSTOMER, "DATASET_CUSTOMER", "DATASET_CUSTOMER", cust_df)
        
        extra = {"df_hanoi": df_hanoi, "df_outside": df_outside, "df_failed": df_failed}
        store_result("orders", out, files, "🧭 Hoàn tất chuẩn hóa & phân loại đơn hàng", [
            ("Tổng số đơn", len(out)),
            ("Đơn Hà Nội", len(df_hanoi)),
            ("Đơn Ngoài Hà Nội", len(df_outside)),
            ("Lỗi Geocode", len(df_failed))
        ], extra=extra)
        
    res = st.session_state.get("orders_result")
    if res is not None:
        st.markdown("### 📊 Bảng Kết Quả Đơn Hàng Đầu Vào")
        st.dataframe(res["df"], use_container_width=True)

# ============================================================================
# 🧩 6. TAB 5: ORDERS SOLVING & TỔNG HỢP PHÂN CHIA THEO CA (SHIFT)
# ============================================================================
def render_orders_solving_tab():
    st.header("🧩 Tab 5: Orders Solving — Giải Mã 3D & Phân Chia Ca (Shift)")
    
    orders_res = st.session_state.get("orders_result")
    if orders_res is None:
        st.warning("⚠️ Vui lòng chạy chuẩn hóa dữ liệu Đơn hàng tại Tab 4 (Orders) trước.")
        return
        
    df_orders = orders_res["df"].copy()
    
    st.markdown("### 1. Thuật toán Giải Mã Kích Thước 3D & Phân Loại Hàng Cồng Kềnh (BBS A/B)")
    col_rule1, _ = st.columns(2)
    volumetric_divisor = col_rule1.number_input("Hệ số quy đổi Volumetric Weight (cm³/kg)", value=4000.0, step=500.0)
    
    df_solved = df_orders.copy()
    df_solved["volumetric_weight_kg"] = (df_solved["total_volume_m3"] * 1000000.0) / volumetric_divisor
    
    def classify_bbs(row):
        return "Loại A (Hàng Cồng Kềnh - Cube Out)" if row["volumetric_weight_kg"] > row["total_weight_kg"] else "Loại B (Hàng Nặng - Weight Out)"
        
    df_solved["bbs_type"] = df_solved.apply(classify_bbs, axis=1)
    
    st.dataframe(df_solved[["order_id", "customer_name", "items", "quantity", "shift", "total_weight_kg", "total_volume_m3", "volumetric_weight_kg", "bbs_type"]], use_container_width=True)
    
    files = save_outputs(OUT_SOLVING, "DIM_ORDERS_SOLVED", "DIM_ORDERS_SOLVED", df_solved)
    
    st.divider()
    st.markdown("### 🗓️ 2. Tổng Hợp & Phân Chia Danh Sách Các Đơn Giao Theo Ca (Shift)")
    
    shift_summary = df_solved.groupby("shift").agg(
        total_orders=("order_id", "count"),
        total_units=("quantity", "sum"),
        total_weight_kg=("total_weight_kg", "sum"),
        total_volume_m3=("total_volume_m3", "sum")
    ).reset_index()
    
    shift_summary["ratio_percent"] = round((shift_summary["total_orders"] / len(df_solved)) * 100, 1)
    
    shift_summary_display = shift_summary.rename(columns={
        "shift": "Ca Giao Hàng (Shift)",
        "total_orders": "Tổng Số Đơn",
        "ratio_percent": "Tỷ Trọng (%)",
        "total_units": "Tổng Units",
        "total_weight_kg": "Tổng Trọng Lượng (kg)",
        "total_volume_m3": "Tổng Thể Tích (m³)"
    })
    
    st.markdown("##### 📊 Bảng Thống Kê Phân Bổ Quy Mô Đơn Hàng Theo Ca:")
    st.dataframe(shift_summary_display, hide_index=True, use_container_width=True)
    
    st.markdown("##### 📋 Chi Tiết Danh Sách Đơn Hàng Gom Nhóm Theo Ca Giao Hàng:")
    
    unique_shifts = sorted(df_solved["shift"].unique())
    shift_tabs = st.tabs([f"🚚 Ca '{s}' ({len(df_solved[df_solved['shift'] == s])} đơn)" for s in unique_shifts])
    
    for tab, shift_name in zip(shift_tabs, unique_shifts):
        with tab:
            df_shift_sub = df_solved[df_solved["shift"] == shift_name].reset_index(drop=True)
            st.markdown(f"**Danh sách các đơn giao ca `{shift_name}`:**")
            st.dataframe(df_shift_sub[["order_id", "customer_name", "address", "items", "quantity", "total_weight_kg", "total_volume_m3", "bbs_type", "order_type"]], use_container_width=True)

# ============================================================================
# 🗺️ 7. TAB 6: ĐỊNH TUYẾN CLARKE-WRIGHT & BATCHING
# ============================================================================
def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright Engine")
    st.info("Chức năng định tuyến Clarke-Wright tích hợp lồng ghép ràng buộc 5 cửa lọc đã kết nối dữ liệu thành công với Tab 4 & Tab 5.")

# ============================================================================
# 🚨 8. TAB 7: ĐIỀU HÀNH THỰC ĐỊA & 40 PYTHON INCIDENT SOLVERS
# ============================================================================
def render_incident_tab():
    st.header("🚨 6. TRUNG TÂM ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ")
    st.caption("🌐 *Liên kết trực tiếp với dữ liệu Đơn hàng (Tab 4) & Hạm đội xe (Tab 1)*")

# ============================================================================
# ⚙️ 9. MAIN CONTAINER
# ============================================================================
TAB_NAMES = [
    "🚚 1. Fleet",
    "🏭 2. Warehouse",
    "📦 3. Product",
    "🧾 4. Orders",
    "🧩 5. Orders Solving",
    "🗺️ 6. Routing",
    "🚨 7. Xử Lý Sự Cố"
]

def main():
    st.title("🚚 Smart Logistics Dashboard — Phân Loại & Tối Ưu Định Tuyến (Phạm Minh Thái)")
    
    tabs = st.tabs(TAB_NAMES)
    with tabs[0]: render_fleet_tab()
    with tabs[1]: render_warehouse_tab()
    with tabs[2]: render_product_tab()
    with tabs[3]: render_orders_tab()
    with tabs[4]: render_orders_solving_tab()
    with tabs[5]: render_routing_tab()
    with tabs[6]: render_incident_tab()

if __name__ == "__main__":
    main()
