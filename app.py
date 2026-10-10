# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import ArcGIS
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process
from unidecode import unidecode

st.set_page_config(page_title="Smart Logistics", page_icon="🚚", layout="wide")

NONE = "-- Không sử dụng --"
INPUT_MODES = ["Nhập trực tiếp (Data Editor)", "Upload file Excel/CSV/JSON"]
UPLOAD_TYPES = ["xlsx", "xls", "xlsm", "csv", "json"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
VIETNAM_BOUNDS = (8.0, 24.0, 102.0, 110.0)
MAP_THRESHOLD = 0.38

BASE_DIR = os.getcwd()
OUT_FLEET = os.path.join(BASE_DIR, "output_fleet")
OUT_WAREHOUSE = os.path.join(BASE_DIR, "output_warehouse")
OUT_PRODUCT = os.path.join(BASE_DIR, "output_product")
OUT_DRIVER = os.path.join(BASE_DIR, "output_driver")
OUT_ORDERS = os.path.join(BASE_DIR, "output_orders")
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")
OUT_SCREEN = os.path.join(BASE_DIR, "output_screening")

DEFAULT_DEPOT = (21.0285, 105.8542)

class UserError(Exception):
    pass

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

def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_exc = None
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:
                header = str(df.columns[0])
                for sep in (";", "\t", "|"):
                    if sep in header: return pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep)
            return df
        except UnicodeDecodeError as exc: last_exc = exc
    raise last_exc

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
            if score > best_score:
                best_col, best_score = col, score
        if best_col is not None and best_score >= threshold:
            result[fld] = (best_col, best_score)
            used.add(best_col)
        else:
            result[fld] = (None, 0.0)
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

def parse_num(val):
    if is_blank(val): return 0.0
    try:
        cleaned = re.sub(r"[^\d.-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except Exception: return 0.0

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
    st.caption(f"🧠 RAM kết quả: {mem_kb(res['df']):,.1f} KB (gốc {res.get('ram_before_kb', 0):,.1f} KB)")
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
# 🚚 TAB 1: FLEET (HẠM ĐỘI XE)
# ============================================================================
VEHICLE_FIELDS = {
    "vehicle_id": ["mã xe", "vehicle id", "vehicle code", "vehicle", "xe", "id xe", "truck id", "mã phương tiện"],
    "license_plate": ["biển số", "bien so", "bsx", "license plate", "plate", "số xe", "so xe"],
    "id_warehouse": ["kho", "warehouse", "wh", "hub", "chi nhánh", "location", "ma kho", "id kho", "khu vực", "warehouse id"],
    "max_weight_kg": ["trọng tải", "trong tai", "weight", "payload", "khối lượng", "khoi luong", "kg", "tấn", "tan", "capacity kg", "max weight"],
    "max_volume_m3": ["thể tích", "the tich", "volume", "m3", "cbm", "capacity m3", "max volume"],
    "average_speed_kmh": ["vận tốc", "van toc", "speed", "vận tốc trung bình", "toc do trung binh", "avg speed", "kmh", "km/h", "average speed"],
    "Max_Distance": ["khoảng cách tối đa", "khoang cach toi da", "max distance", "maximum distance", "distance limit", "quãng đường tối đa", "km tối đa", "max km"],
    "fixed_cost": ["chi phí cố định", "chi phi co dinh", "fixed cost", "cost fix", "fixed"],
    "variable_cost": ["chi phí biến đổi", "chi phi bien doi", "variable cost", "variable", "cost km", "chi phí theo km"],
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
        "vehicle_id": ["VEH_01", "VEH_02"], 
        "license_plate": ["29C-123.45", "29C-678.90"],
        "id_warehouse": ["WH_HN_01", "WH_HN_01"], 
        "max_weight_kg": [5000, 2000],
        "max_volume_m3": [20, 10], 
        "average_speed_kmh": [50, 45],
        "Max_Distance": [80, 100],
        "fixed_cost": [500000, 300000], 
        "variable_cost": [5000, 4000],
    })

def render_fleet_tab():
    st.header("🚚 Quản lý Hạm đội Xe (Fleet)")
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
# 🏭 TAB 2: WAREHOUSE (KHO & TỌA ĐỘ)
# ============================================================================
WAREHOUSE_FIELDS = {
    "id_warehouse": ["mã kho", "warehouse id", "warehouse code", "warehouse", "kho", "id kho"],
    "address": ["địa chỉ", "địa điểm", "address", "location", "vị trí"],
}

def coordinate_in_vietnam(lat, lng):
    try: lat, lng = float(lat), float(lng)
    except Exception: return False
    return VIETNAM_BOUNDS[0] <= lat <= VIETNAM_BOUNDS[1] and VIETNAM_BOUNDS[2] <= lng <= VIETNAM_BOUNDS[3]

ADDRESS_ABBR = [(r"\bTP\.?\b", "Thành phố"), (r"\bQ\.?\b", "Quận"), (r"\bH\.?\b", "Huyện"), (r"\bTX\.?\b", "Thị xã"),
                (r"\bTT\.?\b", "Thị trấn"), (r"\bP\.?\b", "Phường"), (r"\bX\.?\b", "Xã"), (r"\bĐg\.?\b", "Đường")]

def clean_address(address):
    if is_blank(address): return ""
    text = str(address).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u3000]", " ", text)
    text = re.sub(r"\s*[|;→–—]\s*", ", ", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s,./'-]", " ", text)
    for p, r in ADDRESS_ABBR: text = re.sub(p, r, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if text and not re.search(r"\bViệt Nam\b|\bVietnam\b", text, re.I): text += ", Việt Nam"
    return text

def address_quality(address):
    if not address: return 0.0, "❌ Địa chỉ trống"
    score = 0.5 if len(address) >= 10 else 0.25
    if "," in address: score += 0.3
    if re.search(r"\b(Phường|Xã|Quận|Huyện|Thành phố|Tỉnh)\b", address, re.I): score += 0.2
    return min(score, 1.0), "✅ Hợp lệ"

@st.cache_resource(show_spinner=False)
def get_geocoder():
    try: return ArcGIS(user_agent="smart-logistics-warehouse/1.0", timeout=10)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def _geo_cache(): return {}

def geocode_address(address, retries=3):
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
            if not coordinate_in_vietnam(lat, lng): return {"ok": False, "status": "⚠️ Tọa độ ngoài VN"}
            res = {"ok": True, "lat": lat, "lng": lng, "display_name": getattr(loc, "address", "") or "", "score": 1.0}
            cache[address] = res
            return res
        except Exception:
            if attempt == retries: break
            time.sleep(1)
    return {"ok": False, "status": "❌ Lỗi Geocoding"}

def process_warehouse(warehouse_id, address, do_geocode=True):
    raw_address = "" if is_blank(address) else str(address).strip()
    cleaned = clean_address(raw_address)
    quality, clean_status = address_quality(cleaned)
    result = {"id_warehouse": "" if is_blank(warehouse_id) else str(warehouse_id).strip(), "address": cleaned,
              "lat": None, "lng": None, "chất_lượng_địa_chỉ": quality, "trạng_thái_geocode": "—"}
    if is_blank(warehouse_id) or not cleaned or not do_geocode: return result
    geo = geocode_address(cleaned)
    if not geo["ok"]:
        result["trạng_thái_geocode"] = geo["status"]; return result
    result.update({"lat": geo["lat"], "lng": geo["lng"], "trạng_thái_geocode": "✅ Thành công"})
    return result

def warehouse_validate_output(df):
    out = df.copy()
    msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("id_warehouse")): errs.append("Thiếu id")
        if is_blank(row.get("address")): errs.append("Thiếu địa chỉ")
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")): errs.append("Thiếu tọa độ")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_warehouse_tab():
    st.header("🏭 Kho & Tọa độ (Warehouse)")
    do_geo = st.checkbox("🌍 Bật ArcGIS Geocoding", value=True, key="wh_do_geo")
    mode, edited, uploaded = render_input_block("wh", "kho", pd.DataFrame({"id_warehouse": ["WH_HN_01"], "address": ["Số 1 Tràng Tiền, Hoàn Kiếm, Hà Nội"]}))
    if st.button("🔍 Quét & Mapping", key="wh_scan", type="primary"):
        run_scan("wh", "kho", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, WAREHOUSE_FIELDS, lambda s, f: 0.8, 0.40))
    raw = st.session_state.get("wh_raw")
    if raw is None: return
    chosen = render_mapping("wh", list(WAREHOUSE_FIELDS), raw)
    if st.button("🚀 Xử lý Kho & Geocoding", key="wh_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_warehouse(row.get(colmap.get("id_warehouse")), row.get(colmap.get("address")), do_geocode=do_geo) for _, row in raw.iterrows()]
        out = warehouse_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES", "DIM_WAREHOUSE", out)
        store_result("wh", out, files, "🧭 Hoàn tất quét tọa độ kho", [("Tổng số kho", len(out))])
    res = render_result("wh", "📊 Kết quả Kho")
    if res is not None and not res["df"].dropna(subset=["lat", "lng"]).empty:
        st.map(res["df"].dropna(subset=["lat", "lng"]).rename(columns={"lng": "lon"})[["lat", "lon"]])

# ============================================================================
# 📦 TAB 3: PRODUCT (SẢN PHẨM)
# ============================================================================
PRODUCT_FIELDS = {
    "product_id": ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "code", "id"],
    "product_name": ["tên sản phẩm", "product name", "item name", "tên sp", "name", "mô tả"],
    "volume": ["thể tích", "the tich", "volume", "m3", "cbm"],
    "weight": ["trọng lượng", "trong luong", "khối lượng", "weight", "kg"],
    "length": ["dài", "dai", "length", "l"],
    "width": ["rộng", "rong", "width", "w"],
    "height": ["cao", "height", "h"],
    "cost_price": ["giá sản xuất", "cost price", "cost", "giá vốn"],
    "selling_price": ["giá bán", "selling price", "price", "unit price"],
}
PRODUCT_NUMERIC = ["volume", "weight", "length", "width", "height", "cost_price", "selling_price"]

def process_product(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip()) for f in ["product_id", "product_name"]}
    for f in PRODUCT_NUMERIC: out[f] = parse_num(rec.get(f))
    out["trạng_thái"] = "✅ Hợp lệ"
    return out

def product_validate_output(df):
    out = df.copy()
    msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("product_id")): errs.append("Thiếu id")
        if is_blank(row.get("product_name")): errs.append("Thiếu tên")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_product_tab():
    st.header("📦 Quản lý Sản phẩm (Product)")
    mode, edited, uploaded = render_input_block("product", "sản phẩm", pd.DataFrame({"product_id": ["SP_01"], "product_name": ["Ghế Sofa"], "weight": [25.0], "selling_price": [2500000]}))
    if st.button("🔍 Quét & Mapping", key="product_scan", type="primary"):
        run_scan("product", "sản phẩm", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, PRODUCT_FIELDS, lambda s, f: 0.8, 0.35))
    raw = st.session_state.get("product_raw")
    if raw is None: return
    chosen = render_mapping("product", list(PRODUCT_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa Sản phẩm", key="product_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_product({f: pick(r, colmap, f, None) for f in PRODUCT_FIELDS}) for _, r in raw.iterrows()]
        out = product_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_PRODUCT, "DIM_PRODUCT", "DIM_PRODUCT", out)
        store_result("product", out, files, "🧭 Hoàn tất chuẩn hóa sản phẩm", [("Tổng sản phẩm", len(out))])
    render_result("product", "📊 Kết quả Sản phẩm")

# ============================================================================
# 👨‍✈️ TAB 4: DRIVER (TÀI XẾ)
# ============================================================================
DRIVER_FIELDS = {
    "driver_id": ["mã tài xế", "driver id", "driver code", "mã nv", "staff id", "id"],
    "driver_name": ["họ và tên", "tên tài xế", "full name", "name", "họ tên"],
    "license_type": ["loại bằng", "bằng lái", "license", "class"],
    "id_warehouse": ["kho", "warehouse", "trạm", "hub", "id kho"],
    "address": ["địa chỉ", "address"],
    "phone": ["số điện thoại", "phone", "sdt"],
    "role": ["vị trí", "role", "chức vụ"],
}

def process_driver(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip()) for f in DRIVER_FIELDS}
    out["license_type"] = out["license_type"].upper()
    out["trạng_thái"] = "✅ Hợp lệ"
    return out

def driver_validate_output(df):
    out = df.copy()
    msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("driver_id")): errs.append("Thiếu id")
        if is_blank(row.get("driver_name")): errs.append("Thiếu tên")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_driver_tab():
    st.header("👨‍✈️ Quản lý Tài xế (Driver)")
    mode, edited, uploaded = render_input_block("driver", "tài xế", pd.DataFrame({"driver_id": ["DRV_01"], "driver_name": ["Nguyễn Văn A"], "license_type": ["FC"], "id_warehouse": ["WH_HN_01"], "phone": ["0901234567"]}))
    if st.button("🔍 Quét & Mapping", key="driver_scan", type="primary"):
        run_scan("driver", "tài xế", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, DRIVER_FIELDS, lambda s, f: 0.8, 0.35))
    raw = st.session_state.get("driver_raw")
    if raw is None: return
    chosen = render_mapping("driver", list(DRIVER_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa Tài xế", key="driver_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_driver({f: pick(r, colmap, f, None) for f in DRIVER_FIELDS}) for _, r in raw.iterrows()]
        out = driver_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_DRIVER, "DIM_DRIVER", "DIM_DRIVER", out)
        store_result("driver", out, files, "🧭 Hoàn tất chuẩn hóa tài xế", [("Tổng tài xế", len(out))])
    render_result("driver", "📊 Kết quả Tài xế")

# ============================================================================
# 🧾 TAB 5: ORDERS (ĐƠN HÀNG)
# ============================================================================
_ORDER_RAW = {
    "order_id": dict(aliases=["mã đơn", "order id", "order code"], negative=[]),
    "customer_id": dict(aliases=["mã khách", "customer id"], negative=[]),
    "customer_name": dict(aliases=["tên khách", "customer name", "name"], negative=[]),
    "quantity": dict(aliases=["số lượng", "qty", "quantity"], negative=[]),
    "items": dict(aliases=["mặt hàng", "items", "product"], negative=[]),
    "total_weight_kg": dict(aliases=["trọng lượng", "weight", "kg"], negative=[]),
    "total_volume_m3": dict(aliases=["thể tích", "volume", "m3"], negative=[]),
    "address": dict(aliases=["địa chỉ", "address", "destination"], negative=[]),
    "order_status": dict(aliases=["trạng thái", "status"], negative=[]),
    "order_type": dict(aliases=["loại đơn", "order type", "channel"], negative=[]),
    "alert_status": dict(aliases=["cảnh báo", "alert", "priority"], negative=[]),
    "order_date": dict(aliases=["ngày đặt", "order date", "date"], negative=[]),
}
ORDER_FIELDS = {f: dict(label=f, aliases=[f] + v["aliases"], negative=v["negative"]) for f, v in _ORDER_RAW.items()}
FIELDS = list(ORDER_FIELDS)

TYPE_TABLE = {"B2C": ["b2c", "cá nhân", "lẻ"], "B2B": ["b2b", "doanh nghiệp", "công ty", "sỉ"]}
ALERT_TABLE = {"Alert": ["alert", "khẩn", "gấp", "urgent"], "Normal": ["normal", "bình thường"]}
STATUS_TABLE = {"Mới tạo": ["new", "mới"], "Đang xử lý": ["processing", "đang xử lý"], "Đang giao": ["shipping"], "Đã giao": ["delivered"]}

def build_lookup(table):
    exact = {norm_basic(k): l for l, keys in table.items() for k in keys}
    return {"exact": exact, "keys": list(exact.keys())}

TYPE_LK, ALERT_LK, STATUS_LK = build_lookup(TYPE_TABLE), build_lookup(ALERT_TABLE), build_lookup(STATUS_TABLE)

def match_label(value, lk):
    n = norm_basic(str(value))
    if n in lk["exact"]: return lk["exact"][n], "khớp từ điển"
    return None

def parse_measure(val, units):
    if is_blank(val): return None, ""
    try:
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return float(cleaned) if cleaned else 0.0, ""
    except Exception: return None, "lỗi số"

def parse_volume(val):
    return parse_measure(val, {"m3": 1, "cbm": 1})

def parse_date_iso(v):
    if is_blank(v): return ""
    d = pd.to_datetime(v, errors="coerce", dayfirst=True)
    return "" if pd.isna(d) else d.strftime("%Y-%m-%d")

def orders_semantic_mapping(df):
    result = {}
    for fld in FIELDS:
        best_col, best_score = None, 0.0
        for col in df.columns:
            sc = fuzz.token_set_ratio(norm_basic(col), norm_basic(fld)) / 100.0
            if sc > best_score: best_col, best_score = col, sc
        result[fld] = (best_col, best_score, "tên cột") if best_score > 0.3 else (None, 0.0, "")
    return result

def process_order(raw, reports):
    r_name = str(raw.get("customer_name", ""))
    cust_name = r_name or "Khách lẻ"
    t_type = "B2C"
    r_type = str(raw.get("order_type", ""))
    if r_type:
        m = match_label(r_type, TYPE_LK)
        if m: t_type = m[0]
        elif "cong ty" in norm_basic(r_name) or "tnhh" in norm_basic(r_name): t_type = "B2B"
    t_alert = "Normal"
    r_alert = str(raw.get("alert_status", ""))
    if r_alert:
        m = match_label(r_alert, ALERT_LK)
        if m: t_alert = m[0]
    status = str(raw.get("order_status", "")) or "Mới tạo"
    q, _ = parse_measure(raw.get("quantity"), {})
    w, _ = parse_measure(raw.get("total_weight_kg"), {"kg": 1})
    v, _ = parse_volume(raw.get("total_volume_m3"))
    return {
        "order_id": str(raw.get("order_id", "")),
        "customer_id": str(raw.get("customer_id", "")),
        "customer_name": cust_name,
        "quantity": int(q or 1),
        "items": str(raw.get("items", "")),
        "total_weight_kg": round(w or 0.0, 4),
        "total_volume_m3": round(v or 0.0, 6),
        "address": str(raw.get("address", "")),
        "order_status": status,
        "order_type": t_type,
        "alert_status": t_alert,
        "order_date": parse_date_iso(raw.get("order_date")),
        "ghi_chú": "",
    }

def orders_validate_output(df):
    out = df.copy()
    msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("order_id")): errs.append("Thiếu id")
        if is_blank(row.get("address")): errs.append("Thiếu địa chỉ")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_orders_tab():
    st.header("🧾 Quản lý Đơn hàng (Orders)")
    mode, edited, uploaded = render_input_block("orders", "đơn hàng", pd.DataFrame({"order_id": ["ORD_001"], "customer_name": ["Nguyễn Văn A"], "address": ["Hà Nội"], "total_weight_kg": [10.5]}))
    if st.button("🔍 Quét & Mapping", key="orders_scan", type="primary"):
        run_scan("orders", "đơn hàng", mode, uploaded, edited, orders_semantic_mapping)
    raw = st.session_state.get("orders_raw")
    if raw is None: return
    chosen = render_mapping("orders", FIELDS, raw)
    if st.button("🚀 Chuẩn hóa Đơn hàng", key="orders_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        reports, rows = Counter(), []
        for _, row in raw.iterrows():
            rows.append(process_order({f: row.get(c) for f, c in colmap.items()}, reports))
        out = orders_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_ORDERS, "DIM_ORDERS", "DIM_ORDERS", out)
        store_result("orders", out, files, "🧭 Hoàn tất chuẩn hóa đơn hàng", [("Tổng đơn", len(out))])
    render_result("orders", "📊 Kết quả Đơn hàng")

# ============================================================================
# 🗺️ TAB 6: ĐỊNH TUYẾN CLARKE-WRIGHT & RÀNG BUỘC VẬN TẢI
# ============================================================================
SCHEMA_CONTRACT = {
    "output_fleet/DIM_VEHICLE.xlsx": list(VEHICLE_FIELDS),
    "output_warehouse/WAREHOUSE_WITH_COORDINATES.xlsx": ["id_warehouse", "address", "lat", "lng"],
    "output_product/DIM_PRODUCT.xlsx": list(PRODUCT_FIELDS),
    "output_driver/DIM_DRIVER.xlsx": list(DRIVER_FIELDS),
    "output_orders/DIM_ORDERS.xlsx": FIELDS,
}

OC_DEFAULTS = {
    "full_price": 2_500_000.0, "saving_1_price": 800_000.0, "saving_2_price": 300_000.0,
    "full_vehicle_name": "Xe tải thuê ngoài", "full_vehicle_max_weight_kg": 10_000.0, "full_vehicle_max_volume_m3": 40.0,
    "tier2_max_w": 20.0, "tier2_max_v": 1.0,
    "tier1_max_w": 100.0, "tier1_max_v": 5.0,
}

def calculate_effective_max_hours(start_time_str="08:30", max_hours_input=8.0):
    """Tính thời lượng ca làm việc hiệu dụng đến mốc 17:30 chiều."""
    try:
        sh, sm = map(int, start_time_str.split(":"))
        start_in_hours = sh + sm / 60.0
    except Exception:
        start_in_hours = 8.5
        
    cutoff_hours = 17.5  # 17:30
    remaining_hours = max(0.0, cutoff_hours - start_in_hours)
    return min(float(max_hours_input), remaining_hours)

def _full_cls(w, v, oc):
    fw, fv = max(float(oc["full_vehicle_max_weight_kg"]), 1e-9), max(float(oc["full_vehicle_max_volume_m3"]), 1e-9)
    trips = max(1, math.ceil(w / fw - 1e-9), math.ceil(v / fv - 1e-9))
    return {"type": f"Full — {oc['full_vehicle_name']} × {trips} chuyến", "tier": "FULL", "cost": trips * float(oc["full_price"]),
            "trips": trips, "vehicle": oc["full_vehicle_name"], "cap_w": trips * fw}

def classify_outsourcing(excess_w, excess_v, oc=None, force_full=False) -> dict:
    oc = {**OC_DEFAULTS, **(oc or {})}
    w, v = max(float(excess_w), 0.0), max(float(excess_v), 0.0)
    if not force_full:
        if w < oc["tier2_max_w"] and v < oc["tier2_max_v"]:
            return {"type": "Tiết kiệm loại 2", "tier": "SAVING_2", "cost": float(oc["saving_2_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 2", "cap_w": float(oc["tier2_max_w"])}
        if w <= oc["tier1_max_w"] and v <= oc["tier1_max_v"]:
            return {"type": "Tiết kiệm loại 1", "tier": "SAVING_1", "cost": float(oc["saving_1_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 1", "cap_w": float(oc["tier1_max_w"])}
    return _full_cls(w, v, oc)

def evaluate_transportation_constraints(route_or_order, df_vehicles, vehicle_available_counts, outsourcing_config):
    """Đánh giá trọng tải, hạm đội và Max_Distance."""
    oc = {**OC_DEFAULTS, **(outsourcing_config or {})}
    total_w = float(route_or_order.get("total_weight_kg", 0.0))
    total_v = float(route_or_order.get("total_volume_m3", 0.0))
    total_dist = float(route_or_order.get("total_distance_km", 0.0))
    
    def outsourced(action, cls, message):
        return {"status": "OUTSOURCED", "action_type": action, "assigned_vehicle": None, "outsourcing_type": cls["type"],
                "cost": cls["cost"], "cls": cls, "message": message}
                
    if df_vehicles.empty:
        return outsourced("NO_FLEET", _full_cls(total_w, total_v, oc), "⚠️ Hạm đội trống -> Thuê ngoài Full.")
        
    max_fleet_w, max_fleet_v = df_vehicles["max_weight_kg"].max(), df_vehicles["max_volume_m3"].max()
    sum_fleet_w, sum_fleet_v = df_vehicles["max_weight_kg"].sum(), df_vehicles["max_volume_m3"].sum()
    avail = df_vehicles[df_vehicles["vehicle_id"].map(lambda x: vehicle_available_counts.get(x, 1) > 0)]
    
    if total_w > sum_fleet_w or total_v > sum_fleet_v:
        return outsourced("OUTSOURCE_FULL", _full_cls(total_w, total_v, oc),
                        "🚨 Vượt quá tổng sức chứa toàn bộ hạm đội nhà -> Thuê ngoài loại Full theo chuyến.")
                        
    if total_w > max_fleet_w or total_v > max_fleet_v:
        if avail.empty:
            return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE", classify_outsourcing(total_w, total_v, oc),
                            "⚠️ Hết xe khả dụng trong ngày -> thuê ngoài toàn bộ qua màn lọc.")
        big = avail.loc[avail["max_weight_kg"].idxmax()]
        v_id = big["vehicle_id"]
        vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
        cls = classify_outsourcing(total_w - big["max_weight_kg"], total_v - big["max_volume_m3"], oc)
        return {"status": "PARTIAL_SPLIT", "action_type": "MAX_VEHICLE_PLUS_OUTSOURCE", "assigned_vehicle": v_id,
                "outsourcing_type": cls["type"], "cost": cls["cost"], "cls": cls,
                "message": f"⚠️ Vượt xe lớn nhất nhà ({v_id}). Cắt full-fill xe này (trừ 1 xe), phần dư thuê ngoài {cls['type']}."}
                
    feasible = avail[(avail["max_weight_kg"] >= total_w) & (avail["max_volume_m3"] >= total_v)] \
        .sort_values(by=["max_weight_kg", "max_volume_m3"], ascending=True)
        
    if feasible.empty:
        return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE", classify_outsourcing(total_w, total_v, oc),
                        "⚠️ Đủ tải trọng nhưng hết xe khả dụng trong ngày -> thuê ngoài qua màn lọc.")
                        
    veh = feasible.iloc[0]
    v_id, limit = veh["vehicle_id"], float(veh.get("Max_Distance", 100.0))
    if total_dist > limit:
        return {"status": "CUT_REQUIRED", "action_type": "DISTANCE_EXCEEDED_CUT", "assigned_vehicle": v_id,
                "outsourcing_type": None, "cost": 0.0, "cls": None, "max_distance_limit": limit,
                "message": f"⚠️ Tuyến vượt Max_Distance ({total_dist:.1f}km > {limit:.0f}km) của xe {v_id}. Cắt đơn, phần cắt qua màn lọc thuê ngoài."}
                
    vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
    return {"status": "APPROVED", "action_type": "IN_HOUSE_SUCCESS", "assigned_vehicle": v_id, "outsourcing_type": None,
            "cost": 0.0, "cls": None, "message": f"✅ Thỏa mãn trọng tải! Giao xe NHỎ NHẤT khả thi: {v_id} (Đã trừ 1 xe trong ngày)."}

def evaluate_time_constraints(route_or_order, chosen_vehicle=None, service_time_rules=None, max_hours=8.0):
    """Đánh giá thời gian tuyến di chuyển + bốc/dỡ <= max_hours."""
    if service_time_rules is None:
        service_time_rules = {"B2C": {"loading": 25, "unloading": 35}, "B2B": {"loading": 45, "unloading": 60}}
    orders = route_or_order.get("orders", [])
    dist = float(route_or_order.get("total_distance_km", 0.0))
    speed = float(chosen_vehicle.get("average_speed_kmh", 40.0)) if chosen_vehicle is not None else 40.0
    travel = dist / speed if speed > 0 else 0.0
    rule = lambda o: service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])
    total = travel + sum(rule(o)["loading"] + rule(o)["unloading"] for o in orders) / 60.0
    if total <= max_hours + 1e-9:
        return {"status": "APPROVED", "action_type": "TIME_APPROVED", "total_hours": round(total, 2),
                "message": f"✅ Đạt yêu cầu thời gian tuyến: {round(total, 2)}h (<= {max_hours:g}h)."}
    return {"status": "CUT_TIME_EXCEEDED", "action_type": "TIME_WINDOW_CUT", "total_hours": round(total, 2),
            "message": f"⚠️ Tuyến vượt quá {max_hours:g} giờ ({round(total, 2)}h > {max_hours:g}h)! Cắt đơn, phần cắt qua màn lọc thuê ngoài."}

@dataclass
class Config:
    start_time: str = "08:30"
    max_route_hours: float = 8.0
    detour_factor: float = 1.2
    service_min: dict = field(default_factory=lambda: {"B2B": 105, "B2C": 60})
    backup_driver_cost: float = 400_000.0
    vehicle_daily_count: int = 2
    outsourced_speed_kmh: float = 40.0
    full_price: float = OC_DEFAULTS["full_price"]
    saving_1_price: float = OC_DEFAULTS["saving_1_price"]
    saving_2_price: float = OC_DEFAULTS["saving_2_price"]
    full_vehicle_name: str = OC_DEFAULTS["full_vehicle_name"]
    full_vehicle_max_weight_kg: float = OC_DEFAULTS["full_vehicle_max_weight_kg"]
    full_vehicle_max_volume_m3: float = OC_DEFAULTS["full_vehicle_max_volume_m3"]
    tier2_max_w: float = OC_DEFAULTS["tier2_max_w"]
    tier2_max_v: float = OC_DEFAULTS["tier2_max_v"]
    tier1_max_w: float = OC_DEFAULTS["tier1_max_w"]
    tier1_max_v: float = OC_DEFAULTS["tier1_max_v"]
    vehicle_file: str = os.path.join(OUT_FLEET, "DIM_VEHICLE.xlsx")
    warehouse_file: str = os.path.join(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES.xlsx")
    product_file: str = os.path.join(OUT_PRODUCT, "DIM_PRODUCT.xlsx")
    driver_file: str = os.path.join(OUT_DRIVER, "DIM_DRIVER.xlsx")
    order_file: str = os.path.join(OUT_ORDERS, "DIM_ORDERS.xlsx")
    cust_file: str = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.xlsx")
    matrix_file: str = os.path.join(OUT_MATRIX, "DISTANCE_MATRIX_KM.xlsx")

    @property
    def outsourcing(self) -> dict:
        return {
            "full_price": self.full_price,
            "saving_1_price": self.saving_1_price,
            "saving_2_price": self.saving_2_price,
            "full_vehicle_name": self.full_vehicle_name,
            "full_vehicle_max_weight_kg": self.full_vehicle_max_weight_kg,
            "full_vehicle_max_volume_m3": self.full_vehicle_max_volume_m3,
            "tier2_max_w": self.tier2_max_w,
            "tier2_max_v": self.tier2_max_v,
            "tier1_max_w": self.tier1_max_w,
            "tier1_max_v": self.tier1_max_v,
        }

CFG = Config()

def haversine(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def money(x) -> str:
    return f"{float(x):,.0f}"

def resolve_customer_file(cfg: Config) -> str:
    return cfg.cust_file

def read_matrix(path: str) -> pd.DataFrame:
    df = read_dim_file(path, index_col=0).copy()
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    return df

def _num_col(df, col, default=0.0):
    if col not in df.columns: return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)

def load_vehicles(cfg: Config) -> pd.DataFrame:
    df = read_dim_file(cfg.vehicle_file).copy()
    for c in ("vehicle_id", "license_plate", "id_warehouse"):
        df[c] = df[c].astype(str).str.strip() if c in df.columns else ""
    df = df[~df["vehicle_id"].isin(["", "nan", "None"])].copy()
    for c in VEHICLE_NUMERIC: df[c] = _num_col(df, c)
    df.loc[df["average_speed_kmh"] <= 0, "average_speed_kmh"] = 40.0
    df.loc[df["Max_Distance"] <= 0, "Max_Distance"] = 100.0
    df["vehicle_type"] = [f"{w:.0f}kg/{v:g}m3" for w, v in zip(df["max_weight_kg"], df["max_volume_m3"])]
    return df.reset_index(drop=True)

def load_warehouse_coords(cfg: Config) -> dict:
    if not os.path.exists(cfg.warehouse_file): return {}
    out = {}
    for _, r in read_dim_file(cfg.warehouse_file).iterrows():
        wid = "" if is_blank(r.get("id_warehouse")) else str(r["id_warehouse"]).strip()
        if wid and pd.notna(r.get("lat")) and pd.notna(r.get("lng")): out[wid] = (float(r["lat"]), float(r["lng"]))
    return out

def load_customers(cfg: Config) -> dict:
    out = {}
    for _, r in read_dim_file(resolve_customer_file(cfg)).iterrows():
        cid = str(r.get("customer_id", "")).strip()
        if cid and cid != "nan" and pd.notna(r.get("lat")) and pd.notna(r.get("lng")):
            out[cid] = {"lat": float(r["lat"]), "lng": float(r["lng"]), "name": "" if is_blank(r.get("customer_name")) else str(r["customer_name"])}
    return out

def load_drivers(cfg: Config) -> dict:
    out = {}
    for _, r in read_dim_file(cfg.driver_file).iterrows():
        wid = "" if is_blank(r.get("id_warehouse")) else str(r["id_warehouse"]).strip()
        name = r.get("driver_name") if not is_blank(r.get("driver_name")) else r.get("driver_id")
        if not wid or is_blank(name): continue
        role = norm_basic(r.get("role", ""))
        kind = "phụ" if any(k in role for k in ("phu", "assist", "helper")) else "chính"
        out.setdefault(wid, {"chính": [], "phụ": []})[kind].append(str(name).strip())
    return out

def load_orders(cfg: Config, notes: list) -> list:
    df = read_dim_file(cfg.order_file)
    for c in ("order_id", "customer_id", "order_date"):
        if c not in df.columns: raise UserError(f"DIM_ORDERS.xlsx thiếu cột `{c}`.")
    orders, skipped = [], 0
    for _, r in df.iterrows():
        oid, cid, dstr = ("" if is_blank(r.get(k)) else str(r[k]).strip() for k in ("order_id", "customer_id", "order_date"))
        d_iso = parse_date_iso(dstr)
        if not (oid and cid and d_iso): skipped += 1; continue
        otype = str(r.get("order_type", "")).strip().upper()
        orders.append({
            "order_id": oid, "customer_id": cid, "order_type": otype if otype in ("B2B", "B2C") else "B2C",
            "total_weight_kg": parse_num(r.get("total_weight_kg")), "total_volume_m3": parse_num(r.get("total_volume_m3")),
            "address": "" if is_blank(r.get("address")) else str(r["address"]),
            "alert_status": "Normal" if is_blank(r.get("alert_status")) else str(r["alert_status"]),
            "order_status": "" if is_blank(r.get("order_status")) else str(r["order_status"]),
            "order_date": d_iso,
        })
    if skipped: notes.append(f"Bỏ qua {skipped} đơn thiếu order_id / customer_id hoặc order_date sai định dạng.")
    return orders

def load_data(cfg: Config) -> dict:
    notes = []
    veh = load_vehicles(cfg)
    wh_coords = load_warehouse_coords(cfg)
    if not wh_coords:
        wh_coords = {"WH_DEFAULT": DEFAULT_DEPOT}
        notes.append("Chưa có tọa độ kho (Tab 2) → dùng kho mặc định Hà Nội.")
    warehouses = {w: {"lat": la, "lng": lo, "name": w} for w, (la, lo) in wh_coords.items()}
    orphan = sorted(set(veh["id_warehouse"]) - set(warehouses))
    if orphan: notes.append(f"Xe thuộc kho không có tọa độ: {', '.join(orphan)}.")
    return {"vehicles": veh, "drivers": load_drivers(cfg), "dist": read_matrix(cfg.matrix_file), "cust": load_customers(cfg),
            "warehouses": warehouses, "orders": load_orders(cfg, notes), "notes": notes}

_TAB_A_EVENTS = {
    "NO_FLEET": ("HIGH", "KHO KHÔNG CÓ XE", "Bổ sung xe cho kho"),
    "OUTSOURCE_FULL": ("CRITICAL", "VƯỢT TỔNG HẠM ĐỘI", "Thuê xe lớn / tách đơn"),
    "OUTSOURCE_NO_VEHICLE_AVAILABLE": ("MEDIUM", "HẾT XE TRONG NGÀY", "Bổ sung xe / dời lịch"),
    "MAX_VEHICLE_PLUS_OUTSOURCE": ("CRITICAL", "ĐƠN QUÁ CỠ", "Thuê xe lớn"),
}

class RoutePlanner:
    def __init__(self, data: dict, cfg: Config = CFG):
        self.cfg = cfg
        self.veh = data["vehicles"]
        self.drivers = data["drivers"]
        self.dist_df = data["dist"]
        self.cust = data["cust"]
        self.wh = data["warehouses"]
        if self.veh.empty:
            self.catalog = pd.DataFrame(columns=["w", "v", "speed", "fixed", "var", "dist"])
        else:
            self.catalog = (self.veh.groupby("vehicle_type")
                .agg(w=("max_weight_kg", "max"), v=("max_volume_m3", "max"), speed=("average_speed_kmh", "first"),
                     fixed=("fixed_cost", "first"), var=("variable_cost", "first"), dist=("Max_Distance", "max"))
                .sort_values("w"))
        self.max_w = float(self.catalog["w"].max()) if not self.catalog.empty else 1000.0
        self.max_v = float(self.catalog["v"].max()) if not self.catalog.empty else 5.0
        self.max_dist = float(self.catalog["dist"].max()) if not self.catalog.empty else 100.0
        self._fleet = {}
        for rec in self.veh.to_dict("records"): self._fleet.setdefault(rec["id_warehouse"], []).append(rec)
        for lst in self._fleet.values(): lst.sort(key=lambda r: (r["max_weight_kg"], r["max_volume_m3"]))

    def _hav(self, a, b) -> float:
        return haversine(a["lat"], a["lng"], b["lat"], b["lng"]) * self.cfg.detour_factor

    def d_cc(self, i, j) -> float:
        if i in self.dist_df.index and j in self.dist_df.columns: return float(self.dist_df.loc[i, j])
        return self._hav(self.cust[i], self.cust[j]) if (i in self.cust and j in self.cust) else 10.0

    def d_wc(self, wh_id, c) -> float:
        return self._hav(self.wh[wh_id], self.cust[c]) if (wh_id in self.wh and c in self.cust) else 10.0

    def nearest_wh(self, c) -> str:
        return min(self.wh, key=lambda w: self.d_wc(w, c))

    def km(self, route, wh_id) -> float:
        if not route: return 0.0
        return self.d_wc(wh_id, route[0]) + self.d_wc(wh_id, route[-1]) + sum(self.d_cc(a, b) for a, b in zip(route, route[1:]))

    def caps(self, wh_id):
        lst = self._fleet.get(wh_id)
        if not lst: return self.max_w, self.max_v, self.max_dist
        return max(r["max_weight_kg"] for r in lst), max(r["max_volume_m3"] for r in lst), max(r["Max_Distance"] for r in lst)

    def smallest_vehicle(self, wh_id, w, v):
        for r in self._fleet.get(wh_id, []):
            if r["max_weight_kg"] >= w and r["max_volume_m3"] >= v: return r
        return None

    def fit_type(self, w, v):
        if self.catalog.empty: return "Truck"
        ok = self.catalog[(self.catalog["w"] >= w) & (self.catalog["v"] >= v)]
        return ok.index[0] if not ok.empty else self.catalog.index[-1]

    def metrics(self, route, wh_id, demand):
        w = sum(demand[c]["weight"] for c in route)
        v = sum(demand[c]["volume"] for c in route)
        km = self.km(route, wh_id)
        veh = self.smallest_vehicle(wh_id, w, v)
        vtype = veh["vehicle_type"] if veh else self.fit_type(w, v)
        speed = float(veh["average_speed_kmh"]) if veh else (float(self.catalog.loc[vtype, "speed"]) if vtype in self.catalog.index else 35.0)
        service = sum(demand[c]["service_min"] for c in route) / 60.0
        return {"w": w, "v": v, "km": km, "hours": km / speed + service, "vtype": vtype, "speed": speed, "veh": veh}

    def feasible(self, route, wh_id, demand) -> bool:
        cap_w, cap_v, _ = self.caps(wh_id)
        m = self.metrics(route, wh_id, demand)
        if m["w"] > cap_w or m["v"] > cap_v or m["hours"] > self.cfg.max_route_hours: return False
        return m["veh"] is None or m["km"] <= m["veh"]["Max_Distance"]

    def two_opt(self, route, wh_id):
        best, improved = route[:], True
        while improved and len(best) > 2:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if self.km(cand, wh_id) < self.km(best, wh_id) - 1e-9: best, improved = cand, True
        return best

    def clarke_wright(self, custs, wh_id, demand):
        route_of = {c: [c] for c in custs}
        savings = sorted(((self.d_wc(wh_id, a) + self.d_wc(wh_id, b) - self.d_cc(a, b), a, b)
                           for k, a in enumerate(custs) for b in custs[k + 1:]), reverse=True)
        for s, a, b in savings:
            if s <= 0: break
            ra, rb = route_of[a], route_of[b]
            if ra is rb or a not in (ra[0], ra[-1]) or b not in (rb[0], rb[-1]): continue
            ra = ra if ra[-1] == a else ra[::-1]
            rb = rb if rb[0] == b else rb[::-1]
            merged = ra + rb
            if self.feasible(merged, wh_id, demand):
                for c in merged: route_of[c] = merged
        return [self.two_opt(r, wh_id) for r in {id(r): r for r in route_of.values()}.values()]

    def _demand(self, ords):
        demand, by_cust = {}, {}
        for o in ords:
            d = demand.setdefault(o["customer_id"], {"weight": 0.0, "volume": 0.0, "service_min": 0.0})
            d["weight"] += o["total_weight_kg"]; d["volume"] += o["total_volume_m3"]
            d["service_min"] += self.cfg.service_min.get(o["order_type"], 60)
            by_cust.setdefault(o["customer_id"], []).append(o)
        return demand, by_cust

    def _worst_stop(self, r, wh_id):
        base = self.km(r, wh_id)
        return max(r, key=lambda c: base - self.km([x for x in r if x != c], wh_id))

    def _start(self, date_str):
        return dt.datetime.combine(pd.to_datetime(date_str).date(), dt.datetime.strptime(self.cfg.start_time, "%H:%M").time())

    def _outsource_route(self, custs, wh_id, demand, by_cust, cls, events, date_str, km=None):
        cfg = self.cfg
        if km is None:
            custs = self.two_opt(list(custs), wh_id); km = self.km(custs, wh_id)
        w = sum(demand[c]["weight"] for c in custs)
        hours = km / cfg.outsourced_speed_kmh + sum(demand[c]["service_min"] for c in custs) / 60.0
        start = self._start(date_str)
        return {
            "kind": "OUTSOURCE", "id_warehouse": wh_id, "route": list(custs), "orders": [o for c in custs for o in by_cust[c]],
            "vehicle_id": f"3PL-{cls['tier']}", "license_plate": f"Thuê ngoài 3PL ({cls['type']})", "vehicle_type": cls["vehicle"],
            "external": True, "driver_primary": "Đối tác 3PL", "driver_assistant": "—", "is_backup_driver": False,
            "km": round(km, 1), "speed_kmh": cfg.outsourced_speed_kmh,
            "load_factor": min(max(w / cls["cap_w"] if cls["cap_w"] > 0 else 0.5, 0.15), 1.0),
            "fixed_cost": 0.0, "variable_cost": 0.0, "overnight_cost": 0.0, "driver_cost": 0.0,
            "outsourcing_cost": float(cls["cost"]), "outsourcing_type": cls["type"],
            "cut_orders": [], "cut_weight_kg": 0.0, "cut_volume_m3": 0.0,
            "start": start, "end": start + dt.timedelta(hours=hours), "hours": hours, "events": events,
        }

    def _dispatch(self, route, wh_id, demand, by_cust, veh_counts, assign_driver, date_str):
        cfg, oc = self.cfg, self.cfg.outsourcing
        rules = service_rules_from_cfg(cfg)
        fleet = self.veh[self.veh["id_warehouse"] == wh_id]
        r, cut, final = list(route), [], None
        while r:
            m = self.metrics(r, wh_id, demand)
            snap = dict(veh_counts)
            ta = evaluate_transportation_constraints(
                {"total_weight_kg": m["w"], "total_volume_m3": m["v"], "total_distance_km": m["km"]}, fleet, veh_counts, oc)
            action, vid, tb, bad = ta["action_type"], ta.get("assigned_vehicle"), None, None
            vrow = fleet[fleet["vehicle_id"] == vid].iloc[0] if (vid and ta["status"] != "OUTSOURCED") else None
            
            if action == "DISTANCE_EXCEEDED_CUT":
                bad = ("VƯỢT MAX_DISTANCE", ta["message"])
            elif vrow is not None:
                if m["km"] > float(vrow["Max_Distance"]):
                    bad = ("VƯỢT MAX_DISTANCE", f"Tuyến {m['km']:.1f}km > Max_Distance {float(vrow['Max_Distance']):.0f}km của xe {vid}.")
                else:
                    tb = evaluate_time_constraints({"total_distance_km": m["km"], "orders": [o for c in r for o in by_cust[c]],
                                                   "current_date": date_str}, {"average_speed_kmh": float(vrow["average_speed_kmh"])},
                                                   rules, cfg.max_route_hours)
                    if tb["status"] != "APPROVED": bad = ("VƯỢT GIỜ TUYẾN", tb["message"])
            if bad:
                veh_counts.clear(); veh_counts.update(snap)
                c = self._worst_stop(r, wh_id)
                cut.append((c, bad[0], bad[1]))
                r = self.two_opt([x for x in r if x != c], wh_id)
                continue
            final = (m, ta, action, vid, vrow, tb)
            break
        out = []
        if final:
            m, ta, action, vid, vrow, tb = final
            if ta["status"] == "OUTSOURCED":
                sev, kind, propose = _TAB_A_EVENTS[action]
                ev = [{"sev": sev, "kind": kind, "detail": ta["message"], "handled": f"Thuê ngoài {ta['cls']['type']}", "propose": propose}]
                out.append(self._outsource_route(r, wh_id, demand, by_cust, ta["cls"], ev, date_str))
            else:
                split = action == "MAX_VEHICLE_PLUS_OUTSOURCE"
                primary, assistant, backup = assign_driver(wh_id)
                start, hours, cap_w = self._start(date_str), tb["total_hours"], float(vrow["max_weight_kg"])
                ev = []
                if split:
                    sev, kind, propose = _TAB_A_EVENTS[action]
                    ev.append({"sev": sev, "kind": kind, "detail": ta["message"],
                               "handled": f"Cắt full-fill xe {vid} + thuê ngoài {ta['cls']['type']}", "propose": propose})
                orders = [o for c in r for o in by_cust[c]]
                out.append({
                    "kind": "SPLIT" if split else "NORMAL", "id_warehouse": wh_id, "route": list(r), "orders": orders,
                    "vehicle_id": vid, "license_plate": vrow["license_plate"], "vehicle_type": vrow["vehicle_type"], "external": False,
                    "driver_primary": primary, "driver_assistant": assistant, "is_backup_driver": backup,
                    "km": round(m["km"], 1), "speed_kmh": float(vrow["average_speed_kmh"]),
                    "load_factor": min(max(m["w"] / cap_w if cap_w > 0 else 0.5, 0.15), 1.0),
                    "fixed_cost": float(vrow["fixed_cost"]), "variable_cost": float(vrow["variable_cost"]) * m["km"], "overnight_cost": 0.0,
                    "driver_cost": cfg.backup_driver_cost if backup else 0.0,
                    "outsourcing_cost": float(ta["cost"]) if split else 0.0, "outsourcing_type": ta["outsourcing_type"] if split else None,
                    "cut_orders": [o["order_id"] for o in orders] if split else [],
                    "cut_weight_kg": max(m["w"] - cap_w, 0.0) if split else 0.0,
                    "cut_volume_m3": max(m["v"] - float(vrow["max_volume_m3"]), 0.0) if split else 0.0,
                    "start": start, "end": start + dt.timedelta(hours=hours), "hours": hours, "events": ev,
                })
        if cut:
            ids = [c for c, _, _ in cut]
            cls = classify_outsourcing(sum(demand[c]["weight"] for c in ids), sum(demand[c]["volume"] for c in ids), oc)
            ev = [{"sev": "HIGH", "kind": kind, "detail": f"Khách {c}: {detail}", "handled": f"Cắt khỏi tuyến → thuê ngoài {cls['type']}",
                   "propose": "Xem lại Max_Distance / số xe", "ids": [c]} for c, kind, detail in cut]
            out.append(self._outsource_route(ids, wh_id, demand, by_cust, cls, ev, date_str))
        return out

    def plan_day(self, date_str: str, day_orders: list) -> dict:
        cfg, oc = self.cfg, self.cfg.outsourcing
        res = {"routes": [], "overdue_routes": [], "exceptions": [], "day_cost": 0.0}
        def log(r):
            for e in r.pop("events"):
                for o in r["orders"]:
                    if e.get("ids") is None or o["customer_id"] in e["ids"]:
                        res["exceptions"].append({"NGÀY": date_str, "MỨC ĐỘ": e["sev"], "MÃ ĐƠN": o["order_id"], "PHÂN LOẠI": e["kind"],
                                                  "CHI TIẾT": e["detail"], "ĐÃ XỬ LÝ": e["handled"], "ĐỀ XUẤT": e["propose"]})
        placed = [o for o in day_orders if o["customer_id"] in self.cust]
        unplaced = [o for o in day_orders if o["customer_id"] not in self.cust]
        veh_counts = {vid: cfg.vehicle_daily_count for vid in self.veh["vehicle_id"]}
        driver_pointers = {w: {"c": 0, "p": 0} for w in self.wh}
        def assign_driver(wh_id):
            d_info = self.drivers.get(wh_id, {"chính": ["Tài xế chính"], "phụ": ["Phụ xe"]})
            c_list, p_list = d_info["chính"], d_info["phụ"]
            idx_c, idx_p = driver_pointers[wh_id]["c"], driver_pointers[wh_id]["p"]
            if idx_c < len(c_list): primary, backup = c_list[idx_c], False; driver_pointers[wh_id]["c"] += 1
            else: primary, backup = "Tài xế Dự Phòng (Thuê ngoài)", True
            assistant = p_list[idx_p % len(p_list)] if p_list else "Phụ xe"
            driver_pointers[wh_id]["p"] += 1
            return primary, assistant, backup
        by_wh = {}
        for o in placed: by_wh.setdefault(self.nearest_wh(o["customer_id"]), []).append(o)
        for wh_id, ords in by_wh.items():
            demand, by_cust = self._demand(ords)
            for rt in self.clarke_wright(list(demand), wh_id, demand):
                for r in self._dispatch(rt, wh_id, demand, by_cust, veh_counts, assign_driver, date_str):
                    log(r); res["routes"].append(r)
        if unplaced:
            wh0 = next(iter(self.wh))
            demand, by_cust = self._demand(unplaced)
            for c in demand:
                cls = classify_outsourcing(demand[c]["weight"], demand[c]["volume"], oc)
                ev = [{"sev": "MEDIUM", "kind": "THIẾU TỌA ĐỘ", "detail": "Không tìm thấy khách hàng trong DATASET_CUSTOMER",
                       "handled": f"Thuê ngoài {cls['type']} (không thể định tuyến)", "propose": "Bổ sung tọa độ"}]
                r = self._outsource_route([c], wh0, demand, by_cust, cls, ev, date_str, km=0.0)
                log(r); res["routes"].append(r)
        res["day_cost"] = sum(route_total(r) for r in res["routes"])
        return res

def simulate_all(data: dict, cfg: Config = CFG):
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]: by_date.setdefault(o["order_date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    all_r = [r for d in days.values() for r in d["routes"] + d["overdue_routes"]]
    inhouse = [r for r in all_r if not r["external"]]
    delivered = {o["order_id"] for r in all_r for o in r["orders"]}
    n_routes = len(all_r)
    operating = sum(d["day_cost"] for d in days.values())
    kpis = {
        "violations": sum(1 for r in inhouse if r["hours"] > cfg.max_route_hours + 1e-9),
        "operating_cost": operating,
        "external_ratio": sum(r["external"] for r in all_r) / n_routes if n_routes else 0,
        "avg_load_factor": sum(r["load_factor"] for r in inhouse) / len(inhouse) if inhouse else 0,
        "undelivered_orders": len(data["orders"]) - len(delivered),
        "total_orders": len(data["orders"]),
    }
    return planner, days, kpis, operating

def route_total(r):
    return r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r.get("driver_cost", 0) + r.get("outsourcing_cost", 0)

# ----------------------------------------------------------------------------
# PIPELINE TIỆN ÍCH
# ----------------------------------------------------------------------------
def file_status(cfg: Config) -> pd.DataFrame:
    items = [
        ("Tab 1 · Hạm đội xe", cfg.vehicle_file, "Bắt buộc"),
        ("Tab 2 · Kho & Tọa độ", cfg.warehouse_file, "Khuyến nghị"),
        ("Tab 3 · Sản phẩm", cfg.product_file, "Tuỳ chọn"),
        ("Tab 4 · Tài xế", cfg.driver_file, "Bắt buộc"),
        ("Tab 5 · Đơn hàng", cfg.order_file, "Bắt buộc"),
        ("Bước 1 · Khách hàng + tọa độ", cfg.cust_file, "Bắt buộc"),
        ("Bước 2 · Ma trận khoảng cách", cfg.matrix_file, "Bắt buộc"),
    ]
    rows = [{"Nguồn": n, "File": os.path.relpath(p, BASE_DIR), "Trạng thái": "✅ Có" if os.path.exists(p) else "❌ Chưa có", "Yêu cầu": need}
            for n, p, need in items]
    return pd.DataFrame(rows)

def require_files(cfg: Config, which):
    labels = {
        "orders": (cfg.order_file, "Tab 5 (Đơn hàng)"),
        "fleet": (cfg.vehicle_file, "Tab 1 (Hạm đội xe)"),
        "driver": (cfg.driver_file, "Tab 4 (Tài xế)"),
        "matrix": (cfg.matrix_file, "Bước 2 (Ma trận khoảng cách)"),
    }
    missing = [f"`{os.path.relpath(labels[k][0], BASE_DIR)}` (chạy {labels[k][1]})" for k in which if not os.path.exists(labels[k][0])]
    if missing: raise UserError("Thiếu file đầu vào: " + "; ".join(missing))

def step_geocode_customers(cfg: Config, on_progress=None) -> dict:
    require_files(cfg, ["orders"])
    df_orders = pd.read_excel(cfg.order_file)
    if "customer_id" not in df_orders.columns:
        raise UserError("DIM_ORDERS.xlsx không có cột customer_id.")
    cols = [c for c in ["customer_id", "customer_name", "address"] if c in df_orders.columns]
    df_cust = df_orders[cols].dropna(subset=["customer_id"]).copy()
    df_cust["customer_id"] = df_cust["customer_id"].astype(str).str.strip()
    df_cust = df_cust[df_cust["customer_id"] != ""].drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    if df_cust.empty:
        raise UserError("Không có khách hàng hợp lệ trong DIM_ORDERS.xlsx.")
    geolocator = ArcGIS(user_agent="smart_logistics_customer_geocoder/1.0", timeout=15)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5, swallow_exceptions=True)
    cache, lat_list, lng_list, matched_list, status_list = {}, [], [], [], []
    total = len(df_cust)
    for k, (_, row) in enumerate(df_cust.iterrows(), 1):
        addr = str(row.get("address", "")).strip()
        if not addr or addr.lower() == "nan":
            lat_list.append(None); lng_list.append(None); matched_list.append(""); status_list.append("❌ Địa chỉ trống")
        else:
            if addr in cache:
                lat, lng, matched_address, status = cache[addr]
            else:
                query = addr if "việt nam" in addr.lower() else f"{addr}, Việt Nam"
                try:
                    loc = geocode(query)
                    if loc:
                        lat, lng, matched_address = float(loc.latitude), float(loc.longitude), str(loc.address)
                        status = "✅ Geocode hợp lệ" if (8.0 <= lat <= 24.5 and 102.0 <= lng <= 110.0) else "❌ Tọa độ ngoài Việt Nam"
                    else:
                        lat, lng, matched_address, status = None, None, "", "⚠️ Không tìm thấy địa chỉ"
                except Exception as exc:
                    lat, lng, matched_address, status = None, None, "", f"❌ Lỗi: {exc}"
                cache[addr] = (lat, lng, matched_address, status)
            lat_list.append(lat); lng_list.append(lng); matched_list.append(matched_address); status_list.append(status)
        if on_progress: on_progress(k, total)
    df_cust["lat"], df_cust["lng"] = lat_list, lng_list
    df_cust["matched_address"], df_cust["trạng_thái_geocode"] = matched_list, status_list
    files = save_outputs(OUT_CUSTOMER, "DATASET_CUSTOMER", "DATASET_CUSTOMER", df_cust)
    success = int((df_cust["trạng_thái_geocode"] == "✅ Geocode hợp lệ").sum())
    return {"df": df_cust, "files": files, "success": success, "failed": total - success}

def build_distance_matrices(df_valid: pd.DataFrame, detour: float = 1.2):
    cust_ids = df_valid["customer_id"].tolist()
    coords_str = ";".join(f"{row['lng']},{row['lat']}" for _, row in df_valid.iterrows())
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
    note = ""
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok":
                dist = pd.DataFrame([[d / 1000.0 for d in row] for row in data.get("distances")], index=cust_ids, columns=cust_ids)
                dur = pd.DataFrame([[t / 60.0 for t in row] for row in data.get("durations")], index=cust_ids, columns=cust_ids)
                return dist, dur, "OSRM (đường bộ thực tế)", ""
            note = f"OSRM mã lỗi: {data.get('code')}"
        else:
            note = f"HTTP Error Status: {response.status_code}"
    except Exception as exc:
        note = f"Không gọi được OSRM API ({exc})"
    n = len(cust_ids)
    lat, lng = df_valid["lat"].astype(float).tolist(), df_valid["lng"].astype(float).tolist()
    dist_matrix, dur_matrix = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(lat[i], lng[i], lat[j], lng[j]) * detour
                dist_matrix[i][j] = round(d, 2)
                dur_matrix[i][j] = round(d / 40.0 * 60.0, 2)
    return (pd.DataFrame(dist_matrix, index=cust_ids, columns=cust_ids),
            pd.DataFrame(dur_matrix, index=cust_ids, columns=cust_ids),
            f"Haversine × {detour} (dự phòng)", note)

def step_distance_matrix(cfg: Config) -> dict:
    df_cust = pd.read_excel(resolve_customer_file(cfg))
    df_cust["customer_id"] = df_cust["customer_id"].astype(str).str.strip()
    df_valid = df_cust.dropna(subset=["lat", "lng"]).drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    if len(df_valid) < 2:
        raise UserError("Cần ít nhất 2 khách hàng có tọa độ Lat/Lon để tính ma trận!")
    dist, dur, method, note = build_distance_matrices(df_valid, cfg.detour_factor)
    os.makedirs(OUT_MATRIX, exist_ok=True)
    dist_path, dur_path, json_path = os.path.join(OUT_MATRIX, "DISTANCE_MATRIX_KM.xlsx"), os.path.join(OUT_MATRIX, "DURATION_MATRIX_MIN.xlsx"), os.path.join(OUT_MATRIX, "OSRM_MATRIX_RESULT.json")
    dist.to_excel(dist_path); dur.to_excel(dur_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"customers": df_valid["customer_id"].tolist(), "distance_matrix_km": dist.to_dict(), "duration_matrix_min": dur.to_dict()}, fh, ensure_ascii=False, indent=2)
    return {"dist": dist, "dur": dur, "method": method, "note": note, "files": [dist_path, dur_path, json_path], "n": len(df_valid), "skipped": len(df_cust) - len(df_valid)}

def depot_options(cfg: Config) -> dict:
    opts = {f"Mặc định Hà Nội ({DEFAULT_DEPOT[0]}, {DEFAULT_DEPOT[1]})": DEFAULT_DEPOT}
    for wid, (la, lo) in load_warehouse_coords(cfg).items():
        opts[f"Kho {wid} ({la:.4f}, {lo:.4f})"] = (la, lo)
    return opts

def step_savings(cfg: Config, depot) -> dict:
    require_files(cfg, ["matrix"])
    df_dist = read_matrix(cfg.matrix_file)
    df_cust = pd.read_excel(resolve_customer_file(cfg))
    df_cust["customer_id"] = df_cust["customer_id"].astype(str).str.strip()
    depot_lat, depot_lng = depot
    customers = df_dist.index.tolist()
    c0i = {}
    for _, row in df_cust.iterrows():
        c_id = row["customer_id"]
        if c_id in customers and pd.notna(row["lat"]) and pd.notna(row["lng"]):
            lat, lng = row["lat"], row["lng"]
            d = 0.0 if (abs(lat - depot_lat) < 1e-4 and abs(lng - depot_lng) < 1e-4) else haversine(depot_lat, depot_lng, lat, lng) * cfg.detour_factor
            c0i[c_id] = round(d, 2)
    mat = df_dist.values
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            cust_i, cust_j = customers[i], customers[j]
            c_ij = float(mat[i, j])
            s_ij = c0i.get(cust_i, 0.0) + c0i.get(cust_j, 0.0) - c_ij
            savings.append({"pair": (cust_i, cust_j), "savings_km": round(s_ij, 2), "c_0i": c0i.get(cust_i, 0.0), "c_0j": c0i.get(cust_j, 0.0), "c_ij": c_ij})
    savings_sorted = sorted(savings, key=lambda x: x["savings_km"], reverse=True)
    df_savings = pd.DataFrame([{"Cặp khách hàng": f"{s['pair'][0]} - {s['pair'][1]}", "Khách hàng 1": s["pair"][0], "Khách hàng 2": s["pair"][1], "Mức tiết kiệm (km)": s["savings_km"], "Kho -> Khách 1 (c0i)": s["c_0i"], "Kho -> Khách 2 (c0j)": s["c_0j"], "Khách 1 <-> Khách 2 (cij)": s["c_ij"]} for s in savings_sorted])
    os.makedirs(OUT_MATRIX, exist_ok=True)
    out_path = os.path.join(OUT_MATRIX, "DATASET_SORT_SAVING.xlsx")
    df_savings.to_excel(out_path, index=False, sheet_name="SORT_SAVING")
    return {"df": df_savings, "files": [out_path], "pairs": len(df_savings)}

def step_screening(cfg: Config) -> dict:
    require_files(cfg, ["orders", "fleet", "matrix"])
    df_veh, wh = load_vehicles(cfg), load_warehouse_coords(cfg)
    cust = load_customers(cfg) if os.path.exists(cfg.cust_file) else {}
    rows = []
    for _, r in read_dim_file(cfg.order_file).iterrows():
        w, v, cid = parse_num(r.get("total_weight_kg")), parse_num(r.get("total_volume_m3")), str(r.get("customer_id", "")).strip()
        wid, dist = "", 0.0
        if wh and cid in cust:
            wid = min(wh, key=lambda k: haversine(wh[k][0], wh[k][1], cust[cid]["lat"], cust[cid]["lng"]))
            dist = 2 * haversine(wh[wid][0], wh[wid][1], cust[cid]["lat"], cust[cid]["lng"]) * cfg.detour_factor
        fleet = df_veh[df_veh["id_warehouse"] == wid] if wid else df_veh
        counts = {vid: cfg.vehicle_daily_count for vid in fleet["vehicle_id"]}
        ev = evaluate_transportation_constraints({"total_weight_kg": w, "total_volume_m3": v, "total_distance_km": dist}, fleet, counts, cfg.outsourcing)
        rows.append({"order_id": r.get("order_id"), "customer_id": cid, "id_warehouse": wid, "total_weight_kg": w, "total_volume_m3": v,
                     "khoảng_cách_km": round(dist, 1), "status": ev["status"], "action_type": ev["action_type"],
                     "assigned_vehicle": ev.get("assigned_vehicle"), "outsourcing_type": ev.get("outsourcing_type"),
                     "cost": ev.get("cost", 0.0), "số_chuyến_thuê_ngoài": (ev.get("cls") or {}).get("trips"), "message": ev["message"]})
    df = pd.DataFrame(rows)
    return {"df": df, "files": save_outputs(OUT_SCREEN, "DATASET_SCREENING", "DATASET_SCREENING", df)}

def step_clarke_wright(cfg: Config) -> dict:
    require_files(cfg, ["orders", "fleet", "driver", "matrix"])
    data = load_data(cfg)
    planner, days, kpis, total_cost = simulate_all(data, cfg)
    return {"planner": planner, "days": days, "kpis": kpis, "total_cost": total_cost, "notes": data["notes"], "cfg": cfg, "n_orders": len(data["orders"])}

# ============================================================================
# CHUYỂN ĐỔI KẾT QUẢ SANG DATAFRAME
# ============================================================================
def tag_routes(day):
    return [(r, "Thường") for r in day["routes"]] + [(r, "Quá hạn (ưu tiên)") for r in day["overdue_routes"]]

def routes_to_df(tagged_routes, wh_info, cfg: Config) -> pd.DataFrame:
    rows = []
    for idx, (r, kind) in enumerate(tagged_routes, 1):
        wh = wh_info.get(r["id_warehouse"], {"name": r["id_warehouse"]})
        locs = sorted({re.split(r",|\s-\s", o["address"])[-1].strip() for o in r["orders"] if o["address"]})
        rows.append({
            "MÃ TUYẾN": f"Tuyến {wh['name']} #{idx}", "KHU VỰC": ", ".join(locs), "LOẠI": kind,
            "ĐƠN GIAO": ", ".join(o["order_id"] for o in r["orders"]), "SỐ ĐƠN": len(r["orders"]),
            "TÀI XẾ CHÍNH": r["driver_primary"] + (" ⚠️ dự phòng" if r["is_backup_driver"] else ""),
            "PHỤ XE": r["driver_assistant"], "XE": ("🟣 Thuê ngoài 3PL — " if r["external"] else "🚚 ") + str(r["vehicle_type"]),
            "BIỂN SỐ": r["license_plate"], "QUÃNG ĐƯỜNG (km)": r["km"], "LẤP ĐẦY (%)": round(r["load_factor"] * 100, 1),
            "THỜI GIAN (giờ)": round(r["hours"], 2), "TỔNG CHI PHÍ (đ)": round(route_total(r)),
            "BẮT ĐẦU": f"{r['start']:%H:%M}", "KẾT THÚC": f"{r['end']:%H:%M}",
            "TRẠNG THÁI": "🟣 Giao bởi 3PL" if r["external"] else ("✅ Đã tối ưu" if r["hours"] <= cfg.max_route_hours + 1e-9 else "⚠️ Vượt giới hạn giờ"),
        })
    return pd.DataFrame(rows)

def route_orders_df(r, cust) -> pd.DataFrame:
    rows = []
    for k, o in enumerate(r["orders"], 1):
        c = cust.get(o["customer_id"], {})
        rows.append({"THỨ TỰ": k, "MÃ ĐƠN": o["order_id"], "MÃ KHÁCH": o["customer_id"], "LOẠI ĐƠN": o["order_type"],
                     "TRỌNG LƯỢNG (kg)": o["total_weight_kg"], "THỂ TÍCH (m3)": o["total_volume_m3"], "ĐỊA CHỈ": o["address"],
                     "lat": c.get("lat"), "lon": c.get("lng")})
    return pd.DataFrame(rows)

def service_rules_from_cfg(cfg: Config) -> dict:
    base = {"B2C": {"loading": 25, "unloading": 35}, "B2B": {"loading": 45, "unloading": 60}}
    for k in ("B2C", "B2B"):
        total = base[k]["loading"] + base[k]["unloading"]
        want = float(cfg.service_min.get(k, total))
        if abs(want - total) > 1e-9:
            ratio = base[k]["loading"] / total
            base[k] = {"loading": want * ratio, "unloading": want * (1 - ratio)}
    return base

def audit_routes_df(tagged_routes, wh_info, cfg: Config, date_str) -> pd.DataFrame:
    rules = service_rules_from_cfg(cfg)
    rows = []
    for idx, (r, _) in enumerate(tagged_routes, 1):
        wh = wh_info.get(r["id_warehouse"], {"name": r["id_warehouse"]})
        res = evaluate_time_constraints({"orders": r["orders"], "total_distance_km": r["km"], "current_date": date_str},
                                       {"average_speed_kmh": r["speed_kmh"]}, rules, cfg.max_route_hours)
        rows.append({"MÃ TUYẾN": f"Tuyến {wh['name']} #{idx}", "GIỜ (PLANNER)": round(r["hours"], 2),
                     "GIỜ (KIỂM ĐỊNH)": res["total_hours"], "KẾT LUẬN": res["status"], "GHI CHÚ": res["message"]})
    return pd.DataFrame(rows)

def build_plan_workbook(plan: dict) -> bytes:
    planner, days, kpis, cfg = plan["planner"], plan["days"], plan["kpis"], plan["cfg"]
    route_frames, order_frames, exc_rows = [], [], []
    for d, day in days.items():
        tagged = tag_routes(day)
        rdf = routes_to_df(tagged, planner.wh, cfg)
        if not rdf.empty:
            rdf.insert(0, "NGÀY", d)
            route_frames.append(rdf)
        for idx, (r, _) in enumerate(tagged, 1):
            odf = route_orders_df(r, planner.cust).drop(columns=["lat", "lon"])
            odf.insert(0, "MÃ TUYẾN", rdf.iloc[idx - 1]["MÃ TUYẾN"])
            odf.insert(0, "NGÀY", d)
            order_frames.append(odf)
        exc_rows += day["exceptions"]
    kpi_df = pd.DataFrame([
        {"Chỉ số": "Tổng chi phí", "Giá trị": plan["total_cost"]},
        {"Chỉ số": "Chi phí vận hành", "Giá trị": kpis["operating_cost"]},
        {"Chỉ số": "Số tuyến vi phạm giờ", "Giá trị": kpis["violations"]},
        {"Chỉ số": "Tỉ lệ thuê ngoài 3PL", "Giá trị": kpis["external_ratio"]},
        {"Chỉ số": "Lấp đầy trung bình", "Giá trị": kpis["avg_load_factor"]},
        {"Chỉ số": "Đơn chưa giao", "Giá trị": kpis["undelivered_orders"]},
        {"Chỉ số": "Tổng số đơn", "Giá trị": kpis["total_orders"]},
    ])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        kpi_df.to_excel(writer, sheet_name="KPI", index=False)
        if route_frames: pd.concat(route_frames, ignore_index=True).to_excel(writer, sheet_name="ROUTES", index=False)
        if order_frames: pd.concat(order_frames, ignore_index=True).to_excel(writer, sheet_name="ORDERS_IN_ROUTE", index=False)
        if exc_rows: pd.DataFrame(exc_rows).to_excel(writer, sheet_name="EXCEPTIONS", index=False)
    return buf.getvalue()

# ============================================================================
# TAB 6 — UI STREAMLIT
# ============================================================================
def render_routing_settings():
    with st.expander("⚙️ Tham số mô hình, thuê ngoài & kho xuất phát", expanded=True):
        c1, c2, c3 = st.columns(3)
        start_time = c1.text_input("Giờ xuất phát (HH:MM)", "08:30", key="cfg_start")
        max_hours = c2.number_input("Giới hạn giờ / tuyến", 1.0, 24.0, 8.0, 0.5, key="cfg_maxh")
        detour = c3.number_input("Hệ số đường vòng", 1.0, 3.0, 1.2, 0.05, key="cfg_detour")
        c4, c5, c6 = st.columns(3)
        svc_b2b = c4.number_input("Bốc/dỡ B2B (phút)", 0, 600, 105, 5, key="cfg_b2b")
        svc_b2c = c5.number_input("Bốc/dỡ B2C (phút)", 0, 600, 60, 5, key="cfg_b2c")
        backup = c6.number_input("Chi phí tài xế dự phòng (đ)", 0, 10_000_000, 400_000, 50_000, key="cfg_backup")
        st.markdown("##### 🚛 Chi phí thuê ngoài (mỗi lần / mỗi chuyến)")
        o1, o2, o3 = st.columns(3)
        full_p = o1.number_input("Thuê ngoài Full (đ / chuyến)", 0.0, 100_000_000.0, 2_500_000.0, 100_000.0, key="cfg_full_p")
        s1_p = o2.number_input("Tiết kiệm loại 1 (đ)", 0.0, 100_000_000.0, 800_000.0, 50_000.0, key="cfg_s1_p")
        s2_p = o3.number_input("Tiết kiệm loại 2 (đ)", 0.0, 100_000_000.0, 300_000.0, 50_000.0, key="cfg_s2_p")
        st.caption("Màn lọc: < 20 kg và < 1 m³ → loại 2 · 20–100 kg / 1–5 m³ → loại 1 · lớn hơn → thuê ngoài Full theo xe bên dưới.")
        st.markdown("##### 🚚 Giao diện cấu hình tham số xe thuê ngoài Full (Full-fill)")
        f1, f2, f3 = st.columns(3)
        fv_name = f1.text_input("Loại xe thuê ngoài", "Xe tải thuê ngoài", key="cfg_fv_name")
        fv_w = f2.number_input("Khối lượng max (kg)", 1.0, 1_000_000.0, 10_000.0, 500.0, key="cfg_fv_w")
        fv_v = f3.number_input("Thể tích max (m³)", 0.1, 10_000.0, 40.0, 1.0, key="cfg_fv_v")
    base_cfg = Config()
    opts = depot_options(base_cfg)
    if st.session_state.get("cfg_depot") not in opts:
        st.session_state["cfg_depot"] = list(opts)[0]
    depot_label = st.selectbox("Kho xuất phát cho Savings S_ij", list(opts), key="cfg_depot")
    try: dt.datetime.strptime(start_time, "%H:%M")
    except ValueError: start_time = "08:30"
    cfg = Config(start_time=start_time, max_route_hours=float(max_hours), detour_factor=float(detour),
                 service_min={"B2B": int(svc_b2b), "B2C": int(svc_b2c)}, backup_driver_cost=float(backup),
                 full_price=float(full_p), saving_1_price=float(s1_p), saving_2_price=float(s2_p),
                 full_vehicle_name=(fv_name or "").strip() or "Xe tải thuê ngoài",
                 full_vehicle_max_weight_kg=float(fv_w), full_vehicle_max_volume_m3=float(fv_v))
    return cfg, opts[depot_label]

def exec_step(name, fn):
    try:
        st.session_state[f"t6_{name}"] = fn()
        return True
    except UserError as exc: st.error(f"❌ {exc}")
    except Exception as exc: st.error(f"❌ Lỗi ở bước `{name}`: {exc}")
    return False

def render_dashboard(plan):
    planner, days, kpis, total_cost, cfg = plan["planner"], plan["days"], plan["kpis"], plan["total_cost"], plan["cfg"]
    for note in plan["notes"]: st.warning(f"⚠️ {note}")
    dates = list(days)
    if not dates: return st.warning("⚠️ Không tìm thấy đơn hàng nào!")
    if st.session_state.get("t6_date") not in dates: st.session_state["t6_date"] = dates[0]
    date_str = st.selectbox("📅 Chọn ngày điều phối", dates, key="t6_date")
    day = days[date_str]
    tagged = tag_routes(day)
    st.markdown(f"## 🗓️ DASHBOARD ĐIỀU PHỐI — {date_str}")
    g = st.columns(5)
    g[0].metric("🎯 Vi phạm giới hạn giờ", kpis["violations"])
    g[1].metric("👑 Tổng chi phí (VNĐ)", money(total_cost))
    g[2].metric("Lấp đầy TB", f"{kpis['avg_load_factor'] * 100:.1f}%")
    g[3].metric("Tỉ lệ 3PL", f"{kpis['external_ratio'] * 100:.1f}%")
    g[4].metric("Đơn chưa giao", f"{kpis['undelivered_orders']}/{kpis['total_orders']}")
    if not tagged: return st.info("Không có tuyến nào trong ngày.")
    rdf = routes_to_df(tagged, planner.wh, cfg)
    st.dataframe(rdf, hide_index=True)
    labels = rdf["MÃ TUYẾN"].tolist()
    sel = st.selectbox("🔎 Xem chi tiết thứ tự giao", labels, key=f"t6_route_sel_{date_str}")
    r, _ = tagged[labels.index(sel)]
    odf = route_orders_df(r, planner.cust)
    left, right = st.columns([3, 2])
    with left: st.dataframe(odf.drop(columns=["lat", "lon"]), hide_index=True)
    with right:
        wh = planner.wh.get(r["id_warehouse"])
        pts = odf.dropna(subset=["lat", "lon"])[["lat", "lon"]]
        if wh: pts = pd.concat([pd.DataFrame([{"lat": wh["lat"], "lon": wh["lng"]}]), pts], ignore_index=True)
        if not pts.empty: st.map(pts)
    if "xlsx" not in plan: plan["xlsx"] = build_plan_workbook(plan)
    st.download_button("⬇️ Tải kế hoạch tuyến (Excel)", plan["xlsx"], file_name="ROUTE_PLAN.xlsx", mime=XLSX_MIME, key="t6_dl_plan")

def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright Savings")
    cfg, depot = render_routing_settings()
    st.markdown("### 📂 Trạng thái dữ liệu đầu vào")
    st.dataframe(file_status(cfg), hide_index=True)
    if st.button("⚡ Chạy toàn bộ pipeline (bước 1 → 5)", type="primary", key="t6_runall"):
        steps = [
            ("geo", "Bước 1 · Geocode khách hàng", lambda: step_geocode_customers(cfg)),
            ("matrix", "Bước 2 · Ma trận khoảng cách", lambda: step_distance_matrix(cfg)),
            ("savings", "Bước 3 · Tính Savings S_ij", lambda: step_savings(cfg, depot)),
            ("screen", "Bước 4 · Sàng lọc tải trọng", lambda: step_screening(cfg)),
            ("plan", "Bước 5 · Clarke-Wright toàn cục", lambda: step_clarke_wright(cfg)),
        ]
        with st.status("Đang chạy pipeline định tuyến...", expanded=True) as status:
            for name, label, fn in steps:
                try: st.session_state[f"t6_{name}"] = fn()
                except Exception as exc:
                    status.update(label=f"❌ Lỗi ở: {label}", state="error")
                    st.error(f"❌ {exc}"); break
            else: status.update(label="✅ Hoàn tất pipeline", state="complete")
    st.divider()
    if st.button("▶️ Chạy bước 1 (Geocode)", key="t6_b1"): exec_step("geo", lambda: step_geocode_customers(cfg))
    if st.button("▶️ Chạy bước 2 (Ma trận)", key="t6_b2"): exec_step("matrix", lambda: step_distance_matrix(cfg))
    if st.button("▶️ Chạy bước 3 (Savings)", key="t6_b3"): exec_step("savings", lambda: step_savings(cfg, depot))
    if st.button("▶️ Chạy bước 4 (Sàng lọc)", key="t6_b4"): exec_step("screen", lambda: step_screening(cfg))
    if st.button("▶️ Chạy bước 5 (Clarke-Wright)", key="t6_b5", type="primary"): exec_step("plan", lambda: step_clarke_wright(cfg))
    plan = st.session_state.get("t6_plan")
    if plan: render_dashboard(plan)

TAB_NAMES = [
    "🚚 1. Fleet",
    "🏭 2. Warehouse",
    "📦 3. Product",
    "👨‍✈️ 4. Driver",
    "🧾 5. Orders",
    "🗺️ 6. Routing",
]

def main():
    st.title("🚚 Smart Logistics — Hệ thống Chuẩn hóa & Điều phối")
    with st.sidebar:
        st.markdown("### ℹ️ Menu Điều hướng")
        if st.button("🧹 Giải phóng RAM & Cache", key="free_ram"):
            st.cache_data.clear()
            for k in list(st.session_state):
                if k.endswith(("_raw", "_result", "_meta", "_why", "_auto_table")):
                    st.session_state.pop(k, None)
            st.rerun()
    tabs = st.tabs(TAB_NAMES)
    with tabs[0]: render_fleet_tab()
    with tabs[1]: render_warehouse_tab()
    with tabs[2]: render_product_tab()
    with tabs[3]: render_driver_tab()
    with tabs[4]: render_orders_tab()
    with tabs[5]: render_routing_tab()

if __name__ == "__main__":
    main()
