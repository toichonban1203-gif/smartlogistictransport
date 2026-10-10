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
# 🗺️ TAB 6: ĐỊNH TUYẾN & RÀNG BUỘC VẬN TẢI NÂNG CAO (CLARKE-WRIGHT)
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
    fw, fv = max(float(oc["full_vehicle_max_weight_kg"]), 1e-9), max(float(oc["full_vehicle_max_volume_m3"]), 1e-9)
    trips = max(1, math.ceil(w / fw - 1e-9), math.ceil(v / fv - 1e-9))
    return {"type": f"Full — {oc['full_vehicle_name']} × {trips} chuyến", "tier": "FULL", "cost": trips * float(oc["full_price"]),
            "trips": trips, "vehicle": oc["full_vehicle_name"], "cap_w": trips * fw}

def evaluate_transportation_and_time_constraints(
    route_or_order, 
    df_vehicles, 
    vehicle_available_counts, 
    outsourcing_config,
    start_time_str="08:30",
    max_hours_input=8.0,
    service_time_rules=None
):
    if service_time_rules is None:
        service_time_rules = {
            "B2C": {"loading": 25, "unloading": 35}, 
            "B2B": {"loading": 45, "unloading": 60}  
        }

    total_w = float(route_or_order.get("total_weight_kg", route_or_order.get("weight", 0.0)))
    total_v = float(route_or_order.get("total_volume_m3", route_or_order.get("volume", 0.0)))
    total_dist = float(route_or_order.get("total_distance_km", route_or_order.get("km", 0.0)))
    orders_in_route = route_or_order.get("orders", [])
    
    eff_max_hours = calculate_effective_max_hours(start_time_str, max_hours_input)
    has_alert_order = any(str(o.get("alert_status", "")).strip().lower() == "alert" for o in orders_in_route)
    
    # 1. ĐƠN KHẨN CẤP (ALERT)
    if has_alert_order:
        feasible_internal = df_vehicles[
            (df_vehicles["max_weight_kg"] >= total_w) & 
            (df_vehicles["max_volume_m3"] >= total_v)
        ].sort_values(by="max_weight_kg", ascending=True)
        
        chosen_internal = None
        for _, veh in feasible_internal.iterrows():
            v_id = veh["vehicle_id"]
            if vehicle_available_counts.get(v_id, 1) > 0:
                w_fill = total_w / veh["max_weight_kg"] if veh["max_weight_kg"] > 0 else 0
                v_fill = total_v / veh["max_volume_m3"] if veh["max_volume_m3"] > 0 else 0
                if max(w_fill, v_fill) >= 0.70:
                    chosen_internal = veh
                    break
        
        if chosen_internal is not None:
            chosen_v_id = chosen_internal["vehicle_id"]
            vehicle_available_counts[chosen_v_id] -= 1
            v_speed = float(chosen_internal.get("average_speed_kmh", 40.0))
            if v_speed <= 0: v_speed = 40.0
            
            travel_hrs = total_dist / v_speed
            service_mins = sum(
                service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["loading"] + 
                service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["unloading"] 
                for o in orders_in_route
            )
            total_duration = travel_hrs + (service_mins / 60.0)
            
            sh, sm = map(int, start_time_str.split(":"))
            start_hrs = sh + sm / 60.0
            if start_hrs < 12.0 and (start_hrs + total_duration) > 12.0:
                total_duration += 1.0

            if total_duration <= eff_max_hours:
                return {
                    "status": "APPROVED_ALERT_INTERNAL", "action_type": "ALERT_IN_HOUSE",
                    "assigned_vehicle": chosen_v_id, "outsourcing_type": None, "cost": 0.0,
                    "total_hours": round(total_duration, 2),
                    "message": f"🚨 Đơn Alert: Xếp xe nhà riêng {chosen_v_id} (Độ phủ $\\ge 70\\%$ và hoàn thành trước 17:30)."
                }
        
        speed_3pl = 40.0
        travel_hrs = total_dist / speed_3pl
        service_mins = sum(
            service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["loading"] + 
            service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["unloading"] 
            for o in orders_in_route
        )
        total_duration = travel_hrs + (service_mins / 60.0)
        
        sh, sm = map(int, start_time_str.split(":"))
        start_hrs = sh + sm / 60.0
        if start_hrs < 12.0 and (start_hrs + total_duration) > 12.0:
            total_duration += 1.0

        return {
            "status": "OUTSOURCED_ALERT_3PL", "action_type": "ALERT_OUTSOURCE",
            "assigned_vehicle": "3PL_Outsource_Vehicle",
            "outsourcing_type": "Full Outsourcing (Xe riêng Alert)",
            "cost": outsourcing_config.get("full_price", 2500000.0),
            "total_hours": round(total_duration, 2),
            "message": f"🚨 Đơn Alert: Đẩy sang Thuê ngoài 3PL (Tốc độ 40 km/h, do xe nhà lấp đầy < 70% hoặc không kịp giờ)."
        }

    # 2. ĐƠN THƯỜNG (NORMAL)
    if df_vehicles.empty:
        return {
            "status": "OUTSOURCED", "action_type": "NO_FLEET",
            "outsourcing_type": "Full Outsourcing", "cost": outsourcing_config.get("full_price", 2500000.0),
            "message": "⚠️ Hạm đội trống -> Thuê ngoài Full."
        }
        
    max_fleet_w = df_vehicles["max_weight_kg"].max()
    max_fleet_v = df_vehicles["max_volume_m3"].max()
    sum_fleet_w = df_vehicles["max_weight_kg"].sum()
    sum_fleet_v = df_vehicles["max_volume_m3"].sum()
    
    if total_w > sum_fleet_w or total_v > sum_fleet_v:
        return {
            "status": "OUTSOURCED", "action_type": "OUTSOURCE_FULL",
            "outsourcing_type": "Full Outsourcing",
            "cost": outsourcing_config.get("full_price", 2500000.0),
            "message": "🚨 Vượt quá tổng hạm đội nhà -> Thuê ngoài loại Full."
        }

    feasible_vehicles = df_vehicles[
        (df_vehicles["max_weight_kg"] >= total_w) & 
        (df_vehicles["max_volume_m3"] >= total_v)
    ].sort_values(by="max_weight_kg", ascending=True)
    
    chosen_vehicle = None
    for _, veh in feasible_vehicles.iterrows():
        if vehicle_available_counts.get(veh["vehicle_id"], 1) > 0:
            chosen_vehicle = veh
            break

    if chosen_vehicle is None:
        return {
            "status": "OUTSOURCED", "action_type": "OUTSOURCE_NO_VEHICLE_AVAILABLE",
            "outsourcing_type": "Tiết kiệm loại 1",
            "cost": outsourcing_config.get("saving_1_price", 800000.0),
            "message": "⚠️ Hết xe nhà khả dụng -> Thuê ngoài 3PL (Tốc độ 40 km/h)."
        }

    chosen_v_id = chosen_vehicle["vehicle_id"]
    vehicle_available_counts[chosen_v_id] -= 1
    
    v_speed = float(chosen_vehicle.get("average_speed_kmh", 40.0))
    if v_speed <= 0: v_speed = 40.0

    max_dist_limit = float(chosen_vehicle.get("Max_Distance", 100.0))
    if total_dist > max_dist_limit:
        return {
            "status": "BACKLOG_TRIGGERED", "action_type": "DISTANCE_EXCEEDED_BACKLOG",
            "assigned_vehicle": chosen_v_id, "outsourcing_type": "Tiết kiệm loại 2",
            "cost": outsourcing_config.get("saving_2_price", 300000.0),
            "message": f"⚠️ Vượt Max_Distance ({total_dist}km > {max_dist_limit}km) của xe {chosen_v_id}."
        }

    travel_hrs = total_dist / v_speed
    service_mins = sum(
        service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["loading"] + 
        service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["unloading"] 
        for o in orders_in_route
    )
    total_duration = travel_hrs + (service_mins / 60.0)
    
    sh, sm = map(int, start_time_str.split(":"))
    start_hrs = sh + sm / 60.0
    if start_hrs < 12.0 and (start_hrs + total_duration) > 12.0:
        total_duration += 1.0

    if total_duration > eff_max_hours:
        return {
            "status": "BACKLOG_TIME_EXCEEDED", "action_type": "TIME_LIMIT_EXCEEDED",
            "assigned_vehicle": chosen_v_id,
            "total_hours": round(total_duration, 2),
            "eff_max_hours": round(eff_max_hours, 2),
            "message": f"⚠️ Thời gian tuyến ({round(total_duration, 2)}h) vượt trần khả thi ({round(eff_max_hours, 2)}h đến 17:30)!"
        }

    return {
        "status": "APPROVED", "action_type": "IN_HOUSE_SUCCESS",
        "assigned_vehicle": chosen_v_id,
        "total_hours": round(total_duration, 2),
        "message": f"✅ Thỏa mãn hoàn toàn! Xe {chosen_v_id} (Vận tốc {v_speed} km/h, thời gian: {round(total_duration, 2)}h)."
    }

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

def read_matrix(path: str) -> pd.DataFrame:
    df = read_dim_file(path, index_col=0).copy()
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    return df

def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright & Ràng buộc Vận tải")
    
    with st.expander("⚙️ Cấu hình Chi phí Thuê ngoài & Tham số Ca làm việc", expanded=True):
        c1, c2, c3 = st.columns(3)
        full_p = c1.number_input("Giá thuê ngoài Full (VNĐ)", min_value=0.0, value=2500000.0, step=100000.0)
        s1_p = c2.number_input("Giá thuê ngoài Tiết kiệm 1 (VNĐ)", min_value=0.0, value=800000.0, step=50000.0)
        s2_p = c3.number_input("Giá thuê ngoài Tiết kiệm 2 (VNĐ)", min_value=0.0, value=300000.0, step=50000.0)
        
        c4, c5 = st.columns(2)
        start_time_val = c4.text_input("Giờ xuất phát (HH:MM)", value="08:30")
        max_hrs_val = c5.number_input("Giới hạn ca làm việc (Giờ)", min_value=1.0, max_value=12.0, value=8.0)
    
    outsourcing_cfg = {"full_price": full_p, "saving_1_price": s1_p, "saving_2_price": s2_p}
    
    st.markdown("### 📂 Trạng thái dữ liệu đầu vào (Schema Contract)")
    rows = []
    for path, cols in SCHEMA_CONTRACT.items():
        full = os.path.join(BASE_DIR, path)
        status = "✅ Sẵn sàng" if os.path.exists(full) else "❌ Chưa có"
        rows.append({"File": path, "Trạng thái": status, "Cột chuẩn": ", ".join(cols)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    
    if st.button("🚀 Run Preview Test Ràng buộc Vận tải", key="run_test_tab6", type="primary"):
        fleet_path = os.path.join(BASE_DIR, "output_fleet/DIM_VEHICLE.xlsx")
        if not os.path.exists(fleet_path):
            st.error("❌ Chưa tìm thấy file `DIM_VEHICLE.xlsx` trong thư mục `output_fleet`. Vui lòng chạy Tab 1 trước!")
            return
            
        df_veh = pd.read_excel(fleet_path)
        veh_counts = {row["vehicle_id"]: 2 for _, row in df_veh.iterrows()}
        
        # Test Case 1: Đơn thường
        sample_route_normal = {
            "total_weight_kg": 1500.0, "total_volume_m3": 8.0, "total_distance_km": 60.0,
            "orders": [{"order_type": "B2C", "alert_status": "Normal"}]
        }
        res_normal = evaluate_transportation_and_time_constraints(
            sample_route_normal, df_veh.copy(), veh_counts.copy(), outsourcing_cfg,
            start_time_str=start_time_val, max_hours_input=max_hrs_val
        )
        
        # Test Case 2: Đơn Alert
        sample_route_alert = {
            "total_weight_kg": 3000.0, "total_volume_m3": 15.0, "total_distance_km": 80.0,
            "orders": [{"order_type": "B2B", "alert_status": "Alert"}]
        }
        res_alert = evaluate_transportation_and_time_constraints(
            sample_route_alert, df_veh.copy(), veh_counts.copy(), outsourcing_cfg,
            start_time_str=start_time_val, max_hours_input=max_hrs_val
        )
        
        st.success("🏁 Kiểm thử thành công!")
        st.markdown("#### 1. Kết quả Đơn thường (Normal):")
        st.json(res_normal)
        st.markdown("#### 2. Kết quả Đơn khẩn cấp (Alert):")
        st.json(res_alert)

# ============================================================================
# ⚙️ MAIN APP CONTAINER
# ============================================================================
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
