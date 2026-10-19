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
# ⚙️ MÀN HÌNH & TỔNG QUAN HỆ THỐNG
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
OUT_SOLVING = os.path.join(BASE_DIR, "output_solving")
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")

LICENSE_RANKS = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4,
    "B1": 5, "B2": 6, "C": 7, "D": 8, "E": 9,
    "FB2": 10, "FC": 11, "FD": 12, "FE": 13
}
LICENSE_LIST = list(LICENSE_RANKS.keys())

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
# 🚚 TAB 1: FLEET
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
# 🏭 TAB 2: WAREHOUSE
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

def geocode_address(address, ref_wh_lat=21.0285, ref_wh_lng=105.8542, max_valid_km=150.0, retries=3):
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
            if not coordinate_in_vietnam(lat, lng): 
                return {"ok": False, "status": "⚠️ Tọa độ ngoài VN"}
            dist_to_wh = haversine(ref_wh_lat, ref_wh_lng, lat, lng)
            if dist_to_wh > max_valid_km:
                return {"ok": False, "status": f"⚠️ Geocode sai vị trí cách kho {dist_to_wh:.0f}km"}
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
    quality, _ = address_quality(cleaned)
    result = {"id_warehouse": "" if is_blank(warehouse_id) else str(warehouse_id).strip(), "address": cleaned,
              "lat": None, "lng": None, "chất_lượng_địa_chỉ": quality, "trạng_thái_geocode": "—"}
    if is_blank(warehouse_id) or not cleaned or not do_geocode: return result
    geo = geocode_address(cleaned, max_valid_km=10000.0)
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
# 📦 TAB 3: PRODUCT (CẬP NHẬT KÍCH THƯỚC DÀI - RỘNG - CAO CM)
# ============================================================================
PRODUCT_FIELDS = {
    "product_id": ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "code", "id"],
    "product_name": ["tên sản phẩm", "product name", "item name", "tên sp", "name"],
    "length_cm": ["chiều dài", "dài", "length", "length_cm", "dài (cm)"],
    "width_cm": ["chiều rộng", "rộng", "width", "width_cm", "rộng (cm)"],
    "height_cm": ["chiều cao", "cao", "height", "height_cm", "cao (cm)"],
    "weight_kg": ["trọng lượng", "weight", "kg", "trọng lượng (kg)"],
    "cost_price": ["giá sản xuất", "cost price", "cost"],
    "selling_price": ["giá bán", "selling price", "price"],
}
PRODUCT_NUMERIC = ["length_cm", "width_cm", "height_cm", "weight_kg", "cost_price", "selling_price"]

def process_product(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip()) for f in ["product_id", "product_name"]}
    for f in PRODUCT_NUMERIC: out[f] = parse_num(rec.get(f))
    # Tự động tính thể tích m3
    out["volume_m3"] = round((out["length_cm"] * out["width_cm"] * out["height_cm"]) / 1_000_000.0, 6)
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
    seed_product = pd.DataFrame({
        "product_id": ["SP_01", "SP_02"],
        "product_name": ["Ghế Sofa", "Tủ Quần Áo"],
        "length_cm": [120.0, 80.0],
        "width_cm": [60.0, 50.0],
        "height_cm": [80.0, 180.0],
        "weight_kg": [25.0, 45.0],
        "cost_price": [1500000, 2000000],
        "selling_price": [2500000, 3500000]
    })
    mode, edited, uploaded = render_input_block("product", "sản phẩm", seed_product)
    if st.button("🔍 Quét & Mapping", key="product_scan", type="primary"):
        run_scan("product", "sản phẩm", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, PRODUCT_FIELDS, lambda s, f: 0.8, 0.35))
    raw = st.session_state.get("product_raw")
    if raw is None: return
    chosen = render_mapping("product", list(PRODUCT_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa Sản phẩm & Xuất DIM_PRODUCT.xlsx", key="product_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_product({f: pick(r, colmap, f, None) for f in PRODUCT_FIELDS}) for _, r in raw.iterrows()]
        out = product_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_PRODUCT, "DIM_PRODUCT", "DIM_PRODUCT", out)
        store_result("product", out, files, "🧭 Hoàn tất chuẩn hóa sản phẩm với kích thước cm", [("Tổng sản phẩm", len(out))])
    render_result("product", "📊 Kết quả Sản phẩm (DIM_PRODUCT)")

# ============================================================================
# 🧾 TAB 4: ORDERS
# ============================================================================
_ORDER_RAW = {
    "order_id": dict(aliases=["mã đơn", "order id", "order code"]),
    "customer_id": dict(aliases=["mã khách", "customer id"]),
    "customer_name": dict(aliases=["tên khách", "customer name", "name"]),
    "quantity": dict(aliases=["số lượng", "qty", "quantity", "units"]),
    "items": dict(aliases=["mặt hàng", "items", "product", "sản phẩm", "mã sản phẩm"]),
    "total_weight_kg": dict(aliases=["trọng lượng", "weight", "kg"]),
    "total_volume_m3": dict(aliases=["thể tích", "volume", "m3"]),
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

def build_lookup(table):
    exact = {norm_basic(k): l for l, keys in table.items() for k in keys}
    return {"exact": exact, "keys": list(exact.keys())}

TYPE_LK, ALERT_LK = build_lookup(TYPE_TABLE), build_lookup(ALERT_TABLE)

def match_label(value, lk):
    n = norm_basic(str(value))
    if n in lk["exact"]: return lk["exact"][n]
    return None

def orders_semantic_mapping(df):
    result = {}
    for fld in FIELDS:
        best_col, best_score = None, 0.0
        for col in df.columns:
            sc = fuzz.token_set_ratio(norm_basic(col), norm_basic(fld)) / 100.0
            if sc > best_score: best_col, best_score = col, sc
        result[fld] = (best_col, best_score, "tên cột") if best_score > 0.3 else (None, 0.0, "")
    return result

def extract_province(address_str: str) -> str:
    if is_blank(address_str): return "Chưa xác định"
    parts = [p.strip() for p in address_str.split(",")]
    if len(parts) >= 2:
        prov = parts[-1]
        if "Việt Nam" in prov or "Vietnam" in prov:
            prov = parts[-2]
        return prov.strip()
    return "Hà Nội" if "Hà Nội" in address_str or "ha noi" in norm_basic(address_str) else "Tỉnh khác"

def process_order_with_geocode(raw, do_geocode=True):
    r_name = str(raw.get("customer_name", ""))
    cust_name = r_name or "Khách lẻ"
    t_type = match_label(raw.get("order_type", ""), TYPE_LK) or ("B2B" if "cong ty" in norm_basic(r_name) else "B2C")
    t_alert = match_label(raw.get("alert_status", ""), ALERT_LK) or "Normal"
    
    q = parse_qty(raw.get("quantity"))
    w = parse_qty(raw.get("total_weight_kg"))
    v = parse_qty(raw.get("total_volume_m3"), dot_is_decimal=True)
    
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
    if not items_str or items_str == "nan":
        items_str = "SP_01"
        
    province = extract_province(cleaned_addr)
    
    return {
        "order_id": str(raw.get("order_id", "")),
        "customer_id": str(raw.get("customer_id", "")),
        "customer_name": cust_name,
        "quantity": int(q or 1),
        "items": items_str,
        "total_weight_kg": round(w or 0.0, 4),
        "total_volume_m3": round(v or 0.0, 6),
        "address": cleaned_addr,
        "province": province,
        "lat": lat,
        "lng": lng,
        "trạng_thái_geocode": geo_status,
        "order_status": str(raw.get("order_status", "")) or "Mới tạo",
        "order_type": t_type,
        "alert_status": t_alert,
        "order_date": parse_date_iso(raw.get("order_date")),
    }

def orders_validate_output(df):
    out = df.copy()
    msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("order_id")): errs.append("Thiếu id")
        if is_blank(row.get("address")): errs.append("Thiếu địa chỉ/location")
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")): errs.append("Chưa quét được tọa độ")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def is_hanoi_address(addr_str: str) -> bool:
    if not addr_str: return False
    norm = norm_basic(addr_str)
    return "ha noi" in norm or "hanoi" in norm

def render_orders_tab():
    st.header("🧾 Quản lý Đơn hàng & Quét Tọa độ (Orders)")
    do_geo = st.checkbox("🌍 Tự động quét ArcGIS Geocoding cho Location/Address đơn hàng", value=True, key="ord_do_geo")
    mode, edited, uploaded = render_input_block("orders", "đơn hàng", pd.DataFrame({"order_id": ["ORD_003"], "customer_name": ["Nguyễn Văn A"], "items": ["SP_01"], "address": ["Số 1 Tràng Tiền, Hà Nội"], "total_weight_kg": [10.5]}))
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
        out = orders_validate_output(pd.DataFrame(rows))
        
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
        
        cust_df = out.dropna(subset=["lat", "lng"])[["customer_id", "customer_name", "address", "lat", "lng", "trạng_thái_geocode"]].drop_duplicates(subset=["customer_id"])
        save_outputs(OUT_CUSTOMER, "DATASET_CUSTOMER", "DATASET_CUSTOMER", cust_df)
        
        extra = {
            "df_hanoi": optimize_dtypes(df_hanoi, categorize=True),
            "df_outside": optimize_dtypes(df_outside, categorize=True),
            "df_failed": optimize_dtypes(df_failed, categorize=True),
        }
        
        store_result("orders", out, files, "🧭 Hoàn tất chuẩn hóa & phân loại đơn hàng", [
            ("Tổng số đơn", len(out)),
            ("Đơn Hà Nội", len(df_hanoi)),
            ("Đơn Ngoài Hà Nội", len(df_outside)),
            ("Lỗi Geocode", len(df_failed))
        ], extra=extra)
        
    render_result("orders", "📊 Kết quả Đơn hàng")

# ============================================================================
# ⚡ TAB 5 MỚI: ORDERS SOLVED (LEVEL UP: PHÂN LOẠI HÀNG A/B & TỈNH THÀNH)
# ============================================================================
def run_orders_solving(df_orders: pd.DataFrame, df_products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_orders.empty: return pd.DataFrame(), pd.DataFrame()
    
    # Map kích thước từ DIM_PRODUCT
    prod_map = {}
    if not df_products.empty and "product_id" in df_products.columns:
        for _, pr in df_products.iterrows():
            prod_map[str(pr["product_id"]).strip()] = {
                "l": float(pr.get("length_cm", 0.0)),
                "w": float(pr.get("width_cm", 0.0)),
                "h": float(pr.get("height_cm", 0.0))
            }
            
    solved_rows = []
    for _, r in df_orders.iterrows():
        row_dict = dict(r)
        item_id = str(r.get("items", "")).strip()
        qty = int(r.get("quantity", 1))
        real_weight = float(r.get("total_weight_kg", 0.0))
        
        dim = prod_map.get(item_id, {"l": 50.0, "w": 40.0, "h": 30.0}) # Giá trị mặc định nếu thiếu product
        l, w, h = dim["l"], dim["w"], dim["h"]
        
        # CÔNG THỨC: (Dài x Rộng x Cao) x Quantity / 4000
        volumetric_weight = ((l * w * h) * qty) / 4000.0
        
        row_dict["length_cm"] = l
        row_dict["width_cm"] = w
        row_dict["height_cm"] = h
        row_dict["volumetric_weight_kg"] = round(volumetric_weight, 2)
        
        if volumetric_weight > real_weight:
            row_dict["BBS_type"] = "Loại A (Cồng kềnh - BBS)"
            row_dict["BBS_code"] = "A"
        else:
            row_dict["BBS_type"] = "Loại B (Hàng nặng)"
            row_dict["BBS_code"] = "B"
            
        solved_rows.append(row_dict)
        
    df_solved = pd.DataFrame(solved_rows)
    
    # Thống kê theo Tỉnh / Thành phố
    prov_summary = (df_solved.groupby("province")
                    .agg(total_orders=("order_id", "count"),
                         total_weight=("total_weight_kg", "sum"),
                         total_volume=("total_volume_m3", "sum"),
                         bbs_a_count=("BBS_code", lambda x: (x == "A").sum()),
                         bbs_b_count=("BBS_code", lambda x: (x == "B").sum()))
                    .reset_index()
                    .sort_values("total_orders", ascending=False))
    
    prov_summary["ratio_%"] = round(prov_summary["total_orders"] / len(df_solved) * 100, 1)
    return df_solved, prov_summary

def style_bbs(val):
    if "Loại A" in str(val) or val == "A":
        return "background-color: #ffe0b2; color: #e65100; font-weight: bold;" # Màu cam
    elif "Loại B" in str(val) or val == "B":
        return "background-color: #e0e0e0; color: #424242; font-weight: bold;" # Màu xám
    return ""

def render_orders_solving_tab():
    st.header("⚡ Orders Solving — Level Up Phân Loại Hàng A/B & Tỉnh Thành")
    st.caption("🌐 *Kết nối dữ liệu kích thước từ Tab 3 (Product) và vị trí từ Tab 4 (Orders)*")
    
    cfg = Config()
    
    # Đọc dữ liệu Product
    if os.path.exists(cfg.product_file):
        df_prod = read_dim_file(cfg.product_file)
        st.success(f"✅ Đã tải `{len(df_prod)}` sản phẩm từ DIM_PRODUCT.xlsx")
    else:
        df_prod = pd.DataFrame()
        st.warning("⚠️ Chưa có file DIM_PRODUCT.xlsx! Hệ thống sẽ dùng kích thước mặc định (50x40x30 cm).")
        
    # Đọc dữ liệu Orders
    orders_res = st.session_state.get("orders_result")
    if orders_res:
        df_ord = orders_res["df"]
        st.info("⚡ Lấy dữ liệu Đơn hàng từ Session State Tab 4.")
    elif os.path.exists(cfg.order_file):
        df_ord = read_dim_file(cfg.order_file)
        st.info("📂 Đọc dữ liệu Đơn hàng từ file DIM_ORDERS.xlsx.")
    else:
        df_ord = pd.DataFrame()
        st.error("❌ Chưa có dữ liệu Đơn hàng! Vui lòng chạy Tab 4 trước.")
        
    if st.button("🚀 Chạy Phân Loại Hàng A/B & Thống Kê Tỉnh Thành", type="primary", key="btn_run_solving"):
        if df_ord.empty:
            st.error("❌ Không có dữ liệu đơn hàng để xử lý.")
            return
            
        df_solved, prov_summary = run_orders_solving(df_ord, df_prod)
        
        files = save_outputs(OUT_SOLVING, "DIM_ORDERS_SOLVED", "DIM_ORDERS_SOLVED", df_solved, {"PROVINCE_SUMMARY": prov_summary})
        
        a_cnt = int((df_solved["BBS_code"] == "A").sum())
        b_cnt = int((df_solved["BBS_code"] == "B").sum())
        
        store_result("solving", df_solved, files, "⚡ Hoàn tất phân loại hàng BBS & Thống kê Tỉnh thành", [
            ("Tổng đơn", len(df_solved)),
            ("Hàng A (Cồng kềnh - BBS)", a_cnt),
            ("Hàng B (Hàng nặng)", b_cnt),
            ("Số Tỉnh/Thành", len(prov_summary))
        ], extra={"prov_summary": prov_summary})
        
    res = st.session_state.get("solving_result")
    if res:
        st.markdown("### 📊 Kết Quả Phân Loại & Thống Kê")
        prov_df = res["extra"].get("prov_summary", pd.DataFrame())
        
        t1, t2 = st.tabs(["📦 1. Danh Sách Đơn Hàng Solved (Phân loại A/B)", "🌏 2. Thống Kê Số Đơn Theo Tỉnh/Thành Phố"])
        
        with t1:
            st.markdown("##### 📋 Bảng Đơn Hàng Đã Phân Loại BBS")
            styled_df = res["df"].style.map(style_bbs, subset=["BBS_type", "BBS_code"])
            st.dataframe(styled_df, use_container_width=True)
            
        with t2:
            st.markdown("##### 🌏 Thống Kê Khách Đặt Hàng Theo Tỉnh/Thành")
            st.dataframe(prov_df, hide_index=True, use_container_width=True)

# ============================================================================
# 🗺️ TAB 6: ROUTING (CHỈ TÍNH LẤP ĐẦY XE NHÀ & 3 BẢNG RIÊNG)
# ============================================================================
def tag_routes(day):
    return [(r, "Thường") for r in day["routes"]] + [(r, "Quá hạn (ưu tiên)") for r in day["overdue_routes"]]

def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright")
    cfg, depot = render_routing_settings()
    
    st.markdown("### 📥 Chọn Nguồn Dữ Liệu Đơn Hàng Đầu Vào")
    solving_res = st.session_state.get("solving_result")
    orders_res = st.session_state.get("orders_result")
    
    source_options = ["⚡ Dùng Đơn Hàng Từ Tab Orders Solving (Đã phân loại A/B)", "🏢 Chỉ Đơn Hàng Tại Hà Nội", "📂 Đọc Từ File DIM_ORDERS.xlsx"]
    selected_source_label = st.radio("Nguồn đơn hàng:", source_options, horizontal=True, key="routing_src")
    
    selected_df = None
    if "Orders Solving" in selected_source_label and solving_res:
        selected_df = solving_res["df"]
    elif "Hà Nội" in selected_source_label and orders_res and "extra" in orders_res:
        selected_df = orders_res["extra"].get("df_hanoi")
    elif orders_res:
        selected_df = orders_res["df"]

    if st.button("⚡ Chạy Routing Tạm Thời", type="primary", key="t6_run_temp"):
        require_files(cfg, ["fleet"])
        data = load_data_flexible(cfg, source_df=selected_df)
        
        planner, days, kpis, total_cost = simulate_all_flexible(data, cfg)
        st.session_state["t6_temp_plan"] = {"planner": planner, "days": days, "kpis": kpis, "cfg": cfg, "data": data}
        st.session_state.pop("t6_final_plan", None)
        st.success("✅ Đã chạy xong bước tạm thời! Mời xem 3 bảng phân loại bên dưới.")
        
    plan = st.session_state.get("t6_temp_plan")
    if not plan: return
    planner, days, kpis, cfg, data = plan["planner"], plan["days"], plan["kpis"], plan["cfg"], plan["data"]
    dates = list(days)
    if not dates: return
    date_str = st.selectbox("📅 Chọn ngày điều phối", dates, key="t6_date_temp")
    day = days[date_str]
    tagged = tag_routes(day)
    
    st.divider()
    st.markdown(f"## 📊 1. BẢNG ROUTING TẠM THỜI — {date_str}")
    
    # 🌟 TÁCH THÀNH 3 BẢNG THEO YÊU CẦU 🌟
    inhouse_tagged = [(r, k) for r, k in tagged if not r["external"] and r["kind"] != "SPLIT"]
    full_3pl_tagged = [(r, k) for r, k in tagged if (r["external"] and "FULL" in str(r.get("vehicle_id", "")).upper()) or r["kind"] == "SPLIT"]
    saving_3pl_tagged = [(r, k) for r, k in tagged if r["external"] and "FULL" not in str(r.get("vehicle_id", "")).upper()]
    
    t_inh, t_3pl, t_sav = st.tabs([
        f"🚚 1. Các Tuyến Tối Ưu Bằng Xe Nhà ({len(inhouse_tagged)})",
        f"🟣 2. Các Tuyến Thuê Xe 3PL Full ({len(full_3pl_tagged)})",
        f"📦 3. Các Tuyến Giao Hàng Tiết Kiệm Gom Theo Tỉnh ({len(saving_3pl_tagged)})"
    ])
    
    with t_inh:
        st.markdown("##### 🚚 Danh sách các tuyến được phân bổ cho hạm đội xe nhà")
        if inhouse_tagged:
            rdf_inh = routes_to_df(inhouse_tagged, planner.wh, cfg)
            st.dataframe(rdf_inh, hide_index=True, use_container_width=True)
        else:
            st.info("Không có tuyến xe nhà nào trong ngày.")
            
    with t_3pl:
        st.markdown("##### 🟣 Danh sách các tuyến bắt buộc thuê ngoài 3PL Full / Tách chuyến")
        if full_3pl_tagged:
            rdf_3pl = routes_to_df(full_3pl_tagged, planner.wh, cfg)
            st.dataframe(rdf_3pl, hide_index=True, use_container_width=True)
        else:
            st.info("Không có chuyến 3PL Full nào.")
            
    with t_sav:
        st.markdown("##### 📦 Các tuyến Giao hàng tiết kiệm (Đã gom theo Tỉnh/Thành phố)")
        if saving_3pl_tagged:
            # Gom nhóm các tuyến tiết kiệm theo Tỉnh thành
            prov_groups = {}
            for r, _ in saving_3pl_tagged:
                for o in r["orders"]:
                    prov = o.get("province", "Chưa xác định")
                    if prov not in prov_groups:
                        prov_groups[prov] = {"orders": [], "weight": 0.0, "volume": 0.0, "units": 0}
                    prov_groups[prov]["orders"].append(o["order_id"])
                    prov_groups[prov]["weight"] += o["total_weight_kg"]
                    prov_groups[prov]["volume"] += o["total_volume_m3"]
                    prov_groups[prov]["units"] += o.get("quantity", 1)
                    
            saving_rows = []
            for prov, d in prov_groups.items():
                cls = classify_outsourcing(d["weight"], d["volume"], cfg.outsourcing)
                saving_rows.append({
                    "TỈNH / THÀNH PHỐ": prov,
                    "SỐ ĐƠN GOM": len(d["orders"]),
                    "TỔNG UNITS": d["units"],
                    "TỔNG TRỌNG TẢI (kg)": round(d["weight"], 1),
                    "TỔNG THỂ TÍCH (m³)": round(d["volume"], 2),
                    "DỊCH VỤ 3PL ĐỀ XUẤT": cls["type"],
                    "ƯỚC TÍNH CHI PHÍ (đ)": money(cls["cost"]),
                    "DANH SÁCH MÃ ĐƠN": ", ".join(d["orders"])
                })
            st.dataframe(pd.DataFrame(saving_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Không có đơn hàng nào chuyển giao Tiết kiệm.")

    st.divider()
    st.markdown("## 👨‍✈️ 2. PHÂN CÔNG TÀI XẾ CÔNG TY CHO CÁC XE NHÀ")
    
    inhouse_routes = [r for r, _ in tagged if not r["external"]]
    veh_grouped = {}
    for r in inhouse_routes:
        vid = r["vehicle_id"]
        veh_grouped.setdefault(vid, []).append(r)
        
    if not veh_grouped:
        st.info("🎉 Ngày này toàn bộ đơn giao bằng 3PL, không có xe nhà nào cần phân công tài xế.")
    else:
        assignments = []
        has_driver_error = False
        
        for idx, (v_id, r_list) in enumerate(veh_grouped.items(), 1):
            r_list.sort(key=lambda x: x["start"])
            first_r = r_list[0]
            v_wh = first_r["id_warehouse"]
            v_cap_w = first_r["cap_w"]
            suggested_lic = "B2" if v_cap_w < 3500 else ("C" if v_cap_w <= 15000 else "FC")
            
            st.markdown(f"##### 🚚 Xe #{idx}: `{v_id}` (Kho: **{v_wh}** | Tải trọng: **{v_cap_w:,.0f} kg** | **{len(r_list)} chuyến**)")
            driver_choice = st.radio(f"Hình thức tài xế `{v_id}`", ["Nhập tên tài xế công ty", "Thành phần dự phòng (Thuê ngoài)"], horizontal=True, key=f"drv_choice_{v_id}_{idx}")
            
            c1, c2, c3, c4 = st.columns([1.5, 3, 1.5, 2])
            req_lic = c1.selectbox(f"Bằng tối thiểu xe `{v_id}`", LICENSE_LIST, index=LICENSE_LIST.index(suggested_lic), key=f"req_lic_{v_id}_{idx}")
            
            if driver_choice == "Nhập tên tài xế công ty":
                sel_driver = c2.text_input(f"Nhập Tên Tài xế", placeholder="Ví dụ: Nguyễn Văn A", key=f"drv_name_txt_{v_id}_{idx}").strip()
                cur_lic = c3.selectbox(f"Bằng của tài xế", LICENSE_LIST, index=LICENSE_LIST.index("C"), key=f"drv_lic_inp_{v_id}_{idx}")
                cur_role = c4.text_input(f"Vai trò", value="Tài xế chính", key=f"drv_role_inp_{v_id}_{idx}")
                
                if not sel_driver:
                    st.warning("⚠️ Vui lòng nhập tên tài xế!")
                    has_driver_error = True
                else:
                    if LICENSE_RANKS.get(cur_lic, 6) < LICENSE_RANKS.get(req_lic, 6):
                        st.error(f"🔴 **Lỗi Bằng Lái:** Tài xế `{sel_driver}` (Bằng `{cur_lic}`) KHÔNG ĐỦ điều kiện lái xe `{v_id}` (Yêu cầu bằng `{req_lic}`).")
                        has_driver_error = True
            else:
                sel_driver = "Tài xế dự phòng (Thuê ngoài)"
                cur_lic = "C"
                cur_role = "Tài xế dự phòng"
                c2.text_input(f"Tên Tài xế", value=sel_driver, disabled=True, key=f"drv_name_dis_{v_id}_{idx}")
                c3.text_input(f"Bằng của tài xế", value=cur_lic, disabled=True, key=f"drv_lic_dis_{v_id}_{idx}")
                c4.text_input(f"Vai trò", value=cur_role, disabled=True, key=f"drv_role_dis_{v_id}_{idx}")
            assignments.append({"vehicle_id": v_id, "driver_name": sel_driver, "license_type": cur_lic, "role": cur_role})
            st.divider()

    st.divider()
    st.markdown("## 🚛 3. NHẬP LIỆU ĐỘI XE THUÊ NGOÀI (3PL FULL & TÁCH CHUYẾN)")
    full_3pl_routes = [r for r, _ in tagged if (r["external"] and "FULL" in str(r.get("vehicle_id", "")).upper()) or r["kind"] == "SPLIT"]
    user_3pl_inputs = []
    
    if full_3pl_routes:
        for idx, r in enumerate(full_3pl_routes, 1):
            req_w, req_v = route_load(r)
            req_qty = sum(o.get("quantity", 1) for o in r["orders"])
            items_desc = route_items_detail(r)
            
            st.markdown(f"##### 📌 Chuyến 3PL Full #{idx}")
            st.markdown(f"📦 **Yêu cầu:** **{req_qty} Units** | **{req_w:,.1f} kg** | **{req_v:.2f} m³** | `{items_desc}`")
            
            num_rows_key = f"3pl_num_rows_{date_str}_{idx}"
            if num_rows_key not in st.session_state: st.session_state[num_rows_key] = 1
            
            c_add, c_rem = st.columns([2, 8])
            if c_add.button(f"➕ Thêm 1 xe 3PL cho chuyến #{idx}", key=f"add_row_btn_{idx}"):
                st.session_state[num_rows_key] += 1
                st.rerun()
            if st.session_state[num_rows_key] > 1 and c_rem.button(f"➖ Bớt 1 xe 3PL", key=f"rem_row_btn_{idx}"):
                st.session_state[num_rows_key] -= 1
                st.rerun()
                
            total_cap_w, total_cap_v, total_cost, route_trucks = 0.0, 0.0, 0.0, []
            for row_i in range(1, st.session_state[num_rows_key] + 1):
                r1, r2, r3, r4 = st.columns([2, 1.5, 1.5, 2])
                v_name = r1.text_input(f"Tên xe #{row_i}", f"Xe tải 3PL #{row_i}", key=f"3pl_name_{idx}_{row_i}")
                v_w = r2.number_input(f"Tải trọng (kg) #{row_i}", value=float(math.ceil(req_w / st.session_state[num_rows_key])), key=f"3pl_w_{idx}_{row_i}")
                v_v = r3.number_input(f"Thể tích (m³) #{row_i}", value=float(round(req_v / st.session_state[num_rows_key] + 0.2, 1)), key=f"3pl_v_{idx}_{row_i}")
                v_cost = r4.number_input(f"Giá thuê (đ) #{row_i}", value=2_000_000.0, key=f"3pl_cost_{idx}_{row_i}")
                
                total_cap_w += v_w; total_cap_v += v_v; total_cost += v_cost
                route_trucks.append(f"{v_name} ({v_w:,.0f}kg/{v_v:.1f}m³)")
                
            user_3pl_inputs.append({"req_w": req_w, "req_v": req_v, "name": f"{st.session_state[num_rows_key]} xe 3PL [" + " + ".join(route_trucks) + "]", "cap_w": total_cap_w, "cap_v": total_cap_v, "cost": total_cost, "route": r})
            st.divider()

    if st.button("🚀 Kiểm Tra Điều Kiện & Tính Chi Phí Cuối Cùng", type="primary", key="btn_run_final_all"):
        if 'has_driver_error' in locals() and has_driver_error:
            st.error("❌ Vui lòng sửa lỗi tài xế báo đỏ phía trên!")
            return
            
        assign_map = {a["vehicle_id"]: a for a in ('assignments' in locals() and assignments or [])}
        for r, _ in tagged:
            if not r["external"]:
                vid = r["vehicle_id"]
                if vid in assign_map:
                    a = assign_map[vid]
                    r["driver_primary"] = a["driver_name"] if a["driver_name"] != "Tài xế dự phòng (Thuê ngoài)" else "Tài xế Dự Phòng"
                    r["driver_cost"] = 0.0 if a["driver_name"] != "Tài xế dự phòng (Thuê ngoài)" else cfg.backup_driver_cost
                    
        for item in user_3pl_inputs:
            r = item["route"]
            r["vehicle_type"] = item["name"]
            r["cap_w"] = item["cap_w"]
            r["cap_v"] = item["cap_v"]
            r["outsourcing_cost"] = item["cost"]
            
        day["day_cost"] = sum(route_total(rt) for rt in day["routes"])
        st.session_state["t6_final_plan"] = plan
        st.success("🎉 Đã hoàn tất tính toán chi phí cuối cùng!")

    final_plan = st.session_state.get("t6_final_plan")
    if final_plan:
        st.divider()
        st.markdown(f"## 🏆 BÁO CÁO KẾT QUẢ ĐIỀU PHỐI CHÍNH THỨC — {date_str}")
        final_day = final_plan["days"][date_str]
        final_tagged = tag_routes(final_day)
        
        # 🌟 CÔNG THỨC CHỈ TÍNH TỈ LỆ LẤP ĐẦY TRÊN XE NHÀ 🌟
        inhouse_only_routes = [r for r, _ in final_tagged if not r["external"]]
        avg_load_inhouse = (sum(r["load_factor"] for r in inhouse_only_routes) / len(inhouse_only_routes) * 100) if inhouse_only_routes else 0.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 TỔNG CHI PHÍ PHẢI CHI TRONG NGÀY", money(final_day["day_cost"]) + " VNĐ")
        c2.metric("📦 TỔNG ĐƠN ĐÃ PHÂN TUYẾN", len(final_day["orders"]) - len(final_day.get("unrouted_orders", [])))
        c3.metric("📊 TỈ LỆ LẤP ĐẦY TRUNG BÌNH (CHỈ XE NHÀ)", f"{avg_load_inhouse:.1f}%")
        
        final_rdf = routes_to_df(final_tagged, planner.wh, cfg)
        st.dataframe(final_rdf, hide_index=True)

# ============================================================================
# 🚨 TAB 7: INCIDENT CONTROL CENTER
# ============================================================================
def render_incident_tab():
    st.header("🚨 7. TRUNG TÂM ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ")
    st.caption("🌐 *Hệ thống giám sát thực địa thời gian thực*")
    st.info("Tính năng sẵn sàng kết nối cùng các Tab điều phối.")

# ============================================================================
# ⚙️ MAIN CONTAINERS
# ============================================================================
TAB_NAMES = [
    "🚚 1. Fleet",
    "🏭 2. Warehouse",
    "📦 3. Product",
    "🧾 4. Orders",
    "⚡ 5. Orders Solving",
    "🗺️ 6. Routing",
    "🚨 7. Xử Lý Sự Cố"
]

def main():
    st.title("🚚 Smart Logistics — Hệ thống chuẩn hóa & Điều phối Vận tải (của Phạm Minh Thái)")
    with st.sidebar:
        st.markdown("### ℹ️ Menu Điều hướng")
        if st.button("🧹 Giải phóng RAM & Cache", key="free_ram"):
            st.cache_data.clear()
            for k in list(st.session_state):
                if k.endswith(("_raw", "_result", "_meta", "_why", "_auto_table", "_plan", "_temp_plan", "_final_plan", "solving_result")):
                    st.session_state.pop(k, None)
            st.rerun()
            
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
