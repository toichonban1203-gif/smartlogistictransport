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
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")

LICENSE_RANKS = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4,
    "B1": 5, "B2": 6,
    "C": 7,
    "D": 8, "E": 9,
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
    if re.search(r"Tôn Đức Thắng", text, re.I) and re.search(r"Bách Khoa", text, re.I):
        text = re.sub(r"Phường Bách Khoa,?\s*", "", text, flags=re.I)
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
# 📦 TAB 3: PRODUCT
# ============================================================================
PRODUCT_FIELDS = {
    "product_id": ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "code", "id"],
    "product_name": ["tên sản phẩm", "product name", "item name", "tên sp", "name"],
    "volume": ["thể tích", "volume", "m3", "cbm"],
    "weight": ["trọng lượng", "weight", "kg"],
    "cost_price": ["giá sản xuất", "cost price", "cost"],
    "selling_price": ["giá bán", "selling price", "price"],
}
PRODUCT_NUMERIC = ["volume", "weight", "cost_price", "selling_price"]

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
# 🧾 TAB 4: ORDERS
# ============================================================================
_ORDER_RAW = {
    "order_id": dict(aliases=["mã đơn", "order id", "order code"]),
    "customer_id": dict(aliases=["mã khách", "customer id"]),
    "customer_name": dict(aliases=["tên khách", "customer name", "name"]),
    "quantity": dict(aliases=["số lượng", "qty", "quantity", "units"]),
    "items": dict(aliases=["mặt hàng", "items", "product", "sản phẩm"]),
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
        items_str = "Hàng tiêu chuẩn"

    return {
        "order_id": str(raw.get("order_id", "")),
        "customer_id": str(raw.get("customer_id", "")),
        "customer_name": cust_name,
        "quantity": int(q or 1),
        "items": items_str,
        "total_weight_kg": round(w or 0.0, 4),
        "total_volume_m3": round(v or 0.0, 6),
        "address": cleaned_addr,
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
    mode, edited, uploaded = render_input_block("orders", "đơn hàng", pd.DataFrame({"order_id": ["ORD_003"], "customer_name": ["Nguyễn Văn A"], "address": ["Số 1 Tràng Tiền, Hà Nội"], "total_weight_kg": [10.5]}))
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
        
        # 1. Lưu file tổng chuẩn DIM_ORDERS
        files = save_outputs(OUT_ORDERS, "DIM_ORDERS", "DIM_ORDERS", out)
        
        # 2. Phân tách thành 3 bảng với cấu trúc y hệt bảng gốc
        mask_failed = out["trạng_thái_geocode"] != "✅ Thành công"
        df_failed = out[mask_failed].copy().reset_index(drop=True)
        
        df_success = out[~mask_failed].copy()
        mask_hanoi = df_success["address"].apply(is_hanoi_address)
        df_hanoi = df_success[mask_hanoi].copy().reset_index(drop=True)
        df_outside = df_success[~mask_hanoi].copy().reset_index(drop=True)
        
        # Lưu các file phụ trợ
        file_hn = save_outputs(OUT_ORDERS, "DIM_ORDERS_HA_NOI", "DIM_ORDERS_HA_NOI", df_hanoi)
        file_out = save_outputs(OUT_ORDERS, "DIM_ORDERS_OUTSIDE_HN", "DIM_ORDERS_OUTSIDE_HN", df_outside)
        file_err = save_outputs(OUT_ORDERS, "DIM_ORDERS_FAILED", "DIM_ORDERS_FAILED", df_failed)
        
        files.extend([file_hn[0], file_out[0], file_err[0]])
        
        # Lưu thông tin Customer
        cust_df = out.dropna(subset=["lat", "lng"])[["customer_id", "customer_name", "address", "lat", "lng", "trạng_thái_geocode"]].drop_duplicates(subset=["customer_id"])
        save_outputs(OUT_CUSTOMER, "DATASET_CUSTOMER", "DATASET_CUSTOMER", cust_df)
        
        ok_count = int((out["trạng_thái_geocode"] == "✅ Thành công").sum())
        
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
        
    res = st.session_state.get("orders_result")
    if res is not None:
        st.markdown("### 📊 Kết quả Phân Loại Đơn Hàng")
        st.success(res["summary"])
        
        if res["metrics"]:
            cols = st.columns(len(res["metrics"]))
            for c, (lab, val) in zip(cols, res["metrics"]): c.metric(lab, val)
            
        extra = res.get("extra", {})
        df_hn = extra.get("df_hanoi", pd.DataFrame())
        df_out = extra.get("df_outside", pd.DataFrame())
        df_fail = extra.get("df_failed", pd.DataFrame())
        
        t_hn, t_out, t_fail, t_all = st.tabs([
            f"🏢 1. Đơn tại Hà Nội ({len(df_hn)})",
            f"🌏 2. Đơn ngoài Hà Nội ({len(df_out)})",
            f"⚠️ 3. Không quét được tọa độ ({len(df_fail)})",
            f"📋 4. Tất cả đơn tổng ({len(res['df'])})"
        ])
        
        with t_hn:
            st.markdown("##### 🏢 Danh sách các đơn hàng thuộc khu vực Hà Nội")
            st.dataframe(df_hn, use_container_width=True)
            
        with t_out:
            st.markdown("##### 🌏 Danh sách các đơn hàng ngoài khu vực Hà Nội")
            st.dataframe(df_out, use_container_width=True)
            
        with t_fail:
            st.markdown("##### ⚠️ Danh sách các đơn hàng không quét được tọa độ")
            st.dataframe(df_fail, use_container_width=True)
            
        with t_all:
            st.markdown("##### 📋 Bảng tổng hợp tất cả đơn hàng")
            st.dataframe(res["df"], use_container_width=True)

        st.caption(f"🧠 RAM kết quả: {mem_kb(res['df']):,.1f} KB")
        rel = ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"])
        st.info(f"📁 Thư mục lưu các file: {rel}")
        
        cols = st.columns(min(len(res["files"]), 4))
        for i, path in enumerate(res["files"]):
            if os.path.exists(path):
                with open(path, "rb") as fh: data = fh.read()
                mime = XLSX_MIME if path.endswith(".xlsx") else "application/json"
                cols[i % 4].download_button(f"⬇️ Tải {os.path.basename(path)}", data, file_name=os.path.basename(path), mime=mime, key=f"orders_dl_{os.path.basename(path)}_{i}")

# ============================================================================
# 🗺️ TAB 5: ĐỊNH TUYẾN CLARKE-WRIGHT
# ============================================================================
OC_DEFAULTS = {
    "saving_1_price": 800_000.0, "saving_2_price": 300_000.0,
    "tier2_max_w": 20.0, "tier2_max_v": 1.0,
    "tier1_max_w": 100.0, "tier1_max_v": 5.0,
    "full_vehicle_name": "Xe 3PL Full", "full_vehicle_max_weight_kg": 10_000.0, "full_vehicle_max_volume_m3": 40.0, "full_price": 0.0
}

def calculate_effective_max_hours(start_time_str="08:30", max_hours_input=8.0):
    try:
        sh, sm = map(int, start_time_str.split(":"))
        start_in_hours = sh + sm / 60.0
    except Exception:
        start_in_hours = 8.5
    cutoff_hours = 17.5
    return min(float(max_hours_input), max(0.0, cutoff_hours - start_in_hours))

def _full_cls(w, v, oc, why=None):
    fw = max(float(oc.get("full_vehicle_max_weight_kg", 10000.0)), 1e-9)
    fv = max(float(oc.get("full_vehicle_max_volume_m3", 40.0)), 1e-9)
    trips = max(1, math.ceil(w / fw - 1e-9), math.ceil(v / fv - 1e-9))
    if why is None:
        why = f"tải {w:,.0f}kg/{v:.1f}m³ vượt ngưỡng Tiết kiệm 1 → Cần xe 3PL Full ({trips} chuyến)"
    else:
        why = f"Cần xe 3PL Full ({why})"
    return {"type": f"3PL Full — {oc.get('full_vehicle_name', 'Xe 3PL Full')} × {trips} chuyến", "tier": "FULL", "cost": float(oc.get("full_price", 0.0)),
            "trips": trips, "vehicle": oc.get("full_vehicle_name", "Xe 3PL Full"), "cap_w": trips * fw, "cap_v": trips * fv, "why": why}

def classify_outsourcing(excess_w, excess_v, oc=None, force_full=False) -> dict:
    oc = {**OC_DEFAULTS, **(oc or {})}
    w, v = max(float(excess_w), 0.0), max(float(excess_v), 0.0)
    if not force_full:
        if w < oc["tier2_max_w"] and v < oc["tier2_max_v"]:
            return {"type": "Tiết kiệm loại 2", "tier": "SAVING_2", "cost": float(oc["saving_2_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 2", "cap_w": float(oc["tier2_max_w"]), "cap_v": float(oc["tier2_max_v"]),
                    "why": f"tải {w:.1f}kg < {oc['tier2_max_w']:g}kg và {v:.2f}m³ < {oc['tier2_max_v']:g}m³ nên chọn Tiết kiệm 2"}
        if w <= oc["tier1_max_w"] and v <= oc["tier1_max_v"]:
            return {"type": "Tiết kiệm loại 1", "tier": "SAVING_1", "cost": float(oc["saving_1_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 1", "cap_w": float(oc["tier1_max_w"]), "cap_v": float(oc["tier1_max_v"]),
                    "why": f"tải {w:.1f}kg ≤ {oc['tier1_max_w']:g}kg và {v:.2f}m³ ≤ {oc['tier1_max_v']:g}m³ nên chọn Tiết kiệm 1"}
    return _full_cls(w, v, oc, "bắt buộc thuê Full" if force_full else None)

def evaluate_transportation_constraints(route_or_order, df_vehicles, vehicle_available_counts, outsourcing_config):
    oc = {**OC_DEFAULTS, **(outsourcing_config or {})}
    total_w = float(route_or_order.get("total_weight_kg", 0.0))
    total_v = float(route_or_order.get("total_volume_m3", 0.0))
    total_dist = float(route_or_order.get("total_distance_km", 0.0))
    
    def outsourced(action, cls, message):
        return {"status": "OUTSOURCED", "action_type": action, "assigned_vehicle": None, "outsourcing_type": cls["type"],
                "cost": cls["cost"], "cls": cls, "message": message}
                
    if df_vehicles.empty:
        return outsourced("NO_FLEET", _full_cls(total_w, total_v, oc, "bắt buộc khi kho không có xe nhà"), "⚠️ Hạm đội trống -> Thuê ngoài Full.")
    
    max_fleet_w, max_fleet_v = df_vehicles["max_weight_kg"].max(), df_vehicles["max_volume_m3"].max()
    sum_fleet_w, sum_fleet_v = df_vehicles["max_weight_kg"].sum(), df_vehicles["max_volume_m3"].sum()
    avail = df_vehicles[df_vehicles["vehicle_id"].map(lambda x: vehicle_available_counts.get(x, 1) > 0)]
    
    if total_w > 1.5 * max_fleet_w or total_v > 1.5 * max_fleet_v or total_w > sum_fleet_w or total_v > sum_fleet_v:
        return outsourced("OUTSOURCE_FULL", _full_cls(total_w, total_v, oc, f"hàng quá khủng vượt 150% xe lớn nhất nhà ({max_fleet_w:,.0f}kg/{max_fleet_v:.1f}m³)"),
                        f"🚨 Hàng quá khủng (vượt >150% xe lớn nhất nhà {max_fleet_w:,.0f}kg/{max_fleet_v:.1f}m³) -> Thuê ngoài 3PL Full ngay lập tức.")
    if total_w > max_fleet_w or total_v > max_fleet_v:
        if avail.empty:
            return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE", classify_outsourcing(total_w, total_v, oc),
                            "⚠️ Hết xe khả dụng trong ngày -> thuê ngoài qua màn lọc.")
        big = avail.loc[avail["max_weight_kg"].idxmax()]
        v_id = big["vehicle_id"]
        vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
        cls = classify_outsourcing(total_w - big["max_weight_kg"], total_v - big["max_volume_m3"], oc)
        return {"status": "PARTIAL_SPLIT", "action_type": "MAX_VEHICLE_PLUS_OUTSOURCE", "assigned_vehicle": v_id,
                "outsourcing_type": cls["type"], "cost": cls["cost"], "cls": cls,
                "message": f"⚠️ Vượt xe lớn nhất nhà nhưng <= 150% ({v_id}). Cắt xe này, phần dư thuê ngoài {cls['type']}."}
                
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
                "message": f"⚠️ Tuyến vượt Max_Distance ({total_dist:.1f}km > {limit:.0f}km) của xe {v_id}. Cắt theo Units."}
                
    vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
    return {"status": "APPROVED", "action_type": "IN_HOUSE_SUCCESS", "assigned_vehicle": v_id, "outsourcing_type": None,
            "cost": 0.0, "cls": None, "message": f"✅ Thỏa mãn trọng tải! Giao xe NHỎ NHẤT khả thi: {v_id}."}

def evaluate_time_constraints(route_or_order, chosen_vehicle=None, service_time_rules=None, max_hours=8.0):
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
            "message": f"⚠️ Tuyến vượt quá {max_hours:g} giờ ({round(total, 2)}h > {max_hours:g}h)! Cắt theo Units."}

@dataclass
class Config:
    start_time: str = "08:30"
    max_route_hours: float = 8.0
    detour_factor: float = 1.2
    constraint_mode: str = "Cube Out (Thể tích m³)"
    service_min: dict = field(default_factory=lambda: {"B2B": 105, "B2C": 60})
    backup_driver_cost: float = 400_000.0
    vehicle_daily_count: int = 2
    turnaround_min: float = 30.0
    outsourced_speed_kmh: float = 40.0
    saving_1_price: float = OC_DEFAULTS["saving_1_price"]
    saving_2_price: float = OC_DEFAULTS["saving_2_price"]
    tier2_max_w: float = OC_DEFAULTS["tier2_max_w"]
    tier2_max_v: float = OC_DEFAULTS["tier2_max_v"]
    tier1_max_w: float = OC_DEFAULTS["tier1_max_w"]
    tier1_max_v: float = OC_DEFAULTS["tier1_max_v"]
    full_price: float = 0.0
    full_vehicle_name: str = "Xe 3PL Full"
    full_vehicle_max_weight_kg: float = 10000.0
    full_vehicle_max_volume_m3: float = 40.0
    vehicle_file: str = os.path.join(OUT_FLEET, "DIM_VEHICLE.xlsx")
    warehouse_file: str = os.path.join(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES.xlsx")
    product_file: str = os.path.join(OUT_PRODUCT, "DIM_PRODUCT.xlsx")
    order_file: str = os.path.join(OUT_ORDERS, "DIM_ORDERS.xlsx")
    cust_file: str = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.xlsx")
    matrix_file: str = os.path.join(OUT_MATRIX, "DISTANCE_MATRIX_KM.xlsx")

    @property
    def outsourcing(self) -> dict:
        return {
            "saving_1_price": self.saving_1_price,
            "saving_2_price": self.saving_2_price,
            "tier2_max_w": self.tier2_max_w,
            "tier2_max_v": self.tier2_max_v,
            "tier1_max_w": self.tier1_max_w,
            "tier1_max_v": self.tier1_max_v,
            "full_price": self.full_price,
            "full_vehicle_name": self.full_vehicle_name,
            "full_vehicle_max_weight_kg": self.full_vehicle_max_weight_kg,
            "full_vehicle_max_volume_m3": self.full_vehicle_max_volume_m3,
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
    if not os.path.exists(cfg.cust_file): return out
    for _, r in read_dim_file(cfg.cust_file).iterrows():
        cid = str(r.get("customer_id", "")).strip()
        if cid and cid != "nan" and pd.notna(r.get("lat")) and pd.notna(r.get("lng")):
            out[cid] = {"lat": float(r["lat"]), "lng": float(r["lng"]), "name": "" if is_blank(r.get("customer_name")) else str(r["customer_name"])}
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
        
        lat = float(r["lat"]) if pd.notna(r.get("lat")) else None
        lng = float(r["lng"]) if pd.notna(r.get("lng")) else None
        qty = int(parse_qty(r.get("quantity")) or 1)
        items_str = "" if is_blank(r.get("items")) else str(r["items"]).strip()
        
        orders.append({
            "order_id": oid, "customer_id": cid, "order_type": otype if otype in ("B2B", "B2C") else "B2C",
            "quantity": qty,
            "items": items_str or "Hàng tiêu chuẩn",
            "total_weight_kg": parse_qty(r.get("total_weight_kg")), "total_volume_m3": parse_qty(r.get("total_volume_m3"), dot_is_decimal=True),
            "address": "" if is_blank(r.get("address")) else str(r["address"]),
            "lat": lat, "lng": lng,
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
        wh_coords = {"WH_DEFAULT": (21.0285, 105.8542)}
        notes.append("Chưa có tọa độ kho (Tab 2) → dùng kho mặc định Hà Nội.")
    warehouses = {w: {"lat": la, "lng": lo, "name": w} for w, (la, lo) in wh_coords.items()}
    orphan = sorted(set(veh["id_warehouse"]) - set(warehouses))
    if orphan: notes.append(f"Xe thuộc kho không có tọa độ: {', '.join(orphan)}.")
    return {"vehicles": veh,
            "dist": read_matrix(cfg.matrix_file) if os.path.exists(cfg.matrix_file) else pd.DataFrame(), 
            "cust": load_customers(cfg), "warehouses": warehouses, "orders": load_orders(cfg, notes), "notes": notes}

_TAB_A_EVENTS = {
    "NO_FLEET": ("HIGH", "KHO KHÔNG CÓ XE", "Bổ sung xe cho kho"),
    "OUTSOURCE_FULL": ("CRITICAL", "VƯỢT TỔNG HẠM ĐỘI", "Cần xe 3PL Full"),
    "OUTSOURCE_NO_VEHICLE_AVAILABLE": ("MEDIUM", "HẾT XE TRONG NGÀY", "Bổ sung xe / dời lịch"),
    "MAX_VEHICLE_PLUS_OUTSOURCE": ("CRITICAL", "ĐƠN QUÁ CỠ", "Cần xe 3PL Full"),
}

class RoutePlanner:
    def __init__(self, data: dict, cfg: Config = CFG):
        self.cfg = cfg
        self.veh = data["vehicles"]
        self.cust = data["cust"]
        self.wh = data["warehouses"]
        self.detour = cfg.detour_factor
        self.max_hours = calculate_effective_max_hours(cfg.start_time, cfg.max_route_hours)
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
        return haversine(a["lat"], a["lng"], b["lat"], b["lng"]) * self.detour

    def d_cc(self, i, j) -> float:
        if i == j: return 0.0
        if i in self.cust and j in self.cust: 
            return self._hav(self.cust[i], self.cust[j])
        return 0.0

    def d_wc(self, wh_id, c) -> float:
        if wh_id in self.wh and c in self.cust: 
            return self._hav(self.wh[wh_id], self.cust[c])
        return 0.0

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
        qty = sum(demand[c]["qty"] for c in route)
        km = self.km(route, wh_id)
        veh = self.smallest_vehicle(wh_id, w, v)
        vtype = veh["vehicle_type"] if veh else self.fit_type(w, v)
        speed = float(veh["average_speed_kmh"]) if veh else (float(self.catalog.loc[vtype, "speed"]) if vtype in self.catalog.index else 35.0)
        service = sum(demand[c]["service_min"] for c in route) / 60.0
        return {"w": w, "v": v, "qty": qty, "km": km, "hours": km / speed + service, "vtype": vtype, "speed": speed, "veh": veh}

    def feasible(self, route, wh_id, demand) -> bool:
        cap_w, cap_v, _ = self.caps(wh_id)
        m = self.metrics(route, wh_id, demand)
        if m["w"] > cap_w or m["v"] > cap_v or m["hours"] > self.max_hours: return False
        if m["veh"] is None: return not self._fleet.get(wh_id)
        return m["km"] <= m["veh"]["Max_Distance"]

    def route_cost(self, route, wh_id, demand) -> float:
        m = self.metrics(route, wh_id, demand)
        veh = m["veh"]
        if veh is None:
            return float(classify_outsourcing(m["w"], m["v"], self.cfg.outsourcing)["cost"])
        return float(veh.get("fixed_cost", 0.0)) + float(veh.get("variable_cost", 0.0)) * m["km"]

    def two_opt(self, route, wh_id):
        def edge(a, b):
            if a is None: return self.d_wc(wh_id, b)
            if b is None: return self.d_wc(wh_id, a)
            return self.d_cc(a, b)
        best, improved = route[:], True
        while improved and len(best) > 2:
            improved = False
            base = self.km(best, wh_id)
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    prev = None if i == 0 else best[i - 1]
                    nxt = None if j == len(best) - 1 else best[j + 1]
                    old = edge(prev, best[i]) + edge(best[j], nxt)
                    new = edge(prev, best[j]) + edge(best[i], nxt)
                    if new >= old - 1e-9: continue
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    cand_km = self.km(cand, wh_id)
                    if cand_km < base - 1e-9: best, base, improved = cand, cand_km, True
        return best

    def savings_list(self, custs, wh_id):
        out = [(self.d_wc(wh_id, a) + self.d_wc(wh_id, b) - self.d_cc(a, b), a, b)
               for k, a in enumerate(custs) for b in custs[k + 1:]]
        out.sort(key=lambda t: (-t[0], str(t[1]), str(t[2])))
        return out

    def clarke_wright(self, custs, wh_id, demand):
        route_of = {c: [c] for c in custs}
        for s, a, b in self.savings_list(custs, wh_id):
            if s <= 0: break
            ra, rb = route_of[a], route_of[b]
            if ra is rb or a not in (ra[0], ra[-1]) or b not in (rb[0], rb[-1]): continue
            ra = ra if ra[-1] == a else ra[::-1]
            rb = rb if rb[0] == b else rb[::-1]
            merged = ra + rb
            if not self.feasible(merged, wh_id, demand): continue
            if self.route_cost(merged, wh_id, demand) > self.route_cost(ra, wh_id, demand) + self.route_cost(rb, wh_id, demand) + 1e-9: continue
            for c in merged: route_of[c] = merged
        return [self.two_opt(r, wh_id) for r in {id(r): r for r in route_of.values()}.values()]

    def _priority(self, route, wh_id, demand):
        w = sum(demand[c]["weight"] for c in route)
        v = sum(demand[c]["volume"] for c in route)
        return (-float(classify_outsourcing(w, v, self.cfg.outsourcing)["cost"]), -self.km(route, wh_id))

    def _demand(self, ords):
        demand, by_cust = {}, {}
        for o in ords:
            d = demand.setdefault(o["customer_id"], {"weight": 0.0, "volume": 0.0, "qty": 0, "service_min": 0.0})
            d["weight"] += o["total_weight_kg"]; d["volume"] += o["total_volume_m3"]
            d["qty"] += o.get("quantity", 1)
            d["service_min"] += self.cfg.service_min.get(o["order_type"], 60)
            by_cust.setdefault(o["customer_id"], []).append(o)
        return demand, by_cust

    def _worst_stop_by_units(self, r, wh_id, demand):
        mode = getattr(self.cfg, "constraint_mode", "Cube Out (Thể tích m³)")
        base = self.km(r, wh_id)
        if "Weight Out" in mode:
            return max(r, key=lambda c: (demand[c]["weight"] / max(demand[c]["qty"], 1), base - self.km([x for x in r if x != c], wh_id)))
        return max(r, key=lambda c: (demand[c]["volume"] / max(demand[c]["qty"], 1), base - self.km([x for x in r if x != c], wh_id)))

    def _start(self, date_str, start_str=None):
        return dt.datetime.combine(pd.to_datetime(date_str).date(), dt.datetime.strptime(start_str or self.cfg.start_time, "%H:%M").time())

    def _why_outsource(self, action, m, fleet, wh_id):
        load = f"{m['w']:,.0f}kg/{m['v']:.1f}m³ ({m['qty']} units)"
        if action == "NO_FLEET": return f"kho {wh_id} không có xe nhà (tải {load})"
        if action == "OUTSOURCE_FULL":
            return f"tải {load} vượt quá 150% xe lớn nhất nhà"
        return f"kho {wh_id} hết xe nhà phù hợp trong ngày (tải {load})"

    def _calculate_load_factor(self, m, vrow, cls):
        mode = getattr(self.cfg, "constraint_mode", "Cube Out (Thể tích m³)")
        if "Cube Out" in mode:
            cap_v = float(vrow["max_volume_m3"]) if vrow is not None else float(cls.get("cap_v", 40.0))
            return min(m["v"] / cap_v, 1.0) if cap_v > 0 else 0.0
        else:
            cap_w = float(vrow["max_weight_kg"]) if vrow is not None else float(cls.get("cap_w", 10000.0))
            return min(m["w"] / cap_w, 1.0) if cap_w > 0 else 0.0

    def _outsource_route(self, custs, wh_id, demand, by_cust, cls, events, date_str, km=None, reason=None):
        cfg = self.cfg
        if km is None:
            custs = self.two_opt(list(custs), wh_id); km = self.km(custs, wh_id)
        w = sum(demand[c]["weight"] for c in custs)
        v = sum(demand[c]["volume"] for c in custs)
        q = sum(demand[c]["qty"] for c in custs)
        hours = km / cfg.outsourced_speed_kmh + sum(demand[c]["service_min"] for c in custs) / 60.0
        start = self._start(date_str)
        if reason is None:
            reason = "; ".join(e.get("why", e["detail"]) for e in events) or "vượt khả năng xe nhà"
        reason = f"{reason} → {cls['why']}"
        
        load_factor = self._calculate_load_factor({"w": w, "v": v}, None, cls)
        
        return {
            "kind": "OUTSOURCE", "id_warehouse": wh_id, "route": list(custs), "orders": [o for c in custs for o in by_cust[c]],
            "vehicle_id": f"3PL-{cls['tier']}", "license_plate": f"Thuê ngoài 3PL ({cls['type']})", "vehicle_type": cls["vehicle"],
            "external": True, "driver_primary": "Đối tác 3PL", "driver_assistant": "—", "is_backup_driver": False,
            "km": round(km, 1), "speed_kmh": cfg.outsourced_speed_kmh,
            "load_factor": load_factor,
            "fixed_cost": 0.0, "variable_cost": 0.0, "overnight_cost": 0.0, "driver_cost": 0.0,
            "outsourcing_cost": float(cls["cost"]), "outsourcing_type": cls["type"],
            "cut_orders": [], "cut_weight_kg": 0.0, "cut_volume_m3": 0.0,
            "cap_w": float(cls["cap_w"]), "cap_v": float(cls.get("cap_v", 40.0)), "total_weight_kg": w, "total_volume_m3": v, "total_qty": q,
            "reason": reason, "trip_no": None,
            "start": start, "end": start + dt.timedelta(hours=hours), "hours": hours, "events": events,
        }

    def _dispatch(self, route, wh_id, demand, by_cust, veh_counts, date_str, veh_free=None):
        cfg, oc = self.cfg, self.cfg.outsourcing
        veh_free = {} if veh_free is None else veh_free
        rules = service_rules_from_cfg(cfg)
        fleet = self.veh[self.veh["id_warehouse"] == wh_id]
        for vid_, t_ in veh_free.items():
            if calculate_effective_max_hours(t_, cfg.max_route_hours) <= 0: veh_counts[vid_] = 0
        r, cut, final, blocked = list(route), [], None, {}
        while r:
            m = self.metrics(r, wh_id, demand)
            snap = dict(veh_counts)
            ta = evaluate_transportation_constraints(
                {"total_weight_kg": m["w"], "total_volume_m3": m["v"], "total_distance_km": m["km"]}, fleet, veh_counts, oc)
            action, vid, tb, bad = ta["action_type"], ta.get("assigned_vehicle"), None, None
            vrow = fleet[fleet["vehicle_id"] == vid].iloc[0] if (vid and ta["status"] != "OUTSOURCED") else None
            start_str = veh_free.get(vid, cfg.start_time) if vrow is not None else cfg.start_time
            max_h = calculate_effective_max_hours(start_str, cfg.max_route_hours)
            
            if action == "DISTANCE_EXCEEDED_CUT":
                bad = ("VƯỢT MAX_DISTANCE", ta["message"], f"tuyến {m['km']:.1f}km > Max_Distance {ta['max_distance_limit']:.0f}km của xe {vid}")
            elif vrow is not None:
                if m["km"] > float(vrow["Max_Distance"]):
                    bad = ("VƯỢT MAX_DISTANCE", f"Tuyến {m['km']:.1f}km > Max_Distance {float(vrow['Max_Distance']):.0f}km của xe {vid}.",
                           f"tuyến {m['km']:.1f}km > Max_Distance {float(vrow['Max_Distance']):.0f}km của xe {vid}")
                else:
                    tb = evaluate_time_constraints({"total_distance_km": m["km"], "orders": [o for c in r for o in by_cust[c]],
                                                   "current_date": date_str}, {"average_speed_kmh": float(vrow["average_speed_kmh"])},
                                                   rules, max_h)
                    if tb["status"] != "APPROVED" and start_str != cfg.start_time and vid not in blocked:
                        veh_counts.clear(); veh_counts.update(snap)
                        blocked[vid] = veh_counts.get(vid, 0); veh_counts[vid] = 0
                        continue
                    if tb["status"] != "APPROVED":
                        bad = ("VƯỢT GIỜ TUYẾN", tb["message"],
                               f"tuyến {tb['total_hours']:.2f}h > giới hạn {max_h:g}h (xe {vid} xuất phát lúc {start_str})")
            if bad:
                veh_counts.clear(); veh_counts.update(snap)
                c = self._worst_stop_by_units(r, wh_id, demand)
                cut.append((c, bad[0], bad[1], bad[2]))
                r = self.two_opt([x for x in r if x != c], wh_id)
                continue
            final = (m, ta, action, vid, vrow, tb, start_str)
            break
        for v_, n_ in blocked.items(): veh_counts[v_] = n_
        out = []
        if final:
            m, ta, action, vid, vrow, tb, start_str = final
            if ta["status"] == "OUTSOURCED":
                sev, kind, propose = _TAB_A_EVENTS[action]
                why = self._why_outsource(action, m, fleet, wh_id)
                ev = [{"sev": sev, "kind": kind, "detail": ta["message"], "why": why, "handled": f"Thuê ngoài {ta['cls']['type']}", "propose": propose}]
                out.append(self._outsource_route(r, wh_id, demand, by_cust, ta["cls"], ev, date_str, reason=why))
            else:
                split = action == "MAX_VEHICLE_PLUS_OUTSOURCE"
                primary, assistant, backup = "Chờ phân công", "—", False
                start, hours = self._start(date_str, start_str), tb["total_hours"]
                cap_w, cap_v = float(vrow["max_weight_kg"]), float(vrow["max_volume_m3"])
                ev, reason = [], ""
                if split:
                    sev, kind, propose = _TAB_A_EVENTS[action]
                    ex_w, ex_v = max(m["w"] - cap_w, 0.0), max(m["v"] - cap_v, 0.0)
                    reason = (f"tải {m['w']:,.0f}kg/{m['v']:.1f}m³ ({m['qty']} units) vượt xe lớn nhất nhà ({vid}: {cap_w:,.0f}kg/{cap_v:g}m³), "
                              f"phần dư {ex_w:,.0f}kg/{ex_v:.1f}m³ → {ta['cls']['why']}")
                    ev.append({"sev": sev, "kind": kind, "detail": ta["message"], "why": reason,
                               "handled": f"Cắt xe {vid} + thuê ngoài {ta['cls']['type']}", "propose": propose})
                orders = [o for c in r for o in by_cust[c]]
                end_dt = start + dt.timedelta(hours=hours)
                next_start_dt = end_dt + dt.timedelta(minutes=cfg.turnaround_min)
                veh_free[vid] = f"{next_start_dt:%H:%M}" if next_start_dt.date() == start.date() else "23:59"
                
                load_factor = self._calculate_load_factor(m, vrow, {})
                
                out.append({
                    "kind": "SPLIT" if split else "NORMAL", "id_warehouse": wh_id, "route": list(r), "orders": orders,
                    "vehicle_id": vid, "license_plate": vrow["license_plate"], "vehicle_type": vrow["vehicle_type"], "external": False,
                    "driver_primary": primary, "driver_assistant": assistant, "is_backup_driver": backup,
                    "km": round(m["km"], 1), "speed_kmh": float(vrow["average_speed_kmh"]),
                    "load_factor": load_factor,
                    "fixed_cost": float(vrow["fixed_cost"]), "variable_cost": float(vrow["variable_cost"]) * m["km"], "overnight_cost": 0.0,
                    "driver_cost": cfg.backup_driver_cost if backup else 0.0,
                    "outsourcing_cost": float(ta["cost"]) if split else 0.0, "outsourcing_type": ta["outsourcing_type"] if split else None,
                    "cut_orders": [o["order_id"] for o in orders] if split else [],
                    "cut_weight_kg": max(m["w"] - cap_w, 0.0) if split else 0.0,
                    "cut_volume_m3": max(m["v"] - cap_v, 0.0) if split else 0.0,
                    "cap_w": cap_w, "cap_v": cap_v, "total_weight_kg": m["w"], "total_volume_m3": m["v"], "total_qty": m["qty"],
                    "reason": reason if split else "",
                    "trip_no": max(1, cfg.vehicle_daily_count - veh_counts.get(vid, 0)),
                    "start": start, "end": end_dt, "hours": hours, "events": ev,
                })
        if cut:
            ids = [c for c, _, _, _ in cut]
            cls = classify_outsourcing(sum(demand[c]["weight"] for c in ids), sum(demand[c]["volume"] for c in ids), oc)
            
            cut_details = []
            for c, kind, detail, short in cut:
                orders_c = by_cust.get(c, [])
                order_ids_str = ", ".join(o["order_id"] for o in orders_c) if orders_c else c
                addr_str = orders_c[0].get("address", "") if orders_c else ""
                cut_details.append(f"đơn {order_ids_str} ({addr_str}) cắt do {short}")

            reason_cut_str = "; ".join(cut_details)
            ev = [{"sev": "HIGH", "kind": kind, "detail": f"Đơn khách {c}: {detail}", "why": f"đơn {c} bị cắt do {short}",
                   "handled": f"Cắt khỏi tuyến → thuê ngoài {cls['type']}", "propose": "Bổ sung tọa độ hoặc kiểm tra Max_Distance xe", "ids": [c]}
                  for c, kind, detail, short in cut]
            out.append(self._outsource_route(ids, wh_id, demand, by_cust, cls, ev, date_str, reason=reason_cut_str))
        return out

    def plan_day(self, date_str: str, day_orders: list) -> dict:
        cfg = self.cfg
        res = {"routes": [], "overdue_routes": [], "unrouted_orders": [], "exceptions": [], "day_cost": 0.0, "orders": list(day_orders)}
        def log(r):
            for e in r.pop("events"):
                for o in r["orders"]:
                    if e.get("ids") is None or o["customer_id"] in e["ids"]:
                        res["exceptions"].append({"NGÀY": date_str, "MỨC ĐỘ": e["sev"], "MÃ ĐƠN": o["order_id"], "PHÂN LOẠI": e["kind"],
                                                  "CHI TIẾT": e["detail"], "ĐÃ XỬ LÝ": e["handled"], "ĐỀ XUẤT": e["propose"]})
                                                  
        placed = [o for o in day_orders if o.get("lat") is not None and o.get("lng") is not None]
        unplaced = [o for o in day_orders if o.get("lat") is None or o.get("lng") is None]
        
        for o in placed:
            self.cust[o["customer_id"]] = {"lat": float(o["lat"]), "lng": float(o["lng"]), "name": o["customer_id"]}
        
        veh_counts = {vid: cfg.vehicle_daily_count for vid in self.veh["vehicle_id"]}
        veh_free = {}
            
        by_wh = {}
        for o in placed: by_wh.setdefault(self.nearest_wh(o["customer_id"]), []).append(o)
        for wh_id, ords in by_wh.items():
            demand, by_cust = self._demand(ords)
            routes = self.clarke_wright(list(demand), wh_id, demand)
            routes.sort(key=lambda rt: self._priority(rt, wh_id, demand))
            for rt in routes:
                for r in self._dispatch(rt, wh_id, demand, by_cust, veh_counts, date_str, veh_free):
                    log(r); res["routes"].append(r)
                    
        res["unrouted_orders"] = unplaced
        res["day_cost"] = sum(route_total(r) for r in res["routes"])
        return res

def simulate_all(data: dict, cfg: Config = CFG):
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]: by_date.setdefault(o["order_date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    all_r = [r for d in days.values() for r in d["routes"] + d["overdue_routes"]]
    
    valid_routes_for_load = [r for r in all_r if (not r["external"]) or ("FULL" in str(r.get("vehicle_id", "")).upper())]
    avg_load_factor = (sum(r["load_factor"] for r in valid_routes_for_load) / len(valid_routes_for_load)) if valid_routes_for_load else 0.0

    delivered = {o["order_id"] for r in all_r for o in r["orders"]}
    n_routes = len(all_r)
    operating = sum(d["day_cost"] for d in days.values())
    
    unrouted_count = sum(len(d["unrouted_orders"]) for d in days.values())
    
    kpis = {
        "violations": sum(1 for r in all_r if not r["external"] and r["hours"] > cfg.max_route_hours + 1e-9),
        "operating_cost": operating,
        "external_ratio": sum(r["external"] for r in all_r) / n_routes if n_routes else 0,
        "avg_load_factor": avg_load_factor,
        "undelivered_orders": len(data["orders"]) - len(delivered),
        "unrouted_orders": unrouted_count,
        "total_orders": len(data["orders"]),
    }
    return planner, days, kpis, operating

def route_total(r):
    return r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r.get("driver_cost", 0) + r.get("outsourcing_cost", 0)

# ============================================================================
# PIPELINE TIỆN ÍCH & RENDER UI TAB 5
# ============================================================================
def file_status(cfg: Config) -> pd.DataFrame:
    items = [
        ("Tab 1 · Hạm đội xe", cfg.vehicle_file, "Bắt buộc"),
        ("Tab 2 · Kho & Tọa độ", cfg.warehouse_file, "Khuyến nghị"),
        ("Tab 3 · Sản phẩm", cfg.product_file, "Tuỳ chọn"),
        ("Tab 4 · Đơn hàng (Đã geocode)", cfg.order_file, "Bắt buộc"),
    ]
    rows = [{"Nguồn": n, "File": os.path.relpath(p, BASE_DIR), "Trạng thái": "✅ Có" if os.path.exists(p) else "❌ Chưa có", "Yêu cầu": need}
            for n, p, need in items]
    return pd.DataFrame(rows)

def require_files(cfg: Config, which):
    labels = {
        "orders": (cfg.order_file, "Tab 4 (Đơn hàng)"),
        "fleet": (cfg.vehicle_file, "Tab 1 (Hạm đội xe)"),
    }
    missing = [f"`{os.path.relpath(labels[k][0], BASE_DIR)}` (chạy {labels[k][1]})" for k in which if not os.path.exists(labels[k][0])]
    if missing: raise UserError("Thiếu file đầu vào: " + "; ".join(missing))

def depot_options(cfg: Config) -> dict:
    opts = {"Mặc định Hà Nội (21.0285, 105.8542)": (21.0285, 105.8542)}
    for wid, (la, lo) in load_warehouse_coords(cfg).items():
        opts[f"Kho {wid} ({la:.4f}, {lo:.4f})"] = (la, lo)
    return opts

def tag_routes(day):
    return [(r, "Thường") for r in day["routes"]] + [(r, "Quá hạn (ưu tiên)") for r in day["overdue_routes"]]

def route_label(idx, r, wh_info) -> str:
    wh = wh_info.get(r["id_warehouse"], {"name": r["id_warehouse"]})
    return f"Tuyến {wh['name']} #{idx}"

def route_load(r):
    return sum(o["total_weight_kg"] for o in r["orders"]), sum(o["total_volume_m3"] for o in r["orders"])

def route_items_detail(r) -> str:
    item_map = {}
    for o in r["orders"]:
        it_name = o.get("items") or "Hàng tiêu chuẩn"
        q = o.get("quantity", 1)
        w = o.get("total_weight_kg", 0.0)
        v = o.get("total_volume_m3", 0.0)
        
        if it_name not in item_map:
            item_map[it_name] = {"qty": 0, "w": 0.0, "v": 0.0}
        item_map[it_name]["qty"] += q
        item_map[it_name]["w"] += w
        item_map[it_name]["v"] += v
        
    parts = []
    for it, d in item_map.items():
        parts.append(f"{d['qty']} x {it} ({d['w']:,.0f}kg | {d['v']:.2f}m³)")
    return " + ".join(parts)

def route_vehicle_text(r) -> str:
    reason_text = r.get("reason") or "Vượt khả năng xe nhà"
    if r["external"]:
        return f"🟣 Thuê ngoài 3PL — {r['vehicle_type']} (Lý do: {reason_text})"
    if r["kind"] == "SPLIT":
        return f"🚚 {r['vehicle_type']} + 3PL {r['outsourcing_type']} (Lý do: {reason_text})"
    return "🚚 " + str(r["vehicle_type"])

def routes_to_df(tagged_routes, wh_info, cfg: Config) -> pd.DataFrame:
    rows = []
    for idx, (r, kind) in enumerate(tagged_routes, 1):
        locs = sorted({re.split(r",|\s-\s", o["address"])[-1].strip() for o in r["orders"] if o["address"]})
        tw, tv = route_load(r)
        tq = sum(o.get("quantity", 1) for o in r["orders"])
        items_detail = route_items_detail(r)
        
        rows.append({
            "MÃ TUYẾN": route_label(idx, r, wh_info), 
            "KHU VỰC": ", ".join(locs), 
            "LOẠI": kind,
            "THỨ TỰ KHÁCH GHÉP": " → ".join(r["route"]),
            "ĐƠN GIAO": ", ".join(o["order_id"] for o in r["orders"]), 
            "SỐ ĐƠN": len(r["orders"]), 
            "TỔNG UNITS": tq,
            "CHI TIẾT MẶT HÀNG & ĐƠN VỊ": items_detail,
            "TỔNG TRỌNG TẢI (kg)": round(tw, 1), 
            "TỔNG THỂ TÍCH (m³)": round(tv, 2),
            "TẢI TRỌNG XE (kg)": round(float(r.get("cap_w", 0.0)), 1),
            "THỂ TÍCH XE (m³)": round(float(r.get("cap_v", 0.0)), 2),
            "TÀI XẾ CHÍNH": r.get("driver_primary", "Chờ phân công"),
            "PHỤ XE": r.get("driver_assistant", "—"), 
            "XE": route_vehicle_text(r),
            "BIỂN SỐ": r["license_plate"], 
            "CHUYẾN": f"{r['trip_no']}/{cfg.vehicle_daily_count}" if r.get("trip_no") else "—",
            "QUÃNG ĐƯỜNG (km)": r["km"], 
            "LẤP ĐẦY (%)": round(r["load_factor"] * 100, 1),
            "THỜI GIAN (giờ)": round(r["hours"], 2), 
            "TỔNG CHI PHÍ (đ)": round(route_total(r)) if route_total(r) > 0 else "Chờ nhập giá 3PL",
            "BẮT ĐẦU": f"{r['start']:%H:%M}", 
            "KẾT THÚC": f"{r['end']:%H:%M}",
            "TRẠNG THÁI": ("🟣 Cần xe 3PL Full" if (r["external"] and "FULL" in str(r.get("vehicle_id", "")).upper()) else 
                           ("🟣 Giao bởi 3PL Tiết kiệm" if r["external"] else ("🟠 Tách: xe nhà + 3PL" if r["kind"] == "SPLIT" else
                           ("✅ Đã tối ưu" if r["hours"] <= cfg.max_route_hours + 1e-9 else "⚠️ Vượt giới hạn giờ")))),
        })
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

def render_routing_settings():
    with st.expander("⚙️ Tham số mô hình & Ngưỡng thuê ngoài Tiết kiệm", expanded=True):
        c_mode, c1, c2 = st.columns(3)
        constraint_mode = c_mode.selectbox(
            "📦 Đặc thù hàng hóa (Lấp đầy theo)", 
            ["Cube Out (Thể tích m³)", "Weight Out (Trọng tải kg)"],
            index=0, key="cfg_constraint_mode"
        )
        start_time = c1.text_input("Giờ xuất phát (HH:MM)", "08:30", key="cfg_start")
        max_hours = c2.number_input("Giới hạn giờ / tuyến", 1.0, 24.0, 8.0, 0.5, key="cfg_maxh")
        
        c3, c4, c5 = st.columns(3)
        detour = c3.number_input("Hệ số đường vòng (Haversine)", 1.0, 3.0, 1.2, 0.05, key="cfg_detour")
        svc_b2b = c4.number_input("Bốc/dỡ B2B (phút)", 0, 600, 105, 5, key="cfg_b2b")
        svc_b2c = c5.number_input("Bốc/dỡ B2C (phút)", 0, 600, 60, 5, key="cfg_b2c")
        
        c6, c7, c8 = st.columns(3)
        backup = c6.number_input("Chi phí tài xế dự phòng (đ)", 0, 10_000_000, 400_000, 50_000, key="cfg_backup")
        trips = c7.number_input("Số chuyến tối đa / xe / ngày", 1, 5, 2, 1, key="cfg_trips")
        turn = c8.number_input("Thời gian quay đầu giữa 2 chuyến (phút)", 0, 240, 30, 5, key="cfg_turn")
        
        st.markdown("##### 🚛 Chi phí thuê ngoài Tiết kiệm")
        o2, o3 = st.columns(2)
        s1_p = o2.number_input("Tiết kiệm loại 1 (đ)", 0.0, 100_000_000.0, 800_000.0, 50_000.0, key="cfg_s1_p")
        s2_p = o3.number_input("Tiết kiệm loại 2 (đ)", 0.0, 100_000_000.0, 300_000.0, 50_000.0, key="cfg_s2_p")
        
    base_cfg = Config()
    opts = depot_options(base_cfg)
    if st.session_state.get("cfg_depot") not in opts:
        st.session_state["cfg_depot"] = list(opts)[0]
    depot_label = st.selectbox("Kho xuất phát cho Clarke-Wright", list(opts), key="cfg_depot")
    try: dt.datetime.strptime(start_time, "%H:%M")
    except ValueError: start_time = "08:30"
    
    cfg = Config(start_time=start_time, max_route_hours=float(max_hours), detour_factor=float(detour),
                 constraint_mode=constraint_mode,
                 service_min={"B2B": int(svc_b2b), "B2C": int(svc_b2c)}, backup_driver_cost=float(backup), vehicle_daily_count=int(trips), turnaround_min=float(turn),
                 saving_1_price=float(s1_p), saving_2_price=float(s2_p))
    return cfg, opts[depot_label]

def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright")
    cfg, depot = render_routing_settings()
    st.markdown("### 📂 Trạng thái dữ liệu đầu vào")
    st.dataframe(file_status(cfg), hide_index=True)
    
    if st.button("⚡ Chạy Routing Tạm Thời", type="primary", key="t6_run_temp"):
        require_files(cfg, ["orders", "fleet"])
        data = load_data(cfg)
        planner, days, kpis, total_cost = simulate_all(data, cfg)
        st.session_state["t6_temp_plan"] = {"planner": planner, "days": days, "kpis": kpis, "cfg": cfg, "data": data}
        st.session_state.pop("t6_final_plan", None)
        st.success("✅ Đã chạy xong bước tạm thời! Mời phân công tài xế & xe 3PL phía dưới.")

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
    
    routed_orders_count = sum(len(r["orders"]) for r, _ in tagged)
    g = st.columns(2)
    g[0].metric("Tổng số tuyến được phân", len(tagged))
    g[1].metric("Tổng số đơn đã phân tuyến", routed_orders_count)
    
    rdf_temp = routes_to_df(tagged, planner.wh, cfg)
    cols_to_show = [c for c in rdf_temp.columns if c not in ("TÀI XẾ CHÍNH", "PHỤ XE")]
    st.dataframe(rdf_temp[cols_to_show], hide_index=True)
    
    st.divider()
    st.markdown("## 👨‍✈️ 2. PHÂN CÔNG TÀI XẾ CÔNG TY CHO CÁC XE NHÀ HOẠT ĐỘNG TRONG NGÀY")
    
    inhouse_routes = [r for r, _ in tagged if not r["external"]]
    
    # Gom nhóm theo xe nhà để biết lịch chạy từng lần trong ngày
    veh_grouped = {}
    for r in inhouse_routes:
        vid = r["vehicle_id"]
        veh_grouped.setdefault(vid, []).append(r)
        
    if not veh_grouped:
        st.info("🎉 Ngày này toàn bộ đơn giao bằng 3PL, không có xe nhà nào cần phân công tài xế.")
    else:
        st.markdown("Vui lòng nhập tên tài xế hoặc chọn tùy chọn Thuê ngoài tài xế dự phòng bên dưới.")
        
        assignments = []
        has_driver_error = False
        
        for idx, (v_id, r_list) in enumerate(veh_grouped.items(), 1):
            r_list.sort(key=lambda x: x["start"])
            first_r = r_list[0]
            v_wh = first_r["id_warehouse"]
            v_cap_w = first_r["cap_w"]
            
            suggested_lic = "B2" if v_cap_w < 3500 else ("C" if v_cap_w <= 15000 else "FC")
            
            st.markdown(f"##### 🚚 Xe #{idx}: `{v_id}` (Kho: **{v_wh}** | Tải trọng: **{v_cap_w:,.0f} kg** | **{len(r_list)} chuyến trong ngày**)")
            
            # --- HIỂN THỊ LỊCH TRÌNH CHẠY TRONG NGÀY ---
            st.markdown("🕒 **Lịch trình chạy thực tế trong ngày:**")
            for trip_idx, r_item in enumerate(r_list, 1):
                start_s = f"{r_item['start']:%H:%M}"
                end_s = f"{r_item['end']:%H:%M}"
                ord_ids = ", ".join(o["order_id"] for o in r_item["orders"])
                st.caption(f"  • **Lần {trip_idx}:** Từ **{start_s}** đến **{end_s}** ({r_item['hours']:.2f} giờ) — Giao đơn: `{ord_ids}`")
            
            driver_choice = st.radio(
                f"Hình thức tài xế cho xe `{v_id}`",
                ["Nhập tên tài xế công ty", "Thành phần dự phòng (Thuê ngoài)"],
                horizontal=True,
                key=f"drv_choice_{v_id}_{idx}"
            )
            
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
                    req_rank = LICENSE_RANKS.get(req_lic, 6)
                    drv_rank = LICENSE_RANKS.get(cur_lic, 6)
                    if drv_rank < req_rank:
                        st.error(f"🔴 **Lỗi Bằng Lái:** Tài xế `{sel_driver}` (Bằng `{cur_lic}`) KHÔNG ĐỦ điều kiện lái xe `{v_id}` (Yêu cầu bằng `{req_lic}`).")
                        has_driver_error = True
                    else:
                        st.caption(f"✅ Bằng lái hợp lệ: **{cur_lic}** (Đạt yêu cầu ≥ {req_lic})")
            else:
                sel_driver = "Tài xế dự phòng (Thuê ngoài)"
                cur_lic = "C"
                cur_role = "Tài xế dự phòng"
                
                c2.text_input(f"Tên Tài xế", value=sel_driver, disabled=True, key=f"drv_name_dis_{v_id}_{idx}")
                c3.text_input(f"Bằng của tài xế", value=cur_lic, disabled=True, key=f"drv_lic_dis_{v_id}_{idx}")
                c4.text_input(f"Vai trò", value=cur_role, disabled=True, key=f"drv_role_dis_{v_id}_{idx}")
                st.info("ℹ️ Sẽ tính thêm chi phí Thuê tài xế dự phòng.")

            assignments.append({"vehicle_id": v_id, "driver_name": sel_driver, "license_type": cur_lic, "role": cur_role})
            st.divider()

    st.divider()
    st.markdown("## 🚛 3. NHẬP LIỆU ĐỘI XE THUÊ NGOÀI (3PL FULL & TÁCH CHUYẾN)")
    
    full_3pl_routes = [r for r, _ in tagged if (r["external"] and "FULL" in str(r.get("vehicle_id", "")).upper()) or r["kind"] == "SPLIT"]
    
    user_3pl_inputs = []
    if not full_3pl_routes:
        st.info("🎉 Ngày này không có chuyến nào cần thuê ngoài xe 3PL Full!")
    else:
        st.warning(f"⚠️ Phát hiện **{len(full_3pl_routes)} chuyến** cần thuê ngoài 3PL Full. Hãy kiểm tra danh sách mặt hàng & đơn vị bên dưới để chọn/ghép xe 3PL phù hợp.")
        
        for idx, r in enumerate(full_3pl_routes, 1):
            req_w, req_v = route_load(r)
            req_qty = sum(o.get("quantity", 1) for o in r["orders"])
            items_desc = route_items_detail(r)
            is_split = r["kind"] == "SPLIT"
            tag_type = "Tách chuyến xe nhà + 3PL" if is_split else "Thuần Thuê ngoài 3PL Full"
            
            st.markdown(f"##### 📌 Chuyến 3PL Full #{idx} (`{tag_type}`)")
            st.markdown(f"📦 **Tổng yêu cầu:** **{req_qty} đơn vị hàng (Units)** | **{req_w:,.1f} kg** | **{req_v:.2f} m³**")
            st.markdown(f"🛍️ **Danh sách mặt hàng cần tải:** `{items_desc}`")
            st.caption(f"↳ Gồm {len(r['orders'])} đơn: {', '.join(o['order_id'] for o in r['orders'])}")
            
            num_rows_key = f"3pl_num_rows_{date_str}_{idx}"
            if num_rows_key not in st.session_state:
                st.session_state[num_rows_key] = 1
                
            c_add, c_rem = st.columns([2, 8])
            if c_add.button(f"➕ Thêm 1 xe 3PL cho chuyến #{idx}", key=f"add_row_btn_{idx}"):
                st.session_state[num_rows_key] += 1
                st.rerun()
            if st.session_state[num_rows_key] > 1:
                if c_rem.button(f"➖ Bớt 1 xe 3PL", key=f"rem_row_btn_{idx}"):
                    st.session_state[num_rows_key] -= 1
                    st.rerun()

            route_trucks = []
            total_cap_w, total_cap_v, total_cost = 0.0, 0.0, 0.0
            
            for row_i in range(1, st.session_state[num_rows_key] + 1):
                st.caption(f"👉 Hàng xe 3PL #{row_i}:")
                r1, r2, r3, r4 = st.columns([2, 1.5, 1.5, 2])
                
                v_name = r1.text_input(f"Tên xe #{row_i}", f"Xe tải 3PL #{row_i}", key=f"3pl_name_{idx}_{row_i}")
                v_w = r2.number_input(f"Tải trọng (kg) #{row_i}", value=float(math.ceil(req_w / st.session_state[num_rows_key])), step=100.0, key=f"3pl_w_{idx}_{row_i}")
                v_v = r3.number_input(f"Thể tích (m³) #{row_i}", value=float(round(req_v / st.session_state[num_rows_key] + 0.2, 1)), step=0.5, key=f"3pl_v_{idx}_{row_i}")
                v_cost = r4.number_input(f"Giá thuê (đ) #{row_i}", value=2_000_000.0, step=100_000.0, key=f"3pl_cost_{idx}_{row_i}")
                
                total_cap_w += v_w
                total_cap_v += v_v
                total_cost += v_cost
                route_trucks.append(f"{v_name} ({v_w:,.0f}kg/{v_v:.1f}m³)")

            user_3pl_inputs.append({
                "req_w": req_w, "req_v": req_v, "req_qty": req_qty,
                "name": f"{st.session_state[num_rows_key]} xe 3PL [" + " + ".join(route_trucks) + "]",
                "cap_w": total_cap_w, "cap_v": total_cap_v,
                "cost": total_cost, "route": r
            })
            st.divider()

    if st.button("🚀 Kiểm Tra Điều Kiện & Tính Chi Phí Cuối Cùng", type="primary", key="btn_run_final_all"):
        if 'has_driver_error' in locals() and has_driver_error:
            st.error("❌ Vui lòng sửa lại các lỗi phân công tài xế bị báo đỏ phía trên trước khi tính chi phí cuối cùng!")
            return
            
        valid_3pl = True
        for idx, item in enumerate(user_3pl_inputs, 1):
            if item["cap_w"] < item["req_w"] - 1e-5 or item["cap_v"] < item["req_v"] - 1e-5:
                st.error(f"❌ Các xe 3PL ghép cho chuyến #{idx} không đủ tải trọng/thể tích! "
                         f"Cần tối thiểu {item['req_qty']} Đơn vị hàng ({item['req_w']:,.1f}kg / {item['req_v']:.2f}m³), nhưng tổng sức tải xe bạn nhập chỉ đạt {item['cap_w']:,.1f}kg / {item['cap_v']:.2f}m³.")
                valid_3pl = False
        
        if not valid_3pl: return
        
        assign_map = {a["vehicle_id"]: a for a in 'assignments' in locals() and assignments or []}
        for r, _ in tagged:
            if not r["external"]:
                vid = r["vehicle_id"]
                if vid in assign_map:
                    a = assign_map[vid]
                    if a["driver_name"] != "Tài xế dự phòng (Thuê ngoài)":
                        r["driver_primary"] = a["driver_name"]
                        r["is_backup_driver"] = False
                        r["driver_cost"] = 0.0
                    else:
                        r["driver_primary"] = "Tài xế Dự Phòng (Thuê ngoài)"
                        r["is_backup_driver"] = True
                        r["driver_cost"] = cfg.backup_driver_cost

        for item in user_3pl_inputs:
            r = item["route"]
            r["vehicle_type"] = item["name"]
            r["license_plate"] = "3PL Ghép Chuyến"
            r["cap_w"] = item["cap_w"]
            r["cap_v"] = item["cap_v"]
            r["outsourcing_cost"] = item["cost"]
            r["load_factor"] = item["req_v"] / item["cap_v"] if "Cube Out" in cfg.constraint_mode else item["req_w"] / item["cap_w"]
            
        day["day_cost"] = sum(route_total(rt) for rt in day["routes"])
        
        all_day_routes = day["routes"] + day["overdue_routes"]
        valid_load_routes = [rt for rt in all_day_routes if (not rt["external"]) or ("FULL" in str(rt.get("vehicle_id", "")).upper())]
        avg_load_day = (sum(rt["load_factor"] for rt in valid_load_routes) / len(valid_load_routes)) if valid_load_routes else 0.0
        
        kpis["operating_cost"] = sum(d["day_cost"] for d in days.values())
        kpis["avg_load_factor"] = avg_load_day
        
        st.session_state["t6_final_plan"] = plan
        st.success("🎉 Đã phân công tài xế & cập nhật chi phí tổng thành công!")

    final_plan = st.session_state.get("t6_final_plan")
    if final_plan:
        st.divider()
        st.markdown(f"## 🏆 BÁO CÁO KẾT QUẢ ĐIỀU PHỐI CHÍNH THỨC — {date_str}")
        
        final_day = final_plan["days"][date_str]
        final_tagged = tag_routes(final_day)
        
        valid_load_routes = [r for r, _ in final_tagged if (not r["external"]) or ("FULL" in str(r.get("vehicle_id", "")).upper())]
        avg_load = (sum(r["load_factor"] for r in valid_load_routes) / len(valid_load_routes) * 100) if valid_load_routes else 0.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 TỔNG CHI PHÍ PHẢI CHI TRONG NGÀY", money(final_day["day_cost"]) + " VNĐ")
        c2.metric("📦 TỔNG ĐƠN ĐÃ PHÂN TUYẾN", len(final_day["orders"]) - len(final_day.get("unrouted_orders", [])))
        c3.metric("📊 TỈ LỆ LẤP ĐẦY TB (XE NHÀ & 3PL FULL)", f"{avg_load:.1f}%")
        
        final_rdf = routes_to_df(final_tagged, planner.wh, cfg)
        st.dataframe(final_rdf, hide_index=True)

# ============================================================================
# 🚨 TAB 6: ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ (INCIDENT CONTROL CENTER)
# ============================================================================
def handle_road_blocked(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🗺️ [RE-ROUTING] Đường chặn hoàn toàn. OSRM API đã vẽ lộ trình mới né điểm sạt lở cho đơn `{order_id}` (+2.4km)."

def handle_heavy_traffic(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏱️ [TRAFFIC DELAY] Tắc nghẽn >30 phút. Đã tính lại ETA mới và tự động gửi SMS thông báo cho đơn `{order_id}`."

def handle_truck_ban(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚫 [CORRIDOR SWITCH] Phát hiện đường cấm tải. Đã chuyển xe `{vehicle_id}` sang hành lang giao thông phụ cận hợp lệ."

def handle_flooded_road():
    return f"[{dt.datetime.now():%H:%M:%S}] 🌧️ [FLOOD BLACKLIST] Đường ngập sâu. Đã đưa tọa độ vào blacklist tạm thời, điều hướng xe leo đường cao."

def handle_short_blockage():
    return f"[{dt.datetime.now():%H:%M:%S}] ⏸️ [HOLD & WAIT] Chắn tàu/Lễ hội (<10 phút). Kích hoạt chế độ tạm dừng trên app tài xế để tiết kiệm xăng."

def handle_driver_lost(driver_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🧭 [GPS RE-NAVIGATE] Tài xế `{driver_id}` đi nhầm rẽ. Đã phát âm thanh cảnh báo và làm mới bản đồ quay đầu."

def handle_gps_lost():
    return f"[{dt.datetime.now():%H:%M:%S}] 📡 [DEAD RECKONING] Mất sóng GPS hầm. Khóa vị trí Last Known Position và ước tính theo vận tốc quán tính."

def handle_customer_change_address(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🏠 [CLUSTER RE-SEQUENCE] Khách đổi địa chỉ phút chót. Tự động kiểm tra cụm giao và xếp lại thứ tự ghé thăm."

def handle_cust_no_answer(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 📞 [UNREACHABLE CUST] Khách không nghe máy. Ghi nhận lịch sử gọi (Lần 1), cho xe đi đơn tiếp và xếp lịch gọi lại."

def handle_cust_absent(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚪 [CUST ABSENT] Khách vắng mặt. Đánh dấu giao thất bại, tự động phát câu hỏi hẹn lịch giao lại ca sau."

def handle_cust_reject(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ❌ [CUST REJECT] Khách từ chối nhận. Ghi nhận lý do, tạo lệnh thu hồi hàng về kho và báo Điều phối viên."

def handle_cust_request_late(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🕰️ [TIME WINDOW LATE] Khách hẹn giao muộn hơn. Cập nhật Time Window mới, đẩy đơn `{order_id}` xuống cuối tuyến."

def handle_cust_request_early(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚡ [TIME WINDOW EARLY] Khách đòi giao sớm. Đã kiểm tra ETA, đẩy đơn `{order_id}` lên ưu tiên nếu không gãy tuyến khác."

def handle_change_recipient(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 👤 [RECIPIENT SWITCH] Đổi người nhận hàng. Yêu cầu tài xế chụp ảnh CCCD/Xác nhận OTP người nhận mới."

def handle_partial_delivery(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 📦 [PARTIAL DELIVERY] Khách nhận 1 phần hàng. Cập nhật số lượng đã giao, tạo phiếu lưu kho cho phần còn lại."

def handle_address_not_found(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔍 [ADDRESS UNVERIFIED] Không tìm thấy địa chỉ. Đánh dấu điểm nghẽn, gửi link Zalo cho khách ghim vị trí."

def handle_truck_breakdown(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚨 [BREAKDOWN RESCUE] Xe `{vehicle_id}` hỏng giữa đường. Điều xe dự phòng ra cứu hộ & chuyển tải đơn hàng."

def handle_out_of_fuel(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⛽ [FUEL EMERGENCY] Xe `{vehicle_id}` hết nhiên liệu. Định vị trạm cây xăng gần nhất và cập nhật cộng 15phút ETA."

def handle_traffic_accident(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚑 [ACCIDENT EMERGENCY] Xe gặp tai nạn. Ưu tiên an toàn tài xế, kích hoạt bảo hiểm và điều cứu hộ tái phân công đơn."

def handle_vehicle_overload(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚖️ [OVERLOAD DETECTED] Hàng thực tế bị quá tải/quá cỡ. Tiến hành tách đơn `{order_id}` hoặc đổi sang xe 3PL lớn hơn."

def handle_cargo_damaged(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 💔 [CARGO DAMAGED] Hàng bị móp méo/hư hỏng. Tài xế chụp ảnh hiện trạng, cách ly hàng và báo kho đền bù."

def handle_cargo_missing(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ❓ [CARGO MISSING] Thất thoát hàng so với phiếu kho. Đối chiếu dữ liệu quét Barcode cửa kho, lập biên bản đền bù."

def handle_wrong_loading(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔄 [RE-ARRANGE CARGO] Lỗi LIFO xếp sai vị trí hàng. Sắp xếp lại thùng hàng tại điểm dừng để dỡ đúng thứ tự."

def handle_vehicle_disconnect(vehicle_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔌 [DISCONNECT ALERT] Xe mất kết nối IoT. Tạm ngừng điều phối tự động, gọi điện thoại xác thực trực tiếp tài xế."

def handle_route6_full(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 📦 [ROUTE FULL] Tuyến hết chỗ chứa. Kiểm tra đẩy đơn `{order_id}` sang Xe 3PL Thuê ngoài hoặc dời ca sau."

def handle_no_backup_vehicle():
    return f"[{dt.datetime.now():%H:%M:%S}] 🚚 [NO BACKUP FLEET] Hết sạch xe nhà dự phòng. Quét các xe đang chạy còn trống không gian để ghép đơn (Co-loading)."

def handle_route7_overload():
    return f"[{dt.datetime.now():%H:%M:%S}] ⚠️ [CASCADE OVERLOAD] Chuyển đơn làm tuyến kế tiếp gãy tải. Hủy phương án chuyển tuyến, tính giải pháp thuê ngoài."

def handle_cost_too_high():
    return f"[{dt.datetime.now():%H:%M:%S}] 💰 [COST EXCEEDED] Chi phí xử lý quá tốn kém. So sánh chi phí đền bù giao trễ vs thuê ngoài để chọn option rẻ nhất."

def handle_driver_timeout(driver_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏳ [MAX HOURS REACHED] Tài xế `{driver_id}` sắp hết 8h làm việc. Điều xe khác đến đổi ca/bàn giao phần đơn còn lại."

def handle_area_mismatch():
    return f"[{dt.datetime.now():%H:%M:%S}] 🧭 [AREA MISMATCH] Xe trống nhưng lệch vùng. Khóa không ghép đơn nếu quãng đường rỗng (Deadhead) > 10km."

def handle_no_feasible_eta():
    return f"[{dt.datetime.now():%H:%M:%S}] 🚨 [ETA IMPOSSIBLE] Không phương án nào kịp giờ. Cảnh báo tổng đài viên gọi điện đàm phán lại khung giờ với khách."

def handle_redelivery_cost_limit():
    return f"[{dt.datetime.now():%H:%M:%S}] 🛑 [REDELIVERY COST CEILING] Chi phí giao lại vượt giá trị đơn. Chuyển đơn về điểm bưu cục gần nhất cho khách tự đến lấy."

def handle_map_data_conflict():
    return f"[{dt.datetime.now():%H:%M:%S}] 🧐 [DATA CONFLICT] Tài xế báo tắc đường nhưng GIS báo thông thoáng. Yêu cầu gửi ảnh chụp thực địa xác minh."

def handle_duplicate_assignment(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔒 [RACE CONDITION] 2 tài xế cùng nhận đơn `{order_id}`. Khóa phiên bản điều phối, chỉ giữ 1 lệnh hợp lệ đầu tiên."

def handle_driver_unconfirmed():
    return f"[{dt.datetime.now():%H:%M:%S}] 🔔 [UNCONFIRMED COMMAND] Tài xế không bấm xác nhận lệnh. Phát chuông cảnh báo 3 phút, sau đó chuyển Điều phối thủ công."

def handle_driver_ignore_new_route():
    return f"[{dt.datetime.now():%H:%M:%S}] ⚠️ [ROUTE DEVIATION] Tài xế đã nhận lệnh nhưng vẫn chạy đường cũ. So sánh GPS với tuyến mới và phát chuông nhắc nhở."

def handle_order_canceled_midway(order_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚫 [CANCEL MIDWAY] Đơn `{order_id}` bị hủy khi đang re-route. Lập tức hủy tác vụ tính toán, báo tài xế quay đầu."

def handle_infinite_loop_incident():
    return f"[{dt.datetime.now():%H:%M:%S}] 🔄 [LOOP DETECTED] Sự cố lặp lại >3 lần. Giới hạn số lần re-route, yêu cầu Trưởng ca điều hành can thiệp thủ công."

def handle_invalid_order_data():
    return f"[{dt.datetime.now():%H:%M:%S}] ❌ [INVALID DATA] Dữ liệu đơn hàng thiếu m3/kg. Tạm loại đơn khỏi thuật toán tối ưu và gửi ticket sửa dữ liệu."

def handle_solver_timeout():
    return f"[{dt.datetime.now():%H:%M:%S}] ⏱️ [ENGINE TIMEOUT] Thuật toán Clarke-Wright quá tải/Timeout. Giữ kế hoạch hợp lệ gần nhất và báo lỗi Engine."

INCIDENT_SOLVERS = {
    "road_blocked": ("1. Đường bị chặn hoàn toàn", handle_road_blocked),
    "heavy_traffic": ("2. Tắc nghẽn giao thông nghiêm trọng", handle_heavy_traffic),
    "truck_ban": ("3. Đường cấm xe tải", handle_truck_ban),
    "flooded_road": ("4. Đường ngập nước", lambda o, v, d: handle_flooded_road()),
    "short_blockage": ("5. Đường chặn ngắn hạn tạm thời", lambda o, v, d: handle_short_blockage()),
    "driver_lost": ("6. Tài xế đi nhầm đường", lambda o, v, d: handle_driver_lost(d)),
    "gps_lost": ("7. GPS mất tín hiệu", lambda o, v, d: handle_gps_lost()),
    "change_address": ("8. Khách đổi địa chỉ phút chót", handle_customer_change_address),
    "cust_no_answer": ("9. Khách không nghe điện thoại", handle_cust_no_answer),
    "cust_absent": ("10. Khách không có mặt", handle_cust_absent),
    "cust_reject": ("11. Khách từ chối nhận hàng", handle_cust_reject),
    "cust_request_late": ("12. Khách hẹn giao muộn hơn", handle_cust_request_late),
    "cust_request_early": ("13. Khách yêu cầu giao sớm hơn", handle_cust_request_early),
    "change_recipient": ("14. Thay đổi người nhận hàng", handle_change_recipient),
    "partial_delivery": ("15. Yêu cầu giao một phần", handle_partial_delivery),
    "address_not_found": ("16. Không tìm thấy địa chỉ", handle_address_not_found),
    "truck_breakdown": ("17. Xe hỏng giữa đường", lambda o, v, d: handle_truck_breakdown(v)),
    "out_of_fuel": ("18. Xe hết nhiên liệu", lambda o, v, d: handle_out_of_fuel(v)),
    "traffic_accident": ("19. Xe gặp tai nạn", lambda o, v, d: handle_traffic_accident(v)),
    "vehicle_overload": ("20. Xe quá tải do sai dữ liệu", handle_vehicle_overload),
    "cargo_damaged": ("21. Hàng hóa bị hư hỏng", handle_cargo_damaged),
    "cargo_missing": ("22. Hàng hóa bị thiếu", handle_cargo_missing),
    "wrong_loading": ("23. Hàng xếp sai vị trí trên xe", lambda o, v, d: handle_wrong_loading(v)),
    "vehicle_disconnect": ("24. Xe mất kết nối IoT", lambda o, v, d: handle_vehicle_disconnect(v)),
    "route6_full": ("25. Xe Tuyến hết chỗ chứa", handle_route6_full),
    "no_backup_vehicle": ("26. Không còn xe dự phòng", lambda o, v, d: handle_no_backup_vehicle()),
    "route7_overload": ("27. Chuyển đơn làm tuyến kế quá tải", lambda o, v, d: handle_route7_overload()),
    "cost_too_high": ("28. Phương án xử lý quá tốn kém", lambda o, v, d: handle_cost_too_high()),
    "driver_timeout": ("29. Tài xế sắp hết giờ làm việc", lambda o, v, d: handle_driver_timeout(d)),
    "area_mismatch": ("30. Xe còn chỗ nhưng lệch khu vực", lambda o, v, d: handle_area_mismatch()),
    "no_feasible_eta": ("31. Không phương án nào kịp giờ giao", lambda o, v, d: handle_no_feasible_eta()),
    "redelivery_cost_limit": ("32. Chi phí giao lại vượt ngưỡng", lambda o, v, d: handle_redelivery_cost_limit()),
    "map_data_conflict": ("33. Xung đột báo cáo bản đồ", lambda o, v, d: handle_map_data_conflict()),
    "duplicate_assignment": ("34. Trùng lặp 2 tài xế nhận đơn", handle_duplicate_assignment),
    "driver_unconfirmed": ("35. Tài xế không xác nhận lệnh mới", lambda o, v, d: handle_driver_unconfirmed()),
    "driver_ignore_new_route": ("36. Tài xế chạy sai tuyến mới", lambda o, v, d: handle_driver_ignore_new_route()),
    "order_canceled_midway": ("37. Đơn bị hủy khi đang re-route", handle_order_canceled_midway),
    "infinite_loop_incident": ("38. Sự cố lặp lại nhiều lần", lambda o, v, d: handle_infinite_loop_incident()),
    "invalid_order_data": ("39. Dữ liệu đơn hàng không hợp lệ", lambda o, v, d: handle_invalid_order_data()),
    "solver_timeout": ("40. Engine tối ưu bị Timeout/Lỗi", lambda o, v, d: handle_solver_timeout()),
}

def render_incident_tab():
    st.header("🚨 6. TRUNG TÂM ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ")
    st.caption("🌐 *Liên kết trực tiếp với dữ liệu Đơn hàng (Tab 4) & Hạm đội xe (Tab 1)*")
    
    orders_res = st.session_state.get("orders_result")
    fleet_res = st.session_state.get("fleet_result")
    
    order_options = orders_res["df"]["order_id"].tolist() if orders_res else ["ORD_003", "ORD_001", "ORD_002"]
    vehicle_options = fleet_res["df"]["vehicle_id"].tolist() if fleet_res else ["VEH_06", "VEH_01", "VEH_02"]
    driver_options = ["DRV_06", "DRV_01", "DRV_02", "Tài xế dự phòng"]

    if "live_logs" not in st.session_state:
        st.session_state["live_logs"] = [
            {"Thời gian": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "Mã Xe": "VEH_06", "Tài Xế": "DRV_06", 
             "Đang Giao Đơn": "ORD_003", "Vị Trí Hiện Tại": "Kho Tổng Depot Hà Nội", 
             "Trạng Thái": "Đang xuất kho", "Ghi Chú": "Chuẩn bị giao đơn #03 trên Tuyến 6"}
        ]
        
    col_solver, col_form = st.tabs(["🛠️ 1. Kích Hoạt Python Incident Solvers", "📡 2. Báo Cáo Thực Địa & Live Tracking Logs"])
    
    with col_solver:
        st.markdown("### ⚡ Kích Hoạt Tự Động Giải Quyết Sự Cố Thực Địa")
        c1, c2 = st.columns([1.2, 2])
        
        with c1:
            st.markdown("##### 📌 Tùy chọn đối tượng liên quan:")
            sel_order = st.selectbox("Chọn Mã Đơn Hàng (Order ID)", order_options, key="t7_sel_ord")
            sel_vehicle = st.selectbox("Chọn Mã Xe (Vehicle ID)", vehicle_options, key="t7_sel_veh")
            sel_driver = st.selectbox("Chọn Mã Tài Xế (Driver ID)", driver_options, key="t7_sel_drv")
            
            st.markdown("##### 🚨 Chọn Sự Cố Thực Tế Bị Báo:")
            solver_labels = {k: v[0] for k, v in INCIDENT_SOLVERS.items()}
            sel_incident_key = st.selectbox(
                "Danh mục 39 sự cố ngoại lệ", 
                list(solver_labels.keys()), 
                format_func=lambda x: solver_labels[x],
                key="t7_sel_inc"
            )
            
            btn_run_solver = st.button("🚀 Run Tự Động", type="primary", key="btn_run_solver")
            
        with c2:
            st.markdown("##### 🖥️ Màn Hình Log Tự Động Thực Thi (Execution Console)")
            console_placeholder = st.empty()
            
            if btn_run_solver:
                solver_tuple = INCIDENT_SOLVERS[sel_incident_key]
                func = solver_tuple[1]
                
                try:
                    res_msg = func(sel_order, sel_vehicle, sel_driver)
                except TypeError:
                    try:
                        res_msg = func(sel_order)
                    except TypeError:
                        res_msg = func()
                        
                console_placeholder.code(res_msg, language="markdown")
                
                new_log = {
                    "Thời gian": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                    "Mã Xe": sel_vehicle,
                    "Tài Xế": sel_driver,
                    "Đang Giao Đơn": sel_order,
                    "Vị Trí Hiện Tại": "Toạ độ phát sinh sự cố",
                    "Trạng Thái": "Gặp sự cố / Đã re-route",
                    "Ghi Chú": f"Auto Solver [{sel_incident_key}]: {res_msg.split('] ')[-1]}"
                }
                st.session_state["live_logs"].insert(0, new_log)
                st.toast("✅ Đã xử lý sự cố & cập nhật vào Live Logs!", icon="🎉")
            else:
                console_placeholder.info("👈 Chọn loại sự cố bên trái và nhấn nút 'Run Python Solver' để xem kết quả xử lý tự động.")

    with col_form:
        st.markdown("### ✍️ Biểu Mẫu Nhập Báo Cáo Thực Địa Từ Tài Xế / Nhân Viên")
        
        f1, f2 = st.columns([1, 1.5])
        with f1:
            in_veh = st.selectbox("Mã Xe (Vehicle ID)", vehicle_options, key="t7_f_veh")
            in_drv = st.selectbox("Mã Tài Xế (Driver ID)", driver_options, key="t7_f_drv")
            in_ord = st.selectbox("Mã Đơn Hàng (Order ID)", order_options, key="t7_f_ord")
            in_loc = st.text_input("Vị Trí Hiện Tại", value="Đường Cổ Linh, Hà Nội", key="t7_f_loc")
            in_status = st.selectbox("Trạng Thái Tuyến", ["Đang xuất kho Depot", "Đang trên đường giao hàng", "Đã giao thành công", "Gặp sự cố / Tạm dừng"], key="t7_f_st")
            in_note = st.text_area("Ghi Chú Thực Địa", placeholder="Ví dụ: Khách hẹn 10 phút nữa xuống nhận...", key="t7_f_note")
            
            btn_submit_report = st.button("📤 Gửi Báo Cáo Về Hệ Thống", key="btn_sub_rep")
            
            if btn_submit_report:
                rep_log = {
                    "Thời gian": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                    "Mã Xe": in_veh,
                    "Tài Xế": in_drv,
                    "Đang Giao Đơn": in_ord,
                    "Vị Trí Hiện Tại": in_loc,
                    "Trạng Thái": in_status,
                    "Ghi Chú": in_note or "Không có ghi chú"
                }
                st.session_state["live_logs"].insert(0, rep_log)
                st.success("✅ Đã cập nhật nhật ký thực địa thành công!")
                st.rerun()
                
        with f2:
            st.markdown("##### 📊 Bảng Giám Sát Thời Gian Thực (Live Logs)")
            logs_df = pd.DataFrame(st.session_state["live_logs"])
            st.dataframe(logs_df, hide_index=True, use_container_width=True)

# ============================================================================
# ⚙️ MAIN CONTAINERS
# ============================================================================
TAB_NAMES = [
    "🚚 1. Fleet",
    "🏭 2. Warehouse",
    "📦 3. Product",
    "🧾 4. Orders",
    "🗺️ 5. Routing",
    "🚨 6. Xử Lý Sự Cố"
]

def main():
    st.title("🚚 Smart Logistics — Hệ thống chuẩn hóa & Điều phối Vận tải (của Phạm Minh Thái)")
    with st.sidebar:
        st.markdown("### ℹ️ Menu Điều hướng")
        if st.button("🧹 Giải phóng RAM & Cache", key="free_ram"):
            st.cache_data.clear()
            for k in list(st.session_state):
                if k.endswith(("_raw", "_result", "_meta", "_why", "_auto_table", "_plan", "_temp_plan", "_final_plan")):
                    st.session_state.pop(k, None)
            st.rerun()
    tabs = st.tabs(TAB_NAMES)
    with tabs[0]: render_fleet_tab()
    with tabs[1]: render_warehouse_tab()
    with tabs[2]: render_product_tab()
    with tabs[3]: render_orders_tab()
    with tabs[4]: render_routing_tab()
    with tabs[5]: render_incident_tab()

if __name__ == "__main__":
    main()
