# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
import os
import re
import math
import time
import datetime as dt
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
OUT_SOLVING = os.path.join(BASE_DIR, "output_solving")
OUT_ROUTING = os.path.join(BASE_DIR, "output_routing")

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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

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
    "length_cm": ["dài", "chiều dài", "length", "length_cm", "dài (cm)"],
    "width_cm": ["rộng", "chiều rộng", "width", "width_cm", "rộng (cm)"],
    "height_cm": ["cao", "chiều cao", "height", "height_cm", "cao (cm)"],
    "volume": ["thể tích", "volume", "m3", "cbm"],
    "weight": ["trọng lượng", "weight", "kg"],
    "cost_price": ["giá sản xuất", "cost price", "cost"],
    "selling_price": ["giá bán", "selling price", "price"],
}
PRODUCT_NUMERIC = ["length_cm", "width_cm", "height_cm", "volume", "weight", "cost_price", "selling_price"]

def process_product(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip()) for f in ["product_id", "product_name"]}
    for f in PRODUCT_NUMERIC: out[f] = parse_num(rec.get(f))
    if out["volume"] == 0.0 and out["length_cm"] > 0 and out["width_cm"] > 0 and out["height_cm"] > 0:
        out["volume"] = round((out["length_cm"] * out["width_cm"] * out["height_cm"]) / 1_000_000.0, 6)
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
    seed_df = pd.DataFrame({
        "product_id": ["SP_01", "SP_02"], 
        "product_name": ["Ghế Sofa", "Tủ Quần Áo"], 
        "length_cm": [120.0, 150.0],
        "width_cm": [80.0, 60.0],
        "height_cm": [90.0, 200.0],
        "volume": [0.864, 1.8],
        "weight": [25.0, 50.0], 
        "selling_price": [2500000, 4500000]
    })
    mode, edited, uploaded = render_input_block("product", "sản phẩm", seed_df)
    if st.button("🔍 Quét & Mapping", key="product_scan", type="primary"):
        run_scan("product", "sản phẩm", mode, uploaded, edited, lambda df: basic_semantic_mapping(df, PRODUCT_FIELDS, lambda s, f: 0.8, 0.35))
    raw = st.session_state.get("product_raw")
    if raw is None: return
    chosen = render_mapping("product", list(PRODUCT_FIELDS), raw)
    if st.button("🚀 Chuẩn hóa Sản phẩm & Xuất File DIM_PRODUCT", key="product_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows = [process_product({f: pick(r, colmap, f, None) for f in PRODUCT_FIELDS}) for _, r in raw.iterrows()]
        out = product_validate_output(pd.DataFrame(rows))
        ordered_cols = ["product_id", "product_name", "length_cm", "width_cm", "height_cm", "volume", "weight", "cost_price", "selling_price", "trạng_thái", "kiểm_tra"]
        final_cols = [c for c in ordered_cols if c in out.columns] + [c for c in out.columns if c not in ordered_cols]
        out = out[final_cols]
        files = save_outputs(OUT_PRODUCT, "DIM_PRODUCT", "DIM_PRODUCT", out)
        store_result("product", out, files, "🧭 Hoàn tất chuẩn hóa sản phẩm (Đã lưu Length, Width, Height)", [("Tổng sản phẩm", len(out))])
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
    if not items_str or items_str == "nan": items_str = "Hàng tiêu chuẩn"

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
        extra = {"df_hanoi": optimize_dtypes(df_hanoi, categorize=True), "df_outside": optimize_dtypes(df_outside, categorize=True), "df_failed": optimize_dtypes(df_failed, categorize=True)}
        store_result("orders", out, files, "🧭 Hoàn tất chuẩn hóa & phân loại đơn hàng", [("Tổng số đơn", len(out)), ("Đơn Hà Nội", len(df_hanoi)), ("Đơn Ngoài Hà Nội", len(df_outside)), ("Lỗi Geocode", len(df_failed))], extra=extra)
    res = st.session_state.get("orders_result")
    if res is not None:
        st.markdown("### 📊 Kết quả Phân Loại Đơn Hàng")
        st.success(res["summary"])
        if res["metrics"]:
            cols = st.columns(len(res["metrics"]))
            for c, (lab, val) in zip(cols, res["metrics"]): c.metric(lab, val)
        extra = res.get("extra", {})
        df_hn, df_out, df_fail = extra.get("df_hanoi", pd.DataFrame()), extra.get("df_outside", pd.DataFrame()), extra.get("df_failed", pd.DataFrame())
        t_hn, t_out, t_fail, t_all = st.tabs([f"🏢 1. Đơn tại Hà Nội ({len(df_hn)})", f"🌏 2. Đơn ngoài Hà Nội ({len(df_out)})", f"⚠️ 3. Không quét được tọa độ ({len(df_fail)})", f"📋 4. Tất cả đơn tổng ({len(res['df'])})"])
        with t_hn: st.dataframe(df_hn, use_container_width=True)
        with t_out: st.dataframe(df_out, use_container_width=True)
        with t_fail: st.dataframe(df_fail, use_container_width=True)
        with t_all: st.dataframe(res["df"], use_container_width=True)

# ============================================================================
# 📦 TAB 5: ORDERS SOLVING
# ============================================================================
PROVINCE_KEYWORDS = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "An Giang", "Bà Rịa - Vũng Tàu",
    "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre", "Bình Định", "Bình Dương",
    "Bình Phước", "Bình Thuận", "Cà Mau", "Cao Bằng", "Đắk Lắk", "Đắk Nông", "Điện Biên",
    "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Tĩnh", "Hải Dương",
    "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu",
    "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình",
    "Ninh Thuận", "Phú Thọ", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị",
    "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế",
    "Tiền Giang", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]

def extract_province(address_str: str) -> str:
    if is_blank(address_str): return "Không xác định"
    norm_addr = norm_basic(address_str)
    for p in PROVINCE_KEYWORDS:
        if norm_basic(p) in norm_addr: return p
    parts = [part.strip() for part in address_str.split(",") if part.strip()]
    if parts:
        last = re.sub(r"(Việt Nam|Vietnam)\$", "", parts[-1], flags=re.I).strip()
        if last: return last
    return "Chưa rõ Tỉnh/Thành"

def render_orders_solving_tab():
    st.header("⚡ 5. Orders Solving (Level Up Orders Analysis)")
    orders_res = st.session_state.get("orders_result")
    if orders_res is None or orders_res.get("df") is None or orders_res["df"].empty:
        st.warning("⚠️ Chưa có dữ liệu Đơn hàng từ Tab 4 (Orders). Vui lòng thực hiện chuẩn hóa ở Tab 4 trước.")
        return
    df_orders = orders_res["df"].copy()
    product_file_path = os.path.join(OUT_PRODUCT, "DIM_PRODUCT.xlsx")
    prod_map = {}
    if os.path.exists(product_file_path):
        try:
            df_product = pd.read_excel(product_file_path)
            for _, p_row in df_product.iterrows():
                p_id, p_name = str(p_row.get("product_id", "")).strip(), str(p_row.get("product_name", "")).strip()
                l, w, h = parse_num(p_row.get("length_cm")) or 30.0, parse_num(p_row.get("width_cm")) or 20.0, parse_num(p_row.get("height_cm")) or 15.0
                dim_info = {"l": l, "w": w, "h": h}
                if p_id: prod_map[norm_basic(p_id)] = dim_info
                if p_name: prod_map[norm_basic(p_name)] = dim_info
            st.success(f"✅ Đã kết nối thành công tới file `{os.path.relpath(product_file_path, BASE_DIR)}`.")
        except Exception as exc: st.error(f"❌ Lỗi đọc file DIM_PRODUCT.xlsx: {exc}")

    solved_rows = []
    for _, o_row in df_orders.iterrows():
        item_norm = norm_basic(o_row.get("items", ""))
        dim = prod_map.get(item_norm, {"l": 30.0, "w": 20.0, "h": 15.0})
        l, w, h = dim["l"], dim["w"], dim["h"]
        qty = int(parse_num(o_row.get("quantity")) or 1)
        actual_weight = parse_num(o_row.get("total_weight_kg"))
        volumetric_weight = (l * w * h * qty) / 4000.0
        bbs_code = "A" if volumetric_weight > actual_weight else "B"
        bbs_desc = "A (Màu Cam - Hàng Cồng Kềnh)" if bbs_code == "A" else "B (Màu Xám - Hàng Nặng/Tiêu Chuẩn)"
        province = extract_province(o_row.get("address", ""))
        row_dict = o_row.to_dict()
        row_dict.update({
            "Tỉnh_Thành": province,
            "Chiều_Dài_cm": l, "Chiều_Rộng_cm": w, "Chiều_Cao_cm": h,
            "Kích_Thước_DxRxC_cm": f"{l:.0f} x {w:.0f} x {h:.0f}",
            "Volumetric_Weight_kg": round(volumetric_weight, 3),
            "BBS": bbs_code, "BBS_Phân_Loại": bbs_desc
        })
        solved_rows.append(row_dict)
    df_solved = pd.DataFrame(solved_rows)
    total_orders = len(df_solved)
    count_a = int((df_solved["BBS"] == "A").sum())
    count_b = int((df_solved["BBS"] == "B").sum())
    unique_provinces = df_solved["Tỉnh_Thành"].nunique()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    kpi_c1.metric("📦 Tổng Số Đơn Hàng", f"{total_orders:,}")
    kpi_c2.metric("🟧 Đơn Loại A (Cồng kềnh)", f"{count_a:,}")
    kpi_c3.metric("⬜ Đơn Loại B (Nặng)", f"{count_b:,}")
    kpi_c4.metric("🗺️ Số Tỉnh/Thành", f"{unique_provinces}")
    st.divider()
    t_summary, t_detail = st.tabs(["📊 1. Thống Kê Địa Lý Theo Tỉnh/Thành", "📋 2. Bảng Dữ Liệu Chi Tiết Level-Up"])
    with t_summary:
        province_summary = df_solved.groupby("Tỉnh_Thành").agg(
            Tổng_Đơn=("order_id", "count"),
            Đơn_A_Cam=("BBS", lambda s: int((s == "A").sum())),
            Đơn_B_Xám=("BBS", lambda s: int((s == "B").sum()))
        ).reset_index()
        province_summary["Tỷ_Lệ_%"] = (province_summary["Tổng_Đơn"] / total_orders * 100).round(1)
        province_summary = province_summary.sort_values(by="Tổng_Đơn", ascending=False).reset_index(drop=True)
        st.dataframe(province_summary, use_container_width=True)
    with t_detail:
        styled_df = df_solved.style.map(lambda v: "background-color: #ffe0b2; color: #e65100; font-weight: bold;" if v == "A" else ("background-color: #e0e0e0; color: #424242; font-weight: bold;" if v == "B" else ""), subset=["BBS"])
        st.dataframe(styled_df, use_container_width=True)
    files = save_outputs(OUT_SOLVING, "DIM_ORDERS_SOLVED", "DIM_ORDERS_SOLVED", df_solved, extra_sheets={"THONG_KE_TINH": province_summary})
    st.session_state["solved_result"] = {"df": df_solved, "files": files}

# ============================================================================
# 🗺️ TAB 6: ROUTING — CLARKE-WRIGHT (xe nhà / thuê xe 3PL / giao hàng tiết kiệm theo tỉnh)
# ----------------------------------------------------------------------------
# DÁN KHỐI NÀY VÀO CUỐI FILE (sau render_orders_solving_tab) rồi thêm tab:
#     tabs = st.tabs([..., "🗺️ 6. Routing"]);  with tabs[5]: render_routing_tab()
# Phụ thuộc các tên ĐÃ CÓ trong file: st, pd, np, os, re, math, dt, BASE_DIR, XLSX_MIME,
#   OUT_FLEET, OUT_WAREHOUSE, OUT_PRODUCT, OUT_ORDERS, OUT_CUSTOMER, OUT_SOLVING, OUT_ROUTING,
#   UserError, is_blank, norm_basic, parse_num, parse_qty, parse_date_iso, haversine,
#   save_outputs, extract_province, VEHICLE_NUMERIC
# ============================================================================
from dataclasses import dataclass, field

LICENSE_RANKS = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4,
    "B1": 5, "B2": 6,
    "C": 7,
    "D": 8, "E": 9,
    "FB2": 10, "FC": 11, "FD": 12, "FE": 13
}
LICENSE_LIST = list(LICENSE_RANKS.keys())


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


def service_rules_from_cfg(cfg: Config) -> dict:
    base = {"B2C": {"loading": 25, "unloading": 35}, "B2B": {"loading": 45, "unloading": 60}}
    for k in ("B2C", "B2B"):
        total = base[k]["loading"] + base[k]["unloading"]
        want = float(cfg.service_min.get(k, total))
        if abs(want - total) > 1e-9:
            ratio = base[k]["loading"] / total
            base[k] = {"loading": want * ratio, "unloading": want * (1 - ratio)}
    return base


# ============================================================================
# 3.2  PHÂN LOẠI & CHI PHÍ THUÊ NGOÀI 3PL
# ============================================================================
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


# ============================================================================
# 3.3  RÀNG BUỘC VẬN TẢI & THỜI GIAN
# ============================================================================
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
    solved_file: str = os.path.join(OUT_SOLVING, "DIM_ORDERS_SOLVED.xlsx")

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

# ============================================================================
# 📥 NẠP DỮ LIỆU TỪ TẤT CẢ CÁC DIM (Tab 1 → Tab 5)
# ============================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def _read_dim_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_excel(path)

def read_dim(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise UserError(f"Chưa có file `{os.path.relpath(path, BASE_DIR)}`.")
    return _read_dim_cached(path, os.path.getmtime(path))

def _num_col(df, col, default=0.0):
    if col not in df.columns: return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)

def money(x) -> str:
    return f"{float(x):,.0f}"

def load_vehicles(cfg: Config) -> pd.DataFrame:
    """Tab 1 → DIM_VEHICLE.xlsx"""
    df = read_dim(cfg.vehicle_file).copy()
    for c in ("vehicle_id", "license_plate", "id_warehouse"):
        df[c] = df[c].astype(str).str.strip() if c in df.columns else ""
    df = df[~df["vehicle_id"].isin(["", "nan", "None"])].copy()
    for c in VEHICLE_NUMERIC: df[c] = _num_col(df, c)
    df.loc[df["average_speed_kmh"] <= 0, "average_speed_kmh"] = 40.0
    df.loc[df["Max_Distance"] <= 0, "Max_Distance"] = 100.0
    df["vehicle_type"] = [f"{w:.0f}kg/{v:g}m3" for w, v in zip(df["max_weight_kg"], df["max_volume_m3"])]
    return df.reset_index(drop=True)

def load_warehouse_coords(cfg: Config) -> dict:
    """Tab 2 → WAREHOUSE_WITH_COORDINATES.xlsx"""
    if not os.path.exists(cfg.warehouse_file): return {}
    out = {}
    for _, r in read_dim(cfg.warehouse_file).iterrows():
        wid = "" if is_blank(r.get("id_warehouse")) else str(r["id_warehouse"]).strip()
        if wid and pd.notna(r.get("lat")) and pd.notna(r.get("lng")): out[wid] = (float(r["lat"]), float(r["lng"]))
    return out

def load_products(cfg: Config) -> dict:
    """Tab 3 → DIM_PRODUCT.xlsx : tra cân nặng / thể tích 1 đơn vị theo mã hoặc tên sản phẩm."""
    out = {}
    if not os.path.exists(cfg.product_file): return out
    for _, r in read_dim(cfg.product_file).iterrows():
        info = {"weight": parse_num(r.get("weight")), "volume": parse_num(r.get("volume"))}
        for key in (r.get("product_id"), r.get("product_name")):
            if not is_blank(key): out[norm_basic(key)] = info
    return out

def load_customers(cfg: Config) -> dict:
    """Tab 4 → DATASET_CUSTOMER.xlsx : tên + tọa độ khách hàng."""
    out = {}
    if not os.path.exists(cfg.cust_file): return out
    for _, r in read_dim(cfg.cust_file).iterrows():
        cid = str(r.get("customer_id", "")).strip()
        if cid and cid != "nan" and pd.notna(r.get("lat")) and pd.notna(r.get("lng")):
            out[cid] = {"lat": float(r["lat"]), "lng": float(r["lng"]), "name": "" if is_blank(r.get("customer_name")) else str(r["customer_name"])}
    return out

def load_orders(cfg: Config, notes: list, products: dict, customers: dict) -> list:
    """Tab 5 → DIM_ORDERS_SOLVED.xlsx (ưu tiên, có Tỉnh/Thành + BBS); nếu chưa có thì dùng Tab 4 → DIM_ORDERS.xlsx."""
    use_solved = os.path.exists(cfg.solved_file)
    src = cfg.solved_file if use_solved else cfg.order_file
    notes.append(f"Nguồn đơn hàng: `{os.path.relpath(src, BASE_DIR)}`" +
                 ("" if use_solved else " — chưa có DIM_ORDERS_SOLVED nên Tỉnh/Thành được tách trực tiếp từ địa chỉ."))
    df = read_dim(src)
    for c in ("order_id", "customer_id", "order_date"):
        if c not in df.columns: raise UserError(f"`{os.path.basename(src)}` thiếu cột `{c}`.")
    orders, skipped, filled, from_cust = [], 0, 0, 0
    for _, r in df.iterrows():
        oid, cid, dstr = ("" if is_blank(r.get(k)) else str(r[k]).strip() for k in ("order_id", "customer_id", "order_date"))
        d_iso = parse_date_iso(dstr)
        if not (oid and cid and d_iso): skipped += 1; continue
        otype = str(r.get("order_type", "")).strip().upper()
        lat = float(r["lat"]) if pd.notna(r.get("lat")) else None
        lng = float(r["lng"]) if pd.notna(r.get("lng")) else None
        if (lat is None or lng is None) and cid in customers:
            lat, lng, from_cust = customers[cid]["lat"], customers[cid]["lng"], from_cust + 1
        qty = int(parse_qty(r.get("quantity")) or 1)
        items_str = "" if is_blank(r.get("items")) else str(r["items"]).strip()
        w = parse_qty(r.get("total_weight_kg"))
        v = parse_qty(r.get("total_volume_m3"), dot_is_decimal=True)
        pm = products.get(norm_basic(items_str)) if items_str else None
        if pm and (w <= 0 or v <= 0):
            if w <= 0: w = pm["weight"] * qty
            if v <= 0: v = pm["volume"] * qty
            filled += 1
        addr = "" if is_blank(r.get("address")) else str(r["address"])
        prov = r.get("Tỉnh_Thành")
        prov = extract_province(addr) if is_blank(prov) else str(prov).strip()
        cname = "" if is_blank(r.get("customer_name")) else str(r["customer_name"])
        orders.append({
            "order_id": oid, "customer_id": cid, "customer_name": cname or customers.get(cid, {}).get("name", ""),
            "order_type": otype if otype in ("B2B", "B2C") else "B2C",
            "quantity": qty, "items": items_str or "Hàng tiêu chuẩn",
            "total_weight_kg": w, "total_volume_m3": v,
            "address": addr, "lat": lat, "lng": lng, "province": prov,
            "bbs": "" if is_blank(r.get("BBS")) else str(r["BBS"]).strip(),
            "alert_status": "Normal" if is_blank(r.get("alert_status")) else str(r["alert_status"]),
            "order_status": "" if is_blank(r.get("order_status")) else str(r["order_status"]),
            "order_date": d_iso,
        })
    if skipped: notes.append(f"Bỏ qua {skipped} đơn thiếu order_id / customer_id hoặc order_date sai định dạng.")
    if filled: notes.append(f"Đã bù cân nặng/thể tích cho {filled} đơn từ DIM_PRODUCT (đơn thiếu số liệu).")
    if from_cust: notes.append(f"Đã bù tọa độ cho {from_cust} đơn từ DATASET_CUSTOMER.")
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
    customers = load_customers(cfg)
    products = load_products(cfg)
    orders = load_orders(cfg, notes, products, customers)
    return {"vehicles": veh, "cust": dict(customers), "warehouses": warehouses, "orders": orders, "notes": notes}

def require_inputs(cfg: Config):
    missing = []
    if not os.path.exists(cfg.vehicle_file): missing.append(f"`{os.path.relpath(cfg.vehicle_file, BASE_DIR)}` (chạy Tab 1 · Hạm đội xe)")
    if not (os.path.exists(cfg.solved_file) or os.path.exists(cfg.order_file)):
        missing.append(f"`{os.path.relpath(cfg.order_file, BASE_DIR)}` (chạy Tab 4 · Đơn hàng) hoặc `{os.path.relpath(cfg.solved_file, BASE_DIR)}` (Tab 5)")
    if missing: raise UserError("Thiếu file đầu vào: " + "; ".join(missing))


# ============================================================================
# 🧠 ROUTE PLANNER (Clarke-Wright + 2-opt + dispatch)
# ============================================================================
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

    # ------------------------------------------------------------------ khoảng cách
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

    # ------------------------------------------------------------------ xe & chỉ số tuyến
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

    # ------------------------------------------------------------------ 2-opt
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

    # ------------------------------------------------------------------ Clarke-Wright
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

    # ------------------------------------------------------------------ nhu cầu & cắt đơn
    def _demand(self, ords):
        demand, by_cust = {}, {}
        for o in ords:
            d = demand.setdefault(o["customer_id"], {"weight": 0.0, "volume": 0.0, "qty": 0, "service_min": 0.0,
                                                    "province": o.get("province") or "Chưa rõ Tỉnh/Thành"})
            d["weight"] += o["total_weight_kg"]; d["volume"] += o["total_volume_m3"]
            d["qty"] += o.get("quantity", 1)
            d["service_min"] += self.cfg.service_min.get(o["order_type"], 60)
            by_cust.setdefault(o["customer_id"], []).append(o)
        return demand, by_cust

    @staticmethod
    def _group_by_province(custs, demand):
        """Tách danh sách khách theo Tỉnh/Thành (giữ thứ tự) — mỗi tỉnh là 1 gói giao riêng khi thuê ngoài."""
        groups = {}
        for c in custs: groups.setdefault(demand[c].get("province") or "Chưa rõ Tỉnh/Thành", []).append(c)
        return groups

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

    # ------------------------------------------------------------------ tạo tuyến thuê ngoài
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
        provs = sorted({demand[c].get("province") or "Chưa rõ Tỉnh/Thành" for c in custs})
        province = provs[0] if len(provs) == 1 else " / ".join(provs)

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
            "reason": reason, "trip_no": None, "tier": cls["tier"], "province": province,
            "start": start, "end": start + dt.timedelta(hours=hours), "hours": hours, "events": events,
        }

    # ------------------------------------------------------------------ dispatch: gán xe, kiểm ràng buộc, cắt đơn
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
                force_full = action in ("NO_FLEET", "OUTSOURCE_FULL")
                groups = self._group_by_province(r, demand)
                for prov, ids_g in groups.items():
                    if len(groups) == 1: cls = ta["cls"]
                    else:
                        cls = classify_outsourcing(sum(demand[c]["weight"] for c in ids_g), sum(demand[c]["volume"] for c in ids_g),
                                                   oc, force_full=force_full)
                    ev = [{"sev": sev, "kind": kind, "detail": ta["message"], "why": why, "handled": f"Thuê ngoài {cls['type']}", "propose": propose}]
                    rs = why if len(groups) == 1 else f"{why} (gói giao tỉnh {prov})"
                    out.append(self._outsource_route(ids_g, wh_id, demand, by_cust, cls, ev, date_str, reason=rs))
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
                    "outsourcing_cost": 0.0, "outsourcing_type": ta["outsourcing_type"] if split else None,
                    "tier": None, "province": ", ".join(sorted({demand[c].get("province") or "Chưa rõ Tỉnh/Thành" for c in r})),
                    "cut_orders": [o["order_id"] for o in orders] if split else [],
                    "cut_weight_kg": max(m["w"] - cap_w, 0.0) if split else 0.0,
                    "cut_volume_m3": max(m["v"] - cap_v, 0.0) if split else 0.0,
                    "cap_w": cap_w, "cap_v": cap_v, "total_weight_kg": m["w"], "total_volume_m3": m["v"], "total_qty": m["qty"],
                    "reason": reason if split else "",
                    "trip_no": max(1, cfg.vehicle_daily_count - veh_counts.get(vid, 0)),
                    "start": start, "end": end_dt, "hours": hours, "events": ev,
                })
                if split:
                    # Phần dư (vượt xe nhà lớn nhất) trở thành 1 dòng riêng: vào bảng 3PL (Full) hoặc bảng tiết kiệm (theo hạng)
                    frac = min(max(ex_w / m["w"] if m["w"] > 0 else 0.0, ex_v / m["v"] if m["v"] > 0 else 0.0), 1.0)
                    excess = self._outsource_route(
                        r, wh_id, demand, by_cust, ta["cls"], [], date_str, km=m["km"],
                        reason=f"phần dư {ex_w:,.0f}kg/{ex_v:.1f}m³ sau khi xe {vid} chở đầy ({cap_w:,.0f}kg/{cap_v:g}m³)")
                    excess.update({"kind": "SPLIT_EXCESS", "parent_vehicle": vid, "total_weight_kg": ex_w, "total_volume_m3": ex_v,
                                   "total_qty": max(1, math.ceil(m["qty"] * frac)),
                                   "load_factor": self._calculate_load_factor({"w": ex_w, "v": ex_v}, None, ta["cls"])})
                    out.append(excess)
        if cut:
            cut_info = {c: (kind, detail, short) for c, kind, detail, short in cut}
            for prov, ids_g in self._group_by_province([c for c, _, _, _ in cut], demand).items():
                cls = classify_outsourcing(sum(demand[c]["weight"] for c in ids_g), sum(demand[c]["volume"] for c in ids_g), oc)
                cut_details = []
                for c in ids_g:
                    orders_c = by_cust.get(c, [])
                    order_ids_str = ", ".join(o["order_id"] for o in orders_c) if orders_c else c
                    addr_str = orders_c[0].get("address", "") if orders_c else ""
                    cut_details.append(f"đơn {order_ids_str} ({addr_str}) cắt do {cut_info[c][2]}")
                ev = [{"sev": "HIGH", "kind": cut_info[c][0], "detail": f"Đơn khách {c}: {cut_info[c][1]}",
                       "why": f"đơn {c} bị cắt do {cut_info[c][2]}", "handled": f"Cắt khỏi tuyến → thuê ngoài {cls['type']}",
                       "propose": "Bổ sung tọa độ hoặc kiểm tra Max_Distance xe", "ids": [c]} for c in ids_g]
                out.append(self._outsource_route(ids_g, wh_id, demand, by_cust, cls, ev, date_str, reason="; ".join(cut_details)))
        return out

    # ------------------------------------------------------------------ lập kế hoạch 1 ngày
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
            self.cust[o["customer_id"]] = {"lat": float(o["lat"]), "lng": float(o["lng"]), "name": o.get("customer_name") or o["customer_id"]}

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


# ============================================================================
# 📈 MÔ PHỎNG TOÀN BỘ CÁC NGÀY + KPI
# ============================================================================
def route_total(r):
    return r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r.get("driver_cost", 0) + r.get("outsourcing_cost", 0)


def simulate_all(data: dict, cfg: Config = CFG):
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]: by_date.setdefault(o["order_date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    all_r = [r for d in days.values() for r in d["routes"] + d["overdue_routes"]]

    valid_routes_for_load = [r for r in all_r if not r["external"]]  # tỉ lệ lấp đầy: CHỈ tính trên tuyến xe nhà
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

# ============================================================================
# 🧩 TRÌNH BÀY KẾT QUẢ: 3 BẢNG (XE NHÀ / THUÊ XE 3PL / GIAO HÀNG TIẾT KIỆM THEO TỈNH)
# ============================================================================
def split_tables(day):
    """Chia các tuyến trong ngày thành 3 nhóm hiển thị."""
    routes = day["routes"] + day["overdue_routes"]
    home = [r for r in routes if not r["external"]]
    truck3pl = [r for r in routes if r["external"] and r.get("tier") == "FULL"]
    saving = [r for r in routes if r["external"] and r.get("tier") in ("SAVING_1", "SAVING_2")]
    return home, truck3pl, saving

def route_label(prefix, idx, r, wh_info) -> str:
    wh = wh_info.get(r["id_warehouse"], {"name": r["id_warehouse"]})
    return f"{prefix} {wh['name']} #{idx}"

def route_load(r):
    """Trọng tải / thể tích mà tuyến này thực sự phải chở (với phần dư tách chuyến = phần vượt xe nhà)."""
    return r["total_weight_kg"], r["total_volume_m3"]

def route_items_detail(r) -> str:
    item_map = {}
    for o in r["orders"]:
        it_name = o.get("items") or "Hàng tiêu chuẩn"
        d = item_map.setdefault(it_name, {"qty": 0, "w": 0.0, "v": 0.0})
        d["qty"] += o.get("quantity", 1)
        d["w"] += o.get("total_weight_kg", 0.0)
        d["v"] += o.get("total_volume_m3", 0.0)
    return " + ".join(f"{d['qty']} x {it} ({d['w']:,.0f}kg | {d['v']:.2f}m³)" for it, d in item_map.items())

def route_vehicle_text(r) -> str:
    reason_text = r.get("reason") or "Vượt khả năng xe nhà"
    if r["external"]:
        return f"🟣 Thuê ngoài 3PL — {r['vehicle_type']} (Lý do: {reason_text})"
    if r["kind"] == "SPLIT":
        return f"🚚 {r['vehicle_type']} chở đầy xe, phần dư thuê 3PL (Lý do: {reason_text})"
    return "🚚 " + str(r["vehicle_type"])

def _provinces_of(r) -> str:
    return ", ".join(sorted({o.get("province") or "Chưa rõ Tỉnh/Thành" for o in r["orders"]}))

def home_routes_df(routes, wh_info, cfg: Config) -> pd.DataFrame:
    """BẢNG 1 — các tuyến tối ưu được bằng xe nhà."""
    rows = []
    for idx, r in enumerate(routes, 1):
        tw, tv = route_load(r)
        rows.append({
            "MÃ TUYẾN": route_label("Xe nhà", idx, r, wh_info),
            "TỈNH/THÀNH": _provinces_of(r),
            "THỨ TỰ KHÁCH GHÉP": " → ".join(r["route"]),
            "ĐƠN GIAO": ", ".join(o["order_id"] for o in r["orders"]),
            "SỐ ĐƠN": len(r["orders"]),
            "TỔNG UNITS": r["total_qty"],
            "CHI TIẾT MẶT HÀNG & ĐƠN VỊ": route_items_detail(r),
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
            "TỔNG CHI PHÍ (đ)": round(route_total(r)),
            "BẮT ĐẦU": f"{r['start']:%H:%M}",
            "KẾT THÚC": f"{r['end']:%H:%M}",
            "TRẠNG THÁI": ("🟠 Tách: xe nhà + 3PL" if r["kind"] == "SPLIT" else
                           ("✅ Đã tối ưu" if r["hours"] <= cfg.max_route_hours + 1e-9 else "⚠️ Vượt giới hạn giờ")),
        })
    return pd.DataFrame(rows)

def truck3pl_df(routes, wh_info) -> pd.DataFrame:
    """BẢNG 2 — các tuyến thuê xe 3PL (trọn chuyến Full hoặc phần dư tách chuyến)."""
    rows = []
    for idx, r in enumerate(routes, 1):
        tw, tv = route_load(r)
        is_excess = r["kind"] == "SPLIT_EXCESS"
        rows.append({
            "MÃ TUYẾN": route_label("3PL", idx, r, wh_info),
            "LOẠI": f"Phần dư tách chuyến (xe nhà {r.get('parent_vehicle')} đã chở đầy)" if is_excess else "Thuê trọn chuyến 3PL Full",
            "TỈNH/THÀNH": r.get("province", ""),
            "THỨ TỰ KHÁCH GHÉP": " → ".join(r["route"]),
            "ĐƠN GIAO": ", ".join(o["order_id"] + ("*" if is_excess else "") for o in r["orders"]),
            "SỐ ĐƠN": len(r["orders"]),
            "TỔNG UNITS": ("≈ " if is_excess else "") + str(r["total_qty"]),
            "TỔNG TRỌNG TẢI (kg)": round(tw, 1),
            "TỔNG THỂ TÍCH (m³)": round(tv, 2),
            "XE 3PL": route_vehicle_text(r),
            "BIỂN SỐ": r["license_plate"],
            "QUÃNG ĐƯỜNG (km)": r["km"],
            "THỜI GIAN (giờ)": round(r["hours"], 2),
            "TỔNG CHI PHÍ (đ)": round(route_total(r)) if route_total(r) > 0 else "Chờ nhập giá 3PL",
            "TRẠNG THÁI": "🟣 Cần xe 3PL Full" if r["license_plate"].startswith("Thuê ngoài") else "🟣 Đã nhập xe 3PL",
        })
    return pd.DataFrame(rows)

def saving_groups(routes, wh_info):
    """BẢNG 3 — giao hàng tiết kiệm: GOM các tuyến cùng Tỉnh/Thành thành 1 nhóm (tổng trọng tải, tổng số lượng).
    Trả về (bảng nhóm theo tỉnh, bảng chi tiết các tuyến thành viên)."""
    by_prov = {}
    for idx, r in enumerate(routes, 1):
        by_prov.setdefault(r.get("province") or "Chưa rõ Tỉnh/Thành", []).append((idx, r))
    order = sorted(by_prov.items(), key=lambda kv: -sum(route_load(r)[0] for _, r in kv[1]))
    g_rows, m_rows = [], []
    for prov, items in order:
        tw = sum(route_load(r)[0] for _, r in items)
        tv = sum(route_load(r)[1] for _, r in items)
        tq = sum(r["total_qty"] for _, r in items)
        cost = sum(float(r["outsourcing_cost"]) for _, r in items)
        seen, order_ids = set(), []
        for _, r in items:
            for o in r["orders"]:
                oid = o["order_id"] + ("*" if r["kind"] == "SPLIT_EXCESS" else "")
                if oid not in seen: seen.add(oid); order_ids.append(oid)
        tiers = {}
        for _, r in items: tiers[r["outsourcing_type"]] = tiers.get(r["outsourcing_type"], 0) + 1
        tier_txt = ", ".join(f"{k} ×{n}" for k, n in sorted(tiers.items()))
        reasons = []
        for _, r in items:
            if r.get("reason") and r["reason"] not in reasons: reasons.append(r["reason"])
        g_rows.append({
            "TỈNH/THÀNH": prov,
            "SỐ TUYẾN GỘP": len(items),
            "CÁC TUYẾN GỘP": ", ".join(route_label("Tiết kiệm", i, r, wh_info) for i, r in items),
            "SỐ ĐƠN": len(order_ids),
            "ĐƠN GIAO": ", ".join(order_ids),
            "TỔNG UNITS": tq,
            "TỔNG TRỌNG TẢI (kg)": round(tw, 1),
            "TỔNG THỂ TÍCH (m³)": round(tv, 2),
            "HẠNG TIẾT KIỆM": tier_txt,
            "TỔNG CHI PHÍ (đ)": round(cost),
            "HÌNH THỨC GIAO": f"🟣 Giao hàng tiết kiệm — {tier_txt} (Lý do: {' | '.join(reasons)})",
        })
        for i, r in items:
            tw_r, tv_r = route_load(r)
            m_rows.append({
                "TỈNH/THÀNH": prov, "MÃ TUYẾN": route_label("Tiết kiệm", i, r, wh_info), "KHO": r["id_warehouse"],
                "LOẠI": "Phần dư tách chuyến" if r["kind"] == "SPLIT_EXCESS" else "Tuyến bị cắt / không đủ xe nhà",
                "ĐƠN GIAO": ", ".join(o["order_id"] for o in r["orders"]), "TỔNG UNITS": r["total_qty"],
                "TỔNG TRỌNG TẢI (kg)": round(tw_r, 1), "TỔNG THỂ TÍCH (m³)": round(tv_r, 2),
                "HẠNG TIẾT KIỆM": r["outsourcing_type"], "CHI PHÍ (đ)": round(float(r["outsourcing_cost"])), "LÝ DO": r.get("reason", ""),
            })
    return pd.DataFrame(g_rows), pd.DataFrame(m_rows)

def file_status(cfg: Config) -> pd.DataFrame:
    items = [
        ("Tab 1 · Hạm đội xe (DIM_VEHICLE)", cfg.vehicle_file, "Bắt buộc"),
        ("Tab 2 · Kho & Tọa độ", cfg.warehouse_file, "Khuyến nghị"),
        ("Tab 3 · Sản phẩm (DIM_PRODUCT)", cfg.product_file, "Tuỳ chọn — bù cân nặng/thể tích đơn thiếu"),
        ("Tab 4 · Đơn hàng (DIM_ORDERS)", cfg.order_file, "Bắt buộc nếu chưa có file Solved"),
        ("Tab 4 · Khách hàng (DATASET_CUSTOMER)", cfg.cust_file, "Khuyến nghị — bù tọa độ/tên khách"),
        ("Tab 5 · Orders Solved (DIM_ORDERS_SOLVED)", cfg.solved_file, "Khuyến nghị — có Tỉnh/Thành & BBS"),
    ]
    rows = [{"Nguồn": n, "File": os.path.relpath(p, BASE_DIR), "Trạng thái": "✅ Có" if os.path.exists(p) else "❌ Chưa có", "Yêu cầu": need}
            for n, p, need in items]
    return pd.DataFrame(rows)

def _show_table(df, empty_msg, drop=()):
    if df.empty: st.info(empty_msg)
    else: st.dataframe(df.drop(columns=[c for c in drop if c in df.columns]), hide_index=True)

# ============================================================================
# 🗺️ TAB 6: DASHBOARD ĐỊNH TUYẾN
# ============================================================================
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

    try: dt.datetime.strptime(start_time, "%H:%M")
    except ValueError: start_time = "08:30"

    return Config(start_time=start_time, max_route_hours=float(max_hours), detour_factor=float(detour),
                  constraint_mode=constraint_mode,
                  service_min={"B2B": int(svc_b2b), "B2C": int(svc_b2c)}, backup_driver_cost=float(backup),
                  vehicle_daily_count=int(trips), turnaround_min=float(turn),
                  saving_1_price=float(s1_p), saving_2_price=float(s2_p))

def render_routing_tab():
    st.header("🗺️ 6. Dashboard Định tuyến — Clarke-Wright")
    cfg = render_routing_settings()
    st.markdown("### 📂 Trạng thái dữ liệu đầu vào (tất cả các DIM)")
    st.dataframe(file_status(cfg), hide_index=True)

    if st.button("⚡ Chạy Routing Tạm Thời", type="primary", key="t6_run_temp"):
        try:
            require_inputs(cfg)
            data = load_data(cfg)
            planner, days, kpis, total_cost = simulate_all(data, cfg)
        except UserError as exc:
            st.error(f"❌ {exc}"); return
        st.session_state["t6_temp_plan"] = {"planner": planner, "days": days, "kpis": kpis, "cfg": cfg, "data": data}
        for k in ("t6_final_plan", "t6_final_date", "t6_final_files"): st.session_state.pop(k, None)
        st.success("✅ Đã chạy xong bước tạm thời! Mời phân công tài xế & nhập xe 3PL phía dưới.")

    plan = st.session_state.get("t6_temp_plan")
    if not plan: return

    planner, days, kpis, cfg, data = plan["planner"], plan["days"], plan["kpis"], plan["cfg"], plan["data"]
    for note in data["notes"]: st.info(f"ℹ️ {note}")
    dates = list(days)
    if not dates: return
    date_str = st.selectbox("📅 Chọn ngày điều phối", dates, key="t6_date_temp")
    day = days[date_str]
    home, truck3pl, saving = split_tables(day)
    sav_df, sav_members = saving_groups(saving, planner.wh)

    # ------------------------------------------------------------------ 1. BẢNG ROUTING TẠM THỜI (3 BẢNG)
    st.divider()
    st.markdown(f"## 📊 1. BẢNG ROUTING TẠM THỜI — {date_str}")
    n_orders = len({o["order_id"] for r in home + truck3pl + saving for o in r["orders"]})
    g = st.columns(4)
    g[0].metric("🚚 Tuyến xe nhà", len(home))
    g[1].metric("🟣 Tuyến thuê xe 3PL", len(truck3pl))
    g[2].metric("📦 Nhóm giao tiết kiệm (theo tỉnh)", len(sav_df))
    g[3].metric("Tổng số đơn đã phân tuyến", n_orders)
    unrouted = day.get("unrouted_orders", [])
    if unrouted:
        with st.expander(f"⚠️ {len(unrouted)} đơn chưa phân tuyến được (thiếu tọa độ)"):
            st.dataframe(pd.DataFrame([{"MÃ ĐƠN": o["order_id"], "MÃ KHÁCH": o["customer_id"], "TỈNH/THÀNH": o["province"],
                                        "TRỌNG LƯỢNG (kg)": o["total_weight_kg"], "ĐỊA CHỈ": o["address"]} for o in unrouted]), hide_index=True)

    st.markdown("#### 🚚 1A. Các tuyến tối ưu bằng xe nhà")
    _show_table(home_routes_df(home, planner.wh, cfg), "Ngày này không có tuyến xe nhà nào.", drop=("TÀI XẾ CHÍNH", "PHỤ XE"))
    st.markdown("#### 🟣 1B. Các tuyến thuê xe 3PL")
    _show_table(truck3pl_df(truck3pl, planner.wh), "Ngày này không có tuyến nào cần thuê xe 3PL Full.")
    st.markdown("#### 📦 1C. Các tuyến giao hàng tiết kiệm — gom theo Tỉnh/Thành")
    _show_table(sav_df, "Ngày này không có đơn nào giao hàng tiết kiệm.")
    if not sav_df.empty:
        st.caption(f"Tổng nhóm tiết kiệm: {sav_df['TỔNG TRỌNG TẢI (kg)'].sum():,.1f} kg · {int(sav_df['TỔNG UNITS'].sum())} units · {len(sav_df)} tỉnh/thành. "
                   f"Dấu * = phần dư của đơn tách chuyến.")
        with st.expander("🔎 Chi tiết các tuyến thành viên trong từng nhóm tỉnh"):
            st.dataframe(sav_members, hide_index=True)

    # ------------------------------------------------------------------ 2. PHÂN CÔNG TÀI XẾ (như cũ)
    st.divider()
    st.markdown("## 👨‍✈️ 2. PHÂN CÔNG TÀI XẾ CÔNG TY CHO CÁC XE NHÀ HOẠT ĐỘNG TRONG NGÀY")
    veh_grouped = {}
    for r in home: veh_grouped.setdefault(r["vehicle_id"], []).append(r)
    assignments, has_driver_error = [], False

    if not veh_grouped:
        st.info("🎉 Ngày này toàn bộ đơn giao bằng 3PL, không có xe nhà nào cần phân công tài xế.")
    else:
        st.markdown("Vui lòng nhập tên tài xế hoặc chọn tùy chọn Thuê ngoài tài xế dự phòng bên dưới.")
        for idx, (v_id, r_list) in enumerate(veh_grouped.items(), 1):
            r_list.sort(key=lambda x: x["start"])
            first_r = r_list[0]
            v_wh, v_cap_w = first_r["id_warehouse"], first_r["cap_w"]
            suggested_lic = "B2" if v_cap_w < 3500 else ("C" if v_cap_w <= 15000 else "FC")
            kx = f"{date_str}_{v_id}_{idx}"

            st.markdown(f"##### 🚚 Xe #{idx}: `{v_id}` (Kho: **{v_wh}** | Tải trọng: **{v_cap_w:,.0f} kg** | **{len(r_list)} chuyến trong ngày**)")
            st.markdown("🕒 **Lịch trình chạy thực tế trong ngày:**")
            for trip_idx, r_item in enumerate(r_list, 1):
                ord_ids = ", ".join(o["order_id"] for o in r_item["orders"])
                st.caption(f"  • **Lần {trip_idx}:** Từ **{r_item['start']:%H:%M}** đến **{r_item['end']:%H:%M}** ({r_item['hours']:.2f} giờ) — Giao đơn: `{ord_ids}`")

            driver_choice = st.radio(f"Hình thức tài xế cho xe `{v_id}`",
                                     ["Nhập tên tài xế công ty", "Thành phần dự phòng (Thuê ngoài)"], horizontal=True, key=f"drv_choice_{kx}")
            c1, c2, c3, c4 = st.columns([1.5, 3, 1.5, 2])
            req_lic = c1.selectbox(f"Bằng tối thiểu xe `{v_id}`", LICENSE_LIST, index=LICENSE_LIST.index(suggested_lic), key=f"req_lic_{kx}")

            if driver_choice == "Nhập tên tài xế công ty":
                sel_driver = c2.text_input("Nhập Tên Tài xế", placeholder="Ví dụ: Nguyễn Văn A", key=f"drv_name_txt_{kx}").strip()
                cur_lic = c3.selectbox("Bằng của tài xế", LICENSE_LIST, index=LICENSE_LIST.index("C"), key=f"drv_lic_inp_{kx}")
                cur_role = c4.text_input("Vai trò", value="Tài xế chính", key=f"drv_role_inp_{kx}")
                if not sel_driver:
                    st.warning("⚠️ Vui lòng nhập tên tài xế!")
                    has_driver_error = True
                elif LICENSE_RANKS.get(cur_lic, 6) < LICENSE_RANKS.get(req_lic, 6):
                    st.error(f"🔴 **Lỗi Bằng Lái:** Tài xế `{sel_driver}` (Bằng `{cur_lic}`) KHÔNG ĐỦ điều kiện lái xe `{v_id}` (Yêu cầu bằng `{req_lic}`).")
                    has_driver_error = True
                else:
                    st.caption(f"✅ Bằng lái hợp lệ: **{cur_lic}** (Đạt yêu cầu ≥ {req_lic})")
            else:
                sel_driver, cur_lic, cur_role = "Tài xế dự phòng (Thuê ngoài)", "C", "Tài xế dự phòng"
                c2.text_input("Tên Tài xế", value=sel_driver, disabled=True, key=f"drv_name_dis_{kx}")
                c3.text_input("Bằng của tài xế", value=cur_lic, disabled=True, key=f"drv_lic_dis_{kx}")
                c4.text_input("Vai trò", value=cur_role, disabled=True, key=f"drv_role_dis_{kx}")
                st.info("ℹ️ Sẽ tính thêm chi phí Thuê tài xế dự phòng.")

            assignments.append({"vehicle_id": v_id, "driver_name": sel_driver, "license_type": cur_lic, "role": cur_role})
            st.divider()

    # ------------------------------------------------------------------ 3. NHẬP XE 3PL (như cũ — áp dụng cho bảng 1B)
    st.divider()
    st.markdown("## 🚛 3. NHẬP LIỆU ĐỘI XE THUÊ NGOÀI (BẢNG 1B — 3PL FULL & PHẦN DƯ TÁCH CHUYẾN)")
    user_3pl_inputs = []
    if not truck3pl:
        st.info("🎉 Ngày này không có chuyến nào cần thuê ngoài xe 3PL Full!")
    else:
        st.warning(f"⚠️ Phát hiện **{len(truck3pl)} chuyến** cần thuê ngoài 3PL Full. Hãy kiểm tra danh sách mặt hàng & đơn vị bên dưới để chọn/ghép xe 3PL phù hợp.")
        for idx, r in enumerate(truck3pl, 1):
            req_w, req_v = route_load(r)
            req_qty = r["total_qty"]
            is_excess = r["kind"] == "SPLIT_EXCESS"
            tag_type = f"Phần dư tách chuyến (xe nhà {r.get('parent_vehicle')} đã chở đầy)" if is_excess else "Thuần Thuê ngoài 3PL Full"

            st.markdown(f"##### 📌 Chuyến 3PL #{idx} — {r.get('province', '')} (`{tag_type}`)")
            st.markdown(f"📦 **Tổng yêu cầu:** **{'≈ ' if is_excess else ''}{req_qty} đơn vị hàng (Units)** | **{req_w:,.1f} kg** | **{req_v:.2f} m³**")
            st.markdown(f"🛍️ **Danh sách mặt hàng cần tải:** `{route_items_detail(r)}`")
            st.caption(f"↳ Gồm {len(r['orders'])} đơn: {', '.join(o['order_id'] for o in r['orders'])}")

            num_rows_key = f"3pl_num_rows_{date_str}_{idx}"
            if num_rows_key not in st.session_state: st.session_state[num_rows_key] = 1
            c_add, c_rem = st.columns([2, 8])
            if c_add.button(f"➕ Thêm 1 xe 3PL cho chuyến #{idx}", key=f"add_row_btn_{date_str}_{idx}"):
                st.session_state[num_rows_key] += 1
                st.rerun()
            if st.session_state[num_rows_key] > 1:
                if c_rem.button("➖ Bớt 1 xe 3PL", key=f"rem_row_btn_{date_str}_{idx}"):
                    st.session_state[num_rows_key] -= 1
                    st.rerun()

            route_trucks, total_cap_w, total_cap_v, total_cost = [], 0.0, 0.0, 0.0
            n_rows = st.session_state[num_rows_key]
            for row_i in range(1, n_rows + 1):
                st.caption(f"👉 Hàng xe 3PL #{row_i}:")
                r1, r2, r3, r4 = st.columns([2, 1.5, 1.5, 2])
                v_name = r1.text_input(f"Tên xe #{row_i}", f"Xe tải 3PL #{row_i}", key=f"3pl_name_{date_str}_{idx}_{row_i}")
                v_w = r2.number_input(f"Tải trọng (kg) #{row_i}", value=float(math.ceil(req_w / n_rows)), step=100.0, key=f"3pl_w_{date_str}_{idx}_{row_i}")
                v_v = r3.number_input(f"Thể tích (m³) #{row_i}", value=float(round(req_v / n_rows + 0.2, 1)), step=0.5, key=f"3pl_v_{date_str}_{idx}_{row_i}")
                v_cost = r4.number_input(f"Giá thuê (đ) #{row_i}", value=2_000_000.0, step=100_000.0, key=f"3pl_cost_{date_str}_{idx}_{row_i}")
                total_cap_w += v_w; total_cap_v += v_v; total_cost += v_cost
                route_trucks.append(f"{v_name} ({v_w:,.0f}kg/{v_v:.1f}m³)")

            user_3pl_inputs.append({"req_w": req_w, "req_v": req_v, "req_qty": req_qty,
                                    "name": f"{n_rows} xe 3PL [" + " + ".join(route_trucks) + "]",
                                    "cap_w": total_cap_w, "cap_v": total_cap_v, "cost": total_cost, "route": r})
            st.divider()

    # ------------------------------------------------------------------ 4. CHỐT: kiểm tra + tính chi phí cuối
    if st.button("🚀 Kiểm Tra Điều Kiện & Tính Chi Phí Cuối Cùng", type="primary", key="btn_run_final_all"):
        if has_driver_error:
            st.error("❌ Vui lòng sửa lại các lỗi phân công tài xế bị báo đỏ phía trên trước khi tính chi phí cuối cùng!")
            return
        valid_3pl = True
        for idx, item in enumerate(user_3pl_inputs, 1):
            if item["cap_w"] < item["req_w"] - 1e-5 or item["cap_v"] < item["req_v"] - 1e-5:
                st.error(f"❌ Các xe 3PL ghép cho chuyến #{idx} không đủ tải trọng/thể tích! "
                         f"Cần tối thiểu {item['req_qty']} Đơn vị hàng ({item['req_w']:,.1f}kg / {item['req_v']:.2f}m³), "
                         f"nhưng tổng sức tải xe bạn nhập chỉ đạt {item['cap_w']:,.1f}kg / {item['cap_v']:.2f}m³.")
                valid_3pl = False
        if not valid_3pl: return

        assign_map = {a["vehicle_id"]: a for a in assignments}
        for r in home:
            a = assign_map.get(r["vehicle_id"])
            if not a: continue
            if a["driver_name"] != "Tài xế dự phòng (Thuê ngoài)":
                r["driver_primary"], r["is_backup_driver"], r["driver_cost"] = a["driver_name"], False, 0.0
            else:
                r["driver_primary"], r["is_backup_driver"], r["driver_cost"] = "Tài xế Dự Phòng (Thuê ngoài)", True, cfg.backup_driver_cost

        for item in user_3pl_inputs:
            r = item["route"]
            r["vehicle_type"] = item["name"]
            r["license_plate"] = "3PL Ghép Chuyến"
            r["cap_w"], r["cap_v"] = item["cap_w"], item["cap_v"]
            r["outsourcing_cost"] = item["cost"]
            r["load_factor"] = item["req_v"] / item["cap_v"] if "Cube Out" in cfg.constraint_mode else item["req_w"] / item["cap_w"]

        day["day_cost"] = sum(route_total(rt) for rt in day["routes"] + day["overdue_routes"])
        kpis["operating_cost"] = sum(d["day_cost"] for d in days.values())
        home_lf = [rt["load_factor"] for rt in home]               # tỉ lệ lấp đầy: CHỈ tính trên xe nhà
        kpis["avg_load_factor"] = sum(home_lf) / len(home_lf) if home_lf else 0.0

        f_sav_df, f_members = saving_groups(saving, planner.wh)
        files = save_outputs(OUT_ROUTING, f"ROUTING_PLAN_{date_str}", "XE_NHA", home_routes_df(home, planner.wh, cfg),
                             extra_sheets={"XE_3PL": truck3pl_df(truck3pl, planner.wh), "TIET_KIEM_THEO_TINH": f_sav_df, "TIET_KIEM_CHI_TIET": f_members})
        st.session_state["t6_final_plan"], st.session_state["t6_final_date"], st.session_state["t6_final_files"] = plan, date_str, files
        st.success("🎉 Đã phân công tài xế & cập nhật chi phí tổng thành công!")

    # ------------------------------------------------------------------ 5. BÁO CÁO CHÍNH THỨC
    if st.session_state.get("t6_final_plan") and st.session_state.get("t6_final_date") == date_str:
        st.divider()
        st.markdown(f"## 🏆 BÁO CÁO KẾT QUẢ ĐIỀU PHỐI CHÍNH THỨC — {date_str}")
        f_home, f_truck, f_saving = split_tables(day)
        f_sav_df, f_members = saving_groups(f_saving, planner.wh)
        home_lf = [r["load_factor"] for r in f_home]
        avg_load = (sum(home_lf) / len(home_lf) * 100) if home_lf else 0.0
        cost_home = sum(route_total(r) for r in f_home)
        cost_3pl = sum(route_total(r) for r in f_truck)
        cost_sav = sum(route_total(r) for r in f_saving)

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 TỔNG CHI PHÍ PHẢI CHI TRONG NGÀY", money(day["day_cost"]) + " VNĐ")
        c2.metric("📦 TỔNG ĐƠN ĐÃ PHÂN TUYẾN", len({o["order_id"] for r in f_home + f_truck + f_saving for o in r["orders"]}))
        c3.metric("📊 TỈ LỆ LẤP ĐẦY TB (CHỈ XE NHÀ)", f"{avg_load:.1f}%")
        k1, k2, k3 = st.columns(3)
        k1.metric("🚚 Chi phí xe nhà", money(cost_home) + " VNĐ")
        k2.metric("🟣 Chi phí thuê xe 3PL", money(cost_3pl) + " VNĐ")
        k3.metric("📦 Chi phí giao tiết kiệm", money(cost_sav) + " VNĐ")
        st.caption("Tỉ lệ lấp đầy chỉ tính trên các tuyến xe nhà (bảng 1A); xe 3PL và giao hàng tiết kiệm không tham gia chỉ số này.")

        st.markdown("#### 🚚 Các tuyến xe nhà")
        _show_table(home_routes_df(f_home, planner.wh, cfg), "Không có tuyến xe nhà.")
        st.markdown("#### 🟣 Các tuyến thuê xe 3PL")
        _show_table(truck3pl_df(f_truck, planner.wh), "Không có tuyến thuê xe 3PL.")
        st.markdown("#### 📦 Giao hàng tiết kiệm (gom theo Tỉnh/Thành)")
        _show_table(f_sav_df, "Không có đơn giao hàng tiết kiệm.")

        for path in st.session_state.get("t6_final_files", []):
            if path.endswith(".xlsx") and os.path.exists(path):
                with open(path, "rb") as fh: data_bytes = fh.read()
                st.download_button(f"⬇️ Tải {os.path.basename(path)} (3 bảng + chi tiết)", data_bytes,
                                   file_name=os.path.basename(path), mime=XLSX_MIME, key=f"t6_dl_{os.path.basename(path)}")
