# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
import os
import re
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
# 📦 TAB 3: PRODUCT (ĐÃ THÊM DÀI x RỘNG x CAO CM)
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
        "weight": [25.0, 50.0], 
        "selling_price": [2500000, 4500000]
    })
    mode, edited, uploaded = render_input_block("product", "sản phẩm", seed_df)
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
# 📦 TAB 5: ORDERS SOLVING (LEVEL UP VERSION)
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
        if norm_basic(p) in norm_addr:
            return p
    parts = [part.strip() for part in address_str.split(",") if part.strip()]
    if parts:
        last = parts[-1]
        last = re.sub(r"(Việt Nam|Vietnam)\$", "", last, flags=re.I).strip()
        if last: return last
    return "Chưa rõ Tỉnh/Thành"

def render_orders_solving_tab():
    st.header("⚡ 5. Orders Solving (Level Up Orders Analysis)")
    st.caption("🌐 *Kết hợp tra cứu kích thước từ Tab 3 (Product) để phân loại hàng BBS (A/B) và thống kê theo Tỉnh/Thành*")
    
    orders_res = st.session_state.get("orders_result")
    product_res = st.session_state.get("product_result")
    
    if orders_res is None or orders_res.get("df") is None or orders_res["df"].empty:
        st.warning("⚠️ Chưa có dữ liệu Đơn hàng từ Tab 4 (Orders). Vui lòng thực hiện chuẩn hóa ở Tab 4 trước.")
        return
        
    df_orders = orders_res["df"].copy()
    
    if product_res is not None and product_res.get("df") is not None and not product_res["df"].empty:
        df_product = product_res["df"].copy()
    else:
        df_product = pd.DataFrame()
        st.info("💡 Chưa có danh mục sản phẩm ở Tab 3. Hệ thống sẽ sử dụng kích thước mặc định (30cm x 20cm x 15cm).")

    prod_map = {}
    if not df_product.empty:
        for _, p_row in df_product.iterrows():
            p_id = str(p_row.get("product_id", "")).strip()
            p_name = str(p_row.get("product_name", "")).strip()
            l = parse_num(p_row.get("length_cm")) or 30.0
            w = parse_num(p_row.get("width_cm")) or 20.0
            h = parse_num(p_row.get("height_cm")) or 15.0
            
            dim_info = {"l": l, "w": w, "h": h}
            if p_id: prod_map[norm_basic(p_id)] = dim_info
            if p_name: prod_map[norm_basic(p_name)] = dim_info

    solved_rows = []
    for _, o_row in df_orders.iterrows():
        item_str = str(o_row.get("items", "")).strip()
        item_norm = norm_basic(item_str)
        
        dim = prod_map.get(item_norm, {"l": 30.0, "w": 20.0, "h": 15.0})
        l, w, h = dim["l"], dim["w"], dim["h"]
        
        qty = int(parse_num(o_row.get("quantity")) or 1)
        actual_weight = parse_num(o_row.get("total_weight_kg"))
        
        # Công thức Volumetric Weight = (Dài x Rộng x Cao) x quantity / 4000
        volumetric_weight = (l * w * h * qty) / 4000.0
        
        # So sánh với total_weight_kg
        if volumetric_weight > actual_weight:
            bbs_code = "A"
            bbs_desc = "A (Màu Cam - Hàng Cồng Kềnh)"
        else:
            bbs_code = "B"
            bbs_desc = "B (Màu Xám - Hàng Nặng/Tiêu Chuẩn)"
            
        province = extract_province(o_row.get("address", ""))
        
        row_dict = o_row.to_dict()
        row_dict.update({
            "Tỉnh_Thành": province,
            "Kích_Thước_DxRxC_cm": f"{l:.0f} x {w:.0f} x {h:.0f}",
            "Volumetric_Weight_kg": round(volumetric_weight, 3),
            "BBS": bbs_code,
            "BBS_Phân_Loại": bbs_desc
        })
        solved_rows.append(row_dict)
        
    df_solved = pd.DataFrame(solved_rows)
    
    # 1. Thống kê KPI
    total_orders = len(df_solved)
    count_a = int((df_solved["BBS"] == "A").sum())
    count_b = int((df_solved["BBS"] == "B").sum())
    unique_provinces = df_solved["Tỉnh_Thành"].nunique()
    
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    kpi_c1.metric("📦 Tổng Số Đơn Hàng", f"{total_orders:,}")
    kpi_c2.metric("🟧 Đơn Loại A (Cồng kềnh)", f"{count_a:,}", delta=f"{count_a/total_orders:.1%}" if total_orders else "0%")
    kpi_c3.metric("⬜ Đơn Loại B (Nặng/Tiêu chuẩn)", f"{count_b:,}", delta=f"{count_b/total_orders:.1%}" if total_orders else "0%")
    kpi_c4.metric("🗺️ Số Tỉnh/Thành Có Đơn", f"{unique_provinces}")
    
    st.divider()
    
    # 2. Phân tách Tab Xem Chi Tiết
    t_summary, t_detail = st.tabs(["📊 1. Thống Kê Địa Lý Theo Tỉnh/Thành", "📋 2. Bảng Dữ Liệu Chi Tiết Level-Up"])
    
    with t_summary:
        st.markdown("##### 🏙️ Thống kê số lượng đơn và phân loại hàng A/B theo Tỉnh/Thành phố")
        
        province_summary = df_solved.groupby("Tỉnh_Thành").agg(
            Tổng_Đơn=("order_id", "count"),
            Đơn_A_Cam=("BBS", lambda s: int((s == "A").sum())),
            Đơn_B_Xám=("BBS", lambda s: int((s == "B").sum()))
        ).reset_index()
        
        province_summary["Tỷ_Lệ_%"] = (province_summary["Tổng_Đơn"] / total_orders * 100).round(1)
        province_summary = province_summary.sort_values(by="Tổng_Đơn", ascending=False).reset_index(drop=True)
        
        st.dataframe(province_summary, use_container_width=True)
        st.bar_chart(province_summary.set_index("Tỉnh_Thành")[["Đơn_A_Cam", "Đơn_B_Xám"]])

    with t_detail:
        st.markdown("##### 📋 Toàn bộ bảng dữ liệu Đơn hàng đã nâng cấp (Cột BBS được đánh dấu A/B)")
        
        filter_bbs = st.multiselect("Bộ lọc loại BBS:", ["A", "B"], default=["A", "B"], key="filter_bbs_sel")
        df_display = df_solved[df_solved["BBS"].isin(filter_bbs)].copy()
        
        def highlight_bbs(val):
            if val == "A":
                return "background-color: #ffe0b2; color: #e65100; font-weight: bold;"
            elif val == "B":
                return "background-color: #e0e0e0; color: #424242; font-weight: bold;"
            return ""

        styled_df = df_display.style.map(highlight_bbs, subset=["BBS"])
        st.dataframe(styled_df, use_container_width=True)

    # 3. Xuất file
    files = save_outputs(OUT_SOLVING, "DIM_ORDERS_SOLVED", "DIM_ORDERS_SOLVED", df_solved, extra_sheets={"THONG_KE_TINH": province_summary})
    
    st.divider()
    st.info(f"📁 Báo cáo đã được tự động lưu tại: `{os.path.relpath(files[0], BASE_DIR)}`")
    with open(files[0], "rb") as fh:
        st.download_button("⬇️ Tải Báo Cáo DIM_ORDERS_SOLVED.xlsx", fh.read(), file_name="DIM_ORDERS_SOLVED.xlsx", mime=XLSX_MIME, key="dl_orders_solved")

# ============================================================================
# ⚙️ MAIN CONTAINERS
# ============================================================================
TAB_NAMES = [
    "🚚 1. Fleet",
    "🏭 2. Warehouse",
    "📦 3. Product",
    "🧾 4. Orders",
    "⚡ 5. Orders Solving"
]

def main():
    st.title("🚚 Smart Logistics — Hệ thống chuẩn hóa & Phân tích Đơn hàng (Phạm Minh Thái)")
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
    with tabs[3]: render_orders_tab()
    with tabs[4]: render_orders_solving_tab()

if __name__ == "__main__":
    main()
