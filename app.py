# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
import os
import re
import time
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
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
        if seed_key not in st.session_state: st.session_state[seed_key] = seed_df.copy()
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
        "vehicle_id": ["VEH_01", "VEH_02"], "license_plate": ["29C-123.45", "29C-678.90"],
        "id_warehouse": ["WH_HN_01", "WH_HN_01"], "max_weight_kg": [5000, 2000],
        "max_volume_m3": [20, 10], "average_speed_kmh": [50, 45],
        "Max_Distance": [80, 100], "fixed_cost": [500000, 300000], "variable_cost": [5000, 4000],
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
DICT_LOOKUPS = {"order_type": TYPE_LK, "alert_status": ALERT_LK, "order_status": STATUS_LK}

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

def evaluate_transportation_and_time_constraints(route_or_order, df_vehicles, vehicle_available_counts, outsourcing_config):
    """
    Ràng buộc vận tải thông minh:
    1. Vượt toàn bộ hạm đội nhà -> Thuê ngoài Full (tính theo chuyến).
    2. Lớn hơn xe lớn nhất hạm đội -> Cắt full-fill xe lớn nhất (trừ 1 xe trong ngày), phần dư xét thuê ngoài Tiết kiệm 1 hoặc Tiết kiệm 2.
    3. Thỏa hạm đội -> Chọn xe NHỎ NHẤT khả thi (trừ 1 xe khả dụng trong ngày).
    4. Kiểm tra Max_Distance -> Cắt đơn backlog sang thuê ngoài tiết kiệm loại 2.
    """
    total_w = float(route_or_order.get("weight", route_or_order.get("total_weight_kg", 0.0)))
    total_v = float(route_or_order.get("volume", route_or_order.get("total_volume_m3", 0.0)))
    total_dist = float(route_or_order.get("km", route_or_order.get("total_distance_km", 0.0)))
    
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
    
    result = {
        "status": "APPROVED", "action_type": "IN_HOUSE",
        "assigned_vehicle": None, "outsourcing_type": None, "cost": 0.0, "message": ""
    }
    
    if total_w > sum_fleet_w or total_v > sum_fleet_v:
        result.update({
            "status": "OUTSOURCED", "action_type": "OUTSOURCE_FULL",
            "outsourcing_type": "Full Outsourcing",
            "cost": outsourcing_config.get("full_price", 2500000.0),
            "message": "🚨 Vượt quá tổng hạm đội nhà -> Thuê ngoài loại Full theo chuyến."
        })
        return result

    if total_w > max_fleet_w or total_v > max_fleet_v:
        largest_idx = df_vehicles["max_weight_kg"].idxmax()
        largest_vehicle = df_vehicles.loc[largest_idx]
        v_id = largest_vehicle["vehicle_id"]
        
        if vehicle_available_counts.get(v_id, 0) > 0:
            vehicle_available_counts[v_id] -= 1
        
        excess_w = total_w - largest_vehicle["max_weight_kg"]
        excess_v = total_v - largest_vehicle["max_volume_m3"]
        
        out_type, out_cost = "", 0.0
        if (20 <= excess_w <= 100) or (1 <= excess_v <= 5):
            out_type = "Tiết kiệm loại 1"
            out_cost = outsourcing_config.get("saving_1_price", 800000.0)
        elif excess_w < 20 or excess_v < 1:
            out_type = "Tiết kiệm loại 2"
            out_cost = outsourcing_config.get("saving_2_price", 300000.0)
        else:
            out_type = "Tiết kiệm loại 1"
            out_cost = outsourcing_config.get("saving_1_price", 800000.0)
            
        result.update({
            "status": "PARTIAL_SPLIT", "action_type": "MAX_VEHICLE_PLUS_OUTSOURCE",
            "assigned_vehicle": v_id, "outsourcing_type": out_type, "cost": out_cost,
            "message": f"⚠️ Vượt xe lớn nhất nhà ({v_id}). Cắt full-fill xe này (trừ 1 xe), phần dư đẩy sang thuê ngoài {out_type}."
        })
        return result

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
        result.update({
            "status": "OUTSOURCED", "action_type": "OUTSOURCE_NO_VEHICLE_AVAILABLE",
            "outsourcing_type": "Tiết kiệm loại 1",
            "cost": outsourcing_config.get("saving_1_price", 800000.0),
            "message": "⚠️ Đủ tải trọng nhưng hạm đội đã hết xe khả dụng trong ngày -> Thuê ngoài tiết kiệm loại 1."
        })
        return result
    
    chosen_v_id = chosen_vehicle["vehicle_id"]
    vehicle_available_counts[chosen_v_id] = vehicle_available_counts.get(chosen_v_id, 1) - 1
    
    max_distance_limit = float(chosen_vehicle.get("Max_Distance", 100.0))
    if total_dist > max_distance_limit:
        result.update({
            "status": "BACKLOG_TRIGGERED", "action_type": "DISTANCE_EXCEEDED_BACKLOG",
            "assigned_vehicle": chosen_v_id, "outsourcing_type": "Tiết kiệm loại 2 (Phần cắt đơn backlog)",
            "cost": outsourcing_config.get("saving_2_price", 300000.0),
            "message": f"⚠️ Tuyến vượt quá Max_Distance ({total_dist}km > {max_distance_limit}km) của xe {chosen_v_id}. Cắt đơn backlog sang thuê ngoài tiết kiệm loại 2."
        })
        return result
        
    result.update({
        "status": "APPROVED", "action_type": "IN_HOUSE_SUCCESS",
        "assigned_vehicle": chosen_v_id,
        "message": f"✅ Thỏa mãn hoàn toàn! Giao xe NHỎ NHẤT khả thi: {chosen_v_id} (Đã trừ 1 xe trong ngày)."
    })
    return result

def evaluate_route_time_constraint(route_data, service_time_rules=None):
    if service_time_rules is None:
        service_time_rules = {
            "B2C": {"loading": 25, "unloading": 35}, 
            "B2B": {"loading": 45, "unloading": 60}  
        }
    orders_in_route = route_data.get("orders", [])
    total_distance_km = route_data.get("total_distance_km", 0.0)
    vehicle_speed_kmh = route_data.get("vehicle_speed_kmh", 40.0) 
    current_date = route_data.get("current_date", "2026-04-03") 
    
    travel_time_hours = total_distance_km / vehicle_speed_kmh if vehicle_speed_kmh > 0 else 0.0
    total_service_minutes = sum(service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["loading"] + 
                                service_time_rules.get(o.get("order_type", "B2C"), service_time_rules["B2C"])["unloading"] 
                                for o in orders_in_route)
    total_route_duration_hours = travel_time_hours + (total_service_minutes / 60.0)
    
    if total_route_duration_hours <= 8.0:
        return {
            "status": "APPROVED", "message": "✅ Đạt yêu cầu thời gian tuyến (<= 8h)",
            "total_hours": round(total_route_duration_hours, 2), "route": route_data.get("route", [])
        }
    else:
        backlog_orders = [dict(o, backlog_days_count=o.get("backlog_days_count", 0) + 1, original_date=o.get("original_date", current_date)) for o in orders_in_route]
        return {
            "status": "BACKLOG_OR_REPOOL", "message": "⚠️ Tuyến vượt quá giới hạn 8h! Đẩy đơn sang pool xử lý ngầm.",
            "total_hours": round(total_route_duration_hours, 2), "backlog_orders_next_day": backlog_orders
        }

def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright & Ràng buộc Vận tải")
    
    with st.expander("⚙️ Cấu hình Chi phí Thuê ngoài & Tham số", expanded=True):
        c1, c2, c3 = st.columns(3)
        full_p = c1.number_input("Giá thuê ngoài Full (VNĐ)", min_value=0.0, value=2500000.0, step=100000.0)
        s1_p = c2.number_input("Giá thuê ngoài Tiết kiệm 1 (VNĐ)", min_value=0.0, value=800000.0, step=50000.0)
        s2_p = c3.number_input("Giá thuê ngoài Tiết kiệm 2 (VNĐ)", min_value=0.0, value=300000.0, step=50000.0)
    
    outsourcing_cfg = {"full_price": full_p, "saving_1_price": s1_p, "saving_2_price": s2_p}

    st.markdown("### 📂 Trạng thái dữ liệu đầu vào (Schema Contract)")
    rows = []
    for path, cols in SCHEMA_CONTRACT.items():
        full = os.path.join(BASE_DIR, path)
        status = "✅ Sẵn sàng" if os.path.exists(full) else "❌ Chưa có"
        rows.append({"File": path, "Trạng thái": status, "Cột chuẩn": ", ".join(cols)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    
    if st.button("🚀 Kiểm thử Ràng buộc Vận tải & Hạm đội", key="run_test_tab6", type="primary"):
        fleet_path = os.path.join(BASE_DIR, "output_fleet/DIM_VEHICLE.xlsx")
        if not os.path.exists(fleet_path):
            st.error("❌ Chưa tìm thấy file `DIM_VEHICLE.xlsx` trong thư mục `output_fleet`. Vui lòng chạy Tab 1 trước!")
            return
        df_veh = pd.read_excel(fleet_path)
        veh_counts = {row["vehicle_id"]: 2 for _, row in df_veh.iterrows()}
        sample_route = {"total_weight_kg": 2500.0, "total_volume_m3": 12.0, "total_distance_km": 75.0, "orders": [{"order_type": "B2C"}]}
        res = evaluate_transportation_and_time_constraints(sample_route, df_veh, veh_counts, outsourcing_cfg)
        st.success("🏁 Kiểm thử thành công!")
        st.json(res)

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
