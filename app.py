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
import requests
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
OUT_FLEET     = os.path.join(BASE_DIR, "output_fleet")
OUT_WAREHOUSE = os.path.join(BASE_DIR, "output_warehouse")
OUT_PRODUCT   = os.path.join(BASE_DIR, "output_product")
OUT_ORDERS    = os.path.join(BASE_DIR, "output_orders")
OUT_CUSTOMER  = os.path.join(BASE_DIR, "output_customer")
OUT_SOLVING   = os.path.join(BASE_DIR, "output_solving")
OUT_MATRIX    = os.path.join(BASE_DIR, "output_matrix")
OUT_ROUTING   = os.path.join(BASE_DIR, "output_routing")

LICENSE_RANKS = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4,
    "B1": 5, "B2": 6, "C": 7,
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
    js   = os.path.join(out_dir, f"{base}.json")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        for name, sdf in (extra_sheets or {}).items():
            if sdf is not None and len(sdf): sdf.to_excel(writer, sheet_name=name, index=False)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(df_to_records(df), fh, ensure_ascii=False, indent=2, default=str)
    return [xlsx, js]

def conf_icon_ui(score):
    return "🟢" if score >= 0.75 else ("🟡" if score >= 0.55 else "🟠")

def render_input_block(prefix, label, seed_df, column_config=None):
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
            if prefix == "orders" and "shift" not in st.session_state[seed_key].columns:
                st.session_state[seed_key]["shift"] = "Ca Sáng"
        edited = st.data_editor(st.session_state[seed_key], num_rows="dynamic",
                                column_config=column_config, key=f"{prefix}_editor")
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
    st.session_state[f"{prefix}_raw"]  = raw
    st.session_state[f"{prefix}_meta"] = {f: m[1] for f, m in mapping.items()}
    st.session_state[f"{prefix}_why"]  = {f: (m[2] if len(m) > 2 else "") for f, m in mapping.items()}
    for f, m in mapping.items():
        st.session_state[f"{prefix}_col_{f}"] = str(m[0]) if m[0] is not None else NONE
    st.session_state.pop(f"{prefix}_result", None)
    return True

def render_mapping(prefix, fields, raw):
    meta = st.session_state.get(f"{prefix}_meta", {})
    why  = st.session_state.get(f"{prefix}_why",  {})
    choices = [NONE] + [str(c) for c in raw.columns]
    chosen  = {}
    cols    = st.columns(2)
    for i, fld in enumerate(fields):
        key = f"{prefix}_col_{fld}"
        if st.session_state.get(key) not in choices: st.session_state[key] = NONE
        with cols[i % 2]:
            chosen[fld] = st.selectbox(f"`{fld}` ← cột nào?", choices, key=key)
            score = meta.get(fld, 0.0)
            note  = why.get(fld, "")
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
        "df": optimize_dtypes(df, categorize=True), "files": list(files),
        "summary": summary, "metrics": metrics,
        "extra": extra or {}, "ram_before_kb": mem_kb(df),
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
            c.download_button(f"⬇️ Tải {os.path.basename(path)}", data,
                              file_name=os.path.basename(path), mime=mime,
                              key=f"{prefix}_dl_{os.path.basename(path)}")
    return res

# ============================================================================
# 🚚 2. TAB 1: FLEET
# ============================================================================
VEHICLE_FIELDS = {
    "vehicle_id":        ["mã xe","vehicle id","vehicle code","vehicle","xe","id xe","truck id"],
    "license_plate":     ["biển số","bien so","bsx","license plate","plate","số xe"],
    "id_warehouse":      ["kho","warehouse","wh","hub","chi nhánh","location","ma kho","id kho"],
    "max_weight_kg":     ["trọng tải","trong tai","weight","payload","khối lượng","kg","tấn","max weight"],
    "max_volume_m3":     ["thể tích","the tich","volume","m3","cbm","max volume"],
    "average_speed_kmh": ["vận tốc","van toc","speed","vận tốc trung bình","avg speed","kmh"],
    "Max_Distance":      ["khoảng cách tối đa","max distance","quãng đường tối đa","km tối đa","max km"],
    "fixed_cost":        ["chi phí cố định","fixed cost","cost fix"],
    "variable_cost":     ["chi phí biến đổi","variable cost","cost km"],
}
VEHICLE_NUMERIC = ["max_weight_kg","max_volume_m3","average_speed_kmh","Max_Distance","fixed_cost","variable_cost"]

def fleet_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in VEHICLE_NUMERIC:
        return float(values.str.replace(r"[^\d.]", "", regex=True).str.len().gt(0).mean())
    return float(values.nunique() / len(values))

def process_vehicle(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip())
           for f in ["vehicle_id","license_plate","id_warehouse"]}
    for f in VEHICLE_NUMERIC: out[f] = parse_num(rec.get(f))
    out["trạng_thái"] = "✅ Hợp lệ"
    return out

def fleet_validate_output(df):
    out    = df.copy()
    dup_id = out["vehicle_id"].astype(str).duplicated(keep=False)
    dup_pl = out["license_plate"].astype(str).duplicated(keep=False)
    msgs   = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("vehicle_id")):    errs.append("Thiếu vehicle_id")
        if is_blank(row.get("license_plate")): errs.append("Thiếu license_plate")
        if is_blank(row.get("id_warehouse")):  errs.append("Thiếu id_warehouse")
        if dup_id.iloc[i] and not is_blank(row.get("vehicle_id")):    errs.append("Trùng vehicle_id")
        if dup_pl.iloc[i] and not is_blank(row.get("license_plate")): errs.append("Trùng license_plate")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def fleet_empty_table():
    return pd.DataFrame({
        "vehicle_id":        ["VEH_01","VEH_02","VEH_06"],
        "license_plate":     ["29C-123.45","29C-678.90","29C-888.88"],
        "id_warehouse":      ["WH_HN_01","WH_HN_01","WH_HN_01"],
        "max_weight_kg":     [5000,2000,4000],
        "max_volume_m3":     [20,10,15],
        "average_speed_kmh": [50,45,40],
        "Max_Distance":      [80,100,120],
        "fixed_cost":        [500000,300000,400000],
        "variable_cost":     [5000,4000,4500],
    })

def render_fleet_tab():
    st.header("🚚 Quản lý Hạm đội Xe (Fleet Master Data)")
    mode, edited, uploaded = render_input_block("fleet","phương tiện",fleet_empty_table())
    if st.button("🔍 Quét & Semantic Mapping", key="fleet_scan", type="primary"):
        run_scan("fleet","phương tiện",mode,uploaded,edited,
                 lambda df: basic_semantic_mapping(df,VEHICLE_FIELDS,fleet_profile_score,0.35))
    raw = st.session_state.get("fleet_raw")
    if raw is None: return
    chosen = render_mapping("fleet",list(VEHICLE_FIELDS),raw)
    if st.button("🚀 Chuẩn hóa & Xử lý Fleet", key="fleet_process", type="primary"):
        colmap = build_colmap(raw, chosen)
        rows   = [process_vehicle({f: pick(r,colmap,f,None) for f in VEHICLE_FIELDS}) for _,r in raw.iterrows()]
        out    = fleet_validate_output(pd.DataFrame(rows))
        files  = save_outputs(OUT_FLEET,"DIM_VEHICLE","DIM_VEHICLE",out)
        ok     = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
        store_result("fleet",out,files,"🧭 Hoàn tất chuẩn hóa phương tiện",
                     [("Tổng loại xe",len(out)),("Hợp lệ",f"{ok}/{len(out)}")])
    render_result("fleet","📊 Kết quả Hạm đội")

# ============================================================================
# 🏭 3. TAB 2: WAREHOUSE
# ============================================================================
WAREHOUSE_FIELDS = {
    "id_warehouse": ["mã kho","warehouse id","warehouse code","warehouse","kho","id kho"],
    "address":      ["địa chỉ","địa điểm","address","location","vị trí"],
}

def coordinate_in_vietnam(lat, lng):
    try: lat, lng = float(lat), float(lng)
    except Exception: return False
    return VIETNAM_BOUNDS[0] <= lat <= VIETNAM_BOUNDS[1] and VIETNAM_BOUNDS[2] <= lng <= VIETNAM_BOUNDS[3]

ADDRESS_ABBR = [
    (r"\bTP\.?\b","Thành phố"),(r"\bQ\.?\b","Quận"),(r"\bH\.?\b","Huyện"),
    (r"\bTX\.?\b","Thị xã"),(r"\bTT\.?\b","Thị trấn"),(r"\bP\.?\b","Phường"),
    (r"\bX\.?\b","Xã"),(r"\bĐg\.?\b","Đường"),
]

def clean_address(address):
    if is_blank(address): return ""
    text = str(address).replace("\r"," ").replace("\n"," ").replace("\t"," ")
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u3000]"," ",text)
    text = re.sub(r"\s*[|;→–—]\s*",", ",text)
    text = re.sub(r"\s+-\s+",", ",text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s,./'-]"," ",text)
    for p, r in ADDRESS_ABBR: text = re.sub(p,r,text,flags=re.IGNORECASE)
    text = re.sub(r"\s+"," ",text).strip(" ,.")
    if text and not re.search(r"\bViệt Nam\b|\bVietnam\b",text,re.I): text += ", Việt Nam"
    return text

def address_quality(address):
    if not address: return 0.0,"❌ Địa chỉ trống"
    score = 0.5 if len(address) >= 10 else 0.25
    if "," in address: score += 0.3
    if re.search(r"\b(Phường|Xã|Quận|Huyện|Thành phố|Tỉnh)\b",address,re.I): score += 0.2
    return min(score,1.0),"✅ Hợp lệ"

@st.cache_resource(show_spinner=False)
def get_geocoder():
    try: return ArcGIS(user_agent="smart-logistics-warehouse/1.0",timeout=10)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def _geo_cache(): return {}

def geocode_address(address, ref_wh_lat=21.0285, ref_wh_lng=105.8542,
                    max_valid_km=150.0, retries=3):
    if not address: return {"ok":False,"status":"❌ Địa chỉ trống"}
    cache = _geo_cache()
    if address in cache: return cache[address]
    geocoder = get_geocoder()
    if geocoder is None: return {"ok":False,"status":"❌ Không khởi tạo được ArcGIS"}
    for attempt in range(1, retries+1):
        try:
            loc = geocoder.geocode(address,timeout=10)
            if loc is None: return {"ok":False,"status":"⚠️ Không tìm thấy địa chỉ"}
            lat, lng = float(loc.latitude), float(loc.longitude)
            if not coordinate_in_vietnam(lat,lng):
                return {"ok":False,"status":"⚠️ Tọa độ ngoài VN"}
            dist_to_wh = haversine(ref_wh_lat,ref_wh_lng,lat,lng)
            if dist_to_wh > max_valid_km:
                return {"ok":False,"status":f"⚠️ Geocode sai vị trí cách kho {dist_to_wh:.0f}km"}
            res = {"ok":True,"lat":lat,"lng":lng,
                   "display_name":getattr(loc,"address","") or "","score":1.0}
            cache[address] = res
            return res
        except Exception:
            if attempt == retries: break
            time.sleep(1)
    return {"ok":False,"status":"❌ Lỗi Geocoding"}

def process_warehouse(warehouse_id, address, do_geocode=True):
    raw_address = "" if is_blank(address) else str(address).strip()
    cleaned  = clean_address(raw_address)
    quality, _ = address_quality(cleaned)
    result = {"id_warehouse": "" if is_blank(warehouse_id) else str(warehouse_id).strip(),
              "address":cleaned,"lat":None,"lng":None,
              "chất_lượng_địa_chỉ":quality,"trạng_thái_geocode":"—"}
    if is_blank(warehouse_id) or not cleaned or not do_geocode: return result
    geo = geocode_address(cleaned,max_valid_km=10000.0)
    if not geo["ok"]:
        result["trạng_thái_geocode"] = geo["status"]; return result
    result.update({"lat":geo["lat"],"lng":geo["lng"],"trạng_thái_geocode":"✅ Thành công"})
    return result

def warehouse_validate_output(df):
    out = df.copy(); msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("id_warehouse")): errs.append("Thiếu id")
        if is_blank(row.get("address")):      errs.append("Thiếu địa chỉ")
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")): errs.append("Thiếu tọa độ")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_warehouse_tab():
    st.header("🏭 Kho & Tọa độ (Warehouse Master Data)")
    do_geo = st.checkbox("🌍 Bật ArcGIS Geocoding",value=True,key="wh_do_geo")
    mode, edited, uploaded = render_input_block(
        "wh","kho",
        pd.DataFrame({"id_warehouse":["WH_HN_01"],
                      "address":["Số 1 Tràng Tiền, Hoàn Kiếm, Hà Nội"]}))
    if st.button("🔍 Quét & Mapping",key="wh_scan",type="primary"):
        run_scan("wh","kho",mode,uploaded,edited,
                 lambda df: basic_semantic_mapping(df,WAREHOUSE_FIELDS,lambda s,f:0.8,0.40))
    raw = st.session_state.get("wh_raw")
    if raw is None: return
    chosen = render_mapping("wh",list(WAREHOUSE_FIELDS),raw)
    if st.button("🚀 Xử lý Kho & Geocoding",key="wh_process",type="primary"):
        colmap = build_colmap(raw, chosen)
        rows   = [process_warehouse(row.get(colmap.get("id_warehouse")),
                                    row.get(colmap.get("address")),do_geocode=do_geo)
                  for _, row in raw.iterrows()]
        out   = warehouse_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_WAREHOUSE,"WAREHOUSE_WITH_COORDINATES","DIM_WAREHOUSE",out)
        store_result("wh",out,files,"🧭 Hoàn tất quét tọa độ kho",[("Tổng số kho",len(out))])
    res = render_result("wh","📊 Kết quả Kho")
    if res is not None and not res["df"].dropna(subset=["lat","lng"]).empty:
        st.map(res["df"].dropna(subset=["lat","lng"]).rename(columns={"lng":"lon"})[["lat","lon"]])

# ============================================================================
# 📦 4. TAB 3: PRODUCT
# ============================================================================
PRODUCT_FIELDS = {
    "product_id":    ["mã sản phẩm","product id","sku","item code","mã sp","code","id"],
    "product_name":  ["tên sản phẩm","product name","item name","tên sp","name"],
    "volume":        ["thể tích","volume","m3","cbm","cm3"],
    "weight":        ["trọng lượng","weight","kg"],
    "cost_price":    ["giá sản xuất","cost price","cost"],
    "selling_price": ["giá bán","selling price","price"],
}
PRODUCT_NUMERIC = ["volume","weight","cost_price","selling_price"]

def process_product(rec):
    out = {f: ("" if is_blank(rec.get(f)) else str(rec.get(f)).strip())
           for f in ["product_id","product_name"]}
    for f in PRODUCT_NUMERIC: out[f] = parse_num(rec.get(f))
    out["trạng_thái"] = "✅ Hợp lệ"
    return out

def product_validate_output(df):
    out = df.copy(); msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("product_id")):   errs.append("Thiếu id")
        if is_blank(row.get("product_name")): errs.append("Thiếu tên")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def render_product_tab():
    st.header("📦 Quản lý Sản phẩm (Product Master Data)")
    mode, edited, uploaded = render_input_block(
        "product","sản phẩm",
        pd.DataFrame({"product_id":["P022"],"product_name":["Giường đôi 1m8"],
                      "volume":[10497600],"weight":[65.0],"selling_price":[8500000]}))
    if st.button("🔍 Quét & Mapping",key="product_scan",type="primary"):
        run_scan("product","sản phẩm",mode,uploaded,edited,
                 lambda df: basic_semantic_mapping(df,PRODUCT_FIELDS,lambda s,f:0.8,0.35))
    raw = st.session_state.get("product_raw")
    if raw is None: return
    chosen = render_mapping("product",list(PRODUCT_FIELDS),raw)
    if st.button("🚀 Chuẩn hóa Sản phẩm",key="product_process",type="primary"):
        colmap = build_colmap(raw, chosen)
        rows   = [process_product({f: pick(r,colmap,f,None) for f in PRODUCT_FIELDS})
                  for _,r in raw.iterrows()]
        out   = product_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_PRODUCT,"DIM_PRODUCT","DIM_PRODUCT",out)
        store_result("product",out,files,"🧭 Hoàn tất chuẩn hóa sản phẩm",[("Tổng sản phẩm",len(out))])
    render_result("product","📊 Kết quả Sản phẩm")

# ============================================================================
# 🧾 5. TAB 4: ORDERS
# ============================================================================
_ORDER_RAW = {
    "order_id":        dict(aliases=["mã đơn","order id","order code"]),
    "customer_id":     dict(aliases=["mã khách","customer id"]),
    "customer_name":   dict(aliases=["tên khách","customer name","name"]),
    "quantity":        dict(aliases=["số lượng","qty","quantity","units"]),
    "items":           dict(aliases=["mặt hàng","items","product","sản phẩm","product_id"]),
    "total_weight_kg": dict(aliases=["trọng lượng","weight","kg"]),
    "total_volume_m3": dict(aliases=["thể tích","volume","m3"]),
    "shift":           dict(aliases=["shift","ca","kíp","buổi","ca giao","ca làm việc","buổi giao"]),
    "address":         dict(aliases=["địa chỉ","address","destination","location","vị trí"]),
    "order_status":    dict(aliases=["trạng thái","status"]),
    "order_type":      dict(aliases=["loại đơn","order type","channel"]),
    "alert_status":    dict(aliases=["cảnh báo","alert","priority"]),
    "order_date":      dict(aliases=["ngày đặt","order date","date"]),
}
ORDER_FIELDS = {f: dict(label=f,aliases=[f]+v["aliases"]) for f,v in _ORDER_RAW.items()}
FIELDS = list(ORDER_FIELDS)
TYPE_TABLE  = {"B2C":["b2c","cá nhân","lẻ"], "B2B":["b2b","doanh nghiệp","sỉ"]}
ALERT_TABLE = {"Alert":["alert","khẩn","gấp"], "Normal":["normal","bình thường"]}
SHIFT_MAPPING_KEYWORDS = {
    "Ca Sáng": ["sáng","ca sáng","morning","ca 1","am","shift 1"],
    "Ca Chiều": ["chiều","ca chiều","afternoon","ca 2","pm","shift 2"],
    "Ca Đêm":  ["đêm","ca đêm","night","ca 3","shift 3"],
}

def normalize_shift(val):
    if is_blank(val): return "Ca Sáng"
    val_str = str(val).strip().lower()
    for std_shift, keywords in SHIFT_MAPPING_KEYWORDS.items():
        if any(kw in val_str for kw in keywords): return std_shift
    return str(val).strip().title()

def build_lookup(table):
    exact = {norm_basic(k): l for l, keys in table.items() for k in keys}
    return {"exact":exact,"keys":list(exact.keys())}

TYPE_LK, ALERT_LK = build_lookup(TYPE_TABLE), build_lookup(ALERT_TABLE)

def match_label(value, lk):
    n = norm_basic(str(value))
    if n in lk["exact"]: return lk["exact"][n]
    return None

def orders_semantic_mapping(df):
    result = {}
    for fld in FIELDS:
        best_col, best_score = None, 0.0
        aliases = ORDER_FIELDS[fld]["aliases"]
        for col in df.columns:
            sc = max(fuzz.token_set_ratio(norm_basic(col),norm_basic(a)) for a in aliases) / 100.0
            if sc > best_score: best_col, best_score = col, sc
        result[fld] = (best_col,best_score,"tên cột") if best_score > 0.3 else (None,0.0,"")
    return result

def load_product_lookup():
    prod_file = os.path.join(OUT_PRODUCT, "DIM_PRODUCT.xlsx")
    if not os.path.exists(prod_file):
        return {}
    try:
        df_prod = pd.read_excel(prod_file)
        lookup = {}
        for _, r in df_prod.iterrows():
            pid = str(r.get("product_id", "")).strip()
            pname = str(r.get("product_name", "")).strip()
            vol = parse_num(r.get("volume"))
            w = parse_num(r.get("weight"))
            if pid: lookup[pid] = {"volume": vol, "weight": w}
            if pname: lookup[pname] = {"volume": vol, "weight": w}
        return lookup
    except Exception:
        return {}

def process_order_with_geocode(raw, do_geocode=True, product_lookup=None):
    r_name   = str(raw.get("customer_name",""))
    cust_name = r_name or "Khách lẻ"
    t_type   = match_label(raw.get("order_type",""),TYPE_LK) or \
               ("B2B" if "cong ty" in norm_basic(r_name) else "B2C")
    t_alert  = match_label(raw.get("alert_status",""),ALERT_LK) or "Normal"
    q        = parse_qty(raw.get("quantity")) or 1.0
    shift_val= normalize_shift(raw.get("shift"))
    addr     = str(raw.get("address","")).strip()
    cleaned_addr = clean_address(addr)
    
    items_str = str(raw.get("items","")).strip()
    if not items_str or items_str == "nan": items_str = "Hàng tiêu chuẩn"

    w = parse_qty(raw.get("total_weight_kg"))
    v = parse_qty(raw.get("total_volume_m3"), dot_is_decimal=True)

    # ---------------------------------------------------------------------
    # LOGIC TỰ ĐỘNG TÍNH THỂ TÍCH & TRỌNG LƯỢNG TỪ TAB PRODUCT
    # Thể tích = (thể tích cm3 từ Product * quantity) / 1.000.000
    # ---------------------------------------------------------------------
    if product_lookup and items_str in product_lookup:
        p_info = product_lookup[items_str]
        p_vol_cm3 = float(p_info.get("volume", 0.0))
        p_weight  = float(p_info.get("weight", 0.0))
        
        if p_vol_cm3 > 0:
            v = (p_vol_cm3 * q) / 1_000_000.0
        if p_weight > 0:
            w = p_weight * q

    lat,lng,geo_status = None,None,"— Chưa quét"
    if do_geocode and cleaned_addr:
        geo = geocode_address(cleaned_addr)
        if geo["ok"]: lat,lng,geo_status = geo["lat"],geo["lng"],"✅ Thành công"
        else: geo_status = geo["status"]

    return {
        "order_id":     str(raw.get("order_id","")),
        "customer_id":  str(raw.get("customer_id","")),
        "customer_name":cust_name,
        "quantity":     int(q),
        "items":        items_str,
        "total_weight_kg": round(w or 0.0,4),
        "total_volume_m3": round(v or 0.0,6),
        "shift":        shift_val,
        "address":      cleaned_addr,
        "lat":lat,"lng":lng,
        "trạng_thái_geocode":geo_status,
        "order_status": str(raw.get("order_status","")) or "Mới tạo",
        "order_type":   t_type,
        "alert_status": t_alert,
        "order_date":   parse_date_iso(raw.get("order_date")),
    }

def orders_validate_output(df):
    out = df.copy(); msgs = []
    for _, row in out.iterrows():
        errs = []
        if is_blank(row.get("order_id")):    errs.append("Thiếu id")
        if is_blank(row.get("address")):     errs.append("Thiếu địa chỉ/location")
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")): errs.append("Chưa quét được tọa độ")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Hợp lệ")
    out["kiểm_tra"] = msgs
    return out

def is_hanoi_address(addr_str: str) -> bool:
    if not addr_str: return False
    norm = norm_basic(addr_str)
    return "ha noi" in norm or "hanoi" in norm

def render_orders_tab():
    st.header("🧾 Quản lý Đơn hàng & Phân Ca (Orders Master Data)")
    do_geo = st.checkbox("🌍 Tự động quét ArcGIS Geocoding cho Location/Address đơn hàng",
                         value=True,key="ord_do_geo")
    seed_orders = pd.DataFrame({
        "order_id":       ["ORD_001","ORD_002","ORD_003"],
        "customer_name":  ["Công ty A","Nguyễn Văn B","Trần Văn C"],
        "quantity":       [10,5,12],
        "items":          ["P011","P008","P022"],
        "total_weight_kg":[150.0,85.0,210.0],
        "total_volume_m3":[0.0,0.0,0.0],
        "shift":          ["Ca Sáng","Ca Chiều","Ca Sáng"],
        "address":        ["Số 1 Tràng Tiền, Hà Nội","123 Cầu Giấy, Hà Nội","Thành phố Bắc Ninh, Bắc Ninh"],
    })
    orders_column_config = {
        "shift": st.column_config.SelectboxColumn(
            "Ca Giao Hàng (Shift)",
            help="Chọn ca làm việc/giao hàng",
            options=["Ca Sáng","Ca Chiều","Ca Đêm"],
            required=True,
        )
    }
    mode, edited, uploaded = render_input_block("orders","đơn hàng",seed_orders,
                                                column_config=orders_column_config)
    if st.button("🔍 Quét & Mapping",key="orders_scan",type="primary"):
        run_scan("orders","đơn hàng",mode,uploaded,edited,orders_semantic_mapping)
    raw = st.session_state.get("orders_raw")
    if raw is None: return
    chosen = render_mapping("orders",FIELDS,raw)
    if st.button("🚀 Chuẩn hóa & Quét Tọa độ Đơn hàng",key="orders_process",type="primary"):
        colmap = build_colmap(raw, chosen)
        prod_lookup = load_product_lookup()
        rows   = []
        with st.spinner("Đang chuẩn hóa, tra cứu sản phẩm & quét ArcGIS Geocoding địa chỉ..."):
            for _, row in raw.iterrows():
                rows.append(process_order_with_geocode(
                    {f: row.get(c) for f,c in colmap.items()},
                    do_geocode=do_geo,
                    product_lookup=prod_lookup
                ))
        out   = orders_validate_output(pd.DataFrame(rows))
        files = save_outputs(OUT_ORDERS,"DIM_ORDERS","DIM_ORDERS",out)
        mask_failed = out["trạng_thái_geocode"] != "✅ Thành công"
        df_failed   = out[mask_failed].copy().reset_index(drop=True)
        df_success  = out[~mask_failed].copy()
        mask_hanoi  = df_success["address"].apply(is_hanoi_address)
        df_hanoi    = df_success[mask_hanoi].copy().reset_index(drop=True)
        df_outside  = df_success[~mask_hanoi].copy().reset_index(drop=True)
        file_hn  = save_outputs(OUT_ORDERS,"DIM_ORDERS_HA_NOI","DIM_ORDERS_HA_NOI",df_hanoi)
        file_out = save_outputs(OUT_ORDERS,"DIM_ORDERS_OUTSIDE_HN","DIM_ORDERS_OUTSIDE_HN",df_outside)
        file_err = save_outputs(OUT_ORDERS,"DIM_ORDERS_FAILED","DIM_ORDERS_FAILED",df_failed)
        files.extend([file_hn[0],file_out[0],file_err[0]])
        cust_df = out.dropna(subset=["lat","lng"])[
            ["customer_id","customer_name","address","lat","lng","trạng_thái_geocode"]
        ].drop_duplicates(subset=["customer_id"])
        save_outputs(OUT_CUSTOMER,"DATASET_CUSTOMER","DATASET_CUSTOMER",cust_df)
        extra = {"df_hanoi":optimize_dtypes(df_hanoi,categorize=True),
                 "df_outside":optimize_dtypes(df_outside,categorize=True),
                 "df_failed":optimize_dtypes(df_failed,categorize=True)}
        store_result("orders",out,files,"🧭 Hoàn tất chuẩn hóa & phân loại đơn hàng",
                     [("Tổng số đơn",len(out)),("Đơn Hà Nội",len(df_hanoi)),
                      ("Đơn Ngoài Hà Nội",len(df_outside)),("Lỗi Geocode",len(df_failed))],
                     extra=extra)
    res = st.session_state.get("orders_result")
    if res is not None:
        st.markdown("### 📊 Kết quả Phân Loại Đơn Hàng")
        st.success(res["summary"])
        if res["metrics"]:
            cols = st.columns(len(res["metrics"]))
            for c,(lab,val) in zip(cols,res["metrics"]): c.metric(lab,val)
        extra  = res.get("extra",{})
        df_hn  = extra.get("df_hanoi",pd.DataFrame())
        df_out = extra.get("df_outside",pd.DataFrame())
        df_fail= extra.get("df_failed",pd.DataFrame())
        t_hn,t_out,t_fail,t_all = st.tabs([
            f"🏢 1. Đơn tại Hà Nội ({len(df_hn)})",
            f"🌏 2. Đơn ngoài Hà Nội ({len(df_out)})",
            f"⚠️ 3. Không quét được tọa độ ({len(df_fail)})",
            f"📋 4. Tất cả đơn tổng ({len(res['df'])})",
        ])
        with t_hn:   st.dataframe(df_hn,  use_container_width=True)
        with t_out:  st.dataframe(df_out, use_container_width=True)
        with t_fail: st.dataframe(df_fail,use_container_width=True)
        with t_all:  st.dataframe(res["df"],use_container_width=True)
        st.caption(f"🧠 RAM kết quả: {mem_kb(res['df']):,.1f} KB")
        rel = ", ".join(f"`{os.path.relpath(p,BASE_DIR)}`" for p in res["files"])
        st.info(f"📁 Thư mục lưu các file: {rel}")
        cols = st.columns(min(len(res["files"]),4))
        for i,path in enumerate(res["files"]):
            if os.path.exists(path):
                with open(path,"rb") as fh: data = fh.read()
                mime = XLSX_MIME if path.endswith(".xlsx") else "application/json"
                cols[i%4].download_button(
                    f"⬇️ Tải {os.path.basename(path)}",data,
                    file_name=os.path.basename(path),mime=mime,
                    key=f"orders_dl_{os.path.basename(path)}_{i}")

# ============================================================================
# 🧩 6. TAB 5: ORDERS SOLVING
# ============================================================================
def render_orders_solving_tab():
    st.header("🧩 Tab 5: Orders Solving — Giải Mã 3D & Phân Chia Ca (Shift)")
    orders_res = st.session_state.get("orders_result")
    if orders_res is None:
        st.warning("⚠️ Vui lòng chạy chuẩn hóa dữ liệu Đơn hàng tại Tab 4 (Orders) trước.")
        return
    df_orders = orders_res["df"].copy()
    st.markdown("### 1. Thuật toán Giải Mã Kích Thước 3D & Phân Loại Hàng Cồng Kềnh (BBS A/B)")
    col_rule1,_ = st.columns(2)
    volumetric_divisor = col_rule1.number_input(
        "Hệ số quy đổi Volumetric Weight (cm³/kg)",value=4000.0,step=500.0)
    df_solved = df_orders.copy()
    df_solved["volumetric_weight_kg"] = (df_solved["total_volume_m3"] * 1_000_000.0) / volumetric_divisor
    def classify_bbs(row):
        return "Loại A (Hàng Cồng Kềnh - Cube Out)" \
            if row["volumetric_weight_kg"] > row["total_weight_kg"] \
            else "Loại B (Hàng Nặng - Weight Out)"
    df_solved["bbs_type"] = df_solved.apply(classify_bbs,axis=1)
    st.dataframe(df_solved[["order_id","customer_name","items","quantity","shift",
                             "total_weight_kg","total_volume_m3","volumetric_weight_kg","bbs_type"]],
                 use_container_width=True)
    save_outputs(OUT_SOLVING,"DIM_ORDERS_SOLVED","DIM_ORDERS_SOLVED",df_solved)
    st.divider()
    st.markdown("### 🗓️ 2. Tổng Hợp & Phân Chia Danh Sách Các Đơn Giao Theo Ca (Shift)")
    shift_summary = df_solved.groupby("shift").agg(
        total_orders=("order_id","count"),
        total_units=("quantity","sum"),
        total_weight_kg=("total_weight_kg","sum"),
        total_volume_m3=("total_volume_m3","sum"),
    ).reset_index()
    shift_summary["ratio_percent"] = round((shift_summary["total_orders"]/len(df_solved))*100,1)
    st.dataframe(shift_summary.rename(columns={
        "shift":"Ca Giao Hàng","total_orders":"Tổng Số Đơn","ratio_percent":"Tỷ Trọng (%)",
        "total_units":"Tổng Units","total_weight_kg":"Tổng KG","total_volume_m3":"Tổng m³",
    }), hide_index=True, use_container_width=True)
    unique_shifts = sorted(df_solved["shift"].unique())
    shift_tabs = st.tabs([f"🚚 Ca '{s}' ({len(df_solved[df_solved['shift']==s])} đơn)"
                          for s in unique_shifts])
    for tab, shift_name in zip(shift_tabs, unique_shifts):
        with tab:
            df_sub = df_solved[df_solved["shift"]==shift_name].reset_index(drop=True)
            st.dataframe(df_sub[["order_id","customer_name","address","items","quantity",
                                  "total_weight_kg","total_volume_m3","bbs_type","order_type"]],
                         use_container_width=True)

# ============================================================================
# 🗺️ 7. ROUTING ENGINE (SHARED)
# ============================================================================
OC_DEFAULTS = {
    "tier2_max_w": 20.0,  "tier2_max_v": 1.0,
    "tier1_max_w": 100.0, "tier1_max_v": 5.0,
    "full_vehicle_name": "Xe 3PL Full",
    "full_vehicle_max_weight_kg": 10_000.0,
    "full_vehicle_max_volume_m3": 40.0,
    "saving_1_price": 0.0, "saving_2_price": 0.0, "full_price": 0.0,
}

def haversine(lat1, lng1, lat2, lng2) -> float:
    p1,p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2-p1)/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lng2-lng1)/2)**2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def _full_cls(w, v, oc, why=None):
    fw = max(float(oc.get("full_vehicle_max_weight_kg",10000.0)),1e-9)
    fv = max(float(oc.get("full_vehicle_max_volume_m3",40.0)),1e-9)
    trips = max(1,math.ceil(w/fw-1e-9),math.ceil(v/fv-1e-9))
    if why is None:
        why = f"tải {w:,.0f}kg/{v:.1f}m³ vượt ngưỡng Tiết kiệm 1 ({trips} chuyến 3PL Full)"
    return {"type":f"3PL Full — {oc.get('full_vehicle_name','Xe 3PL Full')} × {trips} chuyến",
            "tier":"FULL","cost":float(oc.get("full_price",0.0)),
            "trips":trips,"vehicle":oc.get("full_vehicle_name","Xe 3PL Full"),
            "cap_w":trips*fw,"cap_v":trips*fv,"why":why}

def classify_outsourcing(excess_w, excess_v, oc=None, force_full=False) -> dict:
    oc = {**OC_DEFAULTS,**(oc or {})}
    w,v = max(float(excess_w),0.0), max(float(excess_v),0.0)
    if not force_full:
        if w < oc["tier2_max_w"] and v < oc["tier2_max_v"]:
            return {"type":"Tiết kiệm loại 2","tier":"SAVING_2","cost":float(oc["saving_2_price"]),
                    "trips":1,"vehicle":"Tiết kiệm loại 2",
                    "cap_w":float(oc["tier2_max_w Cor"] if "tier2_max_w Cor" in oc else oc["tier2_max_w"]),"cap_v":float(oc["tier2_max_v"]),
                    "why":f"tải {w:.1f}kg<{oc['tier2_max_w']:g}kg, {v:.2f}m³<{oc['tier2_max_v']:g}m³"}
        if w <= oc["tier1_max_w"] and v <= oc["tier1_max_v"]:
            return {"type":"Tiết kiệm loại 1","tier":"SAVING_1","cost":float(oc["saving_1_price"]),
                    "trips":1,"vehicle":"Tiết kiệm loại 1",
                    "cap_w":float(oc["tier1_max_w"]),"cap_v":float(oc["tier1_max_v"]),
                    "why":f"tải {w:.1f}kg≤{oc['tier1_max_w']:g}kg, {v:.2f}m³≤{oc['tier1_max_v']:g}m³"}
    return _full_cls(w,v,oc,"bắt buộc thuê Full" if force_full else None)

def evaluate_transportation_constraints(route_or_order, df_vehicles, vehicle_available_counts, outsourcing_config):
    oc = {**OC_DEFAULTS,**(outsourcing_config or {})}
    total_w    = float(route_or_order.get("total_weight_kg",0.0))
    total_v    = float(route_or_order.get("total_volume_m3",0.0))
    total_dist = float(route_or_order.get("total_distance_km",0.0))
    def outsourced(action,cls,message):
        return {"status":"OUTSOURCED","action_type":action,"assigned_vehicle":None,
                "outsourcing_type":cls["type"],"cost":cls["cost"],"cls":cls,"message":message}
    if df_vehicles.empty:
        return outsourced("NO_FLEET",_full_cls(total_w,total_v,oc,"bắt buộc khi kho không có xe nhà"),
                          "⚠️ Hạm đội trống -> Thuê ngoài Full.")
    max_fleet_w = df_vehicles["max_weight_kg"].max()
    max_fleet_v = df_vehicles["max_volume_m3"].max()
    sum_fleet_w = df_vehicles["max_weight_kg"].sum()
    sum_fleet_v = df_vehicles["max_volume_m3"].sum()
    avail = df_vehicles[df_vehicles["vehicle_id"].map(lambda x: vehicle_available_counts.get(x,1)>0)]
    if (total_w > 1.5*max_fleet_w or total_v > 1.5*max_fleet_v
            or total_w > sum_fleet_w or total_v > sum_fleet_v):
        return outsourced("OUTSOURCE_FULL",
                          _full_cls(total_w,total_v,oc,f"hàng quá khủng vượt 150% xe lớn nhất nhà"),
                          f"🚨 Hàng quá khủng -> Thuê ngoài 3PL Full.")
    if total_w > max_fleet_w or total_v > max_fleet_v:
        if avail.empty:
            return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE",classify_outsourcing(total_w,total_v,oc),
                              "⚠️ Hết xe khả dụng -> thuê ngoài.")
        big = avail.loc[avail["max_weight_kg"].idxmax()]
        v_id = big["vehicle_id"]
        vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id,1)-1
        cls = classify_outsourcing(total_w-big["max_weight_kg"],total_v-big["max_volume_m3"],oc)
        return {"status":"PARTIAL_SPLIT","action_type":"MAX_VEHICLE_PLUS_OUTSOURCE","assigned_vehicle":v_id,
                "outsourcing_type":cls["type"],"cost":cls["cost"],"cls":cls,
                "message":f"⚠️ Vượt xe lớn nhất nhà ({v_id}). Cắt xe này, phần dư thuê ngoài {cls['type']}."}
    feasible = (avail[(avail["max_weight_kg"]>=total_w)&(avail["max_volume_m3"]>=total_v)]
                .sort_values(by=["max_weight_kg","max_volume_m3"],ascending=True))
    if feasible.empty:
        return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE",classify_outsourcing(total_w,total_v,oc),
                          "⚠️ Hết xe khả dụng -> thuê ngoài.")
    veh = feasible.iloc[0]
    v_id  = veh["vehicle_id"]
    limit = float(veh.get("Max_Distance",100.0))
    if total_dist > limit:
        return {"status":"CUT_REQUIRED","action_type":"DISTANCE_EXCEEDED_CUT","assigned_vehicle":v_id,
                "outsourcing_type":None,"cost":0.0,"cls":None,"max_distance_limit":limit,
                "message":f"⚠️ Tuyến {total_dist:.1f}km > Max_Distance {limit:.0f}km của xe {v_id}. Cắt theo Units."}
    vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id,1)-1
    return {"status":"APPROVED","action_type":"IN_HOUSE_SUCCESS","assigned_vehicle":v_id,
            "outsourcing_type":None,"cost":0.0,"cls":None,
            "message":f"✅ Giao xe nhỏ nhất khả thi: {v_id}."}

def evaluate_time_constraints(route_or_order, chosen_vehicle=None,
                               service_time_rules=None, max_hours=8.0):
    if service_time_rules is None:
        service_time_rules = {"B2C":{"loading":25,"unloading":35},"B2B":{"loading":45,"unloading":60}}
    orders = route_or_order.get("orders",[])
    dist   = float(route_or_order.get("total_distance_km",0.0))
    speed  = float(chosen_vehicle.get("average_speed_kmh",40.0)) if chosen_vehicle else 40.0
    travel = dist/speed if speed > 0 else 0.0
    rule   = lambda o: service_time_rules.get(o.get("order_type","B2C"),service_time_rules["B2C"])
    total  = travel + sum(rule(o)["loading"]+rule(o)["unloading"] for o in orders)/60.0
    if total <= max_hours+1e-9:
        return {"status":"APPROVED","action_type":"TIME_APPROVED","total_hours":round(total,2),
                "message":f"✅ Đạt thời gian: {round(total,2)}h ≤ {max_hours:g}h."}
    return {"status":"CUT_TIME_EXCEEDED","action_type":"TIME_WINDOW_CUT","total_hours":round(total,2),
            "message":f"⚠️ Tuyến {round(total,2)}h > {max_hours:g}h! Cắt theo Units."}

@dataclass
class Config:
    start_time:          str   = "06:00"
    max_route_hours:     float = 8.0
    detour_factor:       float = 1.2
    constraint_mode:     str   = "Cube Out (Thể tích m³)"
    service_min:         dict  = field(default_factory=lambda: {"B2B":105,"B2C":60})
    backup_driver_cost:  float = 0.0
    vehicle_daily_count: int   = 2
    turnaround_min:      float = 30.0
    outsourced_speed_kmh:float = 40.0
    vehicle_file:  str = os.path.join(OUT_FLEET,    "DIM_VEHICLE.xlsx")
    warehouse_file:str = os.path.join(OUT_WAREHOUSE,"WAREHOUSE_WITH_COORDINATES.xlsx")
    product_file:  str = os.path.join(OUT_PRODUCT,  "DIM_PRODUCT.xlsx")
    order_file:    str = os.path.join(OUT_ORDERS,   "DIM_ORDERS.xlsx")
    cust_file:     str = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.xlsx")
    matrix_file:   str = os.path.join(OUT_MATRIX,   "DISTANCE_MATRIX_KM.xlsx")

    @property
    def outsourcing(self) -> dict:
        return {k: OC_DEFAULTS[k] for k in OC_DEFAULTS}

def _num_col(df, col, default=0.0):
    if col not in df.columns: return pd.Series(default,index=df.index,dtype=float)
    return pd.to_numeric(df[col],errors="coerce").fillna(default).astype(float)

def load_vehicles(cfg: Config) -> pd.DataFrame:
    df = read_dim_file(cfg.vehicle_file).copy()
    for c in ("vehicle_id","license_plate","id_warehouse"):
        df[c] = df[c].astype(str).str.strip() if c in df.columns else ""
    df = df[~df["vehicle_id"].isin(["","nan","None"])].copy()
    for c in VEHICLE_NUMERIC: df[c] = _num_col(df,c)
    df.loc[df["average_speed_kmh"]<=0,"average_speed_kmh"] = 40.0
    df.loc[df["Max_Distance"]<=0,"Max_Distance"] = 100.0
    df["vehicle_type"] = [f"{w:.0f}kg/{v:g}m3"
                          for w,v in zip(df["max_weight_kg"],df["max_volume_m3"])]
    return df.reset_index(drop=True)

def load_warehouse_coords(cfg: Config) -> dict:
    if not os.path.exists(cfg.warehouse_file): return {}
    out = {}
    for _,r in read_dim_file(cfg.warehouse_file).iterrows():
        wid = "" if is_blank(r.get("id_warehouse")) else str(r["id_warehouse"]).strip()
        if wid and pd.notna(r.get("lat")) and pd.notna(r.get("lng")):
            out[wid] = (float(r["lat"]),float(r["lng"]))
    return out

def load_customers(cfg: Config) -> dict:
    out = {}
    if not os.path.exists(cfg.cust_file): return out
    for _,r in read_dim_file(cfg.cust_file).iterrows():
        cid = str(r.get("customer_id","")).strip()
        if cid and cid != "nan" and pd.notna(r.get("lat")) and pd.notna(r.get("lng")):
            out[cid] = {"lat":float(r["lat"]),"lng":float(r["lng"]),
                        "name":"" if is_blank(r.get("customer_name")) else str(r["customer_name"])}
    return out

def load_orders_from_file(cfg: Config, notes: list) -> list:
    df = read_dim_file(cfg.order_file)
    for c in ("order_id","customer_id","order_date"):
        if c not in df.columns: raise UserError(f"DIM_ORDERS.xlsx thiếu cột `{c}`.")
    orders, skipped = [], 0
    for _,r in df.iterrows():
        oid,cid,dstr = ("" if is_blank(r.get(k)) else str(r[k]).strip()
                        for k in ("order_id","customer_id","order_date"))
        d_iso = parse_date_iso(dstr)
        if not (oid and cid and d_iso): skipped += 1; continue
        otype     = str(r.get("order_type","")).strip().upper()
        lat       = float(r["lat"]) if pd.notna(r.get("lat")) else None
        lng       = float(r["lng"]) if pd.notna(r.get("lng")) else None
        shift_val = normalize_shift(r.get("shift"))
        items_str = "" if is_blank(r.get("items")) else str(r["items"]).strip()
        orders.append({
            "order_id":      oid,"customer_id":cid,
            "order_type":    otype if otype in ("B2B","B2C") else "B2C",
            "quantity":      int(parse_qty(r.get("quantity")) or 1),
            "items":         items_str or "Hàng tiêu chuẩn",
            "total_weight_kg":parse_qty(r.get("total_weight_kg")),
            "total_volume_m3":parse_qty(r.get("total_volume_m3"),dot_is_decimal=True),
            "shift":         shift_val,
            "address":       "" if is_blank(r.get("address")) else str(r["address"]),
            "customer_name": "" if is_blank(r.get("customer_name")) else str(r["customer_name"]),
            "lat":lat,"lng":lng,
            "alert_status":  "Normal" if is_blank(r.get("alert_status")) else str(r["alert_status"]),
            "order_status":  "" if is_blank(r.get("order_status")) else str(r["order_status"]),
            "order_date":    d_iso,
        })
    if skipped: notes.append(f"Bỏ qua {skipped} đơn thiếu order_id/customer_id/order_date.")
    return orders

def load_data(cfg: Config) -> dict:
    notes = []
    veh       = load_vehicles(cfg)
    wh_coords = load_warehouse_coords(cfg)
    if not wh_coords:
        wh_coords = {"WH_DEFAULT":(21.0285,105.8542)}
        notes.append("Chưa có tọa độ kho (Tab 2) → dùng kho mặc định Hà Nội.")
    warehouses = {w:{"lat":la,"lng":lo,"name":w} for w,(la,lo) in wh_coords.items()}
    return {"vehicles":veh,"cust":load_customers(cfg),"warehouses":warehouses,
            "orders":load_orders_from_file(cfg,notes),"notes":notes}

def service_rules_from_cfg(cfg: Config) -> dict:
    base = {"B2C":{"loading":25,"unloading":35},"B2B":{"loading":45,"unloading":60}}
    for k in ("B2C","B2B"):
        total = base[k]["loading"]+base[k]["unloading"]
        want  = float(cfg.service_min.get(k,total))
        if abs(want-total) > 1e-9:
            ratio = base[k]["loading"]/total
            base[k] = {"loading":want*ratio,"unloading":want*(1-ratio)}
    return base

def calculate_effective_max_hours(start_time_str="06:00", max_hours_input=8.0):
    try:
        sh,sm = map(int,start_time_str.split(":"))
        start_in_hours = sh+sm/60.0
    except Exception:
        start_in_hours = 6.0
    cutoff_hours = 23.5
    return min(float(max_hours_input),max(0.0,cutoff_hours-start_in_hours))

_TAB_A_EVENTS = {
    "NO_FLEET":                    ("HIGH",   "KHO KHÔNG CÓ XE",   "Bổ sung xe cho kho"),
    "OUTSOURCE_FULL":              ("CRITICAL","VƯỢT TỔNG HẠM ĐỘI", "Cần xe 3PL Full"),
    "OUTSOURCE_NO_VEHICLE_AVAILABLE":("MEDIUM","HẾT XE TRONG NGÀY", "Bổ sung xe / dời lịch"),
    "MAX_VEHICLE_PLUS_OUTSOURCE":  ("CRITICAL","ĐƠN QUÁ CỠ",        "Cần xe 3PL Full"),
}

# ============================================================================
# 🗺️ OSRM — MA TRẬN ĐƯỜNG THỰC TỪ OPENSTREETMAP
# ============================================================================
OSRM_PUBLIC_URL = "http://router.project-osrm.org/table/v1/driving"
MAX_OSRM_POINTS = 100

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _fetch_osrm_matrix_cached(point_ids: tuple, coords_flat: tuple) -> dict:
    """Cached OSRM table call. coords_flat = (lat0,lng0,lat1,lng1,...)"""
    n     = len(point_ids)
    coords= [(coords_flat[i*2], coords_flat[i*2+1]) for i in range(n)]
    coord_str = ";".join(f"{lng},{lat}" for lat,lng in coords)
    url   = f"{OSRM_PUBLIC_URL}/{coord_str}?annotations=distance"
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if data.get("code") != "Ok":
            return {}
        distances = data.get("distances") or []
        result = {}
        for i, id_a in enumerate(point_ids):
            row_i = distances[i] if i < len(distances) else []
            for j, id_b in enumerate(point_ids):
                val = row_i[j] if j < len(row_i) else None
                if val is not None:
                    result[(id_a, id_b)] = val / 1000.0   # m → km
        return result
    except requests.Timeout:
        return {}
    except Exception:
        return {}

def build_osrm_matrix(warehouses: dict, customers: dict,
                      orders: list = None) -> tuple[dict, str]:
    all_custs = dict(customers)
    for o in (orders or []):
        cid = o.get("customer_id")
        if cid and o.get("lat") and o.get("lng") and cid not in all_custs:
            all_custs[cid] = {"lat":float(o["lat Corbin"] if "lat Corbin" in o else o["lat"]),"lng":float(o["lng"]),"name":cid}
    ids    = list(warehouses.keys())
    coords = [(v["lat"],v["lng"]) for v in warehouses.values()]
    for cid,c in all_custs.items():
        if c.get("lat") and c.get("lng"):
            ids.append(cid)
            coords.append((c["lat"],c["lng"]))
    if len(ids) < 2:
        return {}, "⚠️ Không đủ điểm để xây dựng ma trận OSRM."
    if len(ids) > MAX_OSRM_POINTS:
        return {}, (f"⚠️ Quá nhiều điểm ({len(ids)} > {MAX_OSRM_POINTS}). "
                    f"Dùng Haversine × detour_factor thay thế.")
    flat   = tuple(v for lat,lng in coords for v in (lat,lng))
    matrix = _fetch_osrm_matrix_cached(tuple(ids), flat)
    if not matrix:
        return {}, "❌ OSRM không phản hồi hoặc lỗi. Dùng Haversine thay thế."
    return matrix, (f"✅ Ma trận OSRM: {len(ids)} điểm — "
                    f"{len(matrix)} khoảng cách đường thực (km)")

# ============================================================================
# 🧠 ROUTE PLANNER
# ============================================================================
class RoutePlanner:
    def __init__(self, data: dict, cfg: Config, osrm_matrix: dict = None):
        self.cfg         = cfg
        self.osrm_matrix = osrm_matrix or {}
        self.veh  = data["vehicles"]
        self.cust = data["cust"]
        self.wh   = data["warehouses"]
        self.detour    = cfg.detour_factor
        self.max_hours = calculate_effective_max_hours(cfg.start_time, cfg.max_route_hours)
        if self.veh.empty:
            self.catalog = pd.DataFrame(columns=["w","v","speed","fixed","var","dist"])
        else:
            self.catalog = (
                self.veh.groupby("vehicle_type")
                .agg(w=("max_weight_kg","max"),v=("max_volume_m3","max"),
                     speed=("average_speed_kmh","first"),
                     fixed=("fixed_cost","first"),var=("variable_cost","first"),
                     dist=("Max_Distance","max"))
                .sort_values("w"))
        self.max_w    = float(self.catalog["w"].max())    if not self.catalog.empty else 1000.0
        self.max_v    = float(self.catalog["v"].max())    if not self.catalog.empty else 5.0
        self.max_dist = float(self.catalog["dist"].max()) if not self.catalog.empty else 100.0
        self._fleet   = {}
        for rec in self.veh.to_dict("records"):
            self._fleet.setdefault(rec["id_warehouse"],[]).append(rec)
        for lst in self._fleet.values():
            lst.sort(key=lambda r:(r["max_weight_kg"],r["max_volume_m3"]))

    def _dist(self, id_a: str, a: dict, id_b: str, b: dict) -> float:
        key = (id_a, id_b)
        if key in self.osrm_matrix:
            return self.osrm_matrix[key]
        return haversine(a["lat"],a["lng"],b["lat"],b["lng"]) * self.detour

    def d_cc(self, i, j) -> float:
        if i == j: return 0.0
        if i in self.cust and j in self.cust:
            return self._dist(i,self.cust[i],j,self.cust[j])
        return 0.0

    def d_wc(self, wh_id, c) -> float:
        if wh_id in self.wh and c in self.cust:
            return self._dist(wh_id,self.wh[wh_id],c,self.cust[c])
        return 0.0

    def nearest_wh(self, c) -> str:
        return min(self.wh, key=lambda w: self.d_wc(w,c))

    def km(self, route, wh_id) -> float:
        if not route: return 0.0
        return (self.d_wc(wh_id,route[0])
                + self.d_wc(wh_id,route[-1])
                + sum(self.d_cc(a,b) for a,b in zip(route,route[1:])))

    def caps(self, wh_id):
        lst = self._fleet.get(wh_id)
        if not lst: return self.max_w, self.max_v, self.max_dist
        return (max(r["max_weight_kg"] for r in lst),
                max(r["max_volume_m3"] for r in lst),
                max(r["Max_Distance"]  for r in lst))

    def smallest_vehicle(self, wh_id, w, v):
        for r in self._fleet.get(wh_id,[]):
            if r["max_weight_kg"] >= w and r["max_volume_m3"] >= v: return r
        return None

    def fit_type(self, w, v):
        if self.catalog.empty: return "Truck"
        ok = self.catalog[(self.catalog["w"]>=w)&(self.catalog["v"]>=v)]
        return ok.index[0] if not ok.empty else self.catalog.index[-1]

    def metrics(self, route, wh_id, demand):
        w   = sum(demand[c]["weight"] for c in route)
        v   = sum(demand[c]["volume"] for c in route)
        qty = sum(demand[c]["qty"]    for c in route)
        km  = self.km(route,wh_id)
        veh   = self.smallest_vehicle(wh_id,w,v)
        vtype = veh["vehicle_type"] if veh else self.fit_type(w,v)
        speed = (float(veh["average_speed_kmh"]) if veh
                 else (float(self.catalog.loc[vtype,"speed"])
                       if vtype in self.catalog.index else 35.0))
        service = sum(demand[c]["service_min"] for c in route)/60.0
        return {"w":w,"v":v,"qty":qty,"km":km,
                "hours":km/speed+service,"vtype":vtype,"speed":speed,"veh":veh}

    def feasible(self, route, wh_id, demand) -> bool:
        cap_w,cap_v,_ = self.caps(wh_id)
        m = self.metrics(route,wh_id,demand)
        if m["w"]>cap_w or m["v"]>cap_v or m["hours"]>self.max_hours: return False
        if m["veh"] is None: return not self._fleet.get(wh_id)
        return m["km"] <= m["veh"]["Max_Distance"]

    def route_cost(self, route, wh_id, demand) -> float:
        m   = self.metrics(route,wh_id,demand)
        veh = m["veh"]
        if veh is None:
            return float(classify_outsourcing(m["w"],m["v"],self.cfg.outsourcing)["cost"])
        return float(veh.get("fixed_cost",0.0))+float(veh.get("variable_cost",0.0))*m["km"]

    def two_opt(self, route, wh_id):
        def edge(a,b):
            if a is None: return self.d_wc(wh_id,b)
            if b is None: return self.d_wc(wh_id,a)
            return self.d_cc(a,b)
        best,improved = route[:],True
        while improved and len(best)>2:
            improved = False
            base = self.km(best,wh_id)
            for i in range(len(best)-1):
                for j in range(i+1,len(best)):
                    prev = None if i==0         else best[i-1]
                    nxt  = None if j==len(best)-1 else best[j+1]
                    old  = edge(prev,best[i])+edge(best[j],nxt)
                    new  = edge(prev,best[j])+edge(best[i],nxt)
                    if new >= old-1e-9: continue
                    cand    = best[:i]+best[i:j+1][::-1]+best[j+1:]
                    cand_km = self.km(cand,wh_id)
                    if cand_km < base-1e-9:
                        best,base,improved = cand,cand_km,True
        return best

    def savings_list(self, custs, wh_id):
        out = [(self.d_wc(wh_id,a)+self.d_wc(wh_id,b)-self.d_cc(a,b),a,b)
               for k,a in enumerate(custs) for b in custs[k+1:]]
        out.sort(key=lambda t:(-t[0],str(t[1]),str(t[2])))
        return out

    def clarke_wright(self, custs, wh_id, demand):
        route_of = {c:[c] for c in custs}
        for s,a,b in self.savings_list(custs,wh_id):
            if s<=0: break
            ra,rb = route_of[a],route_of[b]
            if ra is rb or a not in (ra[0],ra[-1]) or b not in (rb[0],rb[-1]): continue
            ra = ra if ra[-1]==a else ra[::-1]
            rb = rb if rb[0]==b  else rb[::-1]
            merged = ra+rb
            if not self.feasible(merged,wh_id,demand): continue
            if (self.route_cost(merged,wh_id,demand)
                    > self.route_cost(ra,wh_id,demand)+self.route_cost(rb,wh_id,demand)+1e-9): continue
            for c in merged: route_of[c] = merged
        return [self.two_opt(r,wh_id) for r in {id(r):r for r in route_of.values()}.values()]

    def _priority(self, route, wh_id, demand):
        w = sum(demand[c]["weight"] for c in route)
        v = sum(demand[c]["volume"] for c in route)
        return (-float(classify_outsourcing(w,v,self.cfg.outsourcing)["cost"]),-self.km(route,wh_id))

    def _demand(self, ords):
        demand,by_cust = {},{}
        for o in ords:
            d = demand.setdefault(o["customer_id"],{"weight":0.0,"volume":0.0,"qty":0,"service_min":0.0})
            d["weight"] += o["total_weight_kg"]; d["volume"] += o["total_volume_m3"]
            d["qty"]    += o.get("quantity",1)
            d["service_min"] += self.cfg.service_min.get(o["order_type"],60)
            by_cust.setdefault(o["customer_id"],[]).append(o)
        return demand,by_cust

    def _worst_stop_by_units(self, r, wh_id, demand):
        mode = getattr(self.cfg,"constraint_mode","Cube Out (Thể tích m³)")
        base = self.km(r,wh_id)
        if "Weight Out" in mode:
            return max(r,key=lambda c:(demand[c]["weight"]/max(demand[c]["qty"],1),
                                       base-self.km([x for x in r if x!=c],wh_id)))
        return max(r,key=lambda c:(demand[c]["volume"]/max(demand[c]["qty"],1),
                                   base-self.km([x for x in r if x!=c],wh_id)))

    def _start(self, date_str, start_str=None):
        return dt.datetime.combine(
            pd.to_datetime(date_str).date(),
            dt.datetime.strptime(start_str or self.cfg.start_time,"%H:%M").time())

    def _calculate_load_factor(self, m, vrow, cls):
        mode = getattr(self.cfg,"constraint_mode","Cube Out (Thể tích m³)")
        if "Cube Out" in mode:
            cap_v = float(vrow["max_volume_m3"]) if vrow is not None else float(cls.get("cap_v",40.0))
            return min(m["v"]/cap_v,1.0) if cap_v>0 else 0.0
        else:
            cap_w = float(vrow["max_weight_kg"]) if vrow is not None else float(cls.get("cap_w",10000.0))
            return min(m["w"]/cap_w,1.0) if cap_w>0 else 0.0

    def _outsource_route(self, custs, wh_id, demand, by_cust,
                         cls, events, date_str, km=None, reason=None):
        cfg = self.cfg
        if km is None:
            custs = self.two_opt(list(custs),wh_id); km = self.km(custs,wh_id)
        w = sum(demand[c]["weight"] for c in custs)
        v = sum(demand[c]["volume"] for c in custs)
        q = sum(demand[c]["qty"]    for c in custs)
        hours  = km/cfg.outsourced_speed_kmh + sum(demand[c]["service_min"] for c in custs)/60.0
        start  = self._start(date_str)
        reason = reason or ("; ".join(e.get("why",e["detail"]) for e in events) or "vượt khả năng xe nhà")
        reason = f"{reason} → {cls['why']}"
        load_f = self._calculate_load_factor({"w":w,"v":v},None,cls)
        return {"kind":"OUTSOURCE","id_warehouse":wh_id,"route":list(custs),
                "orders":[o for c in custs for o in by_cust[c]],
                "vehicle_id":f"3PL-{cls['tier']}","license_plate":f"Thuê ngoài 3PL ({cls['type']})",
                "vehicle_type":cls["vehicle"],"external":True,
                "km":round(km,1),"speed_kmh":cfg.outsourced_speed_kmh,
                "load_factor":load_f,
                "fixed_cost":0.0,"variable_cost":0.0,"outsourcing_cost":float(cls["cost"]),
                "outsourcing_type":cls["type"],
                "cap_w":float(cls["cap_w"]),"cap_v":float(cls.get("cap_v",40.0)),
                "total_weight_kg":w,"total_volume_m3":v,"total_qty":q,
                "reason":reason,"trip_no":None,
                "start":start,"end":start+dt.timedelta(hours=hours),
                "hours":hours,"events":events}

    def _dispatch(self, route, wh_id, demand, by_cust, veh_counts, date_str, veh_free=None):
        cfg,oc = self.cfg,self.cfg.outsourcing
        veh_free = {} if veh_free is None else veh_free
        rules    = service_rules_from_cfg(cfg)
        fleet    = self.veh[self.veh["id_warehouse"]==wh_id]
        r,cut,final,blocked = list(route),[],None,{}
        while r:
            m    = self.metrics(r,wh_id,demand)
            snap = dict(veh_counts)
            ta   = evaluate_transportation_constraints(
                {"total_weight_kg":m["w"],"total_volume_m3":m["v"],"total_distance_km":m["km"]},
                fleet,veh_counts,oc)
            action,vid = ta["action_type"],ta.get("assigned_vehicle")
            vrow = fleet[fleet["vehicle_id"]==vid].iloc[0] \
                   if (vid and ta["status"] != "OUTSOURCED") else None
            start_str = veh_free.get(vid,cfg.start_time) if vrow is not None else cfg.start_time
            max_h     = calculate_effective_max_hours(start_str,cfg.max_route_hours)
            bad       = None
            if action == "DISTANCE_EXCEEDED_CUT":
                bad = ("VƯỢT MAX_DISTANCE",ta["message"],
                       f"tuyến {m['km']:.1f}km > Max_Distance {ta['max_distance_limit']:.0f}km của xe {vid}")
            elif vrow is not None:
                if m["km"] > float(vrow["Max_Distance"]):
                    bad = ("VƯỢT MAX_DISTANCE",
                           f"Tuyến {m['km']:.1f}km > Max_Distance {float(vrow['Max_Distance']):.0f}km",
                           f"tuyến {m['km']:.1f}km > Max_Distance {float(vrow['Max_Distance']):.0f}km của xe {vid}")
                else:
                    tb = evaluate_time_constraints(
                        {"total_distance_km":m["km"],"orders":[o for c in r for o in by_cust[c]],
                         "current_date":date_str},
                        {"average_speed_kmh":float(vrow["average_speed_kmh"])},rules,max_h)
                    if tb["status"] != "APPROVED" and start_str != cfg.start_time and vid not in blocked:
                        veh_counts.clear(); veh_counts.update(snap)
                        blocked[vid] = veh_counts.get(vid,0); veh_counts[vid] = 0
                        continue
                    if tb["status"] != "APPROVED":
                        bad = ("VƯỢT GIỜ TUYẾN",tb["message"],
                               f"tuyến {tb['total_hours']:.2f}h > {max_h:g}h (xe {vid})")
            if bad:
                veh_counts.clear(); veh_counts.update(snap)
                c   = self._worst_stop_by_units(r,wh_id,demand)
                cut.append((c,bad[0],bad[1],bad[2]))
                r   = self.two_opt([x for x in r if x!=c],wh_id)
                continue
            final = (m,ta,action,vid,vrow,
                     tb if (vrow is not None and not bad) else None, start_str)
            break
        for v_,n_ in blocked.items(): veh_counts[v_] = n_
        out = []
        if final:
            m,ta,action,vid,vrow,tb,start_str = final
            if ta["status"] == "OUTSOURCED":
                sev,kind,propose = _TAB_A_EVENTS[action]
                why = f"kho {wh_id} không có xe phù hợp (tải {m['w']:,.0f}kg/{m['v']:.1f}m³)"
                ev  = [{"sev":sev,"kind":kind,"detail":ta["message"],"why":why,
                        "handled":f"Thuê ngoài {ta['cls']['type']}","propose":propose}]
                out.append(self._outsource_route(r,wh_id,demand,by_cust,ta["cls"],ev,date_str,reason=why))
            else:
                split = action == "MAX_VEHICLE_PLUS_OUTSOURCE"
                start = self._start(date_str,start_str)
                hours = (tb["total_hours"] if tb else
                         m["km"]/float(vrow["average_speed_kmh"])
                         + sum(demand[c]["service_min"] for c in r)/60.0)
                cap_w,cap_v = float(vrow["max_weight_kg"]),float(vrow["max_volume_m3"])
                ev,reason   = [],""
                if split:
                    sev,kind,propose = _TAB_A_EVENTS[action]
                    reason = (f"tải {m['w']:,.0f}kg/{m['v']:.1f}m³ vượt xe lớn nhất nhà "
                              f"({vid}: {cap_w:,.0f}kg/{cap_v:g}m³)")
                    ev.append({"sev":sev,"kind":kind,"detail":ta["message"],"why":reason,
                               "handled":f"Cắt xe {vid} + thuê ngoài {ta['cls']['type']}","propose":propose})
                orders    = [o for c in r for o in by_cust[c]]
                end_dt    = start+dt.timedelta(hours=hours)
                next_start= end_dt+dt.timedelta(minutes=cfg.turnaround_min)
                veh_free[vid] = (f"{next_start:%H:%M}" if next_start.date()==start.date() else "23:59")
                load_f = self._calculate_load_factor(m,vrow,{})
                out.append({
                    "kind":"SPLIT" if split else "NORMAL",
                    "id_warehouse":wh_id,"route":list(r),"orders":orders,
                    "vehicle_id":vid,"license_plate":vrow["license_plate"],
                    "vehicle_type":vrow["vehicle_type"],"external":False,
                    "km":round(m["km"],1),"speed_kmh":float(vrow["average_speed_kmh"]),
                    "load_factor":load_f,
                    "fixed_cost":float(vrow["fixed_cost"]),
                    "variable_cost":float(vrow["variable_cost"])*m["km"],
                    "outsourcing_cost":float(ta["cost"]) if split else 0.0,
                    "outsourcing_type":ta["outsourcing_type"] if split else None,
                    "cap_w":cap_w,"cap_v":cap_v,
                    "total_weight_kg":m["w"],"total_volume_m3":m["v"],"total_qty":m["qty"],
                    "reason":reason if split else "",
                    "trip_no":max(1,self.cfg.vehicle_daily_count-veh_counts.get(vid,0)),
                    "start":start,"end":end_dt,"hours":hours,"events":ev,
                })
        if cut:
            ids  = [c for c,_,_,_ in cut]
            cls  = classify_outsourcing(
                sum(demand[c]["weight"] for c in ids),
                sum(demand[c]["volume"] for c in ids),oc)
            ev   = [{"sev":"HIGH","kind":kind,"detail":f"Đơn {c}: {detail}",
                     "why":f"đơn {c} bị cắt do {short}",
                     "handled":f"Thuê ngoài {cls['type']}","propose":"Kiểm tra Max_Distance xe",
                     "ids":[c]}
                    for c,kind,detail,short in cut]
            out.append(self._outsource_route(ids,wh_id,demand,by_cust,cls,ev,date_str,
                                             reason="; ".join(f"đơn {c} cắt do {s}" for c,_,_,s in cut)))
        return out

    def plan_day(self, date_str: str, day_orders: list) -> dict:
        res = {"routes":[],"overdue_routes":[],"unrouted_orders":[],
               "exceptions":[],"day_cost":0.0,"orders":list(day_orders)}
        def log(r):
            for e in r.pop("events"):
                for o in r["orders"]:
                    if e.get("ids") is None or o["customer_id"] in e["ids"]:
                        res["exceptions"].append({
                            "NGÀY":date_str,"MỨC ĐỘ":e["sev"],"MÃ ĐƠN":o["order_id"],
                            "PHÂN LOẠI":e["kind"],"CHI TIẾT":e["detail"],
                            "ĐÃ XỬ LÝ":e["handled"],"ĐỀ XUẤT":e["propose"]})
        placed   = [o for o in day_orders if o.get("lat") is not None and o.get("lng") is not None]
        unplaced = [o for o in day_orders if o.get("lat") is None or o.get("lng") is None]
        for o in placed:
            self.cust[o["customer_id"]] = {"lat":float(o["lat"]),"lng":float(o["lng"]),"name":o["customer_id"]}
        veh_counts = {vid:self.cfg.vehicle_daily_count for vid in self.veh["vehicle_id"]}
        veh_free   = {}
        by_wh      = {}
        for o in placed: by_wh.setdefault(self.nearest_wh(o["customer_id"]),[]).append(o)
        for wh_id,ords in by_wh.items():
            demand,by_cust = self._demand(ords)
            routes = self.clarke_wright(list(demand),wh_id,demand)
            routes.sort(key=lambda rt:self._priority(rt,wh_id,demand))
            for rt in routes:
                for r in self._dispatch(rt,wh_id,demand,by_cust,veh_counts,date_str,veh_free):
                    log(r); res["routes"].append(r)
        res["unrouted_orders"] = unplaced
        res["day_cost"]        = sum(_route_total(r) for r in res["routes"])
        return res

def _route_total(r):
    return (r.get("fixed_cost",0)+r.get("variable_cost",0)
            +r.get("outsourcing_cost",0)+r.get("driver_cost",0))

# ============================================================================
# 🗺️ TAB 6: ROUTING — HELPER FUNCTIONS
# ============================================================================
def _mins(time_str: str) -> int:
    try:
        h,m = map(int, time_str.strip().split(":"))
        return h*60+m
    except Exception:
        return 0

def plan_by_shifts(data: dict, cfg: Config,
                   cfg_shifts: list, osrm_matrix: dict = None) -> dict:
    if not cfg_shifts:
        cfg_shifts = [{"name":"Ca 1","start":"06:00","end":"17:30"}]
    shift_timing = {}
    for s in cfg_shifts:
        try:
            s_min = _mins(s["start"])
            e_min = _mins(s["end"])
            shift_timing[s["name"]] = {
                "start":     s["start"],
                "max_hours": max((e_min - s_min)/60.0, 0.5),
            }
        except Exception:
            shift_timing[s["name"]] = {"start":"06:00","max_hours":8.0}
    default_shift = cfg_shifts[0]["name"]
    by_date: dict = {}
    for o in data["orders"]:
        by_date.setdefault(o["order_date"],[]).append(o)
    all_day_results: dict = {}
    for date_str in sorted(by_date):
        date_orders = by_date[date_str]
        shift_groups: dict = {}
        for o in date_orders:
            sname = o.get("shift", default_shift)
            if sname not in shift_timing:
                sname = default_shift
            shift_groups.setdefault(sname,[]).append(o)
        tagged_all: list  = []
        shift_results: dict = {}
        all_exceptions: list = []
        for sname in [s["name"] for s in cfg_shifts]:
            s_orders = shift_groups.get(sname, [])
            if not s_orders:
                continue
            timing = shift_timing[sname]
            shift_cfg = Config(
                start_time      = timing["start"],
                max_route_hours = timing["max_hours"],
                detour_factor   = cfg.detour_factor,
                constraint_mode = cfg.constraint_mode,
                service_min     = cfg.service_min,
                vehicle_daily_count = cfg.vehicle_daily_count,
                turnaround_min  = cfg.turnaround_min,
                vehicle_file    = cfg.vehicle_file,
                warehouse_file  = cfg.warehouse_file,
                product_file    = cfg.product_file,
                order_file      = cfg.order_file,
                cust_file       = cfg.cust_file,
            )
            planner = RoutePlanner(data, shift_cfg, osrm_matrix=osrm_matrix or {})
            day     = planner.plan_day(date_str, s_orders)
            for r in day["routes"] + day.get("overdue_routes",[]):
                r["shift_name"] = sname
            shift_results[sname] = {"day":day,"planner":planner}
            tagged_all += [(r,"Thường") for r in day["routes"]]
            tagged_all += [(r,"Quá hạn") for r in day.get("overdue_routes",[])]
            all_exceptions.extend(day.get("exceptions",[]))
        all_day_results[date_str] = {
            "tagged_all":    tagged_all,
            "shifts":        shift_results,
            "exceptions":    all_exceptions,
        }
    return all_day_results

def build_routing_result_tables(tagged_all: list) -> tuple:
    inhouse_rows: list = []
    full_3pl_rows: list = []
    saving_rows: list   = []
    for idx,(r,kind) in enumerate(tagged_all, 1):
        vid        = str(r.get("vehicle_id","") or "")
        shift_name = r.get("shift_name","")
        orders     = r.get("orders",[])
        if r["external"]:
            is_full = vid.startswith("3PL-FULL") or "full" in str(r.get("outsourcing_type","")).lower()
            for o in orders:
                row = {
                    "Mã đơn":          o.get("order_id",""),
                    "Khách hàng":       o.get("customer_name",""),
                    "Ca":               o.get("shift", shift_name),
                    "Địa chỉ giao":     o.get("address",""),
                    "Mặt hàng":         o.get("items",""),
                    "Units":            o.get("quantity",1),
                    "Trọng lượng (kg)": round(float(o.get("total_weight_kg",0)),2),
                    "Thể tích (m³)":    round(float(o.get("total_volume_m3",0)),3),
                    "Loại outsource":   r.get("outsourcing_type","—"),
                    "Lý do":            (r.get("reason","") or "Vượt khả năng xe nhà")[:100],
                }
                if is_full:
                    full_3pl_rows.append(row)
                else:
                    saving_rows.append(row)
        else:
            split_note = ""
            if r.get("kind") == "SPLIT":
                split_note = f" [+3PL phần dư: {r.get('outsourcing_type','')}]"
            cap_w  = round(float(r.get("cap_w",0)),1)
            cap_v  = round(float(r.get("cap_v",0)),3)
            tw     = round(r.get("total_weight_kg",0),1)
            tv     = round(r.get("total_volume_m3",0),3)
            tq     = sum(o.get("quantity",1) for o in orders)
            items_parts: dict = {}
            for o in orders:
                it  = o.get("items","Hàng tiêu chuẩn")
                q   = o.get("quantity",1)
                w   = o.get("total_weight_kg",0)
                v   = o.get("total_volume_m3",0)
                if it not in items_parts:
                    items_parts[it] = {"qty":0,"w":0.0,"v":0.0}
                items_parts[it]["qty"] += q
                items_parts[it]["w"]   += w
                items_parts[it]["v"]   += v
            items_str = " + ".join(
                f"{d['qty']}×{it} ({d['w']:,.0f}kg|{d['v']:.2f}m³)"
                for it,d in items_parts.items())
            inhouse_rows.append({
                "Mã tuyến":           f"R{idx:03d}",
                "Ca":                 shift_name or (orders[0].get("shift","") if orders else ""),
                "Loại tuyến":         kind,
                "Kho":                r.get("id_warehouse",""),
                "Thứ tự điểm giao":   " → ".join(r.get("route",[])),
                "Danh sách đơn":      ", ".join(o.get("order_id","") for o in orders),
                "Số đơn":             len(orders),
                "Tổng units":         tq,
                "Chi tiết mặt hàng":  items_str,
                "Tổng KG":            tw,
                "Tổng m³":            tv,
                "Tải xe (kg)":        cap_w,
                "Thể tích xe (m³)":   cap_v,
                "Tải lấp đầy (%)":    round(r.get("load_factor",0)*100,1),
                "Xe đề xuất":         r.get("vehicle_type","") + split_note,
                "Biển số":            r.get("license_plate",""),
                "Km":                 r.get("km",0),
                "Giờ chạy":           round(r.get("hours",0),2),
                "Xuất phát":          f"{r['start']:%H:%M}" if r.get("start") else "",
                "Dự kiến về":         f"{r['end']:%H:%M}"   if r.get("end")   else "",
                "Trạng thái":         "🟠 Tách tuyến + 3PL" if r.get("kind")=="SPLIT" else "✅ Tối ưu thành công",
            })
    return (pd.DataFrame(inhouse_rows),
            pd.DataFrame(full_3pl_rows),
            pd.DataFrame(saving_rows))

def render_phan_ca_section(tagged_all: list, split_time_str: str, cfg_shifts: list):
    st.markdown(f"### 🔄 Chốt Hạm Đội Tại Giờ Phân Ca: **{split_time_str}**")
    split_min = _mins(split_time_str)
    if split_min == 0:
        st.error("Giờ phân ca không hợp lệ (định dạng HH:MM)"); return
    veh_info: dict = {}
    for r,_ in tagged_all:
        if r["external"]: continue
        vid = r["vehicle_id"]
        if vid not in veh_info:
            veh_info[vid] = {
                "vehicle_id":    vid,
                "license_plate": r.get("license_plate",""),
                "vehicle_type":  r.get("vehicle_type",""),
                "kho":           r.get("id_warehouse",""),
                "routes":        [],
            }
        veh_info[vid]["routes"].append(r)
    completed: list = []
    still_running: list = []
    for vid,info in veh_info.items():
        routes = info["routes"]
        valid_ends = [r["end"] for r in routes if r.get("end")]
        if not valid_ends: continue
        latest_end  = max(valid_ends)
        end_min     = latest_end.hour*60+latest_end.minute
        n_orders    = sum(len(r["orders"]) for r in routes)
        total_km    = round(sum(r.get("km",0) for r in routes),1)
        shifts_done = ", ".join(sorted({r.get("shift_name","?") for r in routes}))
        row = {
            "Mã xe":          vid,
            "Biển số":        info["license_plate"],
            "Loại xe":        info["vehicle_type"],
            "Kho":            info["kho"],
            "Ca đã phục vụ":  shifts_done,
            "Số chuyến":      len(routes),
            "Đơn đã giao":    n_orders,
            "Tổng km":        total_km,
            "Dự kiến về kho": f"{latest_end:%H:%M}",
        }
        if end_min <= split_min:
            row["Trạng thái"] = "✅ Đã về kho — Sẵn sàng ca tiếp"
            completed.append(row)
        else:
            remaining = end_min - split_min
            row["Trạng thái"] = f"🚚 Đang chạy (còn ~{remaining} phút)"
            still_running.append(row)
    next_shift = None
    for s in cfg_shifts:
        if _mins(s.get("start","00:00")) > split_min:
            next_shift = s; break
    mc1,mc2,mc3 = st.columns(3)
    mc1.metric("✅ Xe đã về kho tại giờ phân ca", len(completed))
    mc2.metric("🚚 Xe đang chạy (không liên quan)", len(still_running))
    if next_shift:
        mc3.info(f"🚀 Ca tiếp theo: **{next_shift['name']}** — Xuất phát **{next_shift['start']}**")
    tab_done, tab_run = st.tabs([
        f"✅ Xe sẵn sàng ca tiếp ({len(completed)} xe)",
        f"🚚 Xe đang chạy — không tính vào ca tiếp ({len(still_running)} xe)",
    ])
    with tab_done:
        if completed:
            st.dataframe(pd.DataFrame(completed),hide_index=True,use_container_width=True)
            st.success(f"🏁 **{len(completed)} xe** đã về kho trước {split_time_str} "
                       f"— sẵn sàng nhận lệnh ca tiếp theo.")
        else:
            st.info("Chưa có xe nào về kho tại thời điểm phân ca này.")
    with tab_run:
        if still_running:
            st.dataframe(pd.DataFrame(still_running),hide_index=True,use_container_width=True)
            st.caption("ℹ️ Các xe này vẫn đang trên đường, không tính vào hạm đội ca tiếp.")

# ============================================================================
# 🗺️ TAB 6: ROUTING — SETTINGS UI
# ============================================================================
def render_routing_settings():
    with st.expander("⚙️ Cấu hình Ca Giao Hàng & Tham số Mô hình", expanded=True):
        if "shift_count" not in st.session_state:
            st.session_state["shift_count"] = 2
        st.markdown("**🕐 Cấu hình ca giao hàng trong ngày:**")
        st.caption("Tên ca phải trùng khớp với trường 'Shift' trong dữ liệu Đơn hàng (Tab 4).")
        btn1,btn2,_,split_col = st.columns([1.3,1.3,2.5,2.5])
        if btn1.button("➕ Thêm ca", key="btn_add_shift"):
            if st.session_state["shift_count"] < 6:
                st.session_state["shift_count"] += 1
                st.rerun()
        if btn2.button("➖ Bớt ca", key="btn_rem_shift"):
            if st.session_state["shift_count"] > 1:
                st.session_state["shift_count"] -= 1
                st.rerun()
        split_time = split_col.text_input(
            "🔄 Giờ phân ca (HH:MM)",value="12:00",key="cfg_split_time",
            help="Tại giờ này xe đã về kho sẽ được tập kết chuẩn bị ca tiếp theo. "
                 "Xe chưa về cứ để chạy, không liên quan.")
        _DEFAULTS = [
            ("Ca Sáng","06:00","12:00"),
            ("Ca Chiều","13:00","17:30"),
            ("Ca Đêm","18:00","22:00"),
        ]
        n = st.session_state["shift_count"]
        h0,h1,h2,h3 = st.columns([0.4,2.4,1.5,1.5])
        h0.caption("**#**"); h1.caption("**Tên ca**")
        h2.caption("**Bắt đầu**"); h3.caption("**Kết thúc**")
        cfg_shifts = []
        for i in range(n):
            dn,ds,de = _DEFAULTS[i] if i < len(_DEFAULTS) else (f"Ca {i+1}","08:00","17:00")
            for key,val in [(f"sh_name_{i}",dn),(f"sh_start_{i}",ds),(f"sh_end_{i}",de)]:
                if key not in st.session_state:
                    st.session_state[key] = val
            c0,c1,c2,c3 = st.columns([0.4,2.4,1.5,1.5])
            c0.markdown(f"**{i+1}**")
            name  = c1.text_input(f"name_{i}",  key=f"sh_name_{i}",  label_visibility="collapsed")
            start = c2.text_input(f"start_{i}", key=f"sh_start_{i}", label_visibility="collapsed")
            end   = c3.text_input(f"end_{i}",   key=f"sh_end_{i}",   label_visibility="collapsed")
            cfg_shifts.append({"name":name,"start":start,"end":end})
        st.divider()
        c_mode,c1,c2 = st.columns(3)
        constraint_mode = c_mode.selectbox(
            "📦 Đặc thù hàng hóa (Lấp đầy theo)",
            ["Cube Out (Thể tích m³)","Weight Out (Trọng tải kg)"],
            key="cfg_constraint_mode")
        detour = c1.number_input("Hệ số đường vòng (nếu dùng Haversine)",
                                 1.0,3.0,1.2,0.05,key="cfg_detour")
        trips  = c2.number_input("Số chuyến tối đa / xe / ngày",1,5,2,1,key="cfg_trips")
        c3,c4,c5 = st.columns(3)
        svc_b2b = c3.number_input("Bốc/dỡ B2B (phút)",0,600,105,5,key="cfg_b2b")
        svc_b2c = c4.number_input("Bốc/dỡ B2C (phút)",0,600,60,5, key="cfg_b2c")
        turn    = c5.number_input("Quay đầu giữa 2 chuyến (phút)",0,240,30,5,key="cfg_turn")
    cfg = Config(
        start_time          = cfg_shifts[0]["start"] if cfg_shifts else "06:00",
        max_route_hours     = 8.0,
        detour_factor       = float(detour),
        constraint_mode     = constraint_mode,
        service_min         = {"B2B":int(svc_b2b),"B2C":int(svc_b2c)},
        vehicle_daily_count = int(trips),
        turnaround_min      = float(turn),
    )
    return cfg, cfg_shifts, split_time

def file_status(cfg: Config) -> pd.DataFrame:
    items = [
        ("Tab 1 · Hạm đội xe",           cfg.vehicle_file,   "Bắt buộc"),
        ("Tab 2 · Kho & Tọa độ",          cfg.warehouse_file, "Khuyến nghị"),
        ("Tab 3 · Sản phẩm",              cfg.product_file,   "Tuỳ chọn"),
        ("Tab 4 · Đơn hàng (Đã geocode)", cfg.order_file,     "Bắt buộc"),
    ]
    return pd.DataFrame([
        {"Nguồn":n,"File":os.path.relpath(p,BASE_DIR),
         "Trạng thái":"✅ Có" if os.path.exists(p) else "❌ Chưa có","Yêu cầu":need}
        for n,p,need in items])

def require_files(cfg: Config, which):
    labels = {"orders":(cfg.order_file,"Tab 4 (Đơn hàng)"),
              "fleet": (cfg.vehicle_file,"Tab 1 (Hạm đội xe)")}
    missing = [f"`{os.path.relpath(labels[k][0],BASE_DIR)}` (chạy {labels[k][1]})"
               for k in which if not os.path.exists(labels[k][0])]
    if missing: raise UserError("Thiếu file đầu vào: " + "; ".join(missing))

# ============================================================================
# 🗺️ TAB 6: ROUTING — MAIN RENDER
# ============================================================================
def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Clarke-Wright + OpenStreetMap (OSRM)")
    cfg, cfg_shifts, split_time = render_routing_settings()
    st.markdown("### 📂 Trạng thái dữ liệu đầu vào")
    st.dataframe(file_status(cfg), hide_index=True)
    use_osrm = st.checkbox(
        "🗺️ Lấy ma trận khoảng cách đường thực từ OpenStreetMap (OSRM)",
        value=True, key="use_osrm",
        help=f"Dùng OSRM public API (tối đa {MAX_OSRM_POINTS} điểm). "
             f"Nếu vượt hoặc lỗi, tự động fallback về Haversine × hệ số đường vòng.")
    if st.button("⚡ Chạy Routing", type="primary", key="t6_run"):
        try:
            require_files(cfg, ["orders","fleet"])
            data = load_data(cfg)
            osrm_matrix: dict = {}
            if use_osrm:
                with st.spinner("🗺️ Đang lấy ma trận khoảng cách từ OSRM / OpenStreetMap..."):
                    osrm_matrix, osrm_msg = build_osrm_matrix(
                        data["warehouses"], data["cust"], data["orders"])
                if osrm_matrix:
                    st.success(osrm_msg)
                else:
                    st.warning(f"{osrm_msg} — Tự động dùng Haversine × {cfg.detour_factor}.")
            with st.spinner("⚙️ Đang tối ưu tuyến theo ca (Clarke-Wright)..."):
                day_results = plan_by_shifts(data, cfg, cfg_shifts, osrm_matrix)
            st.session_state["t6_plan"] = {
                "day_results": day_results,
                "cfg":         cfg,
                "cfg_shifts":  cfg_shifts,
                "split_time":  split_time,
                "data":        data,
            }
            n_routes = sum(len(dr["tagged_all"]) for dr in day_results.values())
            st.success(f"✅ Routing hoàn tất! {n_routes} tuyến trên {len(day_results)} ngày.")
        except UserError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Lỗi không mong đợi: {e}")
            raise

    plan = st.session_state.get("t6_plan")
    if not plan:
        return
    day_results = plan["day_results"]
    cfg         = plan["cfg"]
    cfg_shifts  = plan["cfg_shifts"]
    split_time  = plan["split_time"]
    dates = list(day_results.keys())
    if not dates:
        st.warning("Không có ngày nào có đơn hàng trong dữ liệu.")
        return
    date_str   = st.selectbox("📅 Chọn ngày điều phối", dates, key="t6_date")
    tagged_all = day_results[date_str]["tagged_all"]
    if not tagged_all:
        st.info("Không có tuyến nào được tạo cho ngày này.")
        return
    df_inhouse, df_full_3pl, df_saving = build_routing_result_tables(tagged_all)
    st.divider()
    st.markdown(f"## 📊 Kết Quả Routing — {date_str}")
    total_orders   = sum(len(r["orders"])  for r,_ in tagged_all)
    n_inhouse      = len(df_inhouse)
    n_3pl          = len(df_full_3pl)
    n_saving       = len(df_saving)
    unrouted       = day_results[date_str].get("shifts",{})
    n_unrouted     = sum(len(sr["day"].get("unrouted_orders",[]))
                         for sr in unrouted.values())
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Tổng đơn trong ngày",       total_orders)
    k2.metric("✅ Tuyến xe nhà tối ưu",     f"{n_inhouse} tuyến")
    k3.metric("🚛 Đơn cần 3PL Full",        f"{n_3pl} đơn")
    k4.metric("🛵 Đơn giao hàng tiết kiệm", f"{n_saving} đơn")
    k5.metric("❓ Đơn chưa geocode",         f"{n_unrouted} đơn")

    tab1, tab2, tab3 = st.tabs([
        f"✅ Tuyến xe nhà tối ưu ({n_inhouse} tuyến)",
        f"🚛 Đơn cần 3PL Full ({n_3pl} đơn)",
        f"🛵 Đơn giao hàng tiết kiệm ({n_saving} đơn)",
    ])
    with tab1:
        st.markdown("##### Các tuyến được Clarke-Wright tối ưu thành công — xe nhà phụ trách")
        if df_inhouse.empty:
            st.info("Toàn bộ đơn được phân loại sang 3PL hoặc tiết kiệm. Kiểm tra lại hạm đội xe.")
        else:
            st.dataframe(df_inhouse, hide_index=True, use_container_width=True)
        if not df_inhouse.empty:
            buf = io.BytesIO()
            df_inhouse.to_excel(buf, index=False, sheet_name="Tuyến Tối Ưu")
            st.download_button("⬇️ Tải ROUTES_INHOUSE.xlsx", buf.getvalue(),
                               file_name=f"ROUTES_INHOUSE_{date_str}.xlsx",
                               mime=XLSX_MIME, key="dl_inhouse")
            save_outputs(OUT_ROUTING, f"ROUTES_INHOUSE_{date_str}",
                         "Tuyến Tối Ưu", df_inhouse)
    with tab2:
        st.markdown("##### Đơn hàng vượt khả năng hạm đội xe nhà — cần liên hệ đối tác 3PL Full")
        st.caption("ℹ️ Chỉ liệt kê danh sách đơn hàng. Phân tuyến 3PL do đội vận hành quyết định.")
        if df_full_3pl.empty:
            st.success("🎉 Không có đơn nào cần 3PL Full hôm nay!")
        else:
            st.dataframe(df_full_3pl, hide_index=True, use_container_width=True)
        if not df_full_3pl.empty:
            buf2 = io.BytesIO()
            df_full_3pl.to_excel(buf2, index=False, sheet_name="Đơn 3PL Full")
            st.download_button("⬇️ Tải ORDERS_3PL_FULL.xlsx", buf2.getvalue(),
                               file_name=f"ORDERS_3PL_FULL_{date_str}.xlsx",
                               mime=XLSX_MIME, key="dl_3pl")
            save_outputs(OUT_ROUTING, f"ORDERS_3PL_FULL_{date_str}",
                         "Đơn 3PL Full", df_full_3pl)
    with tab3:
        st.markdown("##### Đơn hàng đủ điều kiện dùng dịch vụ giao hàng tiết kiệm")
        st.caption("ℹ️ Chỉ liệt kê danh sách đơn hàng. Không tính số tuyến hay chi phí.")
        if df_saving.empty:
            st.success("🎉 Không có đơn nào phân loại giao tiết kiệm hôm nay!")
        else:
            st.dataframe(df_saving, hide_index=True, use_container_width=True)
        if not df_saving.empty:
            buf3 = io.BytesIO()
            df_saving.to_excel(buf3, index=False, sheet_name="Đơn Tiết Kiệm")
            st.download_button("⬇️ Tải ORDERS_SAVING.xlsx", buf3.getvalue(),
                               file_name=f"ORDERS_SAVING_{date_str}.xlsx",
                               mime=XLSX_MIME, key="dl_saving")
            save_outputs(OUT_ROUTING, f"ORDERS_SAVING_{date_str}",
                         "Đơn Tiết Kiệm", df_saving)
    st.divider()
    render_phan_ca_section(tagged_all, split_time, cfg_shifts)
    all_exc = day_results[date_str].get("exceptions",[])
    if all_exc:
        with st.expander(f"⚠️ Nhật ký sự kiện routing ({len(all_exc)} sự kiện)", expanded=False):
            st.dataframe(pd.DataFrame(all_exc), hide_index=True, use_container_width=True)
    notes = plan["data"].get("notes",[])
    if notes:
        with st.expander("📝 Ghi chú tải dữ liệu", expanded=False):
            for n in notes: st.caption(f"• {n}")

# ============================================================================
# 🚨 8. TAB 7: ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ
# ============================================================================
def handle_road_blocked(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🗺️ [RE-ROUTING] Đường chặn. OSRM đã vẽ lộ trình mới né điểm sạt lở cho đơn `{order_id}` (+2.4km)."
def handle_heavy_traffic(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏱️ [TRAFFIC DELAY] Tắc >30p. Tính lại ETA và gửi SMS khách đơn `{order_id}`."
def handle_truck_ban(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚫 [CORRIDOR SWITCH] Cấm xe tải. Đã chuyển xe `{vehicle_id}` sang hành lang phụ cận."
def handle_flooded_road(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🌧️ [FLOOD BLACKLIST] Đường ngập. Đã blacklist tọa độ tạm thời."
def handle_short_blockage(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏸️ [HOLD & WAIT] Chắn tàu/Lễ hội (<10p). Kích hoạt tạm dừng trên app tài xế."
def handle_driver_lost(order_id,vehicle_id,driver_id):
    return f"[{dt.datetime.now():%H:%M:%S}] 🧭 [GPS RE-NAVIGATE] Tài xế `{driver_id}` đi nhầm. Phát cảnh báo và làm mới bản đồ."
def handle_gps_lost(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 📡 [DEAD RECKONING] Mất GPS. Khóa Last Known Position, ước tính theo vận tốc."
def handle_customer_change_address(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🏠 [CLUSTER RE-SEQUENCE] Khách đổi địa chỉ. Kiểm tra cụm và sắp lại thứ tự ghé thăm đơn `{order_id}`."
def handle_cust_no_answer(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 📞 [UNREACHABLE CUST] Khách không nghe. Ghi nhận lần gọi 1, xe đi tiếp, xếp lịch gọi lại."
def handle_cust_absent(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚪 [CUST ABSENT] Khách vắng. Đánh dấu giao thất bại, gửi câu hỏi hẹn lịch lại."
def handle_cust_reject(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ❌ [CUST REJECT] Khách từ chối. Ghi lý do, tạo lệnh thu hồi về kho đơn `{order_id}`."
def handle_cust_request_late(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🕰️ [TIME WINDOW LATE] Khách hẹn muộn. Cập nhật Time Window, đẩy đơn `{order_id}` xuống cuối tuyến."
def handle_cust_request_early(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚡ [TIME WINDOW EARLY] Khách đòi sớm. Kiểm tra ETA, đẩy đơn `{order_id}` lên ưu tiên."
def handle_change_recipient(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 👤 [RECIPIENT SWITCH] Đổi người nhận. Yêu cầu tài xế chụp CCCD/OTP người nhận mới."
def handle_partial_delivery(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 📦 [PARTIAL DELIVERY] Khách nhận một phần. Cập nhật số lượng, tạo phiếu lưu kho phần còn lại."
def handle_address_not_found(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔍 [ADDRESS UNVERIFIED] Không tìm thấy địa chỉ đơn `{order_id}`. Gửi link Zalo cho khách ghim vị trí."
def handle_truck_breakdown(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚨 [BREAKDOWN RESCUE] Xe `{vehicle_id}` hỏng. Điều xe dự phòng ra cứu hộ & chuyển tải."
def handle_out_of_fuel(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⛽ [FUEL EMERGENCY] Xe `{vehicle_id}` hết nhiên liệu. Định vị trạm xăng gần nhất."
def handle_traffic_accident(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚑 [ACCIDENT EMERGENCY] Xe `{vehicle_id}` gặp tai nạn. Kích hoạt bảo hiểm và cứu hộ."
def handle_vehicle_overload(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚖️ [OVERLOAD DETECTED] Quá tải thực tế đơn `{order_id}`. Tách đơn hoặc đổi xe 3PL lớn hơn."
def handle_cargo_damaged(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 💔 [CARGO DAMAGED] Hàng hư hỏng đơn `{order_id}`. Tài xế chụp ảnh, cách ly hàng, báo kho đền bù."
def handle_cargo_missing(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ❓ [CARGO MISSING] Thất thoát hàng đơn `{order_id}`. Đối chiếu Barcode, lập biên bản đền bù."
def handle_wrong_loading(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔄 [RE-ARRANGE CARGO] Xếp sai vị trí xe `{vehicle_id}`. Sắp xếp lại thùng hàng theo thứ tự dỡ."
def handle_vehicle_disconnect(order_id,vehicle_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔌 [DISCONNECT ALERT] Xe `{vehicle_id}` mất kết nối IoT. Gọi điện xác thực trực tiếp tài xế."
def handle_route6_full(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 📦 [ROUTE FULL] Tuyến hết chỗ. Kiểm tra đẩy đơn `{order_id}` sang xe 3PL hoặc dời ca sau."
def handle_no_backup_vehicle(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚚 [NO BACKUP FLEET] Hết xe dự phòng. Quét xe đang chạy còn trống để ghép đơn (Co-loading)."
def handle_route7_overload(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚠️ [CASCADE OVERLOAD] Chuyển đơn làm tuyến kế quá tải. Hủy phương án, tính thuê ngoài."
def handle_cost_too_high(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 💰 [COST EXCEEDED] Chi phí quá cao. So sánh chi phí đền bù trễ vs thuê ngoài."
def handle_driver_timeout(order_id,vehicle_id,driver_id):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏳ [MAX HOURS REACHED] Tài xế `{driver_id}` sắp hết giờ. Điều xe đổi ca/bàn giao phần đơn còn lại."
def handle_area_mismatch(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🧭 [AREA MISMATCH] Xe còn chỗ nhưng lệch khu vực. Khóa ghép đơn nếu deadhead > 10km."
def handle_no_feasible_eta(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚨 [ETA IMPOSSIBLE] Không phương án nào kịp giờ. Cảnh báo tổng đài đàm phán lại khung giờ."
def handle_redelivery_cost_limit(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🛑 [REDELIVERY COST CEILING] Chi phí giao lại vượt ngưỡng. Chuyển về bưu cục gần nhất."
def handle_map_data_conflict(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🧐 [DATA CONFLICT] Xung đột GIS vs báo thực địa. Yêu cầu tài xế gửi ảnh xác minh."
def handle_duplicate_assignment(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔒 [RACE CONDITION] 2 tài xế cùng nhận đơn `{order_id}`. Khóa phiên bản, giữ lệnh hợp lệ đầu tiên."
def handle_driver_unconfirmed(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔔 [UNCONFIRMED COMMAND] Tài xế không bấm xác nhận. Chuông cảnh báo 3 phút, sau đó điều phối thủ công."
def handle_driver_ignore_new_route(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⚠️ [ROUTE DEVIATION] Tài xế chạy đường cũ. So sánh GPS với tuyến mới, phát chuông nhắc."
def handle_order_canceled_midway(order_id,*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🚫 [CANCEL MIDWAY] Đơn `{order_id}` bị hủy khi đang re-route. Hủy tác vụ, báo tài xế quay đầu."
def handle_infinite_loop_incident(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] 🔄 [LOOP DETECTED] Sự cố lặp lại >3 lần. Giới hạn re-route, yêu cầu Trưởng ca can thiệp thủ công."
def handle_invalid_order_data(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ❌ [INVALID DATA] Dữ liệu đơn thiếu m3/kg. Loại đơn khỏi thuật toán, gửi ticket sửa dữ liệu."
def handle_solver_timeout(*a):
    return f"[{dt.datetime.now():%H:%M:%S}] ⏱️ [ENGINE TIMEOUT] Clarke-Wright Timeout. Giữ kế hoạch gần nhất, báo lỗi Engine."

INCIDENT_SOLVERS = {
    "road_blocked":           ("1. Đường bị chặn hoàn toàn",            handle_road_blocked),
    "heavy_traffic":          ("2. Tắc nghẽn giao thông nghiêm trọng",  handle_heavy_traffic),
    "truck_ban":              ("3. Đường cấm xe tải",                    handle_truck_ban),
    "flooded_road":           ("4. Đường ngập nước",                     handle_flooded_road),
    "short_blockage":         ("5. Đường chặn ngắn hạn tạm thời",       handle_short_blockage),
    "driver_lost":            ("6. Tài xế đi nhầm đường",               handle_driver_lost),
    "gps_lost":               ("7. GPS mất tín hiệu",                    handle_gps_lost),
    "change_address":         ("8. Khách đổi địa chỉ phút chót",        handle_customer_change_address),
    "cust_no_answer":         ("9. Khách không nghe điện thoại",         handle_cust_no_answer),
    "cust_absent":            ("10. Khách không có mặt",                 handle_cust_absent),
    "cust_reject":            ("11. Khách từ chối nhận hàng",            handle_cust_reject),
    "cust_request_late":      ("12. Khách hẹn giao muộn hơn",           handle_cust_request_late),
    "cust_request_early":     ("13. Khách yêu cầu giao sớm hơn",        handle_cust_request_early),
    "change_recipient":       ("14. Thay đổi người nhận hàng",          handle_change_recipient),
    "partial_delivery":       ("15. Yêu cầu giao một phần",             handle_partial_delivery),
    "address_not_found":      ("16. Không tìm thấy địa chỉ",            handle_address_not_found),
    "truck_breakdown":        ("17. Xe hỏng giữa đường",                handle_truck_breakdown),
    "out_of_fuel":            ("18. Xe hết nhiên liệu",                  handle_out_of_fuel),
    "traffic_accident":       ("19. Xe gặp tai nạn",                    handle_traffic_accident),
    "vehicle_overload":       ("20. Xe quá tải do sai dữ liệu",         handle_vehicle_overload),
    "cargo_damaged":          ("21. Hàng hóa bị hư hỏng",               handle_cargo_damaged),
    "cargo_missing":          ("22. Hàng hóa bị thiếu",                 handle_cargo_missing),
    "wrong_loading":          ("23. Hàng xếp sai vị trí trên xe",       handle_wrong_loading),
    "vehicle_disconnect":     ("24. Xe mất kết nối IoT",                handle_vehicle_disconnect),
    "route6_full":            ("25. Tuyến hết chỗ chứa",                handle_route6_full),
    "no_backup_vehicle":      ("26. Không còn xe dự phòng",             handle_no_backup_vehicle),
    "route7_overload":        ("27. Chuyển đơn làm tuyến kế quá tải",  handle_route7_overload),
    "cost_too_high":          ("28. Phương án xử lý quá tốn kém",       handle_cost_too_high),
    "driver_timeout":         ("29. Tài xế sắp hết giờ làm việc",      handle_driver_timeout),
    "area_mismatch":          ("30. Xe còn chỗ nhưng lệch khu vực",    handle_area_mismatch),
    "no_feasible_eta":        ("31. Không phương án nào kịp giờ giao",  handle_no_feasible_eta),
    "redelivery_cost_limit":  ("32. Chi phí giao lại vượt ngưỡng",      handle_redelivery_cost_limit),
    "map_data_conflict":      ("33. Xung đột báo cáo bản đồ",          handle_map_data_conflict),
    "duplicate_assignment":   ("34. Trùng lặp 2 tài xế nhận đơn",      handle_duplicate_assignment),
    "driver_unconfirmed":     ("35. Tài xế không xác nhận lệnh mới",   handle_driver_unconfirmed),
    "driver_ignore_new_route":("36. Tài xế chạy sai tuyến mới",        handle_driver_ignore_new_route),
    "order_canceled_midway":  ("37. Đơn bị hủy khi đang re-route",     handle_order_canceled_midway),
    "infinite_loop_incident": ("38. Sự cố lặp lại nhiều lần",          handle_infinite_loop_incident),
    "invalid_order_data":     ("39. Dữ liệu đơn hàng không hợp lệ",    handle_invalid_order_data),
    "solver_timeout":         ("40. Engine tối ưu bị Timeout/Lỗi",     handle_solver_timeout),
}

def render_incident_tab():
    st.header("🚨 7. TRUNG TÂM ĐIỀU HÀNH THỰC ĐỊA & XỬ LÝ SỰ CỐ")
    st.caption("🌐 Liên kết trực tiếp với dữ liệu Đơn hàng (Tab 4) & Hạm đội xe (Tab 1)")
    orders_res = st.session_state.get("orders_result")
    fleet_res  = st.session_state.get("fleet_result")
    order_options   = orders_res["df"]["order_id"].tolist() if orders_res else ["ORD_003","ORD_001","ORD_002"]
    vehicle_options = fleet_res["df"]["vehicle_id"].tolist() if fleet_res  else ["VEH_06","VEH_01","VEH_02"]
    driver_options  = ["DRV_06","DRV_01","DRV_02","Tài xế dự phòng"]
    if "live_logs" not in st.session_state:
        st.session_state["live_logs"] = [
            {"Thời gian":f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}","Mã Xe":"VEH_06",
             "Tài Xế":"DRV_06","Đang Giao Đơn":"ORD_003","Vị Trí Hiện Tại":"Kho Depot Hà Nội",
             "Trạng Thái":"Đang xuất kho","Ghi Chú":"Chuẩn bị giao đơn #03"}]
    col_solver, col_form = st.tabs([
        "🛠️ 1. Kích Hoạt Python Incident Solvers",
        "📡 2. Báo Cáo Thực Địa & Live Tracking Logs"])
    with col_solver:
        st.markdown("### ⚡ Kích Hoạt Tự Động Giải Quyết Sự Cố Thực Địa")
        c1,c2 = st.columns([1.2,2])
        with c1:
            st.markdown("##### 📌 Tùy chọn đối tượng liên quan:")
            sel_order   = st.selectbox("Chọn Mã Đơn Hàng",   order_options,   key="t7_sel_ord")
            sel_vehicle = st.selectbox("Chọn Mã Xe",          vehicle_options, key="t7_sel_veh")
            sel_driver  = st.selectbox("Chọn Mã Tài Xế",      driver_options,  key="t7_sel_drv")
            st.markdown("##### 🚨 Chọn Sự Cố Thực Tế:")
            solver_labels = {k:v[0] for k,v in INCIDENT_SOLVERS.items()}
            sel_incident_key = st.selectbox(
                "Danh mục 40 sự cố ngoại lệ",list(solver_labels.keys()),
                format_func=lambda x:solver_labels[x],key="t7_sel_inc")
            btn_run = st.button("🚀 Run Tự Động",type="primary",key="btn_run_solver")
        with c2:
            st.markdown("##### 🖥️ Màn Hình Log Tự Động Thực Thi")
            console = st.empty()
            if btn_run:
                func = INCIDENT_SOLVERS[sel_incident_key][1]
                try:    res_msg = func(sel_order,sel_vehicle,sel_driver)
                except Exception: res_msg = func()
                console.code(res_msg,language="markdown")
                st.session_state["live_logs"].insert(0,{
                    "Thời gian":f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                    "Mã Xe":sel_vehicle,"Tài Xế":sel_driver,"Đang Giao Đơn":sel_order,
                    "Vị Trí Hiện Tại":"Toạ độ phát sinh sự cố",
                    "Trạng Thái":"Gặp sự cố / Đã re-route",
                    "Ghi Chú":f"Auto Solver [{sel_incident_key}]: {res_msg.split('] ')[-1]}"})
                st.toast("✅ Đã xử lý sự cố & cập nhật vào Live Logs!",icon="🎉")
            else:
                console.info("👈 Chọn loại sự cố bên trái và nhấn 'Run Tự Động' để xem kết quả.")
    with col_form:
        st.markdown("### ✍️ Biểu Mẫu Nhập Báo Cáo Thực Địa Từ Tài Xế")
        f1,f2 = st.columns([1,1.5])
        with f1:
            in_veh    = st.selectbox("Mã Xe",     vehicle_options,key="t7_f_veh")
            in_drv    = st.selectbox("Mã Tài Xế", driver_options, key="t7_f_drv")
            in_ord    = st.selectbox("Mã Đơn",    order_options,  key="t7_f_ord")
            in_loc    = st.text_input("Vị Trí Hiện Tại",value="Đường Cổ Linh, Hà Nội",key="t7_f_loc")
            in_status = st.selectbox("Trạng Thái Tuyến",
                ["Đang xuất kho Depot","Đang trên đường giao hàng","Đã giao thành công","Gặp sự cố / Tạm dừng"],
                key="t7_f_st")
            in_note   = st.text_area("Ghi Chú Thực Địa",placeholder="Ví dụ: Khách hẹn 10 phút nữa xuống nhận...",key="t7_f_note")
            if st.button("📤 Gửi Báo Cáo Về Hệ Thống",key="btn_sub_rep"):
                st.session_state["live_logs"].insert(0,{
                    "Thời gian":f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                    "Mã Xe":in_veh,"Tài Xế":in_drv,"Đang Giao Đơn":in_ord,
                    "Vị Trí Hiện Tại":in_loc,"Trạng Thái":in_status,
                    "Ghi Chú":in_note or "Không có ghi chú"})
                st.success("✅ Đã cập nhật nhật ký thực địa thành công!")
                st.rerun()
        with f2:
            st.markdown("##### 📊 Bảng Giám Sát Thời Gian Thực (Live Logs)")
            st.dataframe(pd.DataFrame(st.session_state["live_logs"]),
                         hide_index=True,use_container_width=True)

# ============================================================================
# ⚙️ 9. MAIN
# ============================================================================
TAB_NAMES = [
    "🚚 1. Fleet","🏭 2. Warehouse","📦 3. Product",
    "🧾 4. Orders","🧩 5. Orders Solving",
    "🗺️ 6. Routing","🚨 7. Xử Lý Sự Cố",
]

def main():
    st.title("🚚 Smart Logistics — Hệ thống chuẩn hóa & Điều phối Vận tải (Phạm Minh Thái)")
    with st.sidebar:
        st.markdown("### ℹ️ Menu Điều hướng")
        if st.button("🧹 Giải phóng RAM & Cache",key="free_ram"):
            st.cache_data.clear()
            for k in list(st.session_state):
                if k.endswith(("_raw","_result","_meta","_why","_plan","_seed")):
                    st.session_state.pop(k,None)
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
