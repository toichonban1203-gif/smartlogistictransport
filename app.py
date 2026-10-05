# -*- coding: utf-8 -*-
"""
================================================================================
SMART LOGISTICS — STREAMLIT APP (chuyển đổi hoàn toàn từ Gradio / Colab)
================================================================================
6 phân hệ (st.tabs):
  1. Hạm đội xe (Fleet)                      -> output_fleet/
  2. Kho & Tọa độ (Warehouse + ArcGIS)       -> output_warehouse/
  3. Sản phẩm (Product)                      -> output_product/
  4. Tài xế (Driver)                         -> output_driver/
  5. Đơn hàng (Orders + Semantic Mapping)    -> output_orders/
  6. Dashboard Định tuyến (Clarke-Wright)    -> output_customer/, output_matrix/
     (đọc input từ chính các thư mục output_* ở trên)

Mỗi tab dữ liệu có st.radio chọn: "Nhập trực tiếp (Data Editor)" hoặc
"Upload file Excel/CSV/JSON". Không dùng Gradio / ipywidgets / display().

requirements.txt đi kèm:
    streamlit, pandas, numpy, openpyxl, xlrd, rapidfuzz, unidecode, geopy, requests
================================================================================
"""
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
from dataclasses import dataclass, field as dc_field

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

# ============================================================================
# 0. HẰNG SỐ & THƯ MỤC ĐẦU RA
# ============================================================================
NONE = "-- Không sử dụng --"
INPUT_MODES = ["Nhập trực tiếp (Data Editor)", "Upload file Excel/CSV/JSON"]
UPLOAD_TYPES = ["xlsx", "xls", "xlsm", "csv", "json"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
VIETNAM_BOUNDS = (8.0, 24.0, 102.0, 110.0)
MAP_THRESHOLD = 0.38  # ngưỡng semantic mapping của tab Đơn hàng
DEFAULT_DEPOT = (21.0285, 105.8542)  # toạ độ kho mặc định (Hà Nội) như bản gốc

BASE_DIR = os.getcwd()
OUT_FLEET = os.path.join(BASE_DIR, "output_fleet")
OUT_WAREHOUSE = os.path.join(BASE_DIR, "output_warehouse")
OUT_PRODUCT = os.path.join(BASE_DIR, "output_product")
OUT_DRIVER = os.path.join(BASE_DIR, "output_driver")
OUT_ORDERS = os.path.join(BASE_DIR, "output_orders")
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")


class UserError(Exception):
    """Lỗi nghiệp vụ hiển thị cho người dùng (thay cho gr.Error)."""


# ============================================================================
# 1. TIỆN ÍCH DÙNG CHUNG
# ============================================================================
def is_blank(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return str(v).strip() == ""


def norm_basic(v):
    """Chuẩn hóa tên cột cho 4 tab Fleet / Warehouse / Product / Driver."""
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_exc = None
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:  # CSV dùng dấu ; hoặc tab (Excel tiếng Việt)
                header = str(df.columns[0])
                for sep in (";", "\t", "|"):
                    if sep in header:
                        return pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep)
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # pragma: no cover


def read_any(f) -> pd.DataFrame:
    """Đọc file upload của Streamlit (UploadedFile): Excel / CSV / JSON."""
    name = (getattr(f, "name", "") or "").lower()
    data = f.getvalue() if hasattr(f, "getvalue") else f.read()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data))
    elif name.endswith(".json"):
        try:
            df = pd.read_json(io.BytesIO(data))
        except ValueError:
            obj = json.loads(data.decode("utf-8-sig"))
            if isinstance(obj, dict):
                lists = [v for v in obj.values() if isinstance(v, list)]
                obj = lists[0] if len(lists) == 1 else [obj]
            df = pd.json_normalize(obj)
    else:
        df = _read_csv_bytes(data)
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def get_input(mode, uploaded, edited, label):
    """Lấy DataFrame thô từ Data Editor hoặc file upload."""
    if mode == INPUT_MODES[1]:
        if uploaded is None:
            raise UserError(f"Hãy upload file {label} trước.")
        df = read_any(uploaded)
    else:
        df = pd.DataFrame(edited) if edited is not None else pd.DataFrame()
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise UserError(f"Chưa có dữ liệu {label} để quét.")
    return df


def basic_semantic_mapping(df, fields, profile_fn, threshold):
    """Semantic Mapping cột (tên cột fuzzy + keyword + profile nội dung) — logic gốc
    của Fleet / Warehouse / Product / Driver."""
    result, used = {}, set()
    for fld, (_, aliases) in fields.items():
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used:
                continue
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
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")


def save_outputs(out_dir, base, sheet, df, extra_sheets=None):
    """Tự động tạo thư mục đầu ra và ghi Excel + JSON."""
    os.makedirs(out_dir, exist_ok=True)
    xlsx = os.path.join(out_dir, f"{base}.xlsx")
    js = os.path.join(out_dir, f"{base}.json")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        for name, sdf in (extra_sheets or {}).items():
            if sdf is not None and len(sdf):
                sdf.to_excel(writer, sheet_name=name, index=False)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(df_to_records(df), fh, ensure_ascii=False, indent=2, default=str)
    return [xlsx, js]


def conf_icon_ui(score):
    return "🟢" if score >= 0.75 else ("🟡" if score >= 0.55 else "🟠")


# ============================================================================
# 2. UI HELPER DÙNG CHUNG (Dual Input · Mapping · Result)
# ============================================================================
def render_input_block(prefix, label, seed_df):
    """st.radio + st.data_editor / st.file_uploader. Trả về (mode, edited, uploaded)."""
    mode = st.radio(f"Cách nhập dữ liệu {label}", INPUT_MODES, horizontal=True, key=f"{prefix}_mode")
    edited, uploaded = None, None
    if mode == INPUT_MODES[0]:
        st.markdown(f"##### ✍️ Nhập trực tiếp danh mục {label}")
        st.caption("Bấm vào ô để sửa, kéo xuống dòng cuối hoặc bấm ➕ để thêm dòng mới.")
        seed_key = f"{prefix}_seed"
        if seed_key not in st.session_state:
            st.session_state[seed_key] = seed_df.copy()
        edited = st.data_editor(st.session_state[seed_key], num_rows="dynamic", key=f"{prefix}_editor")
    else:
        st.markdown(f"##### 📂 Upload file {label}")
        st.caption("Hỗ trợ **Excel (.xlsx/.xls/.xlsm) / CSV / JSON**.")
        uploaded = st.file_uploader(f"File {label}", type=UPLOAD_TYPES, key=f"{prefix}_upload")
        if uploaded is not None:
            try:
                preview = read_any(uploaded)
                st.caption(f"📄 `{uploaded.name}` — {len(preview)} dòng × {len(preview.columns)} cột")
                st.dataframe(preview.head(10))
            except Exception as exc:
                st.error(f"❌ Không đọc được file: {exc}")
    return mode, edited, uploaded


def run_scan(prefix, label, mode, uploaded, edited, mapping_fn):
    """Xử lý nút 'Quét & Semantic Mapping': lưu raw + gợi ý mapping vào session_state."""
    try:
        raw = get_input(mode, uploaded, edited, label)
        mapping = mapping_fn(raw)
    except UserError as exc:
        st.error(f"❌ {exc}")
        return False
    except Exception as exc:
        st.error(f"❌ Lỗi khi quét dữ liệu: {exc}")
        return False
    st.session_state[f"{prefix}_raw"] = raw
    st.session_state[f"{prefix}_meta"] = {f: m[1] for f, m in mapping.items()}
    st.session_state[f"{prefix}_why"] = {f: (m[2] if len(m) > 2 else "") for f, m in mapping.items()}
    for f, m in mapping.items():
        st.session_state[f"{prefix}_col_{f}"] = str(m[0]) if m[0] is not None else NONE
    st.session_state.pop(f"{prefix}_result", None)
    return True


def render_mapping(prefix, specs, raw):
    """Các st.selectbox để kiểm tra / sửa ánh xạ cột. Trả về {field: tên cột đã chọn}."""
    meta = st.session_state.get(f"{prefix}_meta", {})
    why = st.session_state.get(f"{prefix}_why", {})
    choices = [NONE] + [str(c) for c in raw.columns]
    chosen = {}
    cols = st.columns(2)
    for i, (fld, label) in enumerate(specs):
        key = f"{prefix}_col_{fld}"
        if st.session_state.get(key) not in choices:
            st.session_state[key] = NONE
        with cols[i % 2]:
            chosen[fld] = st.selectbox(f"{label} ← cột nào?", choices, key=key)
            score = meta.get(fld, 0.0)
            note = why.get(fld, "")
            if score <= 0:
                st.caption("⚪ Không tìm thấy cột phù hợp")
            else:
                st.caption(f"{conf_icon_ui(score)} Độ tin cậy: {score:.0%}" + (f" · {note}" if note else ""))
    return chosen


def build_colmap(raw_df, chosen):
    lookup = {str(c): c for c in raw_df.columns}
    return {f: lookup[c] for f, c in chosen.items() if c and c != NONE and c in lookup}


def pick(row, colmap, fld, default=""):
    return row.get(colmap[fld]) if fld in colmap else default


def store_result(prefix, df, files, summary, metrics, extra=None):
    st.session_state[f"{prefix}_result"] = {
        "df": df, "files": list(files), "summary": summary, "metrics": metrics, "extra": extra or {},
    }


def render_result(prefix, title):
    res = st.session_state.get(f"{prefix}_result")
    if not res:
        return None
    st.markdown(f"### {title}")
    st.success(res["summary"])
    if res["metrics"]:
        cols = st.columns(len(res["metrics"]))
        for c, (lab, val) in zip(cols, res["metrics"]):
            c.metric(lab, val)
    st.dataframe(res["df"])
    rel = ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"])
    st.info(f"📁 Đã tự động lưu vào thư mục nguồn: {rel}")
    cols = st.columns(len(res["files"]))
    for c, path in zip(cols, res["files"]):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            mime = XLSX_MIME if path.endswith(".xlsx") else "application/json"
            c.download_button(f"⬇️ Tải {os.path.basename(path)}", data, file_name=os.path.basename(path),
                              mime=mime, key=f"{prefix}_dl_{os.path.basename(path)}")
    return res


# ============================================================================
# TAB 1 — FLEET: logic gốc (Semantic Mapping · làm sạch · validate)
# ============================================================================

VEHICLE_FIELDS = {
    "vehicle_id": ("Mã xe", ["mã xe", "vehicle id", "vehicle code", "vehicle", "xe", "id xe", "truck id", "mã phương tiện"]),
    "license_plate": ("Biển số", ["biển số", "bien so", "bsx", "license plate", "plate", "số xe", "so xe"]),
    "warehouse_id": ("ID kho hoạt động", ["kho", "warehouse", "wh", "hub", "chi nhánh", "location", "ma kho", "id kho", "khu vực"]),
    "max_weight": ("Trọng tải khối lượng (kg)", ["trọng tải", "trong tai", "weight", "payload", "khối lượng", "khoi luong", "kg", "tấn", "tan", "capacity kg"]),
    "max_volume": ("Trọng tải thể tích (m3)", ["thể tích", "the tich", "volume", "m3", "cbm", "capacity m3"]),
    "average_speed": ("Vận tốc trung bình (km/h)", ["vận tốc", "van toc", "speed", "vận tốc trung bình", "toc do trung binh", "avg speed", "kmh", "km/h"]),
    "fixed_cost": ("Chi phí cố định", ["chi phí cố định", "chi phi co dinh", "fixed cost", "cost fix", "fixed"]),
    "variable_cost": ("Chi phí biến đổi", ["chi phí biến đổi", "chi phi bien doi", "variable cost", "variable", "cost km", "chi phí theo km"])
}
def fleet_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in ["max_weight", "max_volume", "average_speed", "fixed_cost", "variable_cost"]:
        return float(values.str.replace(r"[^\d.]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))
def parse_num(val):
    if is_blank(val): return 0.0
    try:
        cleaned = re.sub(r"[^\d.-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except: return 0.0
def process_vehicle(v_id, plate, wh_id, weight, volume, speed, f_cost, v_cost):
    raw_id = "" if is_blank(v_id) else str(v_id).strip()
    raw_plate = "" if is_blank(plate) else str(plate).strip()
    raw_wh = "" if is_blank(wh_id) else str(wh_id).strip()
    result = {
        "vehicle_id": raw_id,
        "license_plate": raw_plate,
        "id_warehouse": raw_wh,
        "max_weight_kg": parse_num(weight),
        "max_volume_m3": parse_num(volume),
        "average_speed_kmh": parse_num(speed),
        "fixed_cost": parse_num(f_cost),
        "variable_cost": parse_num(v_cost),
        "trạng_thái": "✅ Hợp lệ"
    }
    return result
def fleet_validate_output(df):
    out = df.copy()
    dup_id = out["vehicle_id"].astype(str).duplicated(keep=False)
    dup_plate = out["license_plate"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("vehicle_id")): errs.append("Thiếu Mã xe")
        if is_blank(row.get("license_plate")): errs.append("Thiếu Biển số")
        if is_blank(row.get("id_warehouse")): errs.append("Thiếu ID kho hoạt động")
        if dup_id.iloc[i] and not is_blank(row.get("vehicle_id")): errs.append("Trùng Mã xe")
        if dup_plate.iloc[i] and not is_blank(row.get("license_plate")): errs.append("Trùng Biển số")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Đủ dữ liệu phương tiện chuẩn")
    out["kiểm_tra"] = msgs
    return out


def fleet_semantic_mapping(df):
    return basic_semantic_mapping(df, VEHICLE_FIELDS, fleet_profile_score, 0.35)


def fleet_empty_table():
    return pd.DataFrame({
        "Mã xe": ["VEH_01", "VEH_02"], "Biển số": ["29C-123.45", "29C-678.90"],
        "ID kho": ["WH_HN_01", "WH_HN_01"], "Trọng tải (kg)": [5000, 2000],
        "Thể tích (m3)": [20, 10], "Vận tốc (km/h)": [50, 45],
        "Chi phí cố định": [500000, 300000], "Chi phí biến đổi": [5000, 4000],
    })


def fleet_process_all(raw_df, chosen):
    colmap = build_colmap(raw_df, chosen)
    rows = []
    for _, row in raw_df.iterrows():
        rows.append(process_vehicle(
            pick(row, colmap, "vehicle_id", ""), pick(row, colmap, "license_plate", ""),
            pick(row, colmap, "warehouse_id", ""), pick(row, colmap, "max_weight", 0),
            pick(row, colmap, "max_volume", 0), pick(row, colmap, "average_speed", 0),
            pick(row, colmap, "fixed_cost", 0), pick(row, colmap, "variable_cost", 0)))
    return fleet_validate_output(pd.DataFrame(rows))


# ============================================================================
# TAB 1 — UI
# ============================================================================
def render_fleet_tab():
    st.header("🚚 Smart Logistics — Quản lý & Chuẩn hóa Phương tiện")
    st.markdown("**Input → Semantic Mapping → Làm sạch thông số → Validate → Export Excel/JSON**")
    st.info("🔒 Cột nào không có trong file, hệ thống sẽ tự động để trống hoặc mặc định mà không làm gián đoạn.")
    mode, edited, uploaded = render_input_block("fleet", "phương tiện", fleet_empty_table())

    if st.button("🔍 Quét & Semantic Mapping", key="fleet_scan", type="primary"):
        run_scan("fleet", "phương tiện", mode, uploaded, edited, fleet_semantic_mapping)

    raw = st.session_state.get("fleet_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    st.markdown("### 🔗 Kiểm tra ánh xạ cột phương tiện (cột nào không có chọn '-- Không sử dụng --')")
    specs = [(f, v[0]) for f, v in VEHICLE_FIELDS.items()]
    chosen = render_mapping("fleet", specs, raw)

    if st.button("🚀 Chuẩn hóa & Xử lý Fleet", key="fleet_process", type="primary"):
        try:
            out = fleet_process_all(raw, chosen)
            files = save_outputs(OUT_FLEET, "DIM_VEHICLE", "DIM_VEHICLE", out)
            ok = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            store_result("fleet", out, files, "🧭 Hoàn tất chuẩn hóa phương tiện",
                         [("Tổng loại xe", len(out)), ("Dòng hợp lệ", f"{ok}/{len(out)}")])
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    render_result("fleet", "📊 Kết quả phương tiện")


# ============================================================================
# TAB 2 — WAREHOUSE: logic gốc (làm sạch địa chỉ · ArcGIS · validate)
# ============================================================================

def coordinate_in_vietnam(lat, lng):
    try: lat, lng = float(lat), float(lng)
    except: return False
    return VIETNAM_BOUNDS[0] <= lat <= VIETNAM_BOUNDS[1] and VIETNAM_BOUNDS[2] <= lng <= VIETNAM_BOUNDS[3]
ADDRESS_ABBR = [(r"\bTP\.?\b", "Thành phố"), (r"\bQ\.?\b", "Quận"), (r"\bH\.?\b", "Huyện"), (r"\bTX\.?\b", "Thị xã"), (r"\bTT\.?\b", "Thị trấn"), (r"\bP\.?\b", "Phường"), (r"\bX\.?\b", "Xã"), (r"\bĐg\.?\b", "Đường")]
def clean_address(address):
    if is_blank(address): return ""
    text = str(address).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u3000]", " ", text)
    text = re.sub(r"\s*[|;→–—]\s*", ", ", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])\s*/\s*(?=[A-Za-zÀ-ỹ])", ", ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s,./'-]", " ", text)
    for p, r in ADDRESS_ABBR: text = re.sub(p, r, text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if text and not re.search(r"\bViệt Nam\b|\bVietnam\b", text, re.I): text += ", Việt Nam"
    return text
def address_quality(address):
    if not address: return 0.0, "❌ Địa chỉ trống"
    score, notes = 0.0, []
    if len(address) >= 10: score += 0.25
    else: notes.append("địa chỉ ngắn")
    if "," in address: score += 0.20
    else: notes.append("thiếu separator")
    if re.search(r"\d", address): score += 0.15
    if re.search(r"\b(Phường|Xã|Quận|Huyện|Thành phố|Tỉnh|Thị xã)\b", address, re.I): score += 0.30
    else: notes.append("thiếu thành phần hành chính")
    if re.search(r"\bViệt Nam\b", address, re.I): score += 0.10
    return min(score, 1.0), ("✅ Địa chỉ sạch" if not notes else "⚠️ " + "; ".join(notes))
WAREHOUSE_FIELDS = {"warehouse_id": ("Mã kho", ["mã kho", "warehouse id", "warehouse code", "warehouse", "kho", "id kho", "invent id", "invent_id"]), "address": ("Địa chỉ kho", ["địa chỉ", "địa điểm", "address", "location", "vị trí"])}
def wh_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field == "address": return float(0.7 * (values.str.len() >= 10).mean() + 0.3 * values.str.contains(r"[,\-/]").mean())
    return float(values.nunique() / len(values))
def process_warehouse(warehouse_id, address, do_geocode=True):
    raw_address = "" if is_blank(address) else str(address).strip()
    cleaned = clean_address(raw_address)
    quality, clean_status = address_quality(cleaned)
    result = {"id_warehouse": "" if is_blank(warehouse_id) else str(warehouse_id).strip(), "address": cleaned, "lat": None, "lng": None, "địa_chỉ_gốc": raw_address, "chất_lượng_địa_chỉ": quality, "trạng_thái_làm_sạch": clean_status, "địa_chỉ_geocode": "", "geocode_score": None, "trạng_thái_geocode": "—", "nguồn_tọa_độ": ""}
    if is_blank(warehouse_id): result["trạng_thái_geocode"] = "❌ Thiếu Mã kho"; return result
    if not cleaned: result["trạng_thái_geocode"] = "❌ Không có địa chỉ để geocode"; return result
    if not do_geocode: result["trạng_thái_geocode"] = "⏸️ Đã tắt geocoding"; return result
    geo = geocode_address(cleaned)
    if not geo["ok"]: result["trạng_thái_geocode"] = geo["status"]; return result
    result.update({"lat": geo["lat"], "lng": geo["lng"], "địa_chỉ_geocode": geo["display_name"], "geocode_score": geo["score"], "trạng_thái_geocode": f"✅ ArcGIS geocode thành công" + (f" | score {geo['score']:.0f}" if geo["score"] is not None else ""), "nguồn_tọa_độ": "ArcGIS"})
    return result
def warehouse_validate_output(df):
    out = df.copy()
    dup = out["id_warehouse"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("id_warehouse")): errs.append("Thiếu Mã kho")
        if is_blank(row.get("address")): errs.append("Thiếu địa chỉ")
        lat, lng = row.get("lat"), row.get("lng")
        if pd.isna(lat) or pd.isna(lng): errs.append("Chưa có Lat/Lon")
        elif not coordinate_in_vietnam(lat, lng): errs.append("Lat/Lon ngoài Việt Nam")
        if dup.iloc[i] and not is_blank(row.get("id_warehouse")): errs.append("Trùng Mã kho")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Đủ dữ liệu + Lat/Lon hợp lệ")
    out["kiểm_tra"] = msgs
    return out


def wh_semantic_mapping(df):
    return basic_semantic_mapping(df, WAREHOUSE_FIELDS, wh_profile_score, 0.40)


@st.cache_resource(show_spinner=False)
def get_geocoder():
    try:
        return ArcGIS(user_agent="smart-logistics-warehouse/1.0", timeout=10)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _geo_cache():
    return {}


def geocode_address(address, retries=3):
    """Geocode 1 địa chỉ bằng ArcGIS (có cache; không cache lỗi mạng)."""
    if not address:
        return {"ok": False, "status": "❌ Địa chỉ trống"}
    cache = _geo_cache()
    if address in cache:
        return cache[address]
    geocoder = get_geocoder()
    if geocoder is None:
        return {"ok": False, "status": "❌ Không khởi tạo được ArcGIS"}
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            loc = geocoder.geocode(address, timeout=10)
            if loc is None:
                res = {"ok": False, "status": "⚠️ ArcGIS không tìm thấy địa chỉ"}
                cache[address] = res
                return res
            raw = getattr(loc, "raw", {}) or {}
            score = raw.get("score")
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            lat, lng = float(loc.latitude), float(loc.longitude)
            if not coordinate_in_vietnam(lat, lng):
                res = {"ok": False, "status": "⚠️ ArcGIS trả tọa độ ngoài Việt Nam"}
                cache[address] = res
                return res
            res = {"ok": True, "lat": lat, "lng": lng, "display_name": getattr(loc, "address", "") or "", "score": score}
            cache[address] = res
            return res
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1)
    return {"ok": False, "status": f"❌ ArcGIS lỗi sau {retries} lần: {last_error[:150]}"}


def warehouse_empty_table():
    return pd.DataFrame({
        "Mã kho": ["WH_HN_01"],
        "Địa chỉ kho": ["Số 1 Tràng Tiền, Hoàn Kiếm, Hà Nội"],
    })


# ============================================================================
# TAB 2 — UI
# ============================================================================
def render_warehouse_tab():
    st.header("🏭 Smart Logistics — Quét tọa độ kho (ArcGIS Geocoding)")
    st.markdown("**Input → Semantic Mapping → Làm sạch địa chỉ → ArcGIS Geocoding → Lat/Lon → Export**")
    st.info("🔒 Người dùng chỉ nhập **Mã kho + Địa chỉ kho**. Lat/Lon do hệ thống tự động lấy từ ArcGIS.")
    do_geo = st.checkbox("🌍 Bật ArcGIS Geocoding", value=True, key="wh_do_geo")
    mode, edited, uploaded = render_input_block("wh", "kho", warehouse_empty_table())

    if st.button("🔍 Quét & Semantic Mapping", key="wh_scan", type="primary"):
        run_scan("wh", "kho", mode, uploaded, edited, wh_semantic_mapping)

    raw = st.session_state.get("wh_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    st.markdown("### 🔗 Kiểm tra mapping")
    specs = [(f, v[0]) for f, v in WAREHOUSE_FIELDS.items()]
    chosen = render_mapping("wh", specs, raw)

    if st.button("🚀 Làm sạch + Geocoding", key="wh_process", type="primary"):
        try:
            colmap = build_colmap(raw, chosen)
            if "warehouse_id" not in colmap:
                raise UserError("Chưa chọn cột Mã kho.")
            if "address" not in colmap:
                raise UserError("Chưa chọn cột Địa chỉ kho.")
            rows, n = [], len(raw)
            prog = st.progress(0.0, text="Đang làm sạch + geocode kho...")
            for k, (_, row) in enumerate(raw.iterrows(), 1):
                rows.append(process_warehouse(row.get(colmap["warehouse_id"]), row.get(colmap["address"]), do_geocode=do_geo))
                prog.progress(k / n, text=f"Đã xử lý {k}/{n} kho")
            prog.empty()
            out = warehouse_validate_output(pd.DataFrame(rows))
            files = save_outputs(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES", "DIM_WAREHOUSE", out)
            ok = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            geocoded = int(out["lat"].notna().sum())
            store_result("wh", out, files, "🧭 Hoàn tất quét kho",
                         [("Tổng số kho", len(out)), ("Geocode thành công", f"{geocoded}/{len(out)}"),
                          ("Đủ Lat/Lon + hợp lệ", f"{ok}/{len(out)}")])
        except UserError as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    res = render_result("wh", "📊 Kết quả kho")
    if res is not None:
        geo_df = res["df"].dropna(subset=["lat", "lng"])
        if not geo_df.empty:
            st.markdown("##### 📍 Vị trí kho trên bản đồ")
            st.map(geo_df.rename(columns={"lng": "lon"})[["lat", "lon"]])


# ============================================================================
# TAB 3 — PRODUCT: logic gốc
# ============================================================================

PRODUCT_FIELDS = {
    "product_id": ("Mã sản phẩm", ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "code", "id"]),
    "product_name": ("Tên sản phẩm", ["tên sản phẩm", "product name", "item name", "tên sp", "name", "mô tả"]),
    "volume": ("Thể tích", ["thể tích", "the tich", "volume", "m3", "cbm", "capacity"]),
    "weight": ("Trọng lượng / Khối lượng", ["trọng lượng", "trong luong", "khối lượng", "khoi luong", "weight", "kg", "tấn", "mass"]),
    "length": ("Chiều dài", ["dài", "dai", "length", "l", "dim l"]),
    "width": ("Chiều rộng", ["rộng", "rong", "width", "w", "dim w"]),
    "height": ("Chiều cao", ["cao", "height", "h", "dim h"]),
    "cost_price": ("Giá sản xuất", ["giá sản xuất", "gia san xuat", "cost price", "cost", "giá vốn", "giá gốc"]),
    "selling_price": ("Giá bán", ["giá bán", "gia ban", "selling price", "price", "retail price", "unit price"])
}
def product_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in ["volume", "weight", "length", "width", "height", "cost_price", "selling_price"]:
        return float(values.str.replace(r"[^\d.]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))
def process_product(p_id, p_name, vol, wgt, l, w, h, cost, price):
    raw_id = "" if is_blank(p_id) else str(p_id).strip()
    raw_name = "" if is_blank(p_name) else str(p_name).strip()
    return {
        "product_id": raw_id,
        "product_name": raw_name,
        "volume": parse_num(vol),
        "weight": parse_num(wgt),
        "length": parse_num(l),
        "width": parse_num(w),
        "height": parse_num(h),
        "cost_price": parse_num(cost),
        "selling_price": parse_num(price),
        "trạng_thái": "✅ Hợp lệ"
    }
def product_validate_output(df):
    out = df.copy()
    dup_id = out["product_id"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("product_id")): errs.append("Thiếu Mã sản phẩm")
        if is_blank(row.get("product_name")): errs.append("Thiếu Tên sản phẩm")
        if dup_id.iloc[i] and not is_blank(row.get("product_id")): errs.append("Trùng Mã sản phẩm")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Đủ dữ liệu sản phẩm chuẩn")
    out["kiểm_tra"] = msgs
    return out


def product_semantic_mapping(df):
    return basic_semantic_mapping(df, PRODUCT_FIELDS, product_profile_score, 0.35)


def product_empty_table():
    return pd.DataFrame({
        "Mã sản phẩm": ["SP_01"], "Tên sản phẩm": ["Ghế Sofa Gỗ Sồi"],
        "Thể tích (m3)": [0.5], "Trọng lượng (kg)": [25.0],
        "Dài (cm)": [120], "Rộng (cm)": [60], "Cao (cm)": [80],
        "Giá sản xuất": [1200000], "Giá bán": [2500000],
    })


def product_process_all(raw_df, chosen):
    colmap = build_colmap(raw_df, chosen)
    rows = []
    for _, row in raw_df.iterrows():
        rows.append(process_product(
            pick(row, colmap, "product_id", ""), pick(row, colmap, "product_name", ""),
            pick(row, colmap, "volume", 0), pick(row, colmap, "weight", 0),
            pick(row, colmap, "length", 0), pick(row, colmap, "width", 0),
            pick(row, colmap, "height", 0), pick(row, colmap, "cost_price", 0),
            pick(row, colmap, "selling_price", 0)))
    return product_validate_output(pd.DataFrame(rows))


# ============================================================================
# TAB 3 — UI
# ============================================================================
def render_product_tab():
    st.header("📦 Smart Logistics — Quản lý & Chuẩn hóa Sản phẩm")
    st.markdown("**Input → Semantic Mapping → Làm sạch thông số kích thước/giá → Validate → Export Excel/JSON**")
    st.info("🔒 Cột nào không có trong file, bạn có thể để trống hoặc chọn `-- Không sử dụng --` mà không lo gián đoạn.")
    mode, edited, uploaded = render_input_block("product", "sản phẩm", product_empty_table())

    if st.button("🔍 Quét & Semantic Mapping", key="product_scan", type="primary"):
        run_scan("product", "sản phẩm", mode, uploaded, edited, product_semantic_mapping)

    raw = st.session_state.get("product_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    st.markdown("### 🔗 Kiểm tra ánh xạ cột sản phẩm (cột nào không có chọn '-- Không sử dụng --')")
    specs = [(f, v[0]) for f, v in PRODUCT_FIELDS.items()]
    chosen = render_mapping("product", specs, raw)

    if st.button("🚀 Chuẩn hóa & Xử lý Product", key="product_process", type="primary"):
        try:
            out = product_process_all(raw, chosen)
            files = save_outputs(OUT_PRODUCT, "DIM_PRODUCT", "DIM_PRODUCT", out)
            ok = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            store_result("product", out, files, "🧭 Hoàn tất chuẩn hóa sản phẩm",
                         [("Tổng số sản phẩm", len(out)), ("Dòng hợp lệ", f"{ok}/{len(out)}")])
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    render_result("product", "📊 Kết quả sản phẩm")


# ============================================================================
# TAB 4 — DRIVER: logic gốc
# ============================================================================

DRIVER_FIELDS = {
    "driver_id": ("Mã tài xế", ["mã tài xế", "driver id", "driver code", "mã nv", "staff id", "id", "code"]),
    "driver_name": ("Họ và tên", ["họ và tên", "ho va ten", "tên tài xế", "ten tai xe", "full name", "name", "họ tên", "tên nhân viên"]),
    "license_type": ("Loại bằng", ["loại bằng", "loai bang", "bằng lái", "bang lai", "license", "class", "hạng bằng"]),
    "warehouse": ("Kho hoạt động", ["kho", "warehouse", "trạm", "hub", "địa điểm kho", "chi nhánh"]),
    "address": ("Địa chỉ", ["địa chỉ", "dia chi", "address", "nơi ở"]),
    "phone": ("Số điện thoại", ["số điện thoại", "so dien thoai", "phone", "sdt", "mobile", "hotline"]),
    "role": ("Vị trí làm việc", ["vị trí", "vi tri", "role", "chức vụ", "job", "loại nhân sự", "vị trí làm việc"])
}
def driver_profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field == "phone":
        return float(values.str.replace(r"[^\d+]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))
def process_driver(d_id, d_name, lic, wh, addr, phone, role):
    return {
        "driver_id": "" if is_blank(d_id) else str(d_id).strip(),
        "driver_name": "" if is_blank(d_name) else str(d_name).strip(),
        "license_type": "" if is_blank(lic) else str(lic).strip().upper(),
        "id_warehouse": "" if is_blank(wh) else str(wh).strip(),
        "address": "" if is_blank(addr) else str(addr).strip(),
        "phone": "" if is_blank(phone) else str(phone).strip(),
        "role": "" if is_blank(role) else str(role).strip(),
        "trạng_thái": "✅ Hợp lệ"
    }
def driver_validate_output(df):
    out = df.copy()
    dup_id = out["driver_id"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs = []
        if is_blank(row.get("driver_id")): errs.append("Thiếu Mã tài xế")
        if is_blank(row.get("driver_name")): errs.append("Thiếu Họ và tên")
        if dup_id.iloc[i] and not is_blank(row.get("driver_id")): errs.append("Trùng Mã tài xế")
        msgs.append("❌ " + "; ".join(errs) if errs else "✅ Đủ dữ liệu tài xế chuẩn")
    out["kiểm_tra"] = msgs
    return out


def driver_semantic_mapping(df):
    return basic_semantic_mapping(df, DRIVER_FIELDS, driver_profile_score, 0.35)


def driver_empty_table():
    return pd.DataFrame({
        "Mã tài xế": ["DRV_01", "DRV_02"],
        "Họ và tên": ["Nguyễn Văn A", "Trần Văn B"],
        "Loại bằng": ["FC", "C"],
        "Kho hoạt động": ["WH_HN_01", "WH_HN_01"],
        "Địa chỉ": ["Hà Nội", "Hà Nội"],
        "Số điện thoại": ["0901234567", "0987654321"],
        "Vị trí làm việc": ["Tài xế chính", "Hỗ trợ vận chuyển đồ"],
    })


def driver_process_all(raw_df, chosen):
    colmap = build_colmap(raw_df, chosen)
    rows = []
    for _, row in raw_df.iterrows():
        rows.append(process_driver(
            pick(row, colmap, "driver_id", ""), pick(row, colmap, "driver_name", ""),
            pick(row, colmap, "license_type", ""), pick(row, colmap, "warehouse", ""),
            pick(row, colmap, "address", ""), pick(row, colmap, "phone", ""),
            pick(row, colmap, "role", "")))
    return driver_validate_output(pd.DataFrame(rows))


# ============================================================================
# TAB 4 — UI
# ============================================================================
def render_driver_tab():
    st.header("👨‍✈️ Smart Logistics — Quản lý & Chuẩn hóa Tài xế")
    st.markdown("**Input → Semantic Mapping → Làm sạch thông tin nhân sự → Validate → Export Excel/JSON**")
    st.info("🔒 Cột nào không có trong file, bạn có thể để trống hoặc chọn `-- Không sử dụng --` mà không lo gián đoạn.")
    mode, edited, uploaded = render_input_block("driver", "tài xế", driver_empty_table())

    if st.button("🔍 Quét & Semantic Mapping", key="driver_scan", type="primary"):
        run_scan("driver", "tài xế", mode, uploaded, edited, driver_semantic_mapping)

    raw = st.session_state.get("driver_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    st.markdown("### 🔗 Kiểm tra ánh xạ cột tài xế (cột nào không có chọn '-- Không sử dụng --')")
    specs = [(f, v[0]) for f, v in DRIVER_FIELDS.items()]
    chosen = render_mapping("driver", specs, raw)

    if st.button("🚀 Chuẩn hóa & Xử lý Driver", key="driver_process", type="primary"):
        try:
            out = driver_process_all(raw, chosen)
            files = save_outputs(OUT_DRIVER, "DIM_DRIVER", "DIM_DRIVER", out)
            ok = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            store_result("driver", out, files, "🧭 Hoàn tất chuẩn hóa nhân sự tài xế",
                         [("Tổng số nhân sự", len(out)), ("Dòng hợp lệ", f"{ok}/{len(out)}")])
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    render_result("driver", "📊 Kết quả tài xế")


# ============================================================================
# TAB 5 — ORDERS: logic gốc (Semantic Mapping cột + giá trị · B2B/B2C · Alert · đơn vị)
# ============================================================================

def to_text(v):
    """Giá trị bất kỳ -> chuỗi sạch (1.0 -> '1', True -> 'true')."""
    if is_blank(v): return ""
    if isinstance(v, (bool, np.bool_)): return "true" if v else "false"
    if isinstance(v, (float, np.floating)) and float(v).is_integer(): return str(int(v))
    return str(v).strip()
def split_camel(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
def norm(v):
    """Bỏ dấu, thường hóa, tách camelCase/underscore: 'Mã_Đơn' / 'orderID' -> 'ma don' / 'order id'."""
    s = unidecode(split_camel(to_text(v) if not isinstance(v, str) else v)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
ORDER_FIELDS = {
    "order_id": dict(label="Mã đơn", aliases=[
        "mã đơn", "mã đơn hàng", "số đơn", "số đơn hàng", "mã đh", "id đơn", "đơn hàng", "order id", "order no",
        "order number", "order code", "order ref", "order", "oid", "ord id", "ord no", "so number", "sales order",
        "transaction id", "invoice no", "mã giao dịch", "số chứng từ"],
        negative=["customer", "khach", "cus", "client", "product", "sku", "item", "hang", "sp", "status", "type",
                  "date", "ngay", "tinh trang", "loai", "trang thai", "qty", "weight", "volume", "priority"]),
    "customer_id": dict(label="Mã khách", aliases=[
        "mã khách", "mã khách hàng", "mã kh", "kh id", "customer id", "customer code", "customer no", "cust id",
        "cust code", "cus id", "cus code", "client id", "client code", "account id", "account number", "buyer id",
        "mã người mua", "mã đối tác", "mã đại lý", "mã nhà phân phối", "cid"],
        negative=["name", "ten", "order", "don", "address", "dia chi", "type", "loai"]),
    "customer_name": dict(label="Tên khách", aliases=[
        "tên khách", "tên khách hàng", "tên kh", "khách hàng", "khách", "người nhận", "tên người nhận", "người mua",
        "họ tên", "họ và tên", "tên công ty", "tên đơn vị", "customer name", "client name", "buyer name", "recipient",
        "receiver", "consignee", "ship to name", "account name", "company name", "contact name", "name", "customer"],
        negative=["id", "ma", "code", "no", "type", "loai", "address", "dia chi", "item", "product", "san pham",
                  "hang", "sku", "file"]),
    "quantity": dict(label="Số lượng mua", aliases=[
        "số lượng", "số lượng mua", "sl", "sl mua", "tổng số lượng", "số kiện", "số cái", "qty", "quantity",
        "order qty", "total qty", "units", "pcs", "pieces", "count"],
        negative=["weight", "volume", "price", "don gia", "thanh tien", "cost", "trong luong", "the tich"]),
    "items": dict(label="Mặt hàng mua", aliases=[
        "mặt hàng", "mặt hàng mua", "hàng hóa", "tên hàng", "tên hàng hóa", "sản phẩm", "tên sản phẩm", "danh sách hàng",
        "nội dung hàng", "mô tả hàng", "loại hàng", "items", "item", "item name", "product", "products", "product name",
        "sku", "goods", "cargo", "description"],
        negative=["qty", "quantity", "weight", "volume", "so luong"]),
    "total_weight": dict(label="Tổng trọng lượng (kg)", aliases=[
        "tổng trọng lượng", "trọng lượng", "khối lượng", "tổng khối lượng", "khối lượng hàng", "cân nặng", "tải trọng",
        "weight", "total weight", "gross weight", "net weight", "weight kg", "wt", "kg"],
        negative=["volume", "the tich", "m3", "cbm"]),
    "total_volume": dict(label="Tổng thể tích (m3)", aliases=[
        "tổng thể tích", "thể tích", "số khối", "khối hàng", "dung tích", "kích thước", "kích thước kiện",
        "volume", "total volume", "vol", "m3", "cbm", "cubic", "dimension", "dimensions", "size"],
        negative=["weight", "trong luong", "khoi luong", "kg", "luong"]),
    "address": dict(label="Địa chỉ khách", aliases=[
        "địa chỉ", "địa chỉ khách", "địa chỉ khách hàng", "địa chỉ giao hàng", "địa chỉ nhận", "nơi giao", "nơi giao hàng",
        "nơi nhận", "nơi nhận hàng", "điểm giao", "điểm giao hàng", "địa điểm giao", "đích đến", "vị trí",
        "address", "delivery address", "shipping address", "destination address", "ship to", "deliver to",
        "location", "destination", "addr", "street"],
        negative=["email", "mail", "ip", "web", "phone", "dien thoai"]),
    "order_status": dict(label="Tình trạng đơn", aliases=[
        "tình trạng đơn", "trạng thái đơn", "trạng thái", "tình trạng", "trạng thái giao hàng", "trạng thái xử lý",
        "tiến độ", "order status", "status", "state", "delivery status", "fulfillment status", "stage"],
        negative=["alert", "canh bao", "priority", "payment", "thanh toan"]),
    "order_type": dict(label="Loại đơn (B2C/B2B)", aliases=[
        "loại đơn", "loại đơn hàng", "loại khách", "loại khách hàng", "phân loại khách", "nhóm khách hàng", "đối tượng",
        "kênh", "kênh bán", "kênh bán hàng", "loại hình", "order type", "customer type", "customer segment", "segment",
        "client type", "buyer type", "customer group", "sales channel", "channel", "type", "b2b b2c", "b2c", "b2b"],
        negative=["status", "alert", "vehicle", "item", "product", "xe"]),
    "alert_status": dict(label="Tình trạng (Alert/Normal)", aliases=[
        "tình trạng alert", "cảnh báo", "alert", "alert status", "normal alert", "mức độ ưu tiên", "độ ưu tiên", "ưu tiên",
        "mức độ khẩn", "khẩn cấp", "gấp", "priority", "urgent", "severity", "flag", "express", "rush", "sla"],
        negative=[]),
    "order_date": dict(label="Ngày đơn / ngày giao", aliases=[
        "ngày đặt", "ngày đặt hàng", "ngày tạo đơn", "ngày tạo", "ngày đơn", "ngày giao", "ngày giao hàng",
        "ngày giao dự kiến", "thời gian đặt", "ngày", "order date", "created date", "created at", "delivery date",
        "ship date", "expected delivery", "planned date", "due date", "date", "timestamp"],
        negative=[]),
}
FIELDS = list(ORDER_FIELDS)
ALIAS_N = {f: [a for a in (norm(x) for x in v["aliases"]) if a] for f, v in ORDER_FIELDS.items()}
NEG_N = {f: [n for n in (norm(x) for x in v["negative"]) if n] for f, v in ORDER_FIELDS.items()}
# ==========================================================
# 3. TỪ ĐIỂN GIÁ TRỊ (Individual -> B2C, Business -> B2B, ...)
# ==========================================================
TYPE_TABLE = {
    "B2C": ["b2c", "c", "individual", "individuals", "person", "personal", "private", "private customer", "consumer",
            "retail", "retail customer", "end user", "enduser", "household", "home", "home delivery", "d2c",
            "direct to consumer", "walk in", "guest", "residential", "resident", "user", "online customer",
            "cá nhân", "khách cá nhân", "khách hàng cá nhân", "khách lẻ", "khách hàng lẻ", "bán lẻ", "người tiêu dùng",
            "tiêu dùng", "lẻ", "hộ gia đình", "nhà riêng", "tư nhân", "khách vãng lai", "người dùng"],
    "B2B": ["b2b", "b", "business", "businesses", "company", "companies", "corporate", "corporation", "enterprise",
            "organization", "organisation", "institution", "wholesale", "wholesaler", "wholesale customer",
            "distributor", "dealer", "reseller", "retailer", "agent", "agency", "partner", "merchant", "supplier",
            "vendor", "trade", "commercial", "industrial", "sme", "supermarket", "chain", "b2b customer",
            "doanh nghiệp", "khách doanh nghiệp", "khách hàng doanh nghiệp", "công ty", "đại lý", "tổng đại lý",
            "nhà phân phối", "npp", "đối tác", "tổ chức", "cơ quan", "bán buôn", "bán sỉ", "sỉ", "buôn", "khách sỉ",
            "siêu thị", "chuỗi", "cửa hàng", "nhà hàng", "khách sạn"],
}
ALERT_TABLE = {
    "Alert": ["alert", "alerts", "urgent", "urgency", "khẩn", "khẩn cấp", "gấp", "rất gấp", "cảnh báo", "ưu tiên",
              "ưu tiên cao", "priority", "high priority", "high", "rush", "express", "hỏa tốc", "critical",
              "nghiêm trọng", "warning", "warn", "late", "delayed", "trễ", "trễ hạn", "quá hạn", "overdue", "risk",
              "red", "sla breach", "true", "yes", "y", "1", "flag", "flagged", "emergency", "immediate", "asap",
              "same day", "giao nhanh", "nhanh"],
    "Normal": ["normal", "bình thường", "thường", "standard", "regular", "routine", "low", "thấp", "medium", "trung bình",
               "none", "no", "n", "0", "false", "ok", "on time", "đúng hạn", "green", "xanh", "không", "k", "ko",
               "không gấp", "không khẩn"],
}
STATUS_TABLE = {
    "Mới tạo": ["new", "created", "placed", "open", "draft", "new order", "pending", "awaiting", "received",
                "pending confirmation", "unconfirmed", "mới", "mới tạo", "đơn mới", "chờ xác nhận", "chờ duyệt",
                "tiếp nhận", "đã nhận đơn", "mới đặt"],
    "Đang xử lý": ["processing", "in progress", "inprogress", "picking", "packing", "packed", "preparing", "confirmed",
                   "approved", "ready", "ready to ship", "scheduled", "planned", "assigned", "đang xử lý", "xử lý",
                   "đang chuẩn bị", "đã xác nhận", "đang đóng gói", "đang soạn hàng", "chờ giao", "chờ lấy hàng",
                   "đã lên kế hoạch"],
    "Đang giao": ["shipping", "shipped", "delivering", "in transit", "transit", "out for delivery", "dispatched",
                  "on the way", "on delivery", "đang giao", "đang giao hàng", "đang vận chuyển", "vận chuyển",
                  "đã xuất kho", "xuất kho", "đang đi giao"],
    "Đã giao": ["delivered", "completed", "complete", "done", "finished", "closed", "success", "successful",
                "đã giao", "đã giao hàng", "hoàn thành", "thành công", "giao thành công"],
    "Đã hủy": ["cancelled", "canceled", "cancel", "void", "rejected", "hủy", "đã hủy", "bị hủy", "hủy đơn", "từ chối"],
    "Giao thất bại": ["failed", "failed delivery", "undelivered", "giao thất bại", "không giao được",
                      "giao không thành công", "bom hàng"],
    "Trả hàng": ["returned", "return", "refund", "refunded", "trả hàng", "hoàn hàng", "hoàn trả", "hoàn tiền", "đã trả"],
}
def build_lookup(table):
    exact = {}
    for label, keys in table.items():
        for k in keys:
            nk = norm(k)
            if nk: exact.setdefault(nk, label)
    cont = sorted(((k, l, re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"))
                   for k, l in exact.items() if len(k) >= 3), key=lambda x: -len(x[0]))
    return {"exact": exact, "cont": cont, "keys": [k for k in exact if len(k) >= 4]}
TYPE_LK, ALERT_LK, STATUS_LK = build_lookup(TYPE_TABLE), build_lookup(ALERT_TABLE), build_lookup(STATUS_TABLE)
DICT_LOOKUPS = {"order_type": TYPE_LK, "alert_status": ALERT_LK, "order_status": STATUS_LK}
def match_label(value, lk, fuzzy_cut=88):
    """Trả (nhãn chuẩn, cách khớp, từ khóa) hoặc None. Khớp: chính xác > chứa từ khóa dài nhất > fuzzy."""
    n = norm(to_text(value))
    if not n: return None
    if n in lk["exact"]: return lk["exact"][n], "khớp từ điển", n
    hits = [(k, l) for k, l, rx in lk["cont"] if rx.search(n)]
    if hits:
        best = max(len(k) for k, _ in hits)
        top = sorted([(k, l) for k, l in hits if len(k) == best])
        return top[0][1], f"chứa từ khóa '{top[0][0]}'", top[0][0]
    if len(n) >= 4 and lk["keys"]:
        r = rf_process.extractOne(n, lk["keys"], scorer=fuzz.ratio)
        if r and r[1] >= fuzzy_cut:
            return lk["exact"][r[0]], f"gần đúng {r[1]:.0f}% với '{r[0]}'", r[0]
    return None
B2B_NAME_RX = re.compile(r"(?<![a-z0-9])(cong ty|ctcp|tnhh|co phan|joint stock|jsc|ltd|llc|corp|corporation|inc|company|"
                         r"doanh nghiep|dai ly|nha phan phoi|npp|tap doan|nha hang|khach san|sieu thi|cua hang|"
                         r"xi nghiep|nha may|ngan hang|bank|group|holdings|enterprise|trading|logistics)(?![a-z0-9])")
# ==========================================================
# 4. ĐỌC SỐ + ĐƠN VỊ (kg, tấn, g, m3, lít, kích thước DxRxC)
# ==========================================================
NUM_RE = re.compile(r"[-+]?\d[\d.,]*")
WEIGHT_UNITS = {"kg": 1, "kgs": 1, "kilogram": 1, "kilograms": 1, "ky": 1, "g": 1e-3, "gr": 1e-3, "gam": 1e-3,
                "gram": 1e-3, "grams": 1e-3, "mg": 1e-6, "t": 1000, "tan": 1000, "tonne": 1000, "tonnes": 1000,
                "ton": 1000, "tons": 1000, "ta": 100, "yen": 10, "lb": 0.45359237, "lbs": 0.45359237,
                "pound": 0.45359237, "pounds": 0.45359237, "oz": 0.0283495}
VOLUME_UNITS = {"m3": 1, "cbm": 1, "metkhoi": 1, "khoi": 1, "cm3": 1e-6, "cc": 1e-6, "ml": 1e-6, "l": 1e-3,
                "lit": 1e-3, "litre": 1e-3, "liter": 1e-3, "litres": 1e-3, "liters": 1e-3, "dm3": 1e-3,
                "ft3": 0.0283168, "cuft": 0.0283168}
WUNIT_RX = re.compile(r"\d\s*(kg|kgs|g|gr|gam|gram|mg|tan|ta|yen|lb|lbs|t)\b")
VUNIT_RX = re.compile(r"\d\s*(m3|cbm|cm3|cc|ml|dm3|l|lit|ft3)\b|\d\s*[x*]\s*\d")
DIM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[x*]\s*(\d+(?:[.,]\d+)?)\s*[x*]\s*(\d+(?:[.,]\d+)?)\s*([a-z]*)")
LEN_UNITS = {"mm": 1e-3, "cm": 1e-2, "dm": 1e-1, "m": 1.0}
def _to_float(tok):
    t = tok.strip().rstrip(".,")
    neg = t.startswith("-")
    t = t.lstrip("+-")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        thousands = len(parts) > 2 or (len(parts[1]) == 3 and parts[0] not in ("0", ""))
        t = t.replace(",", "") if thousands else t.replace(",", ".")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    try: v = float(t)
    except ValueError: return None
    return -v if neg else v
def _ascii(v):
    return unidecode(str(v)).lower().replace("^", "").replace("×", "x")
def parse_measure(val, units):
    """-> (giá trị đã quy đổi hoặc None, ghi chú quy đổi)."""
    if is_blank(val): return None, ""
    if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
        return float(val), ""
    s = _ascii(val)
    m = NUM_RE.search(s)
    if not m: return None, "không đọc được số"
    num = _to_float(m.group(0))
    if num is None: return None, "không đọc được số"
    rest = re.sub(r"[^a-z0-9]", "", s[m.end():])
    if not rest or not units: return num, ""
    factor = units.get(rest)
    if factor is None:
        cands = [u for u in sorted(units, key=len, reverse=True) if len(u) >= 2 and rest.startswith(u)]
        factor = units[cands[0]] if cands else None
    if factor is None: return num, f"đơn vị lạ '{rest}', giữ nguyên số"
    return num * factor, ("" if factor == 1 else f"quy đổi '{to_text(val)}'")
def parse_volume(val):
    if is_blank(val): return None, ""
    if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
        return float(val), ""
    s = _ascii(val)
    d = DIM_RE.search(s)
    if d:
        a, b, c = (_to_float(d.group(i)) for i in (1, 2, 3))
        if None not in (a, b, c):
            unit = d.group(4)
            f = LEN_UNITS.get(unit, 1e-2 if max(a, b, c) > 5 else 1.0)
            return a * b * c * f ** 3, f"tính từ kích thước '{to_text(val)}'"
    return parse_measure(val, VOLUME_UNITS)
def parse_date_iso(v):
    if is_blank(v): return ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(v, errors="coerce", dayfirst=True)
    return "" if pd.isna(d) else d.strftime("%Y-%m-%d")
# ==========================================================
# 5. CHẤM ĐIỂM ÁNH XẠ CỘT = TÊN CỘT + NỘI DUNG CỘT
# ==========================================================
ID_RE = re.compile(r"^[A-Za-z]{0,10}[-_/ ]?\d{2,}[A-Za-z0-9\-_/]*$")
DATE_RX = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
ADDR_TOKENS = ["duong", "pho", "phuong", "quan", "huyen", "tinh", "tp", "thanh pho", "thon", "ngo", "ngach", "hem",
               "xom", "khu pho", "street", "road", "rd", "ward", "district", "city", "avenue", "lane", "ha noi",
               "ho chi minh", "tphcm", "hcm", "da nang", "viet nam", "vietnam"]
STRONG_CONTENT = {"order_type", "alert_status", "order_status", "order_date", "address"}
def name_score(col, field):
    n = norm(col)
    if not n: return 0.0
    nc, toks = n.replace(" ", ""), n.split()
    best = 0.0
    for a in ALIAS_N[field]:
        ac = a.replace(" ", "")
        if n == a: s = 1.0
        elif nc == ac: s = 0.98
        elif len(a) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", n): s = 0.80 + 0.15 * len(a) / len(n)
        elif len(n) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", a): s = 0.62
        elif len(a) >= 3 and len(n) >= 3: s = 0.8 * max(fuzz.token_sort_ratio(n, a), fuzz.ratio(nc, ac)) / 100
        else: s = 0.0
        best = max(best, s)
    if any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", n) for t in NEG_N[field]):
        best *= 0.45
    return min(best, 1.0)
def _is_addr(t):
    n = " " + norm(t) + " "
    return any(f" {tok} " in n for tok in ADDR_TOKENS) or (t.count(",") >= 2 and len(t) > 15)
def content_score(texts, field):
    """Điểm 0..1 dựa trên GIÁ TRỊ trong cột (texts đã là list chuỗi, tối đa 300 dòng)."""
    n = len(texts)
    if n == 0: return 0.0
    uniq = len(set(texts)) / n
    def frac(pred): return sum(1 for t in texts if pred(t)) / n
    if field == "order_id":
        return frac(lambda t: bool(ID_RE.match(t))) * (1.0 if uniq >= 0.95 else 0.5)
    if field == "customer_id":
        return frac(lambda t: bool(ID_RE.match(t))) * (0.6 + 0.4 * (uniq < 0.95))
    if field == "customer_name":
        return frac(lambda t: len(t.split()) >= 2 and sum(ch.isdigit() for ch in t) / len(t) < 0.2 and not _is_addr(t)) * 0.8
    if field == "quantity":
        vals = [parse_measure(t, {})[0] for t in texts]
        ok = [v for v in vals if v is not None]
        if not ok: return 0.0
        return 0.5 if (all(float(v).is_integer() for v in ok) and max(ok) <= 10000 and len(ok) / n >= 0.9) else 0.2 * len(ok) / n
    if field == "items":
        return frac(lambda t: any(ch.isalpha() for ch in t) and len(t) >= 3 and not _is_addr(t)) * \
               (0.7 if frac(lambda t: "," in t or " x " in t.lower()) >= 0.3 else 0.4)
    if field == "total_weight":
        if frac(lambda t: bool(WUNIT_RX.search(_ascii(t)))) >= 0.5: return 0.95
        return 0.3 * frac(lambda t: parse_measure(t, WEIGHT_UNITS)[0] is not None)
    if field == "total_volume":
        if frac(lambda t: bool(VUNIT_RX.search(_ascii(t)))) >= 0.5: return 0.95
        return 0.3 * frac(lambda t: parse_measure(t, VOLUME_UNITS)[0] is not None)
    if field == "address":
        return frac(_is_addr)
    if field == "order_date":
        def ok(t):
            if not DATE_RX.search(t): return False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return not pd.isna(pd.to_datetime(t, errors="coerce", dayfirst=True))
        return frac(ok)
    if field in DICT_LOOKUPS:                       # type / alert / status: tỉ lệ giá trị khớp từ điển
        vc = Counter(texts)
        hit = sum(c for v, c in vc.items()
                  if len(norm(v)) >= 2 and any(ch.isalpha() for ch in v) and match_label(v, DICT_LOOKUPS[field]))
        return hit / n
    return 0.0
def combine(ns, cs, field):
    s = 0.6 * ns + 0.4 * cs
    if field in STRONG_CONTENT and cs >= 0.8:
        s = max(s, 0.55 * cs + 0.30 * ns + 0.10)
    return s
def orders_semantic_mapping(df):
    """Chọn cặp (trường, cột) điểm cao nhất toàn cục -> {field: (cột, điểm, căn cứ)}."""
    samples = {c: [t for t in (to_text(v) for v in df[c].dropna().head(300)) if t] for c in df.columns}
    cands = []
    for field in FIELDS:
        for col in df.columns:
            ns, cs = name_score(col, field), content_score(samples[col], field)
            cands.append((combine(ns, cs, field), field, col, ns, cs))
    cands.sort(key=lambda x: -x[0])
    result = {f: (None, 0.0, "") for f in FIELDS}
    used_f, used_c = set(), set()
    for score, field, col, ns, cs in cands:
        if score < MAP_THRESHOLD: break
        if field in used_f or col in used_c: continue
        result[field] = (col, score, f"tên cột {ns:.0%} · nội dung {cs:.0%}")
        used_f.add(field); used_c.add(col)
    return result
# ==========================================================
# 6. CHUẨN HÓA 1 ĐƠN
# ==========================================================
def process_order(raw, reports):
    """raw: {field: giá trị gốc hoặc None}. reports: Counter ghi lại các phép chuẩn hóa giá trị."""
    notes = []
    def log(field, original, standard, how):
        if to_text(original) != standard:
            reports[(field, to_text(original), standard, how)] += 1
    r_name = to_text(raw.get("customer_name"))
    name_b2b = bool(B2B_NAME_RX.search(norm(r_name)))
    cust_name = r_name or "Khách lẻ"
    # --- Loại đơn: Individual->B2C, Business->B2B ...
    r_type = to_text(raw.get("order_type"))
    if r_type:
        m = match_label(r_type, TYPE_LK)
        if m:
            t_type = m[0]
            log("order_type", r_type, t_type, m[1])
            if t_type == "B2C" and name_b2b:
                notes.append("⚠️ Tên khách giống doanh nghiệp nhưng loại đơn là B2C")
        elif name_b2b:
            t_type = "B2B"; log("order_type", r_type, t_type, "suy từ tên khách")
            notes.append(f"⚠️ Loại đơn '{r_type}' lạ → suy ra B2B từ tên khách")
        else:
            t_type = "B2C"; log("order_type", r_type, t_type, "mặc định")
            notes.append(f"⚠️ Không nhận diện loại đơn '{r_type}' → mặc định B2C")
    else:
        t_type = "B2B" if name_b2b else "B2C"
        if name_b2b: log("order_type", "(trống)", t_type, "suy từ tên khách")
    # --- Alert / Normal
    r_alert = to_text(raw.get("alert_status"))
    if r_alert:
        m = match_label(r_alert, ALERT_LK)
        if m: t_alert = m[0]; log("alert_status", r_alert, t_alert, m[1])
        else:
            t_alert = "Normal"; log("alert_status", r_alert, t_alert, "mặc định")
            notes.append(f"⚠️ Không nhận diện mức cảnh báo '{r_alert}' → mặc định Normal")
    else:
        t_alert = "Normal"
    # --- Tình trạng đơn
    r_status = to_text(raw.get("order_status"))
    if r_status:
        m = match_label(r_status, STATUS_LK)
        if m: status = m[0]; log("order_status", r_status, status, m[1])
        else: status = r_status
    else:
        status = "Mới tạo"
    # --- Số lượng, trọng lượng, thể tích
    q, _ = parse_measure(raw.get("quantity"), {})
    quantity = max(int(q), 1) if q is not None else 1
    w, w_note = parse_measure(raw.get("total_weight"), WEIGHT_UNITS)
    if w_note and "quy đổi" in w_note: log("total_weight", raw.get("total_weight"), f"{w:g} kg", w_note)
    elif w_note: notes.append(f"⚠️ Trọng lượng: {w_note}")
    v, v_note = parse_volume(raw.get("total_volume"))
    if v_note and ("quy đổi" in v_note or "tính từ" in v_note): log("total_volume", raw.get("total_volume"), f"{v:.4g} m3", v_note)
    elif v_note: notes.append(f"⚠️ Thể tích: {v_note}")
    if not is_blank(raw.get("total_weight")) and w is None: notes.append("⚠️ Không đọc được trọng lượng → 0")
    if not is_blank(raw.get("total_volume")) and v is None: notes.append("⚠️ Không đọc được thể tích → 0")
    rec = {
        "order_id": to_text(raw.get("order_id")),
        "customer_id": to_text(raw.get("customer_id")),
        "customer_name": cust_name,
        "quantity": quantity,
        "items": to_text(raw.get("items")),
        "total_weight_kg": round(w or 0.0, 4),
        "total_volume_m3": round(v or 0.0, 6),
        "address": to_text(raw.get("address")),
        "order_status": status,
        "order_type": t_type,
        "alert_status": t_alert,
    }
    if "order_date" in raw and raw["order_date"] is not None:
        rec["order_date"] = parse_date_iso(raw["order_date"])
    rec["ghi_chú"] = " | ".join(notes)
    return rec
def orders_validate_output(df):
    out = df.copy().reset_index(drop=True)
    dup = out["order_id"].astype(str).duplicated(keep=False)
    msgs = []
    for i, row in out.iterrows():
        errs, warns = [], []
        if is_blank(row.get("order_id")): errs.append("Thiếu Mã đơn")
        if is_blank(row.get("address")): errs.append("Thiếu Địa chỉ")
        if dup.iloc[i] and not is_blank(row.get("order_id")): errs.append("Trùng Mã đơn")
        if is_blank(row.get("customer_id")): warns.append("Thiếu Mã khách")
        if parse_measure(row.get("total_weight_kg"), {})[0] in (None, 0.0): warns.append("Trọng lượng = 0")
        if parse_measure(row.get("total_volume_m3"), {})[0] in (None, 0.0): warns.append("Thể tích = 0")
        warns += [n.replace("⚠️ ", "") for n in str(row.get("ghi_chú", "")).split(" | ") if n.startswith("⚠️")]
        if errs: msgs.append("❌ " + "; ".join(errs + warns))
        elif warns: msgs.append("⚠️ " + "; ".join(warns))
        else: msgs.append("✅ Đủ dữ liệu đơn hàng chuẩn")
    out["kiểm_tra"] = msgs
    return out
def conf_icon(s):
    return "🟢" if s >= 0.75 else ("🟡" if s >= 0.55 else "🟠")


def orders_empty_table():
    return pd.DataFrame({
        "Mã đơn": ["ORD_001", "ORD_002", "ORD_003", "ORD_004"],
        "Mã khách": ["CUS_01", "CUS_02", "CUS_03", "CUS_04"],
        "Tên khách": ["Nguyễn Văn A", "Công ty TNHH Nội Thất Việt", "Trần Thị B", "Đại lý Minh Phát"],
        "Số lượng": [2, 10, 1, 6],
        "Mặt hàng": ["Ghế sofa, Bàn trà", "Bàn làm việc", "Tủ quần áo", "Giường gỗ"],
        "Tổng trọng lượng (kg)": [45.5, 320.0, 80.0, 450.0],
        "Tổng thể tích (m3)": [0.8, 6.5, 1.2, 5.0],
        "Địa chỉ khách": [
            "Số 88 - Đường Cổ Linh - Long Biên - Hà Nội",
            "Số 1 Đường Trần Duy Hưng, Cầu Giấy, Hà Nội",
            "Số 25 Đường Láng Hạ, Đống Đa, Hà Nội",
            "Số 120 Đường Nguyễn Trãi, Thanh Xuân, Hà Nội",
        ],
        "Tình trạng đơn": ["Đang xử lý", "Mới tạo", "New", "Shipping"],
        "Loại đơn": ["Individual", "Business", "Retail", "Distributor"],
        "Tình trạng Alert": ["Normal", "Urgent", "Normal", "High"],
    })


# ============================================================================
# TAB 5 — UI
# ============================================================================
def render_orders_tab():
    st.header("🧾 Smart Logistics — Quản lý & Chuẩn hóa Đơn hàng")
    st.markdown("**Input → Semantic Mapping (cột + giá trị) → Làm sạch → Validate → Export Excel/JSON**")
    st.info("🔒 Tự nhận diện cột theo *tên + nội dung*, và hiểu giá trị như `Individual`→B2C, `Business`→B2B, "
            "`Urgent`→Alert, `Shipping`→Đang giao, `2 tấn`→2000 kg...")
    mode, edited, uploaded = render_input_block("orders", "đơn hàng", orders_empty_table())

    if st.button("🔍 Quét & Semantic Mapping Đơn hàng", key="orders_scan", type="primary"):
        if run_scan("orders", "đơn hàng", mode, uploaded, edited, orders_semantic_mapping):
            meta = st.session_state["orders_meta"]
            why = st.session_state["orders_why"]
            rows = []
            for fld in FIELDS:
                col = st.session_state.get(f"orders_col_{fld}", NONE)
                label = ORDER_FIELDS[fld]["label"]
                if col == NONE:
                    rows.append({"Trường": label, "Cột được chọn": "(không tìm thấy)", "Độ tin cậy": "–", "Căn cứ": "–"})
                else:
                    rows.append({"Trường": label, "Cột được chọn": col,
                                 "Độ tin cậy": f"{conf_icon(meta[fld])} {meta[fld]:.0%}", "Căn cứ": why[fld]})
            st.session_state["orders_auto_table"] = pd.DataFrame(rows)

    raw = st.session_state.get("orders_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    auto = st.session_state.get("orders_auto_table")
    if auto is not None:
        st.dataframe(auto)
        st.caption("🟢 chắc chắn · 🟡 nên kiểm tra · 🟠 độ tin cậy thấp — bạn có thể đổi trong các ô bên dưới.")
    st.markdown("### 🔗 Kiểm tra ánh xạ cột (cột nào không có chọn '-- Không sử dụng --')")
    st.caption("Xem trước dữ liệu gốc (8 dòng đầu)")
    st.dataframe(raw.head(8))
    specs = [(f, ORDER_FIELDS[f]["label"]) for f in FIELDS]
    chosen = render_mapping("orders", specs, raw)

    if st.button("🚀 Chuẩn hóa & Xử lý Đơn hàng", key="orders_process", type="primary"):
        try:
            colmap = build_colmap(raw, chosen)
            reports, rows = Counter(), []
            for _, row in raw.iterrows():
                rows.append(process_order({f: row.get(c) for f, c in colmap.items()}, reports))
            out = orders_validate_output(pd.DataFrame(rows))
            n_ok = int((~out["kiểm_tra"].astype(str).str.startswith("❌")).sum())
            n_warn = int(out["kiểm_tra"].astype(str).str.startswith("⚠️").sum())
            n_b2b = int((out["order_type"] == "B2B").sum())
            n_alert = int((out["alert_status"] == "Alert").sum())
            rep = pd.DataFrame(
                [{"Trường": ORDER_FIELDS[f]["label"], "Giá trị gốc": o, "Chuẩn hóa thành": s,
                  "Cách nhận diện": h, "Số dòng": n} for (f, o, s, h), n in reports.most_common()],
                columns=["Trường", "Giá trị gốc", "Chuẩn hóa thành", "Cách nhận diện", "Số dòng"])
            files = save_outputs(OUT_ORDERS, "DIM_ORDERS", "DIM_ORDERS", out, extra_sheets={"VALUE_MAPPING": rep})
            store_result(
                "orders", out, files,
                f"🧭 Hoàn tất chuẩn hóa đơn hàng — {n_ok}/{len(out)} dòng không có lỗi (trong đó {n_warn} dòng có cảnh báo ⚠️)",
                [("Tổng số đơn", len(out)), ("B2B", n_b2b), ("B2C", len(out) - n_b2b), ("Alert", n_alert)],
                extra={"report": rep})
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    res = render_result("orders", "📊 Kết quả đơn hàng đã chuẩn hóa")
    if res is not None:
        st.markdown("### 🔁 Bảng ánh xạ giá trị (giá trị gốc → giá trị chuẩn)")
        rep = res["extra"].get("report")
        if rep is not None and len(rep):
            st.dataframe(rep)
        else:
            st.caption("Không có giá trị nào cần quy đổi.")


# ============================================================================
# TAB 6 — LOGIC ĐỊNH TUYẾN (đọc input từ các thư mục output_*)
# ============================================================================
@dataclass
class Config:
    order_file: str = os.path.join(OUT_ORDERS, "DIM_ORDERS.xlsx")
    vehicle_file: str = os.path.join(OUT_FLEET, "DIM_VEHICLE.xlsx")
    driver_file: str = os.path.join(OUT_DRIVER, "DIM_DRIVER.xlsx")
    warehouse_file: str = os.path.join(OUT_WAREHOUSE, "WAREHOUSE_WITH_COORDINATES.xlsx")
    product_file: str = os.path.join(OUT_PRODUCT, "DIM_PRODUCT.xlsx")
    matrix_file: str = os.path.join(OUT_MATRIX, "DISTANCE_MATRIX_KM.xlsx")
    cust_file: str = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.xlsx")
    output_daily_file: str = os.path.join(OUT_CUSTOMER, "DIM_CUSTOMER.xlsx")
    start_time: str = "08:30"
    max_route_hours: float = 8.0
    detour_factor: float = 1.2
    service_min: dict = dc_field(default_factory=lambda: {"B2B": 105, "B2C": 60})
    fixed_cost_col: str | None = None
    variable_cost_col: str | None = None
    overnight_cost: float = 300_000
    backup_driver_cost: float = 400_000
    late_penalty_per_day: float = 100_000


CFG = Config()


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def first_col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def money(x) -> str:
    return f"{x:,.0f}"


def _positive(series, default):
    """Số > 0, ngược lại (NaN / 0 / âm) dùng giá trị mặc định."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s > 0).fillna(default)


def _num(x, default=0.0) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(default if pd.isna(v) else v)


def read_matrix(path) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def resolve_customer_file(cfg: Config) -> str:
    """Tìm DATASET_CUSTOMER.xlsx; nếu không có, tự dò file .xlsx bất kỳ trong output_customer."""
    if os.path.exists(cfg.cust_file):
        return cfg.cust_file
    folder = os.path.dirname(cfg.cust_file)
    if os.path.isdir(folder):
        cands = sorted(f for f in os.listdir(folder) if f.endswith(".xlsx") and not f.startswith("~"))
        if cands:
            return os.path.join(folder, cands[0])
    raise UserError("Không tìm thấy dữ liệu khách hàng trong `output_customer/`. Hãy chạy bước 1 (Geocode khách hàng) trước.")


def load_warehouse_coords(cfg: Config) -> dict:
    """Đọc toạ độ kho đã geocode ở Tab 2 -> {id_kho: (lat, lon)}."""
    coords = {}
    if not os.path.exists(cfg.warehouse_file):
        return coords
    df_wh = pd.read_excel(cfg.warehouse_file)
    wid = first_col(df_wh, "id_warehouse", "warehouse_id", "wh_id", "depot_id")
    lat_c = first_col(df_wh, "lat", "latitude", "LATITUDE")
    lon_c = first_col(df_wh, "lng", "lon", "longitude", "LONGITUDE")
    if not (wid and lat_c and lon_c):
        return coords
    for _, r in df_wh.iterrows():
        if pd.notna(r[wid]) and pd.notna(r[lat_c]) and pd.notna(r[lon_c]):
            coords[str(r[wid]).strip()] = (float(r[lat_c]), float(r[lon_c]))
    return coords


def load_data(cfg: Config = CFG) -> dict:
    notes = []
    df_orders = pd.read_excel(cfg.order_file)
    df_veh = pd.read_excel(cfg.vehicle_file).copy()
    df_driver = pd.read_excel(cfg.driver_file).copy()
    df_dist = read_matrix(cfg.matrix_file)
    df_cust = pd.read_excel(resolve_customer_file(cfg))
    wh_coords = load_warehouse_coords(cfg)

    if df_orders.empty:
        raise UserError("DIM_ORDERS.xlsx không có đơn hàng — hãy xử lý đơn hàng ở Tab 5.")
    if df_veh.empty:
        raise UserError("DIM_VEHICLE.xlsx không có xe — hãy khai báo hạm đội ở Tab 1.")

    # --- Trích xuất linh hoạt cột từ dữ liệu xe (bổ sung tên cột của DIM_VEHICLE: id_warehouse,
    #     average_speed_kmh, variable_cost) ---
    veh_id_col = first_col(df_veh, "vehicle_id", "VEHICLE_ID", "id")
    plate_col = first_col(df_veh, "license_plate", "bien_so", "plate", "vehicle_id")
    type_col = first_col(df_veh, "vehicle_type", "type", "vehicle_class", "vehicle_name")
    wh_col = first_col(df_veh, "wh_id", "warehouse_id", "depot_id", "id_warehouse")
    speed_col = first_col(df_veh, "speed_kmh", "speed", "vận_tốc", "average_speed_kmh")
    w_col = first_col(df_veh, "max_weight_kg", "weight_capacity", "max_weight", "capacity_kg")
    v_col = first_col(df_veh, "max_volume_m3", "volume_capacity", "max_volume", "capacity_m3")

    df_veh["vehicle_id"] = df_veh[veh_id_col].astype(str) if veh_id_col else [f"VEH_{i}" for i in range(len(df_veh))]
    df_veh["license_plate"] = df_veh[plate_col].astype(str) if plate_col else df_veh["vehicle_id"]
    wh_series = df_veh[wh_col].astype(str).str.strip() if wh_col else pd.Series("WH_DEFAULT", index=df_veh.index)
    df_veh["wh_id"] = wh_series.replace({"": "WH_DEFAULT", "nan": "WH_DEFAULT", "None": "WH_DEFAULT"})
    df_veh["speed_kmh"] = _positive(df_veh[speed_col], 35.0) if speed_col else 35.0
    df_veh["max_weight_kg"] = _positive(df_veh[w_col], 1000.0) if w_col else 1000.0
    df_veh["max_volume_m3"] = _positive(df_veh[v_col], 5.0) if v_col else 5.0
    if type_col:
        df_veh["vehicle_type"] = df_veh[type_col].astype(str).str.strip()
    else:  # DIM_VEHICLE không có loại xe -> phân loại theo trọng tải
        df_veh["vehicle_type"] = df_veh["max_weight_kg"].map(lambda w: f"Xe {w:g}kg")

    fx_col = cfg.fixed_cost_col or first_col(df_veh, "fixed_cost", "fixed_cost_per_day")
    vr_col = cfg.variable_cost_col or first_col(df_veh, "variable_cost_per_km", "cost_per_km", "variable_cost")
    df_veh["fixed_cost"] = pd.to_numeric(df_veh[fx_col], errors="coerce").fillna(300_000.0) if fx_col else 300_000.0
    df_veh["variable_cost_per_km"] = pd.to_numeric(df_veh[vr_col], errors="coerce").fillna(8_000.0) if vr_col else 8_000.0

    # --- Kho: danh sách kho lấy từ xe, toạ độ lấy từ output_warehouse (nếu có) ---
    warehouses, no_coord = {}, []
    for wh in df_veh["wh_id"].unique():
        if wh in wh_coords:
            lat, lon = wh_coords[wh]
        else:
            lat, lon = DEFAULT_DEPOT
            no_coord.append(wh)
        warehouses[wh] = {"name": f"Kho {wh}", "lat": lat, "lon": lon}
    if no_coord:
        notes.append("Kho chưa có toạ độ trong `output_warehouse` (dùng toạ độ mặc định Hà Nội "
                     f"{DEFAULT_DEPOT}): {', '.join(map(str, no_coord))}. Hãy khớp **ID kho** ở Tab 1 với **Mã kho** ở Tab 2.")

    # --- Tài xế theo kho ---
    d_name_col = first_col(df_driver, "driver_name", "name", "full_name", "TÊN", "driver_id")
    d_role_col = first_col(df_driver, "role", "position", "VAI_TRÒ", "type")
    d_wh_col = first_col(df_driver, "wh_id", "warehouse_id", "depot_id", "KHO", "id_warehouse")
    df_driver["driver_name"] = df_driver[d_name_col].astype(str) if d_name_col else "Tài xế"
    df_driver["role"] = df_driver[d_role_col].astype(str).str.strip().str.capitalize() if d_role_col else "Chính"
    df_driver["wh_id"] = df_driver[d_wh_col].astype(str).str.strip() if d_wh_col else list(warehouses.keys())[0]
    drivers_by_wh = {}
    for wh in warehouses:
        sub = df_driver[df_driver["wh_id"] == wh]
        chính = sub[sub["role"].str.contains("Chính|Primary|Driver", case=False, na=False)]["driver_name"].tolist()
        phụ = sub[sub["role"].str.contains("Phụ|Assistant|Helper|Hỗ trợ|Support", case=False, na=False)]["driver_name"].tolist()
        if not chính:
            chính = sub["driver_name"].tolist() or ["Tài xế chính"]
        if not phụ:
            phụ = ["Phụ xe hỗ trợ"]
        if sub.empty:
            notes.append(f"Kho {wh}: không có tài xế nào khai báo ở Tab 4 (sẽ dùng tài xế dự phòng).")
        drivers_by_wh[wh] = {"chính": chính, "phụ": phụ}

    # --- Khách hàng (chỉ giữ khách có toạ độ hợp lệ; khách thiếu toạ độ sẽ rơi vào nhóm ngoại lệ) ---
    addr_col = first_col(df_cust, "address", "Location", "ADDRESS")
    lat_col = first_col(df_cust, "lat", "LATITUDE", "latitude")
    lon_col = first_col(df_cust, "lng", "lon", "LONGITUDE", "longitude")
    cid_col = first_col(df_cust, "customer_id", "CUSTOMER_ID", "id")
    cust = {}
    for idx, r in df_cust.iterrows():
        cid = str(r[cid_col]).strip() if cid_col and pd.notna(r[cid_col]) else str(idx)
        if not (lat_col and lon_col) or pd.isna(r[lat_col]) or pd.isna(r[lon_col]):
            continue
        cust[cid] = {
            "lat": float(r[lat_col]), "lon": float(r[lon_col]),
            "address": str(r[addr_col]) if addr_col and pd.notna(r[addr_col]) else "",
        }

    # --- Đơn hàng ---
    oid_c = first_col(df_orders, "order_id", "ORDER_ID")
    date_c = first_col(df_orders, "order_date", "delivery_date", "date", "NGÀY")
    wait_c = first_col(df_orders, "waiting_date", "WAITING_DATE")
    orders = []
    for i, r in df_orders.iterrows():
        cid_raw = r.get("customer_id")
        cid = "" if pd.isna(cid_raw) else str(cid_raw).strip()
        dt_val = pd.to_datetime(r[date_c], errors="coerce") if date_c and pd.notna(r.get(date_c)) else pd.NaT
        if pd.isna(dt_val):
            dt_val = pd.Timestamp.today()
        wait_val = pd.to_datetime(r[wait_c], errors="coerce") if wait_c and pd.notna(r[wait_c]) else pd.NaT
        orders.append({
            "ORDER_ID": str(r[oid_c]) if oid_c and pd.notna(r.get(oid_c)) else f"OR{i}",
            "customer_id": cid,
            "weight": _num(r.get("total_weight_kg", 0.0)),
            "volume": _num(r.get("total_volume_m3", 0.0)),
            "order_type": str(r.get("order_type", "B2C")),
            "date": str(dt_val.date()),
            "WAITING_DATE": None if pd.isna(wait_val) else str(wait_val.date()),
            "Location": cust.get(cid, {}).get("address", ""),
        })
    return {"orders": orders, "vehicles": df_veh, "drivers": drivers_by_wh, "dist": df_dist,
            "cust": cust, "warehouses": warehouses, "notes": notes}


# ============================================================================
# RÀNG BUỘC TẢI TRỌNG (screening chi tiết) & THỜI GIAN TUYẾN (<= 8h - Giữ nguyên)
# ============================================================================

import pandas as pd
import numpy as np

def run_master_logistics_optimizer(order_file="output_orders/DIM_ORDERS.xlsx",
                                   vehicle_file="output_fleet/DIM_VEHICLE.xlsx",
                                   matrix_file="output_matrix/DISTANCE_MATRIX_KM.xlsx"):
    """
    HÀM TỔNG MASTER (Cập nhật logic xét tải trọng, số lượng, chọn xe nhỏ nhất và phân loại cự ly/outsource):
    - Quét từng đơn hàng riêng lẻ.
    - Kiểm tra quá khổ > 150% xe lớn nhất -> Outsource 30k/km.
    - Lọc xe nội bộ đủ tải, đủ thể tích và còn xe trong kho (`available_quantity > 0`).
    - Chọn xe nhỏ nhất khả thi.
    - Kiểm tra `max_distance_km`: nếu vượt ngưỡng, xét các mức giá đặc biệt (20kg/0.6m3: 100k | 100kg/3m3: 500k).
    """
    print("🔮 [MASTER PIPELINE] Khởi động hệ thống điều phối logistics thông minh...")

    # 1. Kiểm tra và đọc file dữ liệu
    df_orders = pd.read_excel(order_file)
    df_vehicles = pd.read_excel(vehicle_file)
    df_dist = pd.read_excel(matrix_file, index_col=0)

    # Đảm bảo có cột số lượng xe trong kho (Mặc định gán mỗi loại 5 chiếc nếu chưa có)
    if "available_quantity" not in df_vehicles.columns:
        df_vehicles["available_quantity"] = 5

    valid_orders, oversized_orders, special_delivery_orders = [], [], []

    # Lấy thông số xe lớn nhất xét theo tải trọng và thể tích toàn cục
    max_fleet_weight = df_vehicles["max_weight_kg"].max()
    max_fleet_volume = df_vehicles["max_volume_m3"].max()

    for _, row in df_orders.iterrows():
        dest_node = row.get("destination_node", row.get("customer_id", df_dist.columns[0]))
        w = float(row.get("total_weight_kg", 0.0))
        v = float(row.get("total_volume_m3", 0.0))
        record = row.to_dict()

        # Lấy khoảng cách từ kho (node đầu tiên trong ma trận) đến điểm giao
        origin_node = df_dist.index[0]
        distance_km = float(df_dist.loc[origin_node, dest_node]) if dest_node in df_dist.columns else 0.0
        record["calculated_distance_km"] = distance_km

        # ----------------------------------------------------
        # BƯỚC 1: XÉT QUÁ CỠ TRÊN 150% XE LỚN NHẤT
        # ----------------------------------------------------
        if w > (1.5 * max_fleet_weight) or v > (1.5 * max_fleet_volume):
            record["handling_method"] = "OUTSOURCE"
            record["outsource_cost_vnd"] = distance_km * 30000
            oversized_orders.append(record)
            continue

        # ----------------------------------------------------
        # BƯỚC 2: QUÉT ĐỘI XE NHÀ (Đủ tải, đủ thể tích và còn xe)
        # ----------------------------------------------------
        feasible_vehicles = df_vehicles[
            (df_vehicles["max_weight_kg"] >= w) &
            (df_vehicles["max_volume_m3"] >= v) &
            (df_vehicles["available_quantity"] > 0)
        ].copy()

        if feasible_vehicles.empty:
            record["handling_method"] = "OUTSOURCE_NO_VEHICLE"
            record["outsource_cost_vnd"] = distance_km * 30000
            oversized_orders.append(record)
            continue

        # ----------------------------------------------------
        # BƯỚC 3: CHỌN XE NHỎ NHẤT TRONG CÁC XE KHẢ THI
        # ----------------------------------------------------
        feasible_vehicles["capacity_score"] = feasible_vehicles["max_weight_kg"] * 0.5 + feasible_vehicles["max_volume_m3"] * 0.5
        best_vehicle = feasible_vehicles.sort_values(by="capacity_score", ascending=True).iloc[0]
        
        assigned_vehicle_id = best_vehicle["vehicle_id"]
        max_allowed_distance = float(best_vehicle.get("max_distance_km", 99999))

        # ----------------------------------------------------
        # BƯỚC 4: KIỂM TRA QUÃNG ĐƯỜNG (DISTANCE CONSTRAINT)
        # ----------------------------------------------------
        if distance_km > max_allowed_distance:
            # Phân loại theo quy định khi vượt max_distance của xe
            if w < 20.0 and v < 0.6:
                record["handling_method"] = "GIAO_HANG_TIET_KIEM"
                record["shipping_cost_vnd"] = 100000
                special_delivery_orders.append(record)
            elif w < 100.0 and v < 3.0:
                record["handling_method"] = "CHUYEN_RIENG_500K"
                record["shipping_cost_vnd"] = 500000
                special_delivery_orders.append(record)
            else:
                record["handling_method"] = "OUTSOURCE_OVER_DISTANCE"
                record["outsource_cost_vnd"] = distance_km * 30000
                oversized_orders.append(record)
        else:
            record["handling_method"] = "INHOUSE_FLEET"
            record["assigned_vehicle_id"] = assigned_vehicle_id
            record["estimated_shipping_cost"] = distance_km * float(best_vehicle.get("cost_per_km", 5000))
            valid_orders.append(record)

    print(f"✅ Xe nội bộ tối ưu: {len(valid_orders)} đơn | 📦 Đặc biệt: {len(special_delivery_orders)} đơn | ❌ Outsource: {len(oversized_orders)} đơn")

    return {
        "valid_inhouse_orders": valid_orders,
        "special_delivery_orders": special_delivery_orders,
        "outsource_orders": oversized_orders,
        "distance_matrix": df_dist
    }


def evaluate_route_time_constraint(route_data, service_time_rules=None):
    """
    Hàm xử lý constraint thời gian tuyến (Time Windows <= 8 giờ): (GIỮ NGUYÊN 100%)
    - route_data: Dict chứa thông tin tuyến đường.
    - service_time_rules: Quy định thời gian bốc/dỡ hàng.
    """
    if service_time_rules is None:
        service_time_rules = {
            "B2C": {"loading": 25, "unloading": 35}, # Tổng 60 phút = 1 giờ
            "B2B": {"loading": 45, "unloading": 60}  # Tổng 105 phút = 1.75 giờ
        }
    # 1. Lấy thông tin từ tuyến
    orders_in_route = route_data.get("orders", [])
    total_distance_km = route_data.get("total_distance_km", 0.0)
    vehicle_speed_kmh = route_data.get("vehicle_speed_kmh", 40.0)
    current_date = route_data.get("current_date", "2026-04-03")

    # 2. Tính Travel Time (giờ) = Quãng đường / Vận tốc xe
    travel_time_hours = total_distance_km / vehicle_speed_kmh if vehicle_speed_kmh > 0 else 0.0

    # 3. Tính Service Time (tổng thời gian bốc/dỡ cho tất cả đơn trong tuyến) (đổi ra giờ)
    total_service_minutes = 0.0
    for order in orders_in_route:
        o_specs = service_time_rules.get(order.get("order_type", "B2C"), service_time_rules["B2C"])
        total_service_minutes += (o_specs["loading"] + o_specs["unloading"])
    service_time_hours = total_service_minutes / 60.0

    # 4. Tổng thời gian hoàn thành tuyến (giờ)
    total_route_duration_hours = travel_time_hours + service_time_hours
    MAX_HOURS_ALLOWED = 8.0 # Giới hạn tối đa 8 tiếng/ngày
    result_status = {}

    # --- PHÂN CASE THEO YÊU CẦU ---
    if total_route_duration_hours <= MAX_HOURS_ALLOWED:
        result_status = {
            "status": "APPROVED",
            "message": "✅ Đạt yêu cầu thời gian tuyến (<= 8h)",
            "total_hours": round(total_route_duration_hours, 2),
            "route": route_data.get("route", [])
        }
    else:
        backlog_orders = []
        for order in orders_in_route:
            order_backlog_info = order.copy()
            order_backlog_info["backlog_days_count"] = order.get("backlog_days_count", 0) + 1
            order_backlog_info["original_date"] = order.get("original_date", current_date)
            order_backlog_info["backlog_reason"] = f"Tuyến vượt quá 8h ({total_route_duration_hours:.2f}h)"
            backlog_orders.append(order_backlog_info)
        result_status = {
            "status": "BACKLOG_OR_REPOOL",
            "message": "⚠️ Tuyến vượt quá giới hạn 8h! Đẩy đơn sang pool xử lý ngầm (Backlog ngày tiếp theo / Tái gộp Clarke-Wright)",
            "total_hours": round(total_route_duration_hours, 2),
            "repool_orders": orders_in_route,
            "backlog_orders_next_day": backlog_orders
        }
    return result_status
# ============================================================================
# CLARKE-WRIGHT SAVINGS: RoutePlanner + simulate_all (Cập nhật logic xét tải trọng & phân nhóm mới)
# ============================================================================
class RoutePlanner:
    def __init__(self, data: dict, cfg: Config = CFG):
        self.cfg = cfg
        self.veh = data["vehicles"]
        self.drivers = data["drivers"]
        self.dist_df = data["dist"]
        self.cust = data["cust"]
        self.wh = data["warehouses"]
        
        # Đảm bảo có cột available_quantity trong hạm đội xe để check số lượng tồn kho thực tế
        if "available_quantity" not in self.veh.columns:
            self.veh["available_quantity"] = 5

        cat = (self.veh.groupby("vehicle_type")
               .agg(w=("max_weight_kg", "max"), v=("max_volume_m3", "max"), speed=("speed_kmh", "first"),
                    fixed=("fixed_cost", "first"), var=("variable_cost_per_km", "first"))
               .sort_values("w"))
        self.catalog = cat
        self.max_w = cat["w"].max() if not cat.empty else 1000.0
        self.max_v = cat["v"].max() if not cat.empty else 5.0

    def _hav(self, a, b) -> float:
        return haversine(a["lat"], a["lon"], b["lat"], b["lon"]) * self.cfg.detour_factor

    def d_cc(self, i, j) -> float:
        if i in self.dist_df.index and j in self.dist_df.columns:
            return float(self.dist_df.loc[i, j])
        return self._hav(self.cust[i], self.cust[j]) if (i in self.cust and j in self.cust) else 10.0

    def d_wc(self, wh_id, c) -> float:
        return self._hav(self.wh[wh_id], self.cust[c]) if (wh_id in self.wh and c in self.cust) else 10.0

    def nearest_wh(self, c) -> str:
        return min(self.wh, key=lambda w: self.d_wc(w, c))

    def fit_type(self, w, v):
        """
        Logic chọn xe mới: Xét đồng thời trọng tải, thể tích và số lượng xe còn lại trong kho.
        Chọn xe NHỎ NHẤT trong các xe khả thi.
        """
        if self.veh.empty: return "Truck"
        
        # Lọc các xe đủ tải, đủ thể tích và còn số lượng trong kho (> 0)
        ok_vehicles = self.veh[
            (self.veh["max_weight_kg"] >= w) & 
            (self.veh["max_volume_m3"] >= v) & 
            (self.veh["available_quantity"] > 0)
        ]
        
        if not ok_vehicles.empty:
            # Tính điểm dung tích (capacity score) để chọn chiếc xe NHỎ NHẤT vừa vặn
            ok_vehicles = ok_vehicles.copy()
            ok_vehicles["capacity_score"] = ok_vehicles["max_weight_kg"] * 0.5 + ok_vehicles["max_volume_m3"] * 0.5
            best_vrow = ok_vehicles.sort_values(by="capacity_score", ascending=True).iloc[0]
            return best_vrow["vehicle_type"]
        
        return self.catalog.index[-1] if not self.catalog.empty else "Truck"

    def km(self, route, wh_id) -> float:
        if not route: return 0.0
        return self.d_wc(wh_id, route[0]) + self.d_wc(wh_id, route[-1]) + sum(self.d_cc(a, b) for a, b in zip(route, route[1:]))

    def metrics(self, route, wh_id, demand):
        w = sum(demand[c]["weight"] for c in route)
        v = sum(demand[c]["volume"] for c in route)
        km = self.km(route, wh_id)
        vtype = self.fit_type(w, v)
        speed = self.catalog.loc[vtype, "speed"] if vtype in self.catalog.index else 35.0
        service = sum(self.cfg.service_min.get(demand[c]["order_type"], 60) for c in route) / 60
        return {"w": w, "v": v, "km": km, "hours": km / speed + service, "vtype": vtype, "speed": speed}

    def feasible(self, route, wh_id, demand) -> bool:
        m = self.metrics(route, wh_id, demand)
        return (m["w"] <= self.max_w and m["v"] <= self.max_v and m["hours"] <= self.cfg.max_route_hours)

    def two_opt(self, route, wh_id):
        best, improved = route[:], True
        while improved and len(best) > 2:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if self.km(cand, wh_id) < self.km(best, wh_id) - 1e-9:
                        best, improved = cand, True
        return best

    def clarke_wright(self, custs, wh_id, demand):
        route_of = {c: [c] for c in custs}
        savings = sorted(
            ((self.d_wc(wh_id, a) + self.d_wc(wh_id, b) - self.d_cc(a, b), a, b)
             for k, a in enumerate(custs) for b in custs[k + 1:]),
            reverse=True)
        for s, a, b in savings:
            if s <= 0: break
            ra, rb = route_of[a], route_of[b]
            if ra is rb or a not in (ra[0], ra[-1]) or b not in (rb[0], rb[-1]): continue
            ra = ra if ra[-1] == a else ra[::-1]
            rb = rb if rb[0] == b else rb[::-1]
            merged = ra + rb
            if self.feasible(merged, wh_id, demand):
                for c in merged: route_of[c] = merged
        uniq = {id(r): r for r in route_of.values()}.values()
        return [self.two_opt(r, wh_id) for r in uniq]

    def plan_day(self, date_str: str, day_orders: list) -> dict:
        cfg = self.cfg
        res = {"routes": [], "overdue_routes": [], "exceptions": [], "carried_orders": [], "overdue_backlog_list": [], "day_cost": 0.0, "penalty_cost": 0.0}
        date = pd.to_datetime(date_str)
        def waiting(o): return (date - pd.to_datetime(o["WAITING_DATE"])).days if o["WAITING_DATE"] else 0
        def exc(sev, o, kind, detail, handled, propose):
            res["exceptions"].append({"NGÀY": date_str, "MỨC ĐỘ": sev, "MÃ ĐƠN": o["ORDER_ID"], "PHÂN LOẠI": kind, "CHI TIẾT": detail, "ĐÃ XỬ LÝ": handled, "ĐỀ XUẤT": propose})
        
        normal, overdue = [], []
        max_fleet_w = self.veh["max_weight_kg"].max() if not self.veh.empty else 1000.0
        max_fleet_v = self.veh["max_volume_m3"].max() if not self.veh.empty else 5.0

        for o in day_orders:
            if o["customer_id"] not in self.cust:
                exc("MEDIUM", o, "THIẾU TỌA ĐỘ", "Không tìm thấy khách hàng", "Bỏ qua", "Bổ sung tọa độ")
                res["carried_orders"].append(o)
                continue
            
            # Xét tư duy trọng tải & quá cỡ mới (> 150% xe lớn nhất hoặc không có xe nào đáp ứng)
            w, v = o["weight"], o["volume"]
            matched_veh = self.veh[(self.veh["max_weight_kg"] >= w) & (self.veh["max_volume_m3"] >= v)]
            
            if w > (1.5 * max_fleet_w) or v > (1.5 * max_fleet_v) or matched_veh.empty:
                exc("CRITICAL", o, "ĐƠN QUÁ CỠ / OUTSOURCE", f"{w:.0f}kg vượt giới hạn/hết xe", "Outsource 30k/km", "Thuê ngoài vận chuyển")
                o["handling_method"] = "OUTSOURCE"
                res["carried_orders"].append(o)
            else:
                (overdue if waiting(o) > 1 else normal).append(o)

        res["overdue_backlog_list"] = overdue
        pool = {i: r for i, r in self.veh.iterrows()}
        driver_pointers = {w: {"c": 0, "p": 0} for w in self.wh}
        
        def assign_driver(wh_id):
            d_info = self.drivers.get(wh_id, {"chính": ["Tài xế chính"], "phụ": ["Phụ xe"]})
            c_list, p_list = d_info["chính"], d_info["phụ"]
            idx_c, idx_p = driver_pointers[wh_id]["c"], driver_pointers[wh_id]["p"]
            if idx_c < len(c_list):
                primary = c_list[idx_c]
                driver_pointers[wh_id]["c"] += 1
                backup = False
            else:
                primary = "Tài xế Dự Phòng (Thuê ngoài)"
                backup = True
            assistant = p_list[idx_p % len(p_list)] if p_list else "Phụ xe"
            driver_pointers[wh_id]["p"] += 1
            return primary, assistant, backup

        for bucket, target in ((overdue, res["overdue_routes"]), (normal, res["routes"])):
            by_wh = {}
            for o in bucket: by_wh.setdefault(self.nearest_wh(o["customer_id"]), []).append(o)
            for wh_id, ords in by_wh.items():
                demand, by_cust = {}, {}
                for o in ords:
                    d = demand.setdefault(o["customer_id"], {"weight": 0, "volume": 0, "order_type": o["order_type"]})
                    d["weight"] += o["weight"]; d["volume"] += o["volume"]
                    by_cust.setdefault(o["customer_id"], []).append(o)
                routes = self.clarke_wright(list(demand), wh_id, demand)
                for rt in routes:
                    target.append(self._build_route(rt, wh_id, demand, by_cust, pool, assign_driver, date_str))
                    
        for r in res["routes"] + res["overdue_routes"]:
            res["day_cost"] += r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r["driver_cost"]
        res["penalty_cost"] = sum(max(waiting(o), 0) for o in overdue) * cfg.late_penalty_per_day
        return res

    def _build_route(self, route, wh_id, demand, by_cust, pool, assign_driver, date_str):
        cfg = self.cfg
        m = self.metrics(route, wh_id, demand)
        
        # Lọc các xe khả thi: đủ tải, đủ thể tích, còn xe trong kho VÀ không vượt quá max_distance của xe
        # Nếu quãng đường vượt max_distance -> Phân loại Giao hàng tiết kiệm 1 (100k) hoặc Giao hàng tiết kiệm 2 (500k)
        cands = [(i, r) for i, r in pool.items() if r["wh_id"] == wh_id and r["max_weight_kg"] >= m["w"] and r["max_volume_m3"] >= m["v"] and r["available_quantity"] > 0]
        
        if cands:
            # Chọn xe NHỎ NHẤT trong các xe khả thi
            cands_sorted = sorted(cands, key=lambda t: (t[1]["max_weight_kg"] * 0.5 + t[1]["max_volume_m3"] * 0.5))
            idx, vrow = cands_sorted[0]
            
            # Kiểm tra ràng buộc khoảng cách (max_distance_km) của xe
            route_km = m["km"]
            max_allowed_dist = float(vrow.get("max_distance_km", 99999))
            
            if route_km > max_allowed_dist:
                # Xét các điều kiện đặc biệt khi vượt cự ly giới hạn của xe
                total_w = m["w"]
                total_v = m["v"]
                if total_w < 20.0 and total_v < 0.6:
                    # Giao hàng tiết kiệm 1: 100k
                    external = False
                    vid, plate, vtype = vrow["vehicle_id"], vrow["license_plate"], "Giao hàng tiết kiệm 1 (100k)"
                    speed, fx, vr = vrow["speed_kmh"], 0, 0
                    fixed_override = 100000
                elif total_w < 100.0 and total_v < 3.0:
                    # Giao hàng tiết kiệm 2: 500k
                    external = False
                    vid, plate, vtype = vrow["vehicle_id"], vrow["license_plate"], "Giao hàng tiết kiệm 2 (500k)"
                    speed, fx, vr = vrow["speed_kmh"], 0, 0
                    fixed_override = 500000
                else:
                    # Vượt cự ly và quá lớn -> Đẩy sang thuê ngoài / Outsource 30k/km
                    external = True
                    vt = m["vtype"] if m["vtype"] in self.catalog.index else "Truck"
                    vid, plate, vtype = f"3PL-{vt}", "Thuê ngoài (Vượt cự ly xe)", vt
                    speed = self.catalog.loc[vt, "speed"] if vt in self.catalog.index else 35.0
                    fx = self.catalog.loc[vt, "fixed"] if vt in self.catalog.index else 300_000
                    vr = 30000 / speed # Outsource 30k/km quy đổi biến phí
                    fixed_override = None
            else:
                pool.pop(idx) # Trừ số lượng xe trong kho khi gán xe nội bộ
                external = False
                vid, plate, vtype = vrow["vehicle_id"], vrow["license_plate"], vrow["vehicle_type"]
                speed, fx, vr = vrow["speed_kmh"], vrow["fixed_cost"], vrow["variable_cost_per_km"]
                fixed_override = None
        else:
            vt = m["vtype"] if m["vtype"] in self.catalog.index else (self.catalog.index[-1] if not self.catalog.empty else "Truck")
            external = True
            vid, plate, vtype = f"3PL-{vt}", "Thuê ngoài 3PL (Hết xe / Quá tải)", vt
            speed = self.catalog.loc[vt, "speed"] if vt in self.catalog.index else 35.0
            fx = self.catalog.loc[vt, "fixed"] if vt in self.catalog.index else 300_000
            vr = self.catalog.loc[vt, "var"] if vt in self.catalog.index else 8_000
            fixed_override = None

        primary, assistant, backup = assign_driver(wh_id)
        km, hours = m["km"], m["hours"]
        start = dt.datetime.combine(pd.to_datetime(date_str).date(), dt.datetime.strptime(cfg.start_time, "%H:%M").time())
        max_v_val = self.catalog.loc[vtype, "v"] if vtype in self.catalog.index else m["v"]
        load_f = m["v"] / max_v_val if max_v_val > 0 else 0.5
        
        final_fixed = fixed_override if fixed_override is not None else fx
        final_var = 0.0 if fixed_override is not None else (vr * km)

        return {
            "kind": "NORMAL", "wh_id": wh_id, "route": list(route), "orders": [o for c in route for o in by_cust[c]],
            "vehicle_id": vid, "license_plate": plate, "vehicle_type": vtype, "external": external,
            "driver_primary": primary, "driver_assistant": assistant, "is_backup_driver": backup,
            "km": round(km, 1), "speed_kmh": speed, "load_factor": min(max(load_f, 0.15), 1.0),
            "fixed_cost": final_fixed, "variable_cost": final_var, "overnight_cost": 0.0,
            "driver_cost": cfg.backup_driver_cost if backup else 0.0,
            "cut_orders": [], "start": start, "end": start + dt.timedelta(hours=hours), "hours": hours,
        }

def simulate_all(data: dict, cfg: Config = CFG):
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]: by_date.setdefault(o["date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    all_r = [r for d in days.values() for r in d["routes"] + d["overdue_routes"]]
    n_routes = len(all_r)
    operating = sum(d["day_cost"] for d in days.values())
    penalty = sum(d["penalty_cost"] for d in days.values())
    kpis = {
        "violations": sum(1 for r in all_r if r["hours"] > cfg.max_route_hours + 1e-9),
        "operating_cost": operating, "penalty_cost": penalty,
        "external_ratio": sum(r["external"] for r in all_r) / n_routes if n_routes else 0,
        "avg_load_factor": sum(r["load_factor"] for r in all_r) / len(all_r) if all_r else 0,
        "late_order_days": sum(len(d["overdue_backlog_list"]) for d in days.values()),
        "undelivered_orders": sum(len(d["carried_orders"]) for d in days.values()),
        "total_orders": len(data["orders"]),
    }
    return planner, days, kpis, operating + penalty
