# -*- coding: utf-8 -*-
"""
================================================================================
SMART LOGISTICS — STREAMLIT APP (bản thống nhất 5 tab + ràng buộc Clarke-Wright mới)
================================================================================
Tab 1-5 (Fleet / Warehouse / Product / Driver / Orders) dùng CHUNG một khung:
  - cùng 1 engine Semantic Mapping (tên cột + nội dung cột, chọn toàn cục, negative keywords)
  - cùng 1 bảng "Ánh xạ cột" (Trường | Cột nguồn | Độ tin cậy | Căn cứ)
  - cùng 1 bảng "Ánh xạ giá trị" (giá trị gốc -> giá trị chuẩn)
  - cùng 1 cột ghi_chú / kiểm_tra (✅ / ⚠️ / ❌) và cùng 1 bộ chỉ số tổng hợp
  - cùng 1 cấu trúc file xuất: <output_xxx>/DIM_xxx.xlsx  (3 sheet: DIM_xxx, COLUMN_MAPPING,
    VALUE_MAPPING) + DIM_xxx.json

Tab 6 (Clarke-Wright toàn cục) - trình tự ràng buộc:
  1) THỂ TÍCH (rồi tới trọng tải & giờ) được kiểm tra NGAY TRONG lúc gộp tuyến (song song với Clarke-Wright)
  2) Khi tuyến đã hình thành: ưu tiên XE NHÀ trong output_fleet/DIM_VEHICLE.xlsx
  3) Chọn XE NHỎ NHẤT trong các xe khả thi (thể tích -> trọng tải)
  4) Xét Max_Distance của xe đó: quãng đường tuyến >= Max_Distance -> thuê ngoài
        - hàng < 30kg và < 1m3   -> Giao hàng tiết kiệm loại 1
        - hàng < 100kg và < 5m3  -> Giao hàng tiết kiệm loại 2
        - lớn hơn                -> thuê xe tải 3PL
  5) Tỉ lệ lấp đầy = thể tích hàng / thể tích khoang xe

requirements.txt: streamlit, pandas, numpy, openpyxl, xlrd, rapidfuzz, unidecode, geopy, requests
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
MAP_THRESHOLD = 0.38  # ngưỡng semantic mapping (dùng chung cho cả 5 tab)
DEFAULT_DEPOT = (21.0285, 105.8542)

BASE_DIR = os.getcwd()
OUT_FLEET = os.path.join(BASE_DIR, "output_fleet")
OUT_WAREHOUSE = os.path.join(BASE_DIR, "output_warehouse")
OUT_PRODUCT = os.path.join(BASE_DIR, "output_product")
OUT_DRIVER = os.path.join(BASE_DIR, "output_driver")
OUT_ORDERS = os.path.join(BASE_DIR, "output_orders")
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")


class UserError(Exception):
    """Lỗi nghiệp vụ hiển thị cho người dùng."""


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


def to_text(v):
    """Giá trị bất kỳ -> chuỗi sạch (1.0 -> '1', True -> 'true')."""
    if is_blank(v):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def split_camel(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))


def norm(v):
    """Bỏ dấu, thường hóa, tách camelCase/underscore: 'Mã_Đơn' / 'orderID' -> 'ma don' / 'order id'."""
    s = unidecode(split_camel(to_text(v) if not isinstance(v, str) else v)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_exc = None
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:
                header = str(df.columns[0])
                for sep in (";", "\t", "|"):
                    if sep in header:
                        return pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep)
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # pragma: no cover


def read_any(f) -> pd.DataFrame:
    """Đọc file upload của Streamlit: Excel / CSV / JSON."""
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


def df_to_records(df: pd.DataFrame):
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")


def conf_icon(s):
    return "🟢" if s >= 0.75 else ("🟡" if s >= 0.55 else "🟠")


MAPPING_COLS = ["Trường", "Cột nguồn", "Độ tin cậy", "Căn cứ"]
REPORT_COLS = ["Trường", "Giá trị gốc", "Chuẩn hóa thành", "Cách nhận diện", "Số dòng"]


def save_outputs(out_dir, base, sheet, df, mapping_df, report_df):
    """CẤU TRÚC XUẤT CHUNG CHO MỌI TAB: 3 sheet (dữ liệu / COLUMN_MAPPING / VALUE_MAPPING) + JSON."""
    os.makedirs(out_dir, exist_ok=True)
    xlsx = os.path.join(out_dir, f"{base}.xlsx")
    js = os.path.join(out_dir, f"{base}.json")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        mapping_df.to_excel(writer, sheet_name="COLUMN_MAPPING", index=False)
        report_df.to_excel(writer, sheet_name="VALUE_MAPPING", index=False)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(df_to_records(df), fh, ensure_ascii=False, indent=2, default=str)
    return [xlsx, js]


# ============================================================================
# 2. ĐỌC SỐ + ĐƠN VỊ · LÀM SẠCH ĐỊA CHỈ / SĐT · TỪ ĐIỂN GIÁ TRỊ
# ============================================================================
NUM_RE = re.compile(r"[-+]?\d[\d.,]*")
WEIGHT_UNITS = {"kg": 1, "kgs": 1, "kilogram": 1, "kilograms": 1, "ky": 1, "g": 1e-3, "gr": 1e-3, "gam": 1e-3,
                "gram": 1e-3, "grams": 1e-3, "mg": 1e-6, "t": 1000, "tan": 1000, "tonne": 1000, "tonnes": 1000,
                "ton": 1000, "tons": 1000, "ta": 100, "yen": 10, "lb": 0.45359237, "lbs": 0.45359237,
                "pound": 0.45359237, "pounds": 0.45359237, "oz": 0.0283495}
VOLUME_UNITS = {"m3": 1, "cbm": 1, "metkhoi": 1, "khoi": 1, "cm3": 1e-6, "cc": 1e-6, "ml": 1e-6, "l": 1e-3,
                "lit": 1e-3, "litre": 1e-3, "liter": 1e-3, "litres": 1e-3, "liters": 1e-3, "dm3": 1e-3,
                "ft3": 0.0283168, "cuft": 0.0283168}
LEN_CM_UNITS = {"mm": 0.1, "cm": 1, "dm": 10, "m": 100, "met": 100, "inch": 2.54, "in": 2.54}
SPEED_UNITS = {"kmh": 1, "kmgio": 1, "kmph": 1, "kph": 1, "mph": 1.609344}
DIST_UNITS = {"km": 1, "kms": 1, "m": 1e-3, "met": 1e-3, "mi": 1.609344, "mile": 1.609344, "miles": 1.609344}
MONEY_UNITS = {"d": 1, "vnd": 1, "dong": 1, "k": 1e3, "nghin": 1e3, "ngan": 1e3, "tr": 1e6, "trieu": 1e6, "ty": 1e9}
NUM_UNITS = {"wt": WEIGHT_UNITS, "len": LEN_CM_UNITS, "speed": SPEED_UNITS, "dist": DIST_UNITS,
             "money": MONEY_UNITS, "num": {}}
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
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _ascii(v):
    return unidecode(str(v)).lower().replace("^", "").replace("×", "x")


def parse_measure(val, units):
    """-> (giá trị đã quy đổi hoặc None, ghi chú quy đổi)."""
    if is_blank(val):
        return None, ""
    if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
        return float(val), ""
    s = _ascii(val)
    m = NUM_RE.search(s)
    if not m:
        return None, "không đọc được số"
    num = _to_float(m.group(0))
    if num is None:
        return None, "không đọc được số"
    rest = re.sub(r"[^a-z0-9]", "", s[m.end():])
    if not rest or not units:
        return num, ""
    factor = units.get(rest)
    if factor is None:
        cands = [u for u in sorted(units, key=len, reverse=True) if len(u) >= 2 and rest.startswith(u)]
        factor = units[cands[0]] if cands else None
    if factor is None:
        return num, f"đơn vị lạ '{rest}', giữ nguyên số"
    return num * factor, ("" if factor == 1 else f"quy đổi '{to_text(val)}'")


def parse_volume(val):
    if is_blank(val):
        return None, ""
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
    if is_blank(v):
        return ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(v, errors="coerce", dayfirst=True)
    return "" if pd.isna(d) else d.strftime("%Y-%m-%d")


# ---- Làm sạch địa chỉ (đã sửa lỗi dính chữ "P.12" -> "Phường12") ----
_ABBR_MULTI = {"tp": "Thành phố", "tx": "Thị xã", "tt": "Thị trấn"}
_ABBR_SINGLE = {"P": "Phường", "Q": "Quận", "H": "Huyện", "X": "Xã"}
_WORD = r"[\wÀ-ỹĐđ]"


def expand_address_abbr(text):
    text = re.sub(rf"(?<!{_WORD})(TP|TX|TT)\.?\s*(?={_WORD})",
                  lambda m: _ABBR_MULTI[m.group(1).lower()] + " ", text, flags=re.I)
    text = re.sub(rf"(?<!{_WORD})([PQHX])(?:\.\s*|\s+(?=\d))(?={_WORD})",
                  lambda m: _ABBR_SINGLE[m.group(1)] + " ", text)
    return text


def coordinate_in_vietnam(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
    except Exception:
        return False
    return VIETNAM_BOUNDS[0] <= lat <= VIETNAM_BOUNDS[1] and VIETNAM_BOUNDS[2] <= lng <= VIETNAM_BOUNDS[3]


def clean_address(address):
    if is_blank(address):
        return ""
    text = str(address).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u3000]", " ", text)
    text = re.sub(r"\s*[|;→–—]\s*", ", ", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])\s*/\s*(?=[A-Za-zÀ-ỹ])", ", ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s,./'-]", " ", text)
    text = expand_address_abbr(text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if text and not re.search(r"\bViệt Nam\b|\bVietnam\b", text, re.I):
        text += ", Việt Nam"
    return text


def address_quality(address):
    if not address:
        return 0.0, "❌ Địa chỉ trống"
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


def clean_phone(v):
    s = re.sub(r"[^\d+]", "", to_text(v))
    if s.startswith("+84"):
        s = "0" + s[3:]
    elif s.startswith("84") and len(s) == 11:
        s = "0" + s[2:]
    elif len(s) == 9 and not s.startswith("0"):  # Excel làm rụng số 0 đầu
        s = "0" + s
    return s


# ---- Từ điển giá trị (Individual -> B2C, Business -> B2B, ...) ----
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
            if nk:
                exact.setdefault(nk, label)
    cont = sorted(((k, l, re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"))
                   for k, l in exact.items() if len(k) >= 3), key=lambda x: -len(x[0]))
    return {"exact": exact, "cont": cont, "keys": [k for k in exact if len(k) >= 4]}


TYPE_LK, ALERT_LK, STATUS_LK = build_lookup(TYPE_TABLE), build_lookup(ALERT_TABLE), build_lookup(STATUS_TABLE)
DICT_LOOKUPS = {"order_type": TYPE_LK, "alert_status": ALERT_LK, "order_status": STATUS_LK}


def match_label(value, lk, fuzzy_cut=88):
    """Trả (nhãn chuẩn, cách khớp, từ khóa) hoặc None. Khớp: chính xác > chứa từ khóa dài nhất > fuzzy."""
    n = norm(to_text(value))
    if not n:
        return None
    if n in lk["exact"]:
        return lk["exact"][n], "khớp từ điển", n
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


# ============================================================================
# 3. SEMANTIC MAPPING DÙNG CHUNG (tên cột + nội dung cột) CHO CẢ 5 TAB
# ============================================================================
ID_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*$")
PLATE_RX = re.compile(r"^\d{2}[A-Za-zĐđ][A-Za-z0-9]?[- ]?[\d.\- ]{4,9}$")
PHONE_RX = re.compile(r"\+?[\d\s.\-()]{8,16}")
DATE_RX = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
ADDR_TOKENS = ["duong", "pho", "phuong", "quan", "huyen", "tinh", "tp", "thanh pho", "thon", "ngo", "ngach", "hem",
               "xom", "khu pho", "street", "road", "rd", "ward", "district", "city", "avenue", "lane", "ha noi",
               "ho chi minh", "tphcm", "hcm", "da nang", "viet nam", "vietnam"]
STRONG_CONTENT = {"order_type", "alert_status", "order_status", "order_date", "address", "plate", "phone"}
# kind của trường -> logic chấm nội dung đã có sẵn
KIND_ALIAS = {"id": "order_id", "ref_id": "customer_id", "person": "customer_name", "name_text": "items"}


def name_score(col, aliases, negs):
    n = norm(col)
    if not n:
        return 0.0
    nc = n.replace(" ", "")
    best = 0.0
    for a in aliases:
        ac = a.replace(" ", "")
        if n == a: s = 1.0
        elif nc == ac: s = 0.98
        elif len(a) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", n): s = 0.80 + 0.15 * len(a) / len(n)
        elif len(n) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", a): s = 0.62
        elif len(a) >= 3 and len(n) >= 3: s = 0.8 * max(fuzz.token_sort_ratio(n, a), fuzz.ratio(nc, ac)) / 100
        else: s = 0.0
        best = max(best, s)
    if any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", n) for t in negs):
        best *= 0.45
    return min(best, 1.0)


def _is_addr(t):
    n = " " + norm(t) + " "
    return any(f" {tok} " in n for tok in ADDR_TOKENS) or (t.count(",") >= 2 and len(t) > 15)


def content_score(texts, kind):
    """Điểm 0..1 dựa trên GIÁ TRỊ trong cột (texts đã là list chuỗi, tối đa 300 dòng)."""
    field = KIND_ALIAS.get(kind, kind)
    n = len(texts)
    if n == 0:
        return 0.0
    uniq = len(set(texts)) / n

    def frac(pred):
        return sum(1 for t in texts if pred(t)) / n

    if field == "plate":
        return frac(lambda t: bool(PLATE_RX.match(t.strip())))
    if field == "phone":
        return frac(lambda t: bool(PHONE_RX.fullmatch(t.strip())) and sum(ch.isdigit() for ch in t) >= 8)
    if field == "num":
        return 0.3 * frac(lambda t: parse_measure(t, {})[0] is not None)
    if field == "text":
        return 0.35 * frac(lambda t: any(ch.isalpha() for ch in t) and len(t) <= 40)
    if field == "order_id":
        return frac(lambda t: bool(ID_RE.match(t))) * (1.0 if uniq >= 0.95 else 0.5)
    if field == "customer_id":
        return frac(lambda t: bool(ID_RE.match(t))) * (0.6 + 0.4 * (uniq < 0.95))
    if field == "customer_name":
        return frac(lambda t: len(t.split()) >= 2 and sum(ch.isdigit() for ch in t) / len(t) < 0.2 and not _is_addr(t)) * 0.8
    if field == "quantity":
        vals = [parse_measure(t, {})[0] for t in texts]
        ok = [v for v in vals if v is not None]
        if not ok:
            return 0.0
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
    if field in DICT_LOOKUPS:
        vc = Counter(texts)
        hit = sum(c for v, c in vc.items()
                  if len(norm(v)) >= 2 and any(ch.isalpha() for ch in v) and match_label(v, DICT_LOOKUPS[field]))
        return hit / n
    return 0.0


def combine(ns, cs, kind):
    s = 0.6 * ns + 0.4 * cs
    if kind in STRONG_CONTENT and cs >= 0.8:
        s = max(s, 0.55 * cs + 0.30 * ns + 0.10)
    return s


def semantic_mapping(df, spec):
    """Chọn cặp (trường, cột) điểm cao nhất toàn cục -> {field: (cột, điểm, căn cứ)}."""
    samples = {c: [t for t in (to_text(v) for v in df[c].dropna().head(300)) if t] for c in df.columns}
    cands = []
    for f, d in spec["fields"].items():
        kind = d.get("kind", f)
        for col in df.columns:
            ns = name_score(col, spec["alias_n"][f], spec["neg_n"][f])
            cs = content_score(samples[col], kind)
            cands.append((combine(ns, cs, kind), f, col, ns, cs))
    cands.sort(key=lambda x: -x[0])
    result = {f: (None, 0.0, "") for f in spec["fields"]}
    used_f, used_c = set(), set()
    for score, f, col, ns, cs in cands:
        if score < MAP_THRESHOLD:
            break
        if f in used_f or col in used_c:
            continue
        result[f] = (col, score, f"tên cột {ns:.0%} · nội dung {cs:.0%}")
        used_f.add(f)
        used_c.add(col)
    return result


def mapping_table(spec, cols_by_field, meta, why, auto):
    """BẢNG ÁNH XẠ CỘT THỐNG NHẤT (hiển thị sau khi quét VÀ ghi vào sheet COLUMN_MAPPING)."""
    rows = []
    for f, d in spec["fields"].items():
        col = cols_by_field.get(f, NONE)
        if col in (None, NONE):
            rows.append({"Trường": d["label"], "Cột nguồn": "(không tìm thấy)", "Độ tin cậy": "–", "Căn cứ": "–"})
        elif auto.get(f) == col and meta.get(f, 0) > 0:
            rows.append({"Trường": d["label"], "Cột nguồn": col, "Độ tin cậy": f"{conf_icon(meta[f])} {meta[f]:.0%}",
                         "Căn cứ": why.get(f, "")})
        else:
            rows.append({"Trường": d["label"], "Cột nguồn": col, "Độ tin cậy": "✋ chọn tay", "Căn cứ": "người dùng chọn"})
    return pd.DataFrame(rows, columns=MAPPING_COLS)


# ============================================================================
# 4. ENGINE CHUẨN HÓA + VALIDATE DÙNG CHUNG
# ============================================================================
def process_generic(spec, raw, reports):
    """Chuẩn hóa 1 dòng theo khai báo type của từng trường. raw: {field: giá trị gốc}."""
    rec, notes = {}, []
    for f, d in spec["fields"].items():
        orig, t, o = raw.get(f), d.get("type", "text"), d["out"]
        if t in NUM_UNITS or t == "vol":
            if is_blank(orig):
                rec[o] = 0.0
                continue
            v = orig
            if t == "money" and isinstance(v, str):
                v = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", v)  # 1.200.000 -> 1200000
            x, note = parse_volume(v) if t == "vol" else parse_measure(v, NUM_UNITS[t])
            if x is None:
                notes.append(f"⚠️ {d['label']}: {note or 'không đọc được'} → 0")
                x = 0.0
            elif note.startswith(("quy đổi", "tính từ")):
                reports[(f, to_text(orig), f"{x:g}", note)] += 1
            elif note:
                notes.append(f"⚠️ {d['label']}: {note}")
            rec[o] = round(float(x), 6)
        elif t == "upper":
            rec[o] = to_text(orig).upper()
        elif t == "address":
            rec[o] = clean_address(orig)
        elif t == "phone":
            p = clean_phone(orig)
            if p and p != to_text(orig):
                reports[(f, to_text(orig), p, "chuẩn hóa SĐT")] += 1
            rec[o] = p
        else:
            rec[o] = to_text(orig)
    rec["ghi_chú"] = " | ".join(notes)
    return rec


def validate_entity(spec, df):
    """KIỂM TRA THỐNG NHẤT: ❌ lỗi (thiếu/trùng) · ⚠️ cảnh báo · ✅ sạch."""
    out = df.copy().reset_index(drop=True)
    dups = {c: out[c].astype(str).duplicated(keep=False) for c, _ in spec["unique"]}
    msgs = []
    for i, row in out.iterrows():
        errs, warns = [], []
        for c, lab in spec["required"]:
            if is_blank(row.get(c)):
                errs.append(f"Thiếu {lab}")
        for c, lab in spec["unique"]:
            if dups[c].iloc[i] and not is_blank(row.get(c)):
                errs.append(f"Trùng {lab}")
        for c, msg in spec.get("warn_blank", {}).items():
            if is_blank(row.get(c)):
                warns.append(msg)
        for c, msg in spec.get("warn_zero", {}).items():
            if parse_measure(row.get(c), {})[0] in (None, 0.0):
                warns.append(msg)
        if spec.get("extra_validate"):
            e, w = spec["extra_validate"](row)
            errs += e
            warns += w
        warns += [n.replace("⚠️ ", "") for n in str(row.get("ghi_chú", "")).split(" | ") if n.startswith("⚠️")]
        if errs:
            msgs.append("❌ " + "; ".join(errs + warns))
        elif warns:
            msgs.append("⚠️ " + "; ".join(warns))
        else:
            msgs.append(f"✅ Đủ dữ liệu {spec['noun']} chuẩn")
    out["kiểm_tra"] = msgs
    return out


def process_rows(spec, raw, colmap, opts, progress=None):
    reports, rows, n = Counter(), [], len(raw)
    for k, (_, row) in enumerate(raw.iterrows(), 1):
        rows.append(spec["process"]({f: row.get(c) for f, c in colmap.items()}, reports, opts))
        if progress:
            progress(k, n)
    out = validate_entity(spec, pd.DataFrame(rows))
    rep = pd.DataFrame(
        [{"Trường": spec["fields"].get(f, {}).get("label", f), "Giá trị gốc": o, "Chuẩn hóa thành": s,
          "Cách nhận diện": h, "Số dòng": c} for (f, o, s, h), c in reports.most_common()], columns=REPORT_COLS)
    return out, rep


# ============================================================================
# 5. KHAI BÁO 5 THỰC THỂ
# ============================================================================
ENTITIES: dict = {}


def F(label, aliases, kind, out, typ, neg=()):
    return dict(label=label, aliases=aliases, kind=kind, out=out, type=typ, negative=list(neg))


def register(key, **kw):
    spec = dict(key=key, **kw)
    spec["alias_n"] = {f: [a for a in (norm(x) for x in d["aliases"]) if a] for f, d in spec["fields"].items()}
    spec["neg_n"] = {f: [n for n in (norm(x) for x in d.get("negative", [])) if n] for f, d in spec["fields"].items()}
    ENTITIES[key] = spec
    return spec


# ---------------------------- FLEET ----------------------------
VEHICLE_FIELDS = {
    "vehicle_id": F("Mã xe", ["mã xe", "vehicle id", "vehicle code", "vehicle", "xe", "id xe", "truck id", "mã phương tiện", "mã số xe"],
                    "id", "vehicle_id", "text", neg=["plate", "bien so", "bsx", "kho", "warehouse", "hub"]),
    "license_plate": F("Biển số", ["biển số", "bsx", "license plate", "plate", "số xe", "biển kiểm soát", "bks", "plate number"],
                       "plate", "license_plate", "upper"),
    "warehouse_id": F("ID kho hoạt động", ["kho", "warehouse", "wh", "hub", "chi nhánh", "mã kho", "id kho", "kho hoạt động", "depot", "trạm", "khu vực"],
                      "ref_id", "id_warehouse", "text", neg=["dia chi", "address", "lat", "lng"]),
    "max_weight": F("Trọng tải khối lượng (kg)", ["trọng tải", "tải trọng", "payload", "khối lượng tối đa", "max weight", "weight capacity", "capacity kg", "weight", "kg", "tấn"],
                    "total_weight", "max_weight_kg", "wt", neg=["volume", "the tich", "m3", "cbm", "speed", "cost"]),
    "max_volume": F("Trọng tải thể tích (m3)", ["thể tích", "thể tích khoang", "khoang", "max volume", "volume capacity", "volume", "m3", "cbm", "capacity m3"],
                    "total_volume", "max_volume_m3", "vol", neg=["weight", "trong luong", "kg", "cost"]),
    "average_speed": F("Vận tốc trung bình (km/h)", ["vận tốc", "vận tốc trung bình", "tốc độ", "tốc độ trung bình", "speed", "avg speed", "average speed", "kmh", "km/h"],
                       "num", "average_speed_kmh", "speed"),
    "fixed_cost": F("Chi phí cố định", ["chi phí cố định", "phí cố định", "fixed cost", "cost fix", "fixed", "fixed cost per day"],
                    "num", "fixed_cost", "money", neg=["variable", "bien doi", "km"]),
    "variable_cost": F("Chi phí biến đổi (đ/km)", ["chi phí biến đổi", "phí biến đổi", "variable cost", "variable", "cost km", "cost per km", "chi phí theo km", "đơn giá km"],
                       "num", "variable_cost", "money", neg=["fixed", "co dinh"]),
    "max_distance": F("Quãng đường tối đa (Max_Distance, km)", ["max distance", "quãng đường tối đa", "cự ly tối đa", "tầm hoạt động", "phạm vi hoạt động",
                                                              "giới hạn quãng đường", "distance limit", "max km", "max range"],
                      "num", "Max_Distance", "dist", neg=["cost", "phi"]),
}


def fleet_seed():
    return pd.DataFrame({
        "Mã xe": ["VEH_01", "VEH_02"], "Biển số": ["29C-123.45", "29C-678.90"], "ID kho": ["WH_HN_01", "WH_HN_01"],
        "Trọng tải (kg)": [5000, 2000], "Thể tích (m3)": [20, 10], "Vận tốc (km/h)": [50, 45],
        "Chi phí cố định": [500000, 300000], "Chi phí biến đổi": [5000, 4000], "Max_Distance (km)": [150, 80],
    })


register("fleet", noun="phương tiện", title="🚚 Smart Logistics — Quản lý & Chuẩn hóa Phương tiện",
         info="🔒 Cột nào không có trong file sẽ để 0/trống. **Max_Distance** (km) là quãng đường tối đa của xe nhà; "
              "để 0 = không giới hạn. Tab 6 dùng cột này để quyết định thuê ngoài giao hàng tiết kiệm.",
         fields=VEHICLE_FIELDS, out_dir=OUT_FLEET, base="DIM_VEHICLE", sheet="DIM_VEHICLE", seed=fleet_seed,
         process=lambda raw, rep, opts: process_generic(ENTITIES["fleet"], raw, rep),
         required=[("vehicle_id", "Mã xe"), ("license_plate", "Biển số"), ("id_warehouse", "ID kho hoạt động")],
         unique=[("vehicle_id", "Mã xe"), ("license_plate", "Biển số")],
         warn_zero={"max_weight_kg": "Trọng tải = 0", "max_volume_m3": "Thể tích khoang = 0",
                    "average_speed_kmh": "Vận tốc = 0", "Max_Distance": "Max_Distance = 0 (coi như không giới hạn)"},
         extra_metrics=lambda out: [])

# ---------------------------- WAREHOUSE ----------------------------
WAREHOUSE_FIELDS = {
    "warehouse_id": F("Mã kho", ["mã kho", "warehouse id", "warehouse code", "warehouse", "kho", "id kho", "invent id", "invent_id", "wh id", "depot id"],
                      "id", "id_warehouse", "text", neg=["dia chi", "address"]),
    "address": F("Địa chỉ kho", ["địa chỉ", "địa chỉ kho", "địa điểm", "address", "location", "vị trí", "addr"],
                 "address", "address", "address"),
}


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


def wh_process(raw, reports, opts):
    rec = process_generic(ENTITIES["wh"], raw, reports)
    notes = rec.pop("ghi_chú")
    quality, clean_status = address_quality(rec["address"])
    rec.update({"lat": None, "lng": None, "địa_chỉ_gốc": to_text(raw.get("address")), "chất_lượng_địa_chỉ": quality,
                "trạng_thái_làm_sạch": clean_status, "địa_chỉ_geocode": "", "geocode_score": None,
                "trạng_thái_geocode": "—", "nguồn_tọa_độ": ""})
    if is_blank(rec["id_warehouse"]):
        rec["trạng_thái_geocode"] = "❌ Thiếu Mã kho"
    elif not rec["address"]:
        rec["trạng_thái_geocode"] = "❌ Không có địa chỉ để geocode"
    elif not opts.get("geocode", True):
        rec["trạng_thái_geocode"] = "⏸️ Đã tắt geocoding"
    else:
        geo = geocode_address(rec["address"])
        if not geo["ok"]:
            rec["trạng_thái_geocode"] = geo["status"]
        else:
            rec.update({"lat": geo["lat"], "lng": geo["lng"], "địa_chỉ_geocode": geo["display_name"],
                        "geocode_score": geo["score"], "nguồn_tọa_độ": "ArcGIS",
                        "trạng_thái_geocode": "✅ ArcGIS geocode thành công" +
                        (f" | score {geo['score']:.0f}" if geo["score"] is not None else "")})
    rec["ghi_chú"] = notes
    return rec


def wh_extra_validate(row):
    errs = []
    lat, lng = row.get("lat"), row.get("lng")
    if pd.isna(lat) or pd.isna(lng):
        errs.append("Chưa có Lat/Lon")
    elif not coordinate_in_vietnam(lat, lng):
        errs.append("Lat/Lon ngoài Việt Nam")
    return errs, []


def wh_options_ui(key):
    return {"geocode": st.checkbox("🌍 Bật ArcGIS Geocoding", value=True, key="wh_do_geo")}


register("wh", noun="kho", title="🏭 Smart Logistics — Quét tọa độ kho (ArcGIS Geocoding)",
         info="🔒 Người dùng chỉ nhập **Mã kho + Địa chỉ kho**. Lat/Lon do hệ thống tự lấy từ ArcGIS.",
         fields=WAREHOUSE_FIELDS, out_dir=OUT_WAREHOUSE, base="DIM_WAREHOUSE", sheet="DIM_WAREHOUSE",
         seed=lambda: pd.DataFrame({"Mã kho": ["WH_HN_01"], "Địa chỉ kho": ["Số 1 Tràng Tiền, Hoàn Kiếm, Hà Nội"]}),
         process=wh_process, options_ui=wh_options_ui, extra_validate=wh_extra_validate,
         required=[("id_warehouse", "Mã kho"), ("address", "Địa chỉ")], unique=[("id_warehouse", "Mã kho")],
         extra_metrics=lambda out: [("Geocode thành công", f"{int(out['lat'].notna().sum())}/{len(out)}")])

# ---------------------------- PRODUCT ----------------------------
PRODUCT_FIELDS = {
    "product_id": F("Mã sản phẩm", ["mã sản phẩm", "product id", "sku", "item code", "mã sp", "product code", "mã hàng", "mã vật tư"],
                    "id", "product_id", "text", neg=["name", "ten"]),
    "product_name": F("Tên sản phẩm", ["tên sản phẩm", "product name", "item name", "tên sp", "tên hàng", "name", "mô tả", "description"],
                      "name_text", "product_name", "text", neg=["id", "ma", "code"]),
    "volume": F("Thể tích (m3)", ["thể tích", "volume", "m3", "cbm", "capacity"], "total_volume", "volume_m3", "vol",
                neg=["weight", "kg", "trong luong", "khoi luong"]),
    "weight": F("Trọng lượng (kg)", ["trọng lượng", "khối lượng", "weight", "kg", "tấn", "mass", "gross weight"], "total_weight", "weight_kg", "wt",
                neg=["volume", "m3", "the tich"]),
    "length": F("Chiều dài (cm)", ["chiều dài", "dài", "length", "dim l", "l", "len"], "num", "length_cm", "len", neg=["weight", "volume"]),
    "width": F("Chiều rộng (cm)", ["chiều rộng", "rộng", "width", "dim w", "w"], "num", "width_cm", "len", neg=["weight", "volume"]),
    "height": F("Chiều cao (cm)", ["chiều cao", "cao", "height", "dim h", "h"], "num", "height_cm", "len", neg=["weight", "volume"]),
    "cost_price": F("Giá sản xuất", ["giá sản xuất", "giá vốn", "giá gốc", "cost price", "cost", "import price"], "num", "cost_price", "money",
                    neg=["sell", "ban", "retail"]),
    "selling_price": F("Giá bán", ["giá bán", "selling price", "retail price", "unit price", "price", "đơn giá"], "num", "selling_price", "money",
                       neg=["cost", "von", "san xuat"]),
}


def product_process(raw, reports, opts):
    rec = process_generic(ENTITIES["product"], raw, reports)
    l, w, h = rec["length_cm"], rec["width_cm"], rec["height_cm"]
    if rec["volume_m3"] <= 0 and min(l, w, h) > 0:  # thiếu thể tích -> tính từ dài × rộng × cao
        rec["volume_m3"] = round(l * w * h / 1e6, 6)
        reports[("volume", "(trống)", f"{rec['volume_m3']:.4g}", "tính từ dài × rộng × cao (cm)")] += 1
    return rec


register("product", noun="sản phẩm", title="📦 Smart Logistics — Quản lý & Chuẩn hóa Sản phẩm",
         info="🔒 Cột nào không có trong file có thể để trống. Nếu thiếu thể tích mà có đủ Dài × Rộng × Cao (cm) hệ thống tự tính.",
         fields=PRODUCT_FIELDS, out_dir=OUT_PRODUCT, base="DIM_PRODUCT", sheet="DIM_PRODUCT",
         seed=lambda: pd.DataFrame({
             "Mã sản phẩm": ["SP_01"], "Tên sản phẩm": ["Ghế Sofa Gỗ Sồi"], "Thể tích (m3)": [0.5],
             "Trọng lượng (kg)": [25.0], "Dài (cm)": [120], "Rộng (cm)": [60], "Cao (cm)": [80],
             "Giá sản xuất": [1200000], "Giá bán": [2500000]}),
         process=product_process,
         required=[("product_id", "Mã sản phẩm"), ("product_name", "Tên sản phẩm")], unique=[("product_id", "Mã sản phẩm")],
         warn_zero={"weight_kg": "Trọng lượng = 0", "volume_m3": "Thể tích = 0"},
         extra_metrics=lambda out: [])

# ---------------------------- DRIVER ----------------------------
DRIVER_FIELDS = {
    "driver_id": F("Mã tài xế", ["mã tài xế", "driver id", "driver code", "mã nv", "mã nhân viên", "staff id", "employee id", "id tài xế"],
                   "id", "driver_id", "text", neg=["name", "ten"]),
    "driver_name": F("Họ và tên", ["họ và tên", "tên tài xế", "full name", "họ tên", "tên nhân viên", "driver name", "name"],
                     "person", "driver_name", "text", neg=["id", "ma", "code", "address", "dia chi"]),
    "license_type": F("Loại bằng", ["loại bằng", "bằng lái", "hạng bằng", "license", "license type", "license class", "class"],
                      "text", "license_type", "upper"),
    "warehouse": F("Kho hoạt động", ["kho", "kho hoạt động", "warehouse", "trạm", "hub", "chi nhánh", "depot"],
                   "ref_id", "id_warehouse", "text"),
    "address": F("Địa chỉ", ["địa chỉ", "nơi ở", "address", "quê quán"], "address", "address", "text"),
    "phone": F("Số điện thoại", ["số điện thoại", "sdt", "điện thoại", "phone", "mobile", "hotline", "tel"], "phone", "phone", "phone"),
    "role": F("Vị trí làm việc", ["vị trí làm việc", "vị trí", "chức vụ", "role", "job", "position", "loại nhân sự"],
              "text", "role", "text", neg=["address", "dia chi"]),
}
register("driver", noun="tài xế", title="👨‍✈️ Smart Logistics — Quản lý & Chuẩn hóa Tài xế",
         info="🔒 Cột nào không có trong file có thể để trống. SĐT được chuẩn hóa về dạng 0xxxxxxxxx.",
         fields=DRIVER_FIELDS, out_dir=OUT_DRIVER, base="DIM_DRIVER", sheet="DIM_DRIVER",
         seed=lambda: pd.DataFrame({
             "Mã tài xế": ["DRV_01", "DRV_02"], "Họ và tên": ["Nguyễn Văn A", "Trần Văn B"], "Loại bằng": ["FC", "C"],
             "Kho hoạt động": ["WH_HN_01", "WH_HN_01"], "Địa chỉ": ["Hà Nội", "Hà Nội"],
             "Số điện thoại": ["0901234567", "0987654321"], "Vị trí làm việc": ["Tài xế chính", "Hỗ trợ vận chuyển đồ"]}),
         process=lambda raw, rep, opts: process_generic(ENTITIES["driver"], raw, rep),
         required=[("driver_id", "Mã tài xế"), ("driver_name", "Họ và tên")], unique=[("driver_id", "Mã tài xế")],
         warn_blank={"id_warehouse": "Thiếu Kho hoạt động"}, extra_metrics=lambda out: [])

# ---------------------------- ORDERS ----------------------------
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


def process_order(raw, reports):
    """raw: {field: giá trị gốc hoặc None}. reports: Counter ghi lại các phép chuẩn hóa giá trị."""
    notes = []

    def log(field, original, standard, how):
        if to_text(original) != standard:
            reports[(field, to_text(original), standard, how)] += 1

    r_name = to_text(raw.get("customer_name"))
    name_b2b = bool(B2B_NAME_RX.search(norm(r_name)))
    cust_name = r_name or "Khách lẻ"
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
        if name_b2b:
            log("order_type", "(trống)", t_type, "suy từ tên khách")
    r_alert = to_text(raw.get("alert_status"))
    if r_alert:
        m = match_label(r_alert, ALERT_LK)
        if m:
            t_alert = m[0]; log("alert_status", r_alert, t_alert, m[1])
        else:
            t_alert = "Normal"; log("alert_status", r_alert, t_alert, "mặc định")
            notes.append(f"⚠️ Không nhận diện mức cảnh báo '{r_alert}' → mặc định Normal")
    else:
        t_alert = "Normal"
    r_status = to_text(raw.get("order_status"))
    if r_status:
        m = match_label(r_status, STATUS_LK)
        if m:
            status = m[0]; log("order_status", r_status, status, m[1])
        else:
            status = r_status
    else:
        status = "Mới tạo"
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
    if raw.get("order_date") is not None:
        rec["order_date"] = parse_date_iso(raw["order_date"])
    rec["ghi_chú"] = " | ".join(notes)
    return rec


register("orders", noun="đơn hàng", title="🧾 Smart Logistics — Quản lý & Chuẩn hóa Đơn hàng",
         info="🔒 Tự nhận diện cột theo *tên + nội dung* và hiểu giá trị như `Individual`→B2C, `Business`→B2B, "
              "`Urgent`→Alert, `Shipping`→Đang giao, `2 tấn`→2000 kg...",
         fields=ORDER_FIELDS, out_dir=OUT_ORDERS, base="DIM_ORDERS", sheet="DIM_ORDERS",
         seed=lambda: pd.DataFrame({
             "Mã đơn": ["ORD_001", "ORD_002", "ORD_003", "ORD_004"],
             "Mã khách": ["CUS_01", "CUS_02", "CUS_03", "CUS_04"],
             "Tên khách": ["Nguyễn Văn A", "Công ty TNHH Nội Thất Việt", "Trần Thị B", "Đại lý Minh Phát"],
             "Số lượng": [2, 10, 1, 6],
             "Mặt hàng": ["Ghế sofa, Bàn trà", "Bàn làm việc", "Tủ quần áo", "Giường gỗ"],
             "Tổng trọng lượng (kg)": [45.5, 320.0, 80.0, 450.0],
             "Tổng thể tích (m3)": [0.8, 6.5, 1.2, 5.0],
             "Địa chỉ khách": ["Số 88 - Đường Cổ Linh - Long Biên - Hà Nội", "Số 1 Đường Trần Duy Hưng, Cầu Giấy, Hà Nội",
                               "Số 25 Đường Láng Hạ, Đống Đa, Hà Nội", "Số 120 Đường Nguyễn Trãi, Thanh Xuân, Hà Nội"],
             "Tình trạng đơn": ["Đang xử lý", "Mới tạo", "New", "Shipping"],
             "Loại đơn": ["Individual", "Business", "Retail", "Distributor"],
             "Tình trạng Alert": ["Normal", "Urgent", "Normal", "High"]}),
         process=lambda raw, rep, opts: process_order(raw, rep),
         required=[("order_id", "Mã đơn"), ("address", "Địa chỉ")], unique=[("order_id", "Mã đơn")],
         warn_blank={"customer_id": "Thiếu Mã khách"},
         warn_zero={"total_weight_kg": "Trọng lượng = 0", "total_volume_m3": "Thể tích = 0"},
         extra_metrics=lambda out: [("B2B", int((out["order_type"] == "B2B").sum())),
                                    ("B2C", int((out["order_type"] == "B2C").sum())),
                                    ("Alert", int((out["alert_status"] == "Alert").sum()))])


# ============================================================================
# 6. UI DÙNG CHUNG CHO 5 TAB (CÙNG 1 KHUNG, CÙNG 1 KẾT QUẢ)
# ============================================================================
def render_input_block(prefix, label, seed_df):
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
    st.session_state[f"{prefix}_why"] = {f: m[2] for f, m in mapping.items()}
    st.session_state[f"{prefix}_auto"] = {f: (str(m[0]) if m[0] is not None else NONE) for f, m in mapping.items()}
    for f, col in st.session_state[f"{prefix}_auto"].items():
        st.session_state[f"{prefix}_col_{f}"] = col
    st.session_state.pop(f"{prefix}_result", None)
    return True


def render_mapping(prefix, specs, raw):
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
            if score <= 0:
                st.caption("⚪ Không tìm thấy cột phù hợp")
            else:
                st.caption(f"{conf_icon(score)} Độ tin cậy: {score:.0%}" + (f" · {why.get(fld, '')}" if why.get(fld) else ""))
    return chosen


def build_colmap(raw_df, chosen):
    lookup = {str(c): c for c in raw_df.columns}
    return {f: lookup[c] for f, c in chosen.items() if c and c != NONE and c in lookup}


def store_result(prefix, df, files, summary, metrics, extra=None):
    st.session_state[f"{prefix}_result"] = {"df": df, "files": list(files), "summary": summary,
                                            "metrics": metrics, "extra": extra or {}}


def render_result(prefix, title):
    """KHUNG KẾT QUẢ CHUNG: tóm tắt · chỉ số · bảng dữ liệu · bảng ánh xạ cột · bảng ánh xạ giá trị · tải file."""
    res = st.session_state.get(f"{prefix}_result")
    if not res:
        return None
    extra = res.get("extra") or {}
    if "mapping" not in extra or "report" not in extra:  # kết quả cũ còn sót từ phiên bản trước -> bỏ
        st.session_state.pop(f"{prefix}_result", None)
        return None
    st.markdown(f"### {title}")
    st.success(res["summary"])
    cols = st.columns(len(res["metrics"]))
    for c, (lab, val) in zip(cols, res["metrics"]):
        c.metric(lab, val)
    st.dataframe(res["df"])
    st.markdown("##### 🔗 Bảng ánh xạ cột đã sử dụng")
    st.dataframe(res["extra"]["mapping"], hide_index=True)
    st.markdown("##### 🔁 Bảng ánh xạ giá trị (giá trị gốc → giá trị chuẩn)")
    rep = res["extra"]["report"]
    if len(rep):
        st.dataframe(rep, hide_index=True)
    else:
        st.caption("Không có giá trị nào cần quy đổi.")
    rel = ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"])
    st.info(f"📁 Đã tự động lưu (3 sheet: dữ liệu · COLUMN_MAPPING · VALUE_MAPPING): {rel}")
    cols = st.columns(len(res["files"]))
    for c, path in zip(cols, res["files"]):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            mime = XLSX_MIME if path.endswith(".xlsx") else "application/json"
            c.download_button(f"⬇️ Tải {os.path.basename(path)}", data, file_name=os.path.basename(path),
                              mime=mime, key=f"{prefix}_dl_{os.path.basename(path)}")
    return res


def render_entity_tab(key):
    spec = ENTITIES[key]
    noun = spec["noun"]
    st.header(spec["title"])
    st.markdown("**Input → Semantic Mapping (cột + giá trị) → Làm sạch → Validate → Export Excel/JSON**")
    st.info(spec["info"])
    opts = spec["options_ui"](key) if spec.get("options_ui") else {}
    mode, edited, uploaded = render_input_block(key, noun, spec["seed"]())

    if st.button(f"🔍 Quét & Semantic Mapping {noun}", key=f"{key}_scan", type="primary"):
        if run_scan(key, noun, mode, uploaded, edited, lambda df: semantic_mapping(df, spec)):
            auto = st.session_state[f"{key}_auto"]
            st.session_state[f"{key}_auto_table"] = mapping_table(
                spec, auto, st.session_state[f"{key}_meta"], st.session_state[f"{key}_why"], auto)

    raw = st.session_state.get(f"{key}_raw")
    if raw is None:
        return
    st.success(f"🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**")
    auto_tbl = st.session_state.get(f"{key}_auto_table")
    if auto_tbl is not None:
        st.markdown("##### 🧭 Kết quả quét ánh xạ")
        st.dataframe(auto_tbl, hide_index=True)
        st.caption("🟢 chắc chắn · 🟡 nên kiểm tra · 🟠 độ tin cậy thấp — bạn có thể đổi trong các ô bên dưới.")
    st.markdown("### 🔗 Kiểm tra ánh xạ cột (cột nào không có chọn '-- Không sử dụng --')")
    st.caption("Xem trước dữ liệu gốc (8 dòng đầu)")
    st.dataframe(raw.head(8))
    chosen = render_mapping(key, [(f, d["label"]) for f, d in spec["fields"].items()], raw)

    if st.button(f"🚀 Chuẩn hóa & Xử lý {noun}", key=f"{key}_process", type="primary"):
        try:
            colmap = build_colmap(raw, chosen)
            prog = st.progress(0.0, text=f"Đang xử lý {noun}...")
            out, rep = process_rows(spec, raw, colmap, opts,
                                    lambda k, n: prog.progress(k / n, text=f"Đã xử lý {k}/{n} dòng"))
            prog.empty()
            mdf = mapping_table(spec, chosen, st.session_state.get(f"{key}_meta", {}),
                                st.session_state.get(f"{key}_why", {}), st.session_state.get(f"{key}_auto", {}))
            files = save_outputs(spec["out_dir"], spec["base"], spec["sheet"], out, mdf, rep)
            chk = out["kiểm_tra"].astype(str)
            n, n_err, n_warn = len(out), int(chk.str.startswith("❌").sum()), int(chk.str.startswith("⚠️").sum())
            metrics = [("Tổng số dòng", n), ("✅ Sạch", n - n_err - n_warn), ("⚠️ Cảnh báo", n_warn),
                       ("❌ Lỗi", n_err)] + spec["extra_metrics"](out)
            store_result(key, out, files,
                         f"🧭 Hoàn tất chuẩn hóa {noun} — {n - n_err}/{n} dòng không có lỗi (trong đó {n_warn} dòng có cảnh báo ⚠️)",
                         metrics, extra={"mapping": mdf, "report": rep})
        except UserError as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:
            st.error(f"❌ Lỗi chi tiết: {exc}")
    res = render_result(key, f"📊 Kết quả {noun}")
    if key == "wh" and res is not None:
        geo_df = res["df"].dropna(subset=["lat", "lng"])
        if not geo_df.empty:
            st.markdown("##### 📍 Vị trí kho trên bản đồ")
            st.map(geo_df.rename(columns={"lng": "lon"})[["lat", "lon"]])


# ============================================================================
# 7. TAB 6 — CẤU HÌNH & DỮ LIỆU
# ============================================================================
@dataclass
class Config:
    order_file: str = os.path.join(OUT_ORDERS, "DIM_ORDERS.xlsx")
    vehicle_file: str = os.path.join(OUT_FLEET, "DIM_VEHICLE.xlsx")
    driver_file: str = os.path.join(OUT_DRIVER, "DIM_DRIVER.xlsx")
    warehouse_file: str = os.path.join(OUT_WAREHOUSE, "DIM_WAREHOUSE.xlsx")
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
    # ---- Thuê ngoài "Giao hàng tiết kiệm" (kích hoạt khi quãng đường >= Max_Distance của xe) ----
    ghtk1_max_kg: float = 30.0
    ghtk1_max_m3: float = 1.0
    ghtk1_base: float = 25_000      # đ / chuyến
    ghtk1_per_km: float = 3_000     # đ / km
    ghtk2_max_kg: float = 100.0
    ghtk2_max_m3: float = 5.0
    ghtk2_base: float = 60_000
    ghtk2_per_km: float = 6_000


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
    if os.path.exists(cfg.cust_file):
        return cfg.cust_file
    folder = os.path.dirname(cfg.cust_file)
    if os.path.isdir(folder):
        cands = sorted(f for f in os.listdir(folder) if f.endswith(".xlsx") and not f.startswith("~"))
        if cands:
            return os.path.join(folder, cands[0])
    raise UserError("Không tìm thấy dữ liệu khách hàng trong `output_customer/`. Hãy chạy bước 1 (Geocode khách hàng) trước.")


def load_warehouse_coords(cfg: Config) -> dict:
    coords = {}
    if not os.path.exists(cfg.warehouse_file):
        return coords
    df_wh = pd.read_excel(cfg.warehouse_file)  # sheet đầu tiên = DIM_WAREHOUSE
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

    veh_id_col = first_col(df_veh, "vehicle_id", "VEHICLE_ID", "id")
    plate_col = first_col(df_veh, "license_plate", "bien_so", "plate", "vehicle_id")
    type_col = first_col(df_veh, "vehicle_type", "type", "vehicle_class", "vehicle_name")
    wh_col = first_col(df_veh, "wh_id", "warehouse_id", "depot_id", "id_warehouse")
    speed_col = first_col(df_veh, "speed_kmh", "speed", "vận_tốc", "average_speed_kmh")
    w_col = first_col(df_veh, "max_weight_kg", "weight_capacity", "max_weight", "capacity_kg")
    v_col = first_col(df_veh, "max_volume_m3", "volume_capacity", "max_volume", "capacity_m3")
    md_col = first_col(df_veh, "Max_Distance", "max_distance", "MAX_DISTANCE", "max_distance_km")

    df_veh["vehicle_id"] = df_veh[veh_id_col].astype(str) if veh_id_col else [f"VEH_{i}" for i in range(len(df_veh))]
    df_veh["license_plate"] = df_veh[plate_col].astype(str) if plate_col else df_veh["vehicle_id"]
    wh_series = df_veh[wh_col].astype(str).str.strip() if wh_col else pd.Series("WH_DEFAULT", index=df_veh.index)
    df_veh["wh_id"] = wh_series.replace({"": "WH_DEFAULT", "nan": "WH_DEFAULT", "None": "WH_DEFAULT"})
    df_veh["speed_kmh"] = _positive(df_veh[speed_col], 35.0) if speed_col else 35.0
    df_veh["max_weight_kg"] = _positive(df_veh[w_col], 1000.0) if w_col else 1000.0
    df_veh["max_volume_m3"] = _positive(df_veh[v_col], 5.0) if v_col else 5.0
    if md_col:  # Max_Distance <= 0 / trống -> không giới hạn
        s_md = pd.to_numeric(df_veh[md_col], errors="coerce")
        df_veh["max_distance_km"] = s_md.where(s_md > 0).fillna(np.inf)
    else:
        df_veh["max_distance_km"] = np.inf
        notes.append("DIM_VEHICLE.xlsx chưa có cột **Max_Distance** → xe nhà được coi là không giới hạn quãng đường "
                     "(sẽ không bật thuê ngoài Giao hàng tiết kiệm).")
    if type_col:
        df_veh["vehicle_type"] = df_veh[type_col].astype(str).str.strip()
    else:
        df_veh["vehicle_type"] = df_veh["max_weight_kg"].map(lambda w: f"Xe {w:g}kg")

    fx_col = cfg.fixed_cost_col or first_col(df_veh, "fixed_cost", "fixed_cost_per_day")
    vr_col = cfg.variable_cost_col or first_col(df_veh, "variable_cost_per_km", "cost_per_km", "variable_cost")
    df_veh["fixed_cost"] = pd.to_numeric(df_veh[fx_col], errors="coerce").fillna(300_000.0) if fx_col else 300_000.0
    df_veh["variable_cost_per_km"] = pd.to_numeric(df_veh[vr_col], errors="coerce").fillna(8_000.0) if vr_col else 8_000.0

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

    d_name_col = first_col(df_driver, "driver_name", "name", "full_name", "TÊN", "driver_id")
    d_role_col = first_col(df_driver, "role", "position", "VAI_TRÒ", "type")
    d_wh_col = first_col(df_driver, "wh_id", "warehouse_id", "depot_id", "KHO", "id_warehouse")
    df_driver["driver_name"] = df_driver[d_name_col].astype(str) if d_name_col else "Tài xế"
    df_driver["role"] = df_driver[d_role_col].astype(str).str.strip().str.capitalize() if d_role_col else "Chính"
    df_driver["wh_id"] = df_driver[d_wh_col].astype(str).str.strip() if d_wh_col else list(warehouses.keys())[0]
    drivers_by_wh = {}
    for wh in warehouses:
        sub = df_driver[df_driver["wh_id"] == wh]
        chinh = sub[sub["role"].str.contains("Chính|Primary|Driver", case=False, na=False)]["driver_name"].tolist()
        phu = sub[sub["role"].str.contains("Phụ|Assistant|Helper|Hỗ trợ|Support", case=False, na=False)]["driver_name"].tolist()
        if not chinh:
            chinh = sub["driver_name"].tolist() or ["Tài xế chính"]
        if not phu:
            phu = ["Phụ xe hỗ trợ"]
        if sub.empty:
            notes.append(f"Kho {wh}: không có tài xế nào khai báo ở Tab 4 (sẽ dùng tài xế dự phòng).")
        drivers_by_wh[wh] = {"chính": chinh, "phụ": phu}

    addr_col = first_col(df_cust, "address", "Location", "ADDRESS")
    lat_col = first_col(df_cust, "lat", "LATITUDE", "latitude")
    lon_col = first_col(df_cust, "lng", "lon", "LONGITUDE", "longitude")
    cid_col = first_col(df_cust, "customer_id", "CUSTOMER_ID", "id")
    cust = {}
    for idx, r in df_cust.iterrows():
        cid = str(r[cid_col]).strip() if cid_col and pd.notna(r[cid_col]) else str(idx)
        if not (lat_col and lon_col) or pd.isna(r[lat_col]) or pd.isna(r[lon_col]):
            continue
        cust[cid] = {"lat": float(r[lat_col]), "lon": float(r[lon_col]),
                     "address": str(r[addr_col]) if addr_col and pd.notna(r[addr_col]) else ""}

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
# 8. SÀNG LỌC THỂ TÍCH -> TRỌNG TẢI (bước 4) & KIỂM ĐỊNH THỜI GIAN TUYẾN
# ============================================================================
def run_master_logistics_optimizer(order_file="output_orders/DIM_ORDERS.xlsx",
                                   vehicle_file="output_fleet/DIM_VEHICLE.xlsx",
                                   matrix_file="output_matrix/DISTANCE_MATRIX_KM.xlsx"):
    """Quét đơn: THỂ TÍCH trước, rồi TRỌNG TẢI. Đơn vượt xe lớn nhất bị tách riêng."""
    df_orders = pd.read_excel(order_file)
    df_vehicles = pd.read_excel(vehicle_file)
    df_dist = pd.read_excel(matrix_file, index_col=0)
    max_v = df_vehicles["max_volume_m3"].max()
    max_w = df_vehicles["max_weight_kg"].max()
    valid_orders, oversized_orders = [], []
    for _, row in df_orders.iterrows():
        w = float(row.get("total_weight_kg", 0.0))
        v = float(row.get("total_volume_m3", 0.0))
        record = row.to_dict()
        if v > max_v:
            record["reason"] = f"Thể tích {v:g}m3 > xe lớn nhất {max_v:g}m3"
            oversized_orders.append(record)
        elif w > max_w:
            record["reason"] = f"Trọng lượng {w:g}kg > xe lớn nhất {max_w:g}kg"
            oversized_orders.append(record)
        else:
            record["eligible_vehicles"] = df_vehicles[
                (df_vehicles["max_volume_m3"] >= v) & (df_vehicles["max_weight_kg"] >= w)]["vehicle_id"].tolist()
            valid_orders.append(record)
    return {"valid_orders": valid_orders, "oversized_orders": oversized_orders, "distance_matrix": df_dist,
            "fleet_max_specs": {"max_weight": max_w, "max_volume": max_v}}


def evaluate_route_time_constraint(route_data, service_time_rules=None):
    if service_time_rules is None:
        service_time_rules = {"B2C": {"loading": 25, "unloading": 35}, "B2B": {"loading": 45, "unloading": 60}}
    orders_in_route = route_data.get("orders", [])
    total_distance_km = route_data.get("total_distance_km", 0.0)
    vehicle_speed_kmh = route_data.get("vehicle_speed_kmh", 40.0)
    current_date = route_data.get("current_date", "2026-04-03")
    travel_time_hours = total_distance_km / vehicle_speed_kmh if vehicle_speed_kmh > 0 else 0.0
    total_service_minutes = 0.0
    for order in orders_in_route:
        o_specs = service_time_rules.get(order.get("order_type", "B2C"), service_time_rules["B2C"])
        total_service_minutes += (o_specs["loading"] + o_specs["unloading"])
    total_h = travel_time_hours + total_service_minutes / 60.0
    if total_h <= 8.0:
        return {"status": "APPROVED", "message": "✅ Đạt yêu cầu thời gian tuyến (<= 8h)",
                "total_hours": round(total_h, 2), "route": route_data.get("route", [])}
    backlog_orders = []
    for order in orders_in_route:
        b = order.copy()
        b["backlog_days_count"] = order.get("backlog_days_count", 0) + 1
        b["original_date"] = order.get("original_date", current_date)
        b["backlog_reason"] = f"Tuyến vượt quá 8h ({total_h:.2f}h)"
        backlog_orders.append(b)
    return {"status": "BACKLOG_OR_REPOOL",
            "message": "⚠️ Tuyến vượt quá giới hạn 8h! Đẩy đơn sang pool xử lý ngầm (Backlog ngày tiếp theo / Tái gộp Clarke-Wright)",
            "total_hours": round(total_h, 2), "repool_orders": orders_in_route, "backlog_orders_next_day": backlog_orders}


# ============================================================================
# 9. CLARKE-WRIGHT SAVINGS: RoutePlanner + simulate_all
# ============================================================================
class RoutePlanner:
    def __init__(self, data: dict, cfg: Config = CFG):
        self.cfg = cfg
        self.veh = data["vehicles"]
        self.drivers = data["drivers"]
        self.dist_df = data["dist"]
        self.cust = data["cust"]
        self.wh = data["warehouses"]
        cat = (self.veh.groupby("vehicle_type")
               .agg(w=("max_weight_kg", "max"), v=("max_volume_m3", "max"), speed=("speed_kmh", "first"),
                    fixed=("fixed_cost", "first"), var=("variable_cost_per_km", "first"))
               .sort_values(["v", "w"]))  # xếp theo THỂ TÍCH trước
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
        """Loại xe nhỏ nhất đủ chỗ: THỂ TÍCH trước, rồi trọng tải."""
        if self.catalog.empty:
            return "Truck"
        ok = self.catalog[(self.catalog["v"] >= v) & (self.catalog["w"] >= w)]
        return ok.index[0] if not ok.empty else self.catalog.index[-1]

    def km(self, route, wh_id) -> float:
        if not route:
            return 0.0
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
        """RÀNG BUỘC KHI GỘP TUYẾN (song song Clarke-Wright): THỂ TÍCH trước -> trọng tải -> thời gian."""
        if sum(demand[c]["volume"] for c in route) > self.max_v:
            return False
        if sum(demand[c]["weight"] for c in route) > self.max_w:
            return False
        return self.metrics(route, wh_id, demand)["hours"] <= self.cfg.max_route_hours

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
            if s <= 0:
                break
            ra, rb = route_of[a], route_of[b]
            if ra is rb or a not in (ra[0], ra[-1]) or b not in (rb[0], rb[-1]):
                continue
            ra = ra if ra[-1] == a else ra[::-1]
            rb = rb if rb[0] == b else rb[::-1]
            merged = ra + rb
            if self.feasible(merged, wh_id, demand):
                for c in merged:
                    route_of[c] = merged
        uniq = {id(r): r for r in route_of.values()}.values()
        return [self.two_opt(r, wh_id) for r in uniq]

    # ---- Thuê ngoài Giao hàng tiết kiệm ----
    def ghtk_tier(self, w, v):
        c = self.cfg
        if w < c.ghtk1_max_kg and v < c.ghtk1_max_m3:
            return 1
        if w < c.ghtk2_max_kg and v < c.ghtk2_max_m3:
            return 2
        return 0  # quá lớn -> thuê xe tải 3PL

    def plan_day(self, date_str: str, day_orders: list) -> dict:
        cfg = self.cfg
        res = {"routes": [], "overdue_routes": [], "exceptions": [], "carried_orders": [], "overdue_backlog_list": [],
               "day_cost": 0.0, "penalty_cost": 0.0}
        date = pd.to_datetime(date_str)

        def waiting(o):
            return (date - pd.to_datetime(o["WAITING_DATE"])).days if o["WAITING_DATE"] else 0

        def exc(sev, o, kind, detail, handled, propose):
            res["exceptions"].append({"NGÀY": date_str, "MỨC ĐỘ": sev, "MÃ ĐƠN": o["ORDER_ID"], "PHÂN LOẠI": kind,
                                      "CHI TIẾT": detail, "ĐÃ XỬ LÝ": handled, "ĐỀ XUẤT": propose})

        normal, overdue = [], []
        for o in day_orders:
            if o["customer_id"] not in self.cust:
                exc("MEDIUM", o, "THIẾU TỌA ĐỘ", "Không tìm thấy khách hàng", "Bỏ qua", "Bổ sung tọa độ")
                res["carried_orders"].append(o)
            elif o["volume"] > self.max_v or o["weight"] > self.max_w:
                why = (f"{o['volume']:.2f}m3 vượt khoang xe lớn nhất" if o["volume"] > self.max_v
                       else f"{o['weight']:.0f}kg vượt tải xe lớn nhất")
                exc("CRITICAL", o, "ĐƠN QUÁ CỠ", why, "Tách riêng", "Thuê xe lớn")
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
            for o in bucket:
                by_wh.setdefault(self.nearest_wh(o["customer_id"]), []).append(o)
            for wh_id, ords in by_wh.items():
                demand, by_cust = {}, {}
                for o in ords:
                    d = demand.setdefault(o["customer_id"], {"weight": 0, "volume": 0, "order_type": o["order_type"]})
                    d["weight"] += o["weight"]
                    d["volume"] += o["volume"]
                    by_cust.setdefault(o["customer_id"], []).append(o)
                routes = self.clarke_wright(list(demand), wh_id, demand)
                for rt in routes:
                    target.append(self._build_route(rt, wh_id, demand, by_cust, pool, assign_driver, date_str))
        for r in res["routes"] + res["overdue_routes"]:
            res["day_cost"] += r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r["driver_cost"]
        res["penalty_cost"] = sum(max(waiting(o), 0) for o in overdue) * cfg.late_penalty_per_day
        return res

    def _build_route(self, route, wh_id, demand, by_cust, pool, assign_driver, date_str):
        """TRÌNH TỰ CHỌN XE: (1) xe nhà trong kho -> (2) xe nhỏ nhất khả thi -> (3) kiểm tra Max_Distance
        -> (4) vượt thì thuê ngoài: GHTK loại 1/2 theo hàng, quá lớn thì 3PL xe tải."""
        cfg = self.cfg
        m = self.metrics(route, wh_id, demand)
        km, hours = m["km"], m["hours"]
        # (1)+(2) xe nhà khả thi (thể tích -> trọng tải) rồi chọn xe NHỎ NHẤT
        cands = [(i, r) for i, r in pool.items()
                 if r["wh_id"] == wh_id and r["max_volume_m3"] >= m["v"] and r["max_weight_kg"] >= m["w"]]
        mode, reason, idx, vrow = "OWN", "", None, None
        if cands:
            idx, vrow = min(cands, key=lambda t: (t[1]["max_volume_m3"], t[1]["max_weight_kg"], t[1]["fixed_cost"]))
            # (3) Max_Distance của chính xe được chọn
            if km >= vrow["max_distance_km"]:
                tier = self.ghtk_tier(m["w"], m["v"])
                mode = f"GHTK-{tier}" if tier else "3PL"
                reason = (f"Quãng đường {km:.1f}km ≥ Max_Distance {vrow['max_distance_km']:g}km của xe {vrow['vehicle_id']}"
                          + ("" if tier else " (hàng quá lớn cho GHTK → thuê xe tải)"))
            else:
                pool.pop(idx)
        else:
            mode, reason = "3PL", "Hết xe nhà phù hợp trong kho"

        start = dt.datetime.combine(pd.to_datetime(date_str).date(), dt.datetime.strptime(cfg.start_time, "%H:%M").time())
        base = {"kind": "NORMAL", "wh_id": wh_id, "route": list(route), "orders": [o for c in route for o in by_cust[c]],
                "km": round(km, 1), "overnight_cost": 0.0, "cut_orders": [], "start": start,
                "end": start + dt.timedelta(hours=hours), "hours": hours, "ext_kind": mode,
                "outsource_reason": reason, "load_kg": m["w"], "load_m3": m["v"]}
        if mode == "OWN":
            primary, assistant, backup = assign_driver(wh_id)
            load_f = m["v"] / vrow["max_volume_m3"] if vrow["max_volume_m3"] > 0 else 0.0   # LẤP ĐẦY THEO THỂ TÍCH
            base.update({"vehicle_id": vrow["vehicle_id"], "license_plate": vrow["license_plate"],
                         "vehicle_type": vrow["vehicle_type"], "external": False, "driver_primary": primary,
                         "driver_assistant": assistant, "is_backup_driver": backup, "speed_kmh": vrow["speed_kmh"],
                         "fixed_cost": vrow["fixed_cost"], "variable_cost": vrow["variable_cost_per_km"] * km,
                         "driver_cost": cfg.backup_driver_cost if backup else 0.0})
        elif mode.startswith("GHTK"):
            tier = int(mode[-1])
            cap_v = cfg.ghtk1_max_m3 if tier == 1 else cfg.ghtk2_max_m3
            base_fee = cfg.ghtk1_base if tier == 1 else cfg.ghtk2_base
            per_km = cfg.ghtk1_per_km if tier == 1 else cfg.ghtk2_per_km
            load_f = m["v"] / cap_v if cap_v > 0 else 0.0
            base.update({"vehicle_id": f"GHTK-L{tier}", "license_plate": f"Thuê ngoài GHTK loại {tier}",
                         "vehicle_type": f"Giao hàng tiết kiệm loại {tier}", "external": True,
                         "driver_primary": "— (GHTK đảm nhận)", "driver_assistant": "—", "is_backup_driver": False,
                         "speed_kmh": m["speed"], "fixed_cost": base_fee, "variable_cost": per_km * km,
                         "driver_cost": 0.0})
        else:  # 3PL xe tải
            vt = m["vtype"] if m["vtype"] in self.catalog.index else (self.catalog.index[-1] if not self.catalog.empty else "Truck")
            speed = self.catalog.loc[vt, "speed"] if vt in self.catalog.index else 35.0
            fx = self.catalog.loc[vt, "fixed"] if vt in self.catalog.index else 300_000
            vr = self.catalog.loc[vt, "var"] if vt in self.catalog.index else 8_000
            cap_v = self.catalog.loc[vt, "v"] if vt in self.catalog.index else m["v"]
            load_f = m["v"] / cap_v if cap_v > 0 else 0.0
            primary, assistant, backup = assign_driver(wh_id)
            base.update({"vehicle_id": f"3PL-{vt}", "license_plate": "Thuê ngoài 3PL (Hết xe nhà)", "vehicle_type": vt,
                         "external": True, "driver_primary": primary, "driver_assistant": assistant,
                         "is_backup_driver": backup, "speed_kmh": speed, "fixed_cost": fx, "variable_cost": vr * km,
                         "driver_cost": cfg.backup_driver_cost if backup else 0.0})
        base["load_factor"] = min(max(load_f, 0.0), 1.0)
        return base


def simulate_all(data: dict, cfg: Config = CFG):
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]:
        by_date.setdefault(o["date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    all_r = [r for d in days.values() for r in d["routes"] + d["overdue_routes"]]
    n_routes = len(all_r)
    operating = sum(d["day_cost"] for d in days.values())
    penalty = sum(d["penalty_cost"] for d in days.values())
    kpis = {
        "violations": sum(1 for r in all_r if r["hours"] > cfg.max_route_hours + 1e-9),
        "operating_cost": operating, "penalty_cost": penalty,
        "external_ratio": sum(r["external"] for r in all_r) / n_routes if n_routes else 0,
        "ghtk_routes": sum(1 for r in all_r if r["ext_kind"].startswith("GHTK")),
        "avg_load_factor": sum(r["load_factor"] for r in all_r) / len(all_r) if all_r else 0,
        "late_order_days": sum(len(d["overdue_backlog_list"]) for d in days.values()),
        "undelivered_orders": sum(len(d["carried_orders"]) for d in days.values()),
        "total_orders": len(data["orders"]),
    }
    return planner, days, kpis, operating + penalty


def route_total(r):
    return r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r.get("driver_cost", 0)


# ============================================================================
# 10. CÁC BƯỚC PIPELINE
# ============================================================================
def file_status(cfg: Config) -> pd.DataFrame:
    items = [
        ("Tab 1 · Hạm đội xe (có Max_Distance)", cfg.vehicle_file, "Bắt buộc"),
        ("Tab 2 · Kho & Tọa độ", cfg.warehouse_file, "Khuyến nghị (toạ độ kho)"),
        ("Tab 3 · Sản phẩm", cfg.product_file, "Tuỳ chọn"),
        ("Tab 4 · Tài xế", cfg.driver_file, "Bắt buộc"),
        ("Tab 5 · Đơn hàng", cfg.order_file, "Bắt buộc"),
        ("Bước 1 · Khách hàng + toạ độ", cfg.cust_file, "Bắt buộc"),
        ("Bước 2 · Ma trận khoảng cách", cfg.matrix_file, "Bắt buộc"),
    ]
    return pd.DataFrame([{"Nguồn": n, "File": os.path.relpath(p, BASE_DIR),
                          "Trạng thái": "✅ Có" if os.path.exists(p) else "❌ Chưa có", "Yêu cầu": need}
                         for n, p, need in items])


def require_files(cfg: Config, which):
    labels = {
        "orders": (cfg.order_file, "Tab 5 (Đơn hàng)"),
        "fleet": (cfg.vehicle_file, "Tab 1 (Hạm đội xe)"),
        "driver": (cfg.driver_file, "Tab 4 (Tài xế)"),
        "matrix": (cfg.matrix_file, "Bước 2 (Ma trận khoảng cách)"),
    }
    missing = [f"`{os.path.relpath(labels[k][0], BASE_DIR)}` (chạy {labels[k][1]})" for k in which if not os.path.exists(labels[k][0])]
    if missing:
        raise UserError("Thiếu file đầu vào: " + "; ".join(missing))


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
        raise UserError("Không có khách hàng hợp lệ (thiếu customer_id) trong DIM_ORDERS.xlsx.")

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
                query = clean_address(addr) or addr  # làm sạch địa chỉ trước khi geocode
                try:
                    loc = geocode(query)
                    if loc:
                        lat, lng, matched_address = float(loc.latitude), float(loc.longitude), str(loc.address)
                        if 8.0 <= lat <= 24.5 and 102.0 <= lng <= 110.0:
                            status = "✅ Geocode hợp lệ"
                        else:
                            lat, lng, status = None, None, "❌ Tọa độ ngoài Việt Nam"
                    else:
                        lat, lng, matched_address, status = None, None, "", "⚠️ Không tìm thấy địa chỉ"
                except Exception as exc:
                    lat, lng, matched_address, status = None, None, "", f"❌ Lỗi: {exc}"
                cache[addr] = (lat, lng, matched_address, status)
            lat_list.append(lat); lng_list.append(lng); matched_list.append(matched_address); status_list.append(status)
        if on_progress:
            on_progress(k, total)

    df_cust["lat"], df_cust["lng"] = lat_list, lng_list
    df_cust["matched_address"], df_cust["trạng_thái_geocode"] = matched_list, status_list
    os.makedirs(OUT_CUSTOMER, exist_ok=True)
    x = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.xlsx")
    j = os.path.join(OUT_CUSTOMER, "DATASET_CUSTOMER.json")
    df_cust.to_excel(x, index=False, sheet_name="DATASET_CUSTOMER")
    with open(j, "w", encoding="utf-8") as fh:
        json.dump(df_to_records(df_cust), fh, ensure_ascii=False, indent=2, default=str)
    success = int((df_cust["trạng_thái_geocode"] == "✅ Geocode hợp lệ").sum())
    return {"df": df_cust, "files": [x, j], "success": success, "failed": total - success}


def build_distance_matrices(df_valid: pd.DataFrame, detour: float = 1.2):
    """OSRM Table API; mất mạng / lỗi -> Haversine x hệ số đường vòng."""
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
            note = f"OSRM trả về mã lỗi: {data.get('code')}"
        else:
            note = f"HTTP Error Status: {response.status_code}"
    except Exception as exc:
        note = f"Không gọi được OSRM API trực tuyến ({exc})"

    n = len(cust_ids)
    lat = df_valid["lat"].astype(float).tolist()
    lng = df_valid["lng"].astype(float).tolist()
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
    dist_path = os.path.join(OUT_MATRIX, "DISTANCE_MATRIX_KM.xlsx")
    dur_path = os.path.join(OUT_MATRIX, "DURATION_MATRIX_MIN.xlsx")
    json_path = os.path.join(OUT_MATRIX, "OSRM_MATRIX_RESULT.json")
    dist.to_excel(dist_path)
    dur.to_excel(dur_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"customers": df_valid["customer_id"].tolist(), "distance_matrix_km": dist.to_dict(),
                   "duration_matrix_min": dur.to_dict()}, fh, ensure_ascii=False, indent=2)
    return {"dist": dist, "dur": dur, "method": method, "note": note, "files": [dist_path, dur_path, json_path],
            "n": len(df_valid), "skipped": len(df_cust) - len(df_valid)}


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
            if abs(lat - depot_lat) < 1e-4 and abs(lng - depot_lng) < 1e-4:
                d = 0.0
            else:
                d = haversine(depot_lat, depot_lng, lat, lng) * cfg.detour_factor
            c0i[c_id] = round(d, 2)
    mat = df_dist.values
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            cust_i, cust_j = customers[i], customers[j]
            c_ij = float(mat[i, j])
            s_ij = c0i.get(cust_i, 0.0) + c0i.get(cust_j, 0.0) - c_ij
            savings.append({"pair": (cust_i, cust_j), "savings_km": round(s_ij, 2),
                            "c_0i": c0i.get(cust_i, 0.0), "c_0j": c0i.get(cust_j, 0.0), "c_ij": c_ij})
    savings_sorted = sorted(savings, key=lambda x: x["savings_km"], reverse=True)
    df_savings = pd.DataFrame([{
        "Cặp khách hàng": f"{s['pair'][0]} - {s['pair'][1]}",
        "Khách hàng 1": s["pair"][0], "Khách hàng 2": s["pair"][1],
        "Mức tiết kiệm (km)": s["savings_km"],
        "Kho -> Khách 1 (c0i)": s["c_0i"], "Kho -> Khách 2 (c0j)": s["c_0j"],
        "Khách 1 <-> Khách 2 (cij)": s["c_ij"],
    } for s in savings_sorted])
    os.makedirs(OUT_MATRIX, exist_ok=True)
    out_path = os.path.join(OUT_MATRIX, "DATASET_SORT_SAVING.xlsx")
    df_savings.to_excel(out_path, index=False, sheet_name="SORT_SAVING")
    return {"df": df_savings, "files": [out_path], "pairs": len(df_savings)}


def step_screening(cfg: Config) -> dict:
    require_files(cfg, ["orders", "fleet", "matrix"])
    return run_master_logistics_optimizer(cfg.order_file, cfg.vehicle_file, cfg.matrix_file)


def step_clarke_wright(cfg: Config) -> dict:
    require_files(cfg, ["orders", "fleet", "driver", "matrix"])
    data = load_data(cfg)
    planner, days, kpis, total_cost = simulate_all(data, cfg)
    return {"planner": planner, "days": days, "kpis": kpis, "total_cost": total_cost,
            "notes": data["notes"], "cfg": cfg, "n_orders": len(data["orders"])}


# ============================================================================
# 11. CHUYỂN KẾT QUẢ SANG BẢNG
# ============================================================================
def tag_routes(day):
    return [(r, "Thường") for r in day["routes"]] + [(r, "Quá hạn (ưu tiên)") for r in day["overdue_routes"]]


VEHICLE_ICON = {"OWN": "🚚 ", "3PL": "🟣 Thuê ngoài 3PL — ", "GHTK-1": "📦 ", "GHTK-2": "📦 "}


def routes_to_df(tagged_routes, wh_info, cfg: Config) -> pd.DataFrame:
    rows = []
    for idx, (r, kind) in enumerate(tagged_routes, 1):
        wh = wh_info.get(r["wh_id"], {"name": r["wh_id"]})
        locs = sorted({re.split(r",|\s-\s", o["Location"])[-1].strip() for o in r["orders"] if o["Location"]})
        rows.append({
            "MÃ TUYẾN": f"Tuyến {wh['name']} #{idx}",
            "KHU VỰC": ", ".join(locs),
            "LOẠI": kind,
            "ĐƠN GIAO": ", ".join(o["ORDER_ID"] for o in r["orders"]),
            "SỐ ĐƠN": len(r["orders"]),
            "TÀI XẾ CHÍNH": r["driver_primary"] + (" ⚠️ dự phòng" if r["is_backup_driver"] else ""),
            "PHỤ XE": r["driver_assistant"],
            "XE": VEHICLE_ICON.get(r["ext_kind"], "🚚 ") + str(r["vehicle_type"]),
            "BIỂN SỐ": r["license_plate"],
            "TỔNG KHỐI LƯỢNG (kg)": round(r["load_kg"], 1),
            "TỔNG THỂ TÍCH (m3)": round(r["load_m3"], 3),
            "QUÃNG ĐƯỜNG (km)": r["km"],
            "LẤP ĐẦY THỂ TÍCH (%)": round(r["load_factor"] * 100, 1),
            "THỜI GIAN (giờ)": round(r["hours"], 2),
            "TỔNG CHI PHÍ (đ)": round(route_total(r)),
            "BẮT ĐẦU": f"{r['start']:%H:%M}",
            "KẾT THÚC": f"{r['end']:%H:%M}",
            "LÝ DO THUÊ NGOÀI": r["outsource_reason"],
            "TRẠNG THÁI": "✅ Đã tối ưu" if r["hours"] <= cfg.max_route_hours + 1e-9 else "⚠️ Vượt giới hạn giờ",
        })
    return pd.DataFrame(rows)


def route_orders_df(r, cust) -> pd.DataFrame:
    rows = []
    for k, o in enumerate(r["orders"], 1):
        c = cust.get(o["customer_id"], {})
        rows.append({"THỨ TỰ": k, "MÃ ĐƠN": o["ORDER_ID"], "MÃ KHÁCH": o["customer_id"], "LOẠI ĐƠN": o["order_type"],
                     "TRỌNG LƯỢNG (kg)": o["weight"], "THỂ TÍCH (m3)": o["volume"], "ĐỊA CHỈ": o["Location"],
                     "lat": c.get("lat"), "lon": c.get("lon")})
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
        wh = wh_info.get(r["wh_id"], {"name": r["wh_id"]})
        res = evaluate_route_time_constraint({
            "orders": r["orders"], "total_distance_km": r["km"], "vehicle_speed_kmh": r["speed_kmh"],
            "current_date": date_str, "route": r.get("route", []),
        }, rules)
        rows.append({"MÃ TUYẾN": f"Tuyến {wh['name']} #{idx}", "GIỜ (PLANNER)": round(r["hours"], 2),
                     "GIỜ (KIỂM ĐỊNH THEO ĐƠN)": res["total_hours"], "KẾT LUẬN": res["status"], "GHI CHÚ": res["message"]})
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
        {"Chỉ số": "Tổng chi phí (vận hành + phạt)", "Giá trị": plan["total_cost"]},
        {"Chỉ số": "Chi phí vận hành", "Giá trị": kpis["operating_cost"]},
        {"Chỉ số": "Chi phí phạt", "Giá trị": kpis["penalty_cost"]},
        {"Chỉ số": "Số tuyến vi phạm giới hạn giờ", "Giá trị": kpis["violations"]},
        {"Chỉ số": "Tỉ lệ thuê ngoài (GHTK + 3PL)", "Giá trị": kpis["external_ratio"]},
        {"Chỉ số": "Số tuyến thuê Giao hàng tiết kiệm", "Giá trị": kpis["ghtk_routes"]},
        {"Chỉ số": "Lấp đầy thể tích trung bình", "Giá trị": kpis["avg_load_factor"]},
        {"Chỉ số": "Đơn chưa giao", "Giá trị": kpis["undelivered_orders"]},
        {"Chỉ số": "Tổng số đơn", "Giá trị": kpis["total_orders"]},
    ])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        kpi_df.to_excel(writer, sheet_name="KPI", index=False)
        if route_frames:
            pd.concat(route_frames, ignore_index=True).to_excel(writer, sheet_name="ROUTES", index=False)
        if order_frames:
            pd.concat(order_frames, ignore_index=True).to_excel(writer, sheet_name="ORDERS_IN_ROUTE", index=False)
        if exc_rows:
            pd.DataFrame(exc_rows).to_excel(writer, sheet_name="EXCEPTIONS", index=False)
    return buf.getvalue()


# ============================================================================
# 12. TAB 6 — UI (BẢNG NHẬP THAM SỐ CLARKE-WRIGHT nằm trong render_routing_settings)
# ============================================================================
def render_routing_settings():
    with st.expander("⚙️ Tham số mô hình & kho xuất phát (Savings)", expanded=False):
        c1, c2, c3 = st.columns(3)
        start_time = c1.text_input("Giờ xuất phát (HH:MM)", "08:30", key="cfg_start")
        max_hours = c2.number_input("Giới hạn giờ / tuyến", min_value=1.0, max_value=24.0, value=8.0, step=0.5, key="cfg_maxh")
        detour = c3.number_input("Hệ số đường vòng (Haversine)", min_value=1.0, max_value=3.0, value=1.2, step=0.05, key="cfg_detour")
        c4, c5, c6, c7 = st.columns(4)
        svc_b2b = c4.number_input("Bốc/dỡ B2B (phút/điểm)", min_value=0, max_value=600, value=105, step=5, key="cfg_b2b")
        svc_b2c = c5.number_input("Bốc/dỡ B2C (phút/điểm)", min_value=0, max_value=600, value=60, step=5, key="cfg_b2c")
        backup = c6.number_input("Chi phí tài xế dự phòng (đ)", min_value=0, max_value=10_000_000, value=400_000, step=50_000, key="cfg_backup")
        penalty = c7.number_input("Phạt trễ / đơn / ngày (đ)", min_value=0, max_value=10_000_000, value=100_000, step=10_000, key="cfg_penalty")

        st.markdown("**📦 Thuê ngoài Giao hàng tiết kiệm** — kích hoạt khi *quãng đường tuyến ≥ Max_Distance* của xe nhà "
                    "đã chọn (cột `Max_Distance` trong `output_fleet/DIM_VEHICLE.xlsx`). Hàng vượt cả loại 2 → thuê xe tải 3PL.")
        g1 = st.columns(4)
        k1_kg = g1[0].number_input("Loại 1: hàng dưới (kg)", min_value=0.0, value=30.0, step=1.0, key="cfg_g1kg")
        k1_m3 = g1[1].number_input("Loại 1: hàng dưới (m³)", min_value=0.0, value=1.0, step=0.1, key="cfg_g1m3")
        k1_base = g1[2].number_input("Loại 1: phí cố định (đ/chuyến)", min_value=0, value=25_000, step=5_000, key="cfg_g1base")
        k1_km = g1[3].number_input("Loại 1: phí theo km (đ/km)", min_value=0, value=3_000, step=500, key="cfg_g1km")
        g2 = st.columns(4)
        k2_kg = g2[0].number_input("Loại 2: hàng dưới (kg)", min_value=0.0, value=100.0, step=5.0, key="cfg_g2kg")
        k2_m3 = g2[1].number_input("Loại 2: hàng dưới (m³)", min_value=0.0, value=5.0, step=0.5, key="cfg_g2m3")
        k2_base = g2[2].number_input("Loại 2: phí cố định (đ/chuyến)", min_value=0, value=60_000, step=5_000, key="cfg_g2base")
        k2_km = g2[3].number_input("Loại 2: phí theo km (đ/km)", min_value=0, value=6_000, step=500, key="cfg_g2km")

        base_cfg = Config()
        opts = depot_options(base_cfg)
        if st.session_state.get("cfg_depot") not in opts:
            st.session_state["cfg_depot"] = list(opts)[0]
        depot_label = st.selectbox("Kho xuất phát dùng cho bảng Savings S_ij (bước 3)", list(opts), key="cfg_depot")
    try:
        dt.datetime.strptime(start_time, "%H:%M")
    except ValueError:
        st.warning("Giờ xuất phát không đúng định dạng HH:MM — dùng mặc định 08:30.")
        start_time = "08:30"
    cfg = Config(start_time=start_time, max_route_hours=float(max_hours), detour_factor=float(detour),
                 service_min={"B2B": int(svc_b2b), "B2C": int(svc_b2c)},
                 backup_driver_cost=float(backup), late_penalty_per_day=float(penalty),
                 ghtk1_max_kg=float(k1_kg), ghtk1_max_m3=float(k1_m3), ghtk1_base=float(k1_base), ghtk1_per_km=float(k1_km),
                 ghtk2_max_kg=float(k2_kg), ghtk2_max_m3=float(k2_m3), ghtk2_base=float(k2_base), ghtk2_per_km=float(k2_km))
    return cfg, opts[depot_label]


def exec_step(name, fn):
    try:
        st.session_state[f"t6_{name}"] = fn()
        return True
    except UserError as exc:
        st.error(f"❌ {exc}")
    except Exception as exc:
        st.error(f"❌ Lỗi ở bước `{name}`: {exc}")
    return False


def render_dashboard(plan):
    planner, days, kpis, total_cost, cfg = plan["planner"], plan["days"], plan["kpis"], plan["total_cost"], plan["cfg"]
    for note in plan["notes"]:
        st.warning(f"⚠️ {note}")
    dates = list(days)
    if not dates:
        st.warning("⚠️ Không tìm thấy đơn hàng nào trong dữ liệu!")
        return
    if st.session_state.get("t6_date") not in dates:
        st.session_state["t6_date"] = dates[0]
    date_str = st.selectbox("📅 Chọn ngày điều phối", dates, key="t6_date")
    day = days[date_str]
    tagged = tag_routes(day)

    st.markdown(f"## 🗓️ DASHBOARD ĐIỀU PHỐI ĐỘNG — {date_str}")
    st.caption("Clarke-Wright Savings + 2-opt · ràng buộc: thể tích → trọng tải → giờ · chọn xe nhà nhỏ nhất → Max_Distance → thuê ngoài GHTK/3PL.")
    g = st.columns(5)
    g[0].metric("🎯 Vi phạm giới hạn giờ", kpis["violations"])
    g[1].metric("👑 Tổng chi phí (VNĐ)", money(total_cost),
                help=f"Vận hành {money(kpis['operating_cost'])} · Phạt {money(kpis['penalty_cost'])}")
    g[2].metric("Lấp đầy thể tích TB", f"{kpis['avg_load_factor'] * 100:.1f}%")
    g[3].metric("Tỉ lệ thuê ngoài", f"{kpis['external_ratio'] * 100:.1f}%", delta=f"GHTK: {kpis['ghtk_routes']} tuyến", delta_color="off")
    g[4].metric("Đơn chưa giao", f"{kpis['undelivered_orders']}/{kpis['total_orders']}")
    n_ext = sum(r["external"] for r, _ in tagged)
    n_ghtk = sum(r["ext_kind"].startswith("GHTK") for r, _ in tagged)
    d = st.columns(4)
    d[0].metric("Chi phí ngày (VNĐ)", money(day["day_cost"]))
    d[1].metric("Phạt trễ ngày (VNĐ)", money(day["penalty_cost"]))
    d[2].metric("Tổng chuyến", len(tagged), delta=f"Thuê ngoài: {n_ext} (GHTK: {n_ghtk})", delta_color="off")
    d[3].metric("Đơn quá hạn / ngoại lệ", f"{len(day['overdue_backlog_list'])} / {len(day['exceptions'])}")

    if kpis["violations"]:
        st.error(f"⚠️ Có {kpis['violations']} tuyến vượt giới hạn {cfg.max_route_hours:g}h.")
    else:
        st.success(f"✅ Tất cả tuyến đều thỏa ràng buộc thời gian ≤ {cfg.max_route_hours:g}h và sức chứa xe.")

    if not tagged:
        st.info("Không có tuyến nào được lập trong ngày này.")
    else:
        rdf = routes_to_df(tagged, planner.wh, cfg)
        st.markdown("### 🚚 Danh sách tuyến")
        st.dataframe(rdf, hide_index=True, column_config={
            "LẤP ĐẦY THỂ TÍCH (%)": st.column_config.ProgressColumn("LẤP ĐẦY THỂ TÍCH (%)", min_value=0, max_value=100, format="%.0f%%"),
            "TỔNG CHI PHÍ (đ)": st.column_config.NumberColumn("TỔNG CHI PHÍ (đ)", format="%d"),
            "QUÃNG ĐƯỜNG (km)": st.column_config.NumberColumn("QUÃNG ĐƯỜNG (km)", format="%.1f"),
        })
        labels = rdf["MÃ TUYẾN"].tolist()
        sel = st.selectbox("🔎 Xem chi tiết thứ tự giao của tuyến", labels, key=f"t6_route_sel_{date_str}")
        r, _kind = tagged[labels.index(sel)]
        odf = route_orders_df(r, planner.cust)
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(odf.drop(columns=["lat", "lon"]), hide_index=True)
            st.caption(f"🚚 {r['vehicle_type']} · {r['license_plate']} · 👤 {r['driver_primary']} · 🤝 {r['driver_assistant']} "
                       f"· ⏰ {r['start']:%H:%M} ➔ {r['end']:%H:%M}")
            if r["outsource_reason"]:
                st.caption(f"📌 {r['outsource_reason']}")
        with right:
            wh = planner.wh.get(r["wh_id"])
            pts = odf.dropna(subset=["lat", "lon"])[["lat", "lon"]]
            if wh:
                pts = pd.concat([pd.DataFrame([{"lat": wh["lat"], "lon": wh["lon"]}]), pts], ignore_index=True)
            if not pts.empty:
                st.map(pts)

        with st.expander("🧮 Kiểm định ràng buộc thời gian tuyến (evaluate_route_time_constraint)"):
            st.caption("Thời gian = quãng đường / vận tốc xe + bốc dỡ theo từng đơn (B2C/B2B). "
                       f"Vượt {cfg.max_route_hours:g}h -> BACKLOG_OR_REPOOL (đẩy lại pool / sang ngày sau).")
            st.dataframe(audit_routes_df(tagged, planner.wh, cfg, date_str), hide_index=True)

    if day["exceptions"]:
        st.markdown("### 🚨 Ngoại lệ trong ngày")
        st.dataframe(pd.DataFrame(day["exceptions"]), hide_index=True)
    if day["overdue_backlog_list"]:
        st.markdown("### ⏳ Đơn quá hạn (ưu tiên xếp tuyến)")
        st.dataframe(pd.DataFrame(day["overdue_backlog_list"]), hide_index=True)
    if day["carried_orders"]:
        st.markdown("### 📌 Đơn chưa giao / chuyển sang ngày sau")
        st.dataframe(pd.DataFrame(day["carried_orders"]), hide_index=True)

    if "xlsx" not in plan:
        plan["xlsx"] = build_plan_workbook(plan)
    st.download_button("⬇️ Tải toàn bộ kế hoạch tuyến (Excel)", plan["xlsx"], file_name="ROUTE_PLAN.xlsx",
                       mime=XLSX_MIME, key="t6_dl_plan")


def render_routing_tab():
    st.header("🗺️ Dashboard Định tuyến — Thuật toán Clarke-Wright Savings")
    st.markdown("**Đọc input từ các thư mục `output_*` → Geocode khách hàng → Ma trận khoảng cách → "
                "Savings S_ij → Sàng lọc thể tích/trọng tải → Clarke-Wright + 2-opt → Chọn xe → Max_Distance → Thuê ngoài**")
    st.latex(r"S_{ij} = c_{0i} + c_{0j} - c_{ij}")
    cfg, depot = render_routing_settings()

    st.markdown("### 📂 Trạng thái dữ liệu đầu vào (đọc trực tiếp từ các thư mục output_*)")
    st.dataframe(file_status(cfg), hide_index=True)

    if st.button("⚡ Chạy toàn bộ pipeline (bước 1 → 5)", type="primary", key="t6_runall"):
        steps = [
            ("geo", "Bước 1 · Geocode khách hàng (ArcGIS)", lambda: step_geocode_customers(cfg)),
            ("matrix", "Bước 2 · Ma trận khoảng cách (OSRM / Haversine)", lambda: step_distance_matrix(cfg)),
            ("savings", "Bước 3 · Tính & sắp xếp Savings S_ij", lambda: step_savings(cfg, depot)),
            ("screen", "Bước 4 · Sàng lọc thể tích → trọng tải", lambda: step_screening(cfg)),
            ("plan", "Bước 5 · Clarke-Wright toàn cục", lambda: step_clarke_wright(cfg)),
        ]
        with st.status("Đang chạy pipeline định tuyến...", expanded=True) as status:
            for name, label, fn in steps:
                st.write(f"▶️ {label}...")
                try:
                    st.session_state[f"t6_{name}"] = fn()
                except UserError as exc:
                    status.update(label=f"❌ Dừng ở: {label}", state="error")
                    st.error(f"❌ {exc}")
                    break
                except Exception as exc:
                    status.update(label=f"❌ Lỗi ở: {label}", state="error")
                    st.error(f"❌ {exc}")
                    break
            else:
                status.update(label="✅ Hoàn tất pipeline", state="complete")

    st.divider()

    st.markdown("### 1️⃣ Geocode khách hàng (ArcGIS) — `output_orders` → `output_customer`")
    if st.button("▶️ Chạy bước 1", key="t6_b1"):
        def _geo():
            prog = st.progress(0.0, text="Đang geocode khách hàng...")
            try:
                return step_geocode_customers(cfg, lambda k, n: prog.progress(k / n, text=f"Đã geocode {k}/{n} khách hàng"))
            finally:
                prog.empty()
        exec_step("geo", _geo)
    res = st.session_state.get("t6_geo")
    if res:
        m = st.columns(3)
        m[0].metric("👥 Tổng khách hàng", len(res["df"]))
        m[1].metric("✅ Hợp lệ", res["success"])
        m[2].metric("⚠️ Chưa xác định", res["failed"])
        st.dataframe(res["df"], hide_index=True)
        ok = res["df"].dropna(subset=["lat", "lng"])
        if not ok.empty:
            st.map(ok.rename(columns={"lng": "lon"})[["lat", "lon"]])
        st.caption("📁 " + ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"]))

    st.markdown("### 2️⃣ Ma trận khoảng cách — `output_customer` → `output_matrix`")
    if st.button("▶️ Chạy bước 2", key="t6_b2"):
        exec_step("matrix", lambda: step_distance_matrix(cfg))
    res = st.session_state.get("t6_matrix")
    if res:
        if res["note"]:
            st.warning(f"🔄 Dùng phương án dự phòng: **{res['method']}**. Lý do: {res['note']}")
        else:
            st.success(f"✅ Kết nối OSRM thành công — {res['method']}")
        st.caption(f"{res['n']} khách hàng có tọa độ (bỏ qua {res['skipped']} khách chưa có tọa độ). Xem trước tối đa 15×15:")
        st.dataframe(res["dist"].iloc[:15, :15].round(2))
        st.caption("📁 " + ", ".join(f"`{os.path.relpath(p, BASE_DIR)}`" for p in res["files"]))

    st.markdown("### 3️⃣ Savings S_ij sắp xếp giảm dần — `output_matrix`")
    if st.button("▶️ Chạy bước 3", key="t6_b3"):
        exec_step("savings", lambda: step_savings(cfg, depot))
    res = st.session_state.get("t6_savings")
    if res:
        st.caption(f"{res['pairs']} cặp khách hàng · hiển thị 200 cặp có mức tiết kiệm lớn nhất · "
                   f"📁 `{os.path.relpath(res['files'][0], BASE_DIR)}`")
        st.dataframe(res["df"].head(200), hide_index=True)

    st.markdown("### 4️⃣ Sàng lọc thể tích → trọng tải — `output_orders` + `output_fleet`")
    if st.button("▶️ Chạy bước 4", key="t6_b4"):
        exec_step("screen", lambda: step_screening(cfg))
    res = st.session_state.get("t6_screen")
    if res:
        m = st.columns(4)
        m[0].metric("✅ Đơn hợp lệ", len(res["valid_orders"]))
        m[1].metric("❌ Đơn quá cỡ", len(res["oversized_orders"]))
        m[2].metric("Khoang xe lớn nhất (m³)", f"{res['fleet_max_specs']['max_volume']:,.1f}")
        m[3].metric("Tải xe lớn nhất (kg)", f"{res['fleet_max_specs']['max_weight']:,.0f}")
        if res["oversized_orders"]:
            st.error("Các đơn dưới đây vượt thể tích / trọng tải xe lớn nhất (tách riêng, cần thuê xe lớn):")
            st.dataframe(pd.DataFrame(res["oversized_orders"]), hide_index=True)
        if res["valid_orders"]:
            vdf = pd.DataFrame(res["valid_orders"])
            if "eligible_vehicles" in vdf.columns:
                vdf["eligible_vehicles"] = vdf["eligible_vehicles"].apply(lambda x: ", ".join(map(str, x)))
            st.dataframe(vdf, hide_index=True)

    st.markdown("### 5️⃣ Clarke-Wright Savings toàn cục — Dashboard điều phối")
    if st.button("▶️ Chạy bước 5 (Clarke-Wright)", key="t6_b5", type="primary"):
        exec_step("plan", lambda: step_clarke_wright(cfg))
    plan = st.session_state.get("t6_plan")
    if plan:
        render_dashboard(plan)
    else:
        st.info("Bấm **Chạy bước 5** (hoặc **Chạy toàn bộ pipeline**) để lập tuyến và xem dashboard.")


# ============================================================================
# MAIN
# ============================================================================
TAB_NAMES = [
    "🚚 1. Hạm đội xe (Fleet)",
    "🏭 2. Kho & Tọa độ (Warehouse)",
    "📦 3. Sản phẩm (Product)",
    "👨‍✈️ 4. Tài xế (Driver)",
    "🧾 5. Đơn hàng (Orders)",
    "🗺️ 6. Dashboard Định tuyến (Clarke-Wright)",
]
TAB_KEYS = ["fleet", "wh", "product", "driver", "orders"]


def main():
    st.title("🚚 Smart Logistics — Quản lý dữ liệu & Tối ưu định tuyến")
    with st.sidebar:
        st.markdown("### ℹ️ Hướng dẫn nhanh")
        st.markdown("1. Điền / upload dữ liệu ở **Tab 1 → 5**, bấm *Quét* rồi *Chuẩn hóa & Xử lý*.\n"
                    "2. Cả 5 tab cho cùng một khung kết quả và cùng cấu trúc file `DIM_*.xlsx` "
                    "(sheet dữ liệu · `COLUMN_MAPPING` · `VALUE_MAPPING`) + `DIM_*.json` trong `output_*`.\n"
                    "3. **Tab 6** đọc trực tiếp các thư mục này để geocode khách, lập ma trận "
                    "(`output_matrix`) và chạy Clarke-Wright.")
        st.caption("⚠️ Ổ đĩa của Streamlit Cloud là tạm thời: file `output_*` mất khi app khởi động lại "
                   "và được dùng chung giữa các phiên — hãy tải file về nếu cần lưu lâu dài.")
    tabs = st.tabs(TAB_NAMES)
    for tab, key in zip(tabs[:5], TAB_KEYS):
        with tab:
            render_entity_tab(key)
    with tabs[5]:
        render_routing_tab()


if __name__ == "__main__":
    main()
