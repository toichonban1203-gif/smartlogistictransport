# ================================================================
# Block nhập liệu khai nguồn lực về hạm đội xe doanh nghiệp sở hữu
# ================================================================
# -*- coding: utf-8 -*-
import importlib, json, os, re, subprocess, sys, tempfile
REQUIRED = {"gradio": "gradio>=5.0", "pandas": "pandas", "numpy": "numpy", "openpyxl": "openpyxl", "rapidfuzz": "rapidfuzz", "unidecode": "unidecode"}
missing = [pkg for mod, pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
import numpy as np, pandas as pd, gradio as gr
from rapidfuzz import fuzz
from unidecode import unidecode

NONE = "-- Không sử dụng --"

def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except: pass
    return str(v).strip() == ""

def norm(v):
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()

def read_any(f):
    path = f if isinstance(f, str) else f.name
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")): df = pd.read_excel(path)
    elif low.endswith(".json"): df = pd.read_json(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding="cp1258")
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

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

def profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in ["max_weight", "max_volume", "average_speed", "fixed_cost", "variable_cost"]:
        return float(values.str.replace(r"[^\d.]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))

def semantic_mapping(df):
    result, used = {}, set()
    for field, (_, aliases) in VEHICLE_FIELDS.items():
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used: continue
            col_norm = norm(col)
            fuzzy_score = max(fuzz.token_set_ratio(col_norm, norm(a)) for a in aliases) / 100.0
            keyword_score = float(any(norm(a) in col_norm for a in aliases if len(norm(a)) >= 3))
            score = 0.75 * (0.7 * fuzzy_score + 0.3 * keyword_score) + 0.25 * profile_score(df[col], field)
            if score > best_score: best_col, best_score = col, score
        if best_col is not None and best_score >= 0.35:
            result[field] = (best_col, best_score)
            used.add(best_col)
        else: result[field] = (None, 0.0)
    return result

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

def validate_output(df):
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

def empty_table():
    return pd.DataFrame({"Mã xe": ["VEH_01"], "Biển số": ["29C-123.45"], "ID kho": ["WH_HN_01"], "Trọng tải (kg)": [5000], "Thể tích (m3)": [20], "Vận tốc (km/h)": [50], "Chi phí cố định": [500000], "Chi phí biến đổi": [5000]})

def get_input(mode, file_obj, manual):
    df = read_any(file_obj) if mode == "Upload file" else pd.DataFrame(manual)
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise gr.Error("Chưa có dữ liệu phương tiện để quét.")
    return df

with gr.Blocks(title="Smart Logistics - Fleet Management") as app:
    gr.Markdown("# 🚚 Smart Logistics — Quản lý & Chuẩn hóa Phương tiện\n\n**Input → Semantic Mapping → Làm sạch thông số → Validate → Export Excel/JSON**\n\n> 🔒 Cột nào không có trong file, hệ thống sẽ tự động để trống hoặc mặc định mà không làm gián đoạn.")
    mode = gr.Radio(["Nhập tay", "Upload file"], value="Nhập tay", label="Cách nhập dữ liệu phương tiện")

    with gr.Column() as manual_box:
        gr.Markdown("### ✍️ Nhập trực tiếp danh mục xe")
        manual = gr.Dataframe(value=empty_table(), headers=["Mã xe", "Biển số", "ID kho", "Trọng tải (kg)", "Thể tích (m3)", "Vận tốc (km/h)", "Chi phí cố định", "Chi phí biến đổi"], datatype=["str", "str", "str", "number", "number", "number", "number", "number"], interactive=True, row_count=(1, "dynamic"), wrap=True)

    with gr.Column(visible=False) as file_box:
        gr.Markdown("### 📂 Upload\nHỗ trợ **Excel / CSV / JSON**.")
        file_obj = gr.File(label="File danh mục xe", file_types=[".xlsx", ".xls", ".xlsm", ".csv", ".json"])

    mode.change(lambda m: (gr.update(visible=m == "Nhập tay"), gr.update(visible=m == "Upload file")), mode, [manual_box, file_box])

    scan_btn = gr.Button("🔍 Quét & Semantic Mapping", variant="primary")
    scan_msg = gr.Markdown()

    with gr.Column(visible=False) as mapping_box:
        gr.Markdown("### 🔗 Kiểm tra ánh xạ cột phương tiện (Cột nào không có chọn '-- Không sử dụng --')")
        v_id_col = gr.Dropdown(choices=[NONE], value=NONE, label="Mã xe ← cột nào?")
        plate_col = gr.Dropdown(choices=[NONE], value=NONE, label="Biển số ← cột nào?")
        wh_id_col = gr.Dropdown(choices=[NONE], value=NONE, label="ID kho hoạt động ← cột nào?")
        weight_col = gr.Dropdown(choices=[NONE], value=NONE, label="Trọng tải khối lượng (kg) ← cột nào?")
        volume_col = gr.Dropdown(choices=[NONE], value=NONE, label="Trọng tải thể tích (m3) ← cột nào?")
        speed_col = gr.Dropdown(choices=[NONE], value=NONE, label="Vận tốc trung bình (km/h) ← cột nào?")
        f_cost_col = gr.Dropdown(choices=[NONE], value=NONE, label="Chi phí cố định ← cột nào?")
        v_cost_col = gr.Dropdown(choices=[NONE], value=NONE, label="Chi phí biến đổi ← cột nào?")
        process_btn = gr.Button("🚀 Chuẩn hóa & Xử lý Fleet", variant="primary")

    raw_state = gr.State()
    gr.Markdown("### 📊 Kết quả phương tiện")
    result = gr.Dataframe(headers=["vehicle_id", "license_plate", "id_warehouse", "max_weight_kg", "max_volume_m3", "average_speed_kmh", "fixed_cost", "variable_cost", "trạng_thái", "kiểm_tra"], interactive=False, wrap=True)
    result_msg = gr.Markdown()

    export_btn = gr.Button("📦 Xuất Excel + JSON", variant="secondary")
    export_msg = gr.Markdown()
    export_files = gr.Files(label="File kết quả phương tiện")

    def scan(m, f, man):
        raw = get_input(m, f, man)
        mapping = semantic_mapping(raw)
        choices = [NONE] + [str(c) for c in raw.columns]
        def get_val(key):
            col, score = mapping[key]
            return str(col) if col else NONE, f"Độ tin cậy: {score:.0%}"
        id_v, id_i = get_val("vehicle_id")
        pl_v, pl_i = get_val("license_plate")
        wh_v, wh_i = get_val("warehouse_id")
        wt_v, wt_i = get_val("max_weight")
        vl_v, vl_i = get_val("max_volume")
        sp_v, sp_i = get_val("average_speed")
        fc_v, fc_i = get_val("fixed_cost")
        vc_v, vc_i = get_val("variable_cost")
        return raw, f"### 🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**", gr.update(visible=True), gr.update(choices=choices, value=id_v, info=id_i), gr.update(choices=choices, value=pl_v, info=pl_i), gr.update(choices=choices, value=wh_v, info=wh_i), gr.update(choices=choices, value=wt_v, info=wt_i), gr.update(choices=choices, value=vl_v, info=vl_i), gr.update(choices=choices, value=sp_v, info=sp_i), gr.update(choices=choices, value=fc_v, info=fc_i), gr.update(choices=choices, value=vc_v, info=vc_i)

    scan_btn.click(scan, [mode, file_obj, manual], [raw_state, scan_msg, mapping_box, v_id_col, plate_col, wh_id_col, weight_col, volume_col, speed_col, f_cost_col, v_cost_col])

    def process(raw, i_col, p_col, wh_col, w_col, v_col, s_col, fc_col, vc_col):
        try:
            if raw is None: raise gr.Error("Hãy quét dữ liệu trước.")
            raw_df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
            rows = []
            for _, row in raw_df.iterrows():
                rows.append(process_vehicle(
                    row.get(i_col) if i_col and i_col != NONE else "",
                    row.get(p_col) if p_col and p_col != NONE else "",
                    row.get(wh_col) if wh_col and wh_col != NONE else "",
                    row.get(w_col) if w_col and w_col != NONE else 0,
                    row.get(v_col) if v_col and v_col != NONE else 0,
                    row.get(s_col) if s_col and s_col != NONE else 0,
                    row.get(fc_col) if fc_col and fc_col != NONE else 0,
                    row.get(vc_col) if vc_col and vc_col != NONE else 0
                ))
            out = validate_output(pd.DataFrame(rows))
            success = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            msg = f"### 🧭 Hoàn tất chuẩn hóa phương tiện\n- Tổng loại xe: **{len(out)}**\n- Dòng hợp lệ: **{success}/{len(out)}**"
            return out, msg
        except Exception as e:
            return pd.DataFrame({"Lỗi hệ thống": [str(e)]}), f"❌ **Lỗi chi tiết:** `{str(e)}`"

    process_btn.click(process, [raw_state, v_id_col, plate_col, wh_id_col, weight_col, volume_col, speed_col, f_cost_col, v_cost_col], [result, result_msg])

    export_btn.click(lambda df: (lambda checked: (os.makedirs(d := os.path.join(os.getcwd(), "output_fleet"), exist_ok=True),
                       checked.to_excel(os.path.join(d, "DIM_VEHICLE.xlsx"), sheet_name="DIM_VEHICLE", index=False),
                       json.dump(checked.to_dict("records"), open(os.path.join(d, "DIM_VEHICLE.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str),
                       f"### 💾 Xuất thành công danh mục phương tiện!\n📁 File đã được lưu thẳng vào thư mục nguồn `./output_fleet` trong Colab ✨",
                       [os.path.join(d, "DIM_VEHICLE.xlsx"), os.path.join(d, "DIM_VEHICLE.json")])[3:5])(validate_output(pd.DataFrame(df))), [result], [export_msg, export_files])

if __name__ == "__main__":
    app.launch(share=True, theme=gr.themes.Soft())

# ===========================================================
# Block nhập liệu thông tin về các kho của doanh nghiệp
# ===========================================================

# -*- coding: utf-8 -*-
import importlib, json, os, re, subprocess, sys, tempfile, time
from functools import lru_cache
REQUIRED = {"gradio": "gradio>=5.0", "pandas": "pandas", "numpy": "numpy", "openpyxl": "openpyxl", "rapidfuzz": "rapidfuzz", "geopy": "geopy", "unidecode": "unidecode"}
missing = [pkg for mod, pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
import numpy as np, pandas as pd, gradio as gr
from rapidfuzz import fuzz
from unidecode import unidecode
from geopy.geocoders import ArcGIS

NONE = "-- Không sử dụng --"
VIETNAM_BOUNDS = (8.0, 24.0, 102.0, 110.0)

def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except: pass
    return str(v).strip() == ""

def norm(v):
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()

def read_any(f):
    path = f if isinstance(f, str) else f.name
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")): df = pd.read_excel(path)
    elif low.endswith(".json"): df = pd.read_json(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding="cp1258")
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

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

def profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field == "address": return float(0.7 * (values.str.len() >= 10).mean() + 0.3 * values.str.contains(r"[,\-/]").mean())
    return float(values.nunique() / len(values))

def semantic_mapping(df):
    result, used = {}, set()
    for field, (_, aliases) in WAREHOUSE_FIELDS.items():
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used: continue
            col_norm = norm(col)
            fuzzy_score = max(fuzz.token_set_ratio(col_norm, norm(a)) for a in aliases) / 100.0
            keyword_score = float(any(norm(a) in col_norm for a in aliases if len(norm(a)) >= 3))
            score = 0.75 * (0.7 * fuzzy_score + 0.3 * keyword_score) + 0.25 * profile_score(df[col], field)
            if score > best_score: best_col, best_score = col, score
        if best_col is not None and best_score >= 0.40:
            result[field] = (best_col, best_score)
            used.add(best_col)
        else: result[field] = (None, 0.0)
    return result

try: GEOCODER = ArcGIS(user_agent="smart-logistics-warehouse/1.0", timeout=10)
except: GEOCODER = None

@lru_cache(maxsize=2048)
def geocode_address(address, retries=3):
    if not address: return {"ok": False, "status": "❌ Địa chỉ trống"}
    if GEOCODER is None: return {"ok": False, "status": "❌ Không khởi tạo được ArcGIS"}
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            loc = GEOCODER.geocode(address, timeout=10)
            if loc is None: return {"ok": False, "status": "⚠️ ArcGIS không tìm thấy địa chỉ"}
            raw = getattr(loc, "raw", {}) or {}
            score = raw.get("score")
            try: score = float(score) if score is not None else None
            except: score = None
            lat, lng = float(loc.latitude), float(loc.longitude)
            if not coordinate_in_vietnam(lat, lng): return {"ok": False, "status": "⚠️ ArcGIS trả tọa độ ngoài Việt Nam"}
            return {"ok": True, "lat": lat, "lng": lng, "display_name": getattr(loc, "address", "") or "", "score": score}
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries: time.sleep(1)
    return {"ok": False, "status": f"❌ ArcGIS lỗi sau {retries} lần: {last_error[:150]}"}

def process_warehouse(warehouse_id, address, do_geocode=True):
    raw_address = "" if is_blank(address) else str(address).strip()
    cleaned = clean_address(raw_address)
    quality, clean_status = address_quality(cleaned)
    result = {"id_warehouse": warehouse_id, "address": cleaned, "lat": None, "lng": None, "địa_chỉ_gốc": raw_address, "chất_lượng_địa_chỉ": quality, "trạng_thái_làm_sạch": clean_status, "địa_chỉ_geocode": "", "geocode_score": None, "trạng_thái_geocode": "—", "nguồn_tọa_độ": ""}
    if is_blank(warehouse_id): result["trạng_thái_geocode"] = "❌ Thiếu Mã kho"; return result
    if not cleaned: result["trạng_thái_geocode"] = "❌ Không có địa chỉ để geocode"; return result
    if not do_geocode: result["trạng_thái_geocode"] = "⏸️ Đã tắt geocoding"; return result
    geo = geocode_address(cleaned)
    if not geo["ok"]: result["trạng_thái_geocode"] = geo["status"]; return result
    result.update({"lat": geo["lat"], "lng": geo["lng"], "địa_chỉ_geocode": geo["display_name"], "geocode_score": geo["score"], "trạng_thái_geocode": f"✅ ArcGIS geocode thành công" + (f" | score {geo['score']:.0f}" if geo["score"] is not None else ""), "nguồn_tọa_độ": "ArcGIS"})
    return result

def validate_output(df):
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

def empty_table():
    return pd.DataFrame({"Mã kho": ["", "", ""], "Địa chỉ kho": ["", "", ""]})

def get_input(mode, file_obj, manual):
    df = read_any(file_obj) if mode == "Upload file" else pd.DataFrame(manual)
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise gr.Error("Chưa có dữ liệu kho để quét.")
    return df

with gr.Blocks(title="Smart Logistics - Warehouse Geocoding", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏭 Smart Logistics — Quét tọa độ kho\n\n**Input → Semantic Mapping → Làm sạch địa chỉ → ArcGIS Geocoding → Lat/Lon → Export**\n\n> 🔒 Người dùng chỉ nhập **Mã kho + Địa chỉ kho**. Lat/Lon do hệ thống tự động lấy từ ArcGIS.")

    do_geocode = gr.Checkbox(value=True, label="🌍 Bật ArcGIS Geocoding")
    mode = gr.Radio(["Nhập tay", "Upload file"], value="Nhập tay", label="Cách nhập dữ liệu kho")

    with gr.Column() as manual_box:
        gr.Markdown("### ✍️ Nhập trực tiếp")
        manual = gr.Dataframe(value=empty_table(), headers=["Mã kho", "Địa chỉ kho"], datatype=["str", "str"], interactive=True, row_count=(3, "dynamic"), column_count=(2, "fixed"), wrap=True)

    with gr.Column(visible=False) as file_box:
        gr.Markdown("### 📂 Upload\nHỗ trợ **Excel / CSV / JSON**.")
        file_obj = gr.File(label="File dữ liệu kho", file_types=[".xlsx", ".xls", ".xlsm", ".csv", ".json"])

    mode.change(lambda m: (gr.update(visible=m == "Nhập tay"), gr.update(visible=m == "Upload file")), mode, [manual_box, file_box])

    scan_btn = gr.Button("🔍 Quét & Semantic Mapping", variant="primary")
    scan_msg = gr.Markdown()

    with gr.Column(visible=False) as mapping_box:
        gr.Markdown("### 🔗 Kiểm tra mapping")
        warehouse_id_col = gr.Dropdown(choices=[NONE], value=NONE, label="Mã kho ← cột nào?")
        address_col = gr.Dropdown(choices=[NONE], value=NONE, label="Địa chỉ kho ← cột nào?")
        process_btn = gr.Button("🚀 Làm sạch + Geocoding", variant="primary")

    raw_state = gr.State()
    gr.Markdown("### 📊 Kết quả")
    result = gr.Dataframe(headers=["id_warehouse", "address", "lat", "lng", "địa_chỉ_gốc", "chất_lượng_địa_chỉ", "trạng_thái_làm_sạch", "địa_chỉ_geocode", "geocode_score", "trạng_thái_geocode", "nguồn_tọa_độ", "kiểm_tra"], interactive=False, wrap=True)
    result_msg = gr.Markdown()

    export_btn = gr.Button("📦 Xuất Excel + JSON", variant="secondary")
    export_msg = gr.Markdown()
    export_files = gr.Files(label="File kết quả")

    def scan(m, f, man):
        raw = get_input(m, f, man)
        mapping = semantic_mapping(raw)
        choices = [NONE] + [str(c) for c in raw.columns]
        id_col, id_score = mapping["warehouse_id"]
        addr_col, addr_score = mapping["address"]
        return raw, f"### 🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**", gr.update(visible=True), gr.update(choices=choices, value=str(id_col) if id_col else NONE, info=f"Độ tin cậy: {id_score:.0%}"), gr.update(choices=choices, value=str(addr_col) if addr_col else NONE, info=f"Độ tin cậy: {addr_score:.0%}")

    scan_btn.click(scan, [mode, file_obj, manual], [raw_state, scan_msg, mapping_box, warehouse_id_col, address_col])

    def process(raw, do_geo, id_col, addr_col):
        try:
            if raw is None: raise gr.Error("Hãy quét dữ liệu trước.")
            if id_col in {None, NONE}: raise gr.Error("Chưa chọn cột Mã kho.")
            if addr_col in {None, NONE}: raise gr.Error("Chưa chọn cột Địa chỉ kho.")
            raw_df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
            rows = [process_warehouse(row.get(id_col), row.get(addr_col), do_geocode=do_geo) for _, row in raw_df.iterrows()]
            out = validate_output(pd.DataFrame(rows))
            success = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            geocoded = int(out["lat"].notna().sum())
            msg = f"### 🧭 Hoàn tất quét kho\n- Tổng số kho: **{len(out)}**\n- Geocode thành công: **{geocoded}/{len(out)}**\n- Dòng đủ Lat/Lon + hợp lệ: **{success}/{len(out)}**"
            return out, msg
        except Exception as e:
            return pd.DataFrame({"Lỗi hệ thống": [str(e)]}), f"❌ **Lỗi chi tiết:** `{str(e)}`"

    process_btn.click(process, [raw_state, do_geocode, warehouse_id_col, address_col], [result, result_msg])

    def export_result(df):
        if df is None or len(pd.DataFrame(df)) == 0: raise gr.Error("Chưa có kết quả để xuất.")
        checked = validate_output(pd.DataFrame(df))

        # 1. Tạo thư mục nguồn ./output_warehouse trong Colab Workspace
        out_dir = os.path.join(os.getcwd(), "output_warehouse")
        os.makedirs(out_dir, exist_ok=True)

        xlsx = os.path.join(out_dir, "WAREHOUSE_WITH_COORDINATES.xlsx")
        js = os.path.join(out_dir, "WAREHOUSE_WITH_COORDINATES.json")

        # Ghi file trực tiếp vào thư mục nguồn
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            checked.to_excel(writer, sheet_name="DIM_WAREHOUSE", index=False)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(checked.to_dict("records"), f, ensure_ascii=False, indent=2, default=str)

        success = int((checked["lat"].notna() & checked["lng"].notna()).sum())
        return f"### 💾 Xuất thành công!\n📁 File đã được lưu thẳng vào thư mục nguồn `./output_warehouse` trong Colab của cậu ✨\n- Có Lat/Lon: **{success}/{len(checked)}** kho", [xlsx, js]

    export_btn.click(export_result, [result], [export_msg, export_files])

if __name__ == "__main__":
    app.launch(share=True)

# =============================================================
# Block nhập liệu khai thông tin danh sách sản phẩm của công ty
# =============================================================
# -*- coding: utf-8 -*-
import importlib, json, os, re, subprocess, sys, tempfile
REQUIRED = {"gradio": "gradio>=5.0", "pandas": "pandas", "numpy": "numpy", "openpyxl": "openpyxl", "rapidfuzz": "rapidfuzz", "unidecode": "unidecode"}
missing = [pkg for mod, pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
import numpy as np, pandas as pd, gradio as gr
from rapidfuzz import fuzz
from unidecode import unidecode

NONE = "-- Không sử dụng --"

def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except: pass
    return str(v).strip() == ""

def norm(v):
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()

def read_any(f):
    path = f if isinstance(f, str) else f.name
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")): df = pd.read_excel(path)
    elif low.endswith(".json"): df = pd.read_json(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding="cp1258")
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

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

def profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field in ["volume", "weight", "length", "width", "height", "cost_price", "selling_price"]:
        return float(values.str.replace(r"[^\d.]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))

def semantic_mapping(df):
    result, used = {}, set()
    for field, (_, aliases) in PRODUCT_FIELDS.items():
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used: continue
            col_norm = norm(col)
            fuzzy_score = max(fuzz.token_set_ratio(col_norm, norm(a)) for a in aliases) / 100.0
            keyword_score = float(any(norm(a) in col_norm for a in aliases if len(norm(a)) >= 3))
            score = 0.75 * (0.7 * fuzzy_score + 0.3 * keyword_score) + 0.25 * profile_score(df[col], field)
            if score > best_score: best_col, best_score = col, score
        if best_col is not None and best_score >= 0.35:
            result[field] = (best_col, best_score)
            used.add(best_col)
        else: result[field] = (None, 0.0)
    return result

def parse_num(val):
    if is_blank(val): return 0.0
    try:
        cleaned = re.sub(r"[^\d.-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except: return 0.0

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

def validate_output(df):
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

def empty_table():
    return pd.DataFrame({
        "Mã sản phẩm": ["SP_01"], "Tên sản phẩm": ["Ghế Sofa Gỗ Sồi"],
        "Thể tích (m3)": [0.5], "Trọng lượng (kg)": [25.0],
        "Dài (cm)": [120], "Rộng (cm)": [60], "Cao (cm)": [80],
        "Giá sản xuất": [1200000], "Giá bán": [2500000]
    })

def get_input(mode, file_obj, manual):
    df = read_any(file_obj) if mode == "Upload file" else pd.DataFrame(manual)
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise gr.Error("Chưa có dữ liệu sản phẩm để quét.")
    return df

with gr.Blocks(title="Smart Logistics - Product Management", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📦 Smart Logistics — Quản lý & Chuẩn hóa Sản phẩm\n\n**Input → Semantic Mapping → Làm sạch thông số kích thước/giá → Validate → Export Excel/JSON**\n\n> 🔒 Cột nào không có trong file, bạn có thể để trống hoặc chọn `-- Không sử dụng --` mà không lo gián đoạn.")

    mode = gr.Radio(["Nhập tay", "Upload file"], value="Nhập tay", label="Cách nhập dữ liệu sản phẩm")

    with gr.Column() as manual_box:
        gr.Markdown("### ✍️ Nhập trực tiếp danh mục sản phẩm")
        manual = gr.Dataframe(value=empty_table(), headers=["Mã sản phẩm", "Tên sản phẩm", "Thể tích (m3)", "Trọng lượng (kg)", "Dài (cm)", "Rộng (cm)", "Cao (cm)", "Giá sản xuất", "Giá bán"], datatype=["str", "str", "number", "number", "number", "number", "number", "number", "number"], interactive=True, row_count=(1, "dynamic"), wrap=True)

    with gr.Column(visible=False) as file_box:
        gr.Markdown("### 📂 Upload\nHỗ trợ **Excel / CSV / JSON**.")
        file_obj = gr.File(label="File danh mục sản phẩm", file_types=[".xlsx", ".xls", ".xlsm", ".csv", ".json"])

    mode.change(lambda m: (gr.update(visible=m == "Nhập tay"), gr.update(visible=m == "Upload file")), mode, [manual_box, file_box])

    scan_btn = gr.Button("🔍 Quét & Semantic Mapping", variant="primary")
    scan_msg = gr.Markdown()

    with gr.Column(visible=False) as mapping_box:
        gr.Markdown("### 🔗 Kiểm tra ánh xạ cột sản phẩm (Cột nào không có chọn '-- Không sử dụng --')")
        p_id_col = gr.Dropdown(choices=[NONE], value=NONE, label="Mã sản phẩm ← cột nào?")
        p_name_col = gr.Dropdown(choices=[NONE], value=NONE, label="Tên sản phẩm ← cột nào?")
        vol_col = gr.Dropdown(choices=[NONE], value=NONE, label="Thể tích ← cột nào?")
        wgt_col = gr.Dropdown(choices=[NONE], value=NONE, label="Trọng lượng ← cột nào?")
        l_col = gr.Dropdown(choices=[NONE], value=NONE, label="Dài ← cột nào?")
        w_col = gr.Dropdown(choices=[NONE], value=NONE, label="Rộng ← cột nào?")
        h_col = gr.Dropdown(choices=[NONE], value=NONE, label="Cao ← cột nào?")
        cost_col = gr.Dropdown(choices=[NONE], value=NONE, label="Giá sản xuất ← cột nào?")
        price_col = gr.Dropdown(choices=[NONE], value=NONE, label="Giá bán ← cột nào?")
        process_btn = gr.Button("🚀 Chuẩn hóa & Xử lý Product", variant="primary")

    raw_state = gr.State()
    gr.Markdown("### 📊 Kết quả sản phẩm")
    result = gr.Dataframe(headers=["product_id", "product_name", "volume", "weight", "length", "width", "height", "cost_price", "selling_price", "trạng_thái", "kiểm_tra"], interactive=False, wrap=True)
    result_msg = gr.Markdown()

    export_btn = gr.Button("📦 Xuất Excel + JSON", variant="secondary")
    export_msg = gr.Markdown()
    export_files = gr.Files(label="File kết quả sản phẩm")

    def scan(m, f, man):
        raw = get_input(m, f, man)
        mapping = semantic_mapping(raw)
        choices = [NONE] + [str(c) for c in raw.columns]
        def get_val(key):
            col, score = mapping[key]
            return str(col) if col else NONE, f"Độ tin cậy: {score:.0%}"
        id_v, id_i = get_val("product_id")
        nm_v, nm_i = get_val("product_name")
        vl_v, vl_i = get_val("volume")
        wg_v, wg_i = get_val("weight")
        l_v, l_i = get_val("length")
        w_v, w_i = get_val("width")
        h_v, h_i = get_val("height")
        c_v, c_i = get_val("cost_price")
        p_v, p_i = get_val("selling_price")
        return raw, f"### 🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**", gr.update(visible=True), gr.update(choices=choices, value=id_v, info=id_i), gr.update(choices=choices, value=nm_v, info=nm_i), gr.update(choices=choices, value=vl_v, info=vl_i), gr.update(choices=choices, value=wg_v, info=wg_i), gr.update(choices=choices, value=l_v, info=l_i), gr.update(choices=choices, value=w_v, info=w_i), gr.update(choices=choices, value=h_v, info=h_i), gr.update(choices=choices, value=c_v, info=c_i), gr.update(choices=choices, value=p_v, info=p_i)

    scan_btn.click(scan, [mode, file_obj, manual], [raw_state, scan_msg, mapping_box, p_id_col, p_name_col, vol_col, wgt_col, l_col, w_col, h_col, cost_col, price_col])

    def process(raw, i_col, n_col, vl_col, wg_col, l_c, w_c, h_c, c_col, p_col):
        try:
            if raw is None: raise gr.Error("Hãy quét dữ liệu trước.")
            raw_df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
            rows = []
            for _, row in raw_df.iterrows():
                rows.append(process_product(
                    row.get(i_col) if i_col and i_col != NONE else "",
                    row.get(n_col) if n_col and n_col != NONE else "",
                    row.get(vl_col) if vl_col and vl_col != NONE else 0,
                    row.get(wg_col) if wg_col and wg_col != NONE else 0,
                    row.get(l_c) if l_c and l_c != NONE else 0,
                    row.get(w_c) if w_c and w_c != NONE else 0,
                    row.get(h_c) if h_c and h_c != NONE else 0,
                    row.get(c_col) if c_col and c_col != NONE else 0,
                    row.get(p_col) if p_col and p_col != NONE else 0
                ))
            out = validate_output(pd.DataFrame(rows))
            success = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            msg = f"### 🧭 Hoàn tất chuẩn hóa sản phẩm\n- Tổng số sản phẩm: **{len(out)}**\n- Dòng hợp lệ: **{success}/{len(out)}**"
            return out, msg
        except Exception as e:
            return pd.DataFrame({"Lỗi hệ thống": [str(e)]}), f"❌ **Lỗi chi tiết:** `{str(e)}`"

    process_btn.click(process, [raw_state, p_id_col, p_name_col, vol_col, wgt_col, l_col, w_col, h_col, cost_col, price_col], [result, result_msg])

    def export_result(df):
        if df is None or len(pd.DataFrame(df)) == 0: raise gr.Error("Chưa có kết quả để xuất.")
        checked = validate_output(pd.DataFrame(df))

        # Lưu trực tiếp vào thư mục nguồn ./output_product trong Colab Workspace
        out_dir = os.path.join(os.getcwd(), "output_product")
        os.makedirs(out_dir, exist_ok=True)
        xlsx = os.path.join(out_dir, "DIM_PRODUCT.xlsx")
        js = os.path.join(out_dir, "DIM_PRODUCT.json")

        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            checked.to_excel(writer, sheet_name="DIM_PRODUCT", index=False)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(checked.to_dict("records"), f, ensure_ascii=False, indent=2, default=str)

        return f"### 💾 Xuất thành công danh mục sản phẩm!\n📁 File đã được lưu thẳng vào thư mục nguồn `./output_product` trong Colab của cậu ✨", [xlsx, js]

    export_btn.click(export_result, [result], [export_msg, export_files])

if __name__ == "__main__":
    app.launch(share=True)

# =============================================================
# Block kê khai thông tin của đội nhân viên tài xế
# =============================================================
# -*- coding: utf-8 -*-
import importlib, json, os, re, subprocess, sys, tempfile
REQUIRED = {"gradio": "gradio>=5.0", "pandas": "pandas", "numpy": "numpy", "openpyxl": "openpyxl", "rapidfuzz": "rapidfuzz", "unidecode": "unidecode"}
missing = [pkg for mod, pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
import numpy as np, pandas as pd, gradio as gr
from rapidfuzz import fuzz
from unidecode import unidecode

NONE = "-- Không sử dụng --"

def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except: pass
    return str(v).strip() == ""

def norm(v):
    return re.sub(r"[^a-z0-9 ]+", " ", unidecode(str(v)).lower()).strip()

def read_any(f):
    path = f if isinstance(f, str) else f.name
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")): df = pd.read_excel(path)
    elif low.endswith(".json"): df = pd.read_json(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding="cp1258")
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

DRIVER_FIELDS = {
    "driver_id": ("Mã tài xế", ["mã tài xế", "driver id", "driver code", "mã nv", "staff id", "id", "code"]),
    "driver_name": ("Họ và tên", ["họ và tên", "ho va ten", "tên tài xế", "ten tai xe", "full name", "name", "họ tên", "tên nhân viên"]),
    "license_type": ("Loại bằng", ["loại bằng", "loai bang", "bằng lái", "bang lai", "license", "class", "hạng bằng"]),
    "warehouse": ("Kho hoạt động", ["kho", "warehouse", "trạm", "hub", "địa điểm kho", "chi nhánh"]),
    "address": ("Địa chỉ", ["địa chỉ", "dia chi", "address", "nơi ở"]),
    "phone": ("Số điện thoại", ["số điện thoại", "so dien thoai", "phone", "sdt", "mobile", "hotline"]),
    "role": ("Vị trí làm việc", ["vị trí", "vi tri", "role", "chức vụ", "job", "loại nhân sự", "vị trí làm việc"])
}

def profile_score(series, field):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty: return 0.0
    if field == "phone":
        return float(values.str.replace(r"[^\d+]", "", regex=True).notna().mean())
    return float(values.nunique() / len(values))

def semantic_mapping(df):
    result, used = {}, set()
    for field, (_, aliases) in DRIVER_FIELDS.items():
        best_col, best_score = None, 0.0
        for col in df.columns:
            if col in used: continue
            col_norm = norm(col)
            fuzzy_score = max(fuzz.token_set_ratio(col_norm, norm(a)) for a in aliases) / 100.0
            keyword_score = float(any(norm(a) in col_norm for a in aliases if len(norm(a)) >= 3))
            score = 0.75 * (0.7 * fuzzy_score + 0.3 * keyword_score) + 0.25 * profile_score(df[col], field)
            if score > best_score: best_col, best_score = col, score
        if best_col is not None and best_score >= 0.35:
            result[field] = (best_col, best_score)
            used.add(best_col)
        else: result[field] = (None, 0.0)
    return result

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

def validate_output(df):
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

def empty_table():
    return pd.DataFrame({
        "Mã tài xế": ["DRV_01", "DRV_02"],
        "Họ và tên": ["Nguyễn Văn A", "Trần Văn B"],
        "Loại bằng": ["FC", "C"],
        "Kho hoạt động": ["Kho Miền Bắc", "Kho Miền Nam"],
        "Địa chỉ": ["Hà Nội", "TP. Hồ Chí Minh"],
        "Số điện thoại": ["0901234567", "0987654321"],
        "Vị trí làm việc": ["Tài xế chính", "Hỗ trợ vận chuyển đồ"]
    })

def get_input(mode, file_obj, manual):
    df = read_any(file_obj) if mode == "Upload file" else pd.DataFrame(manual)
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise gr.Error("Chưa có dữ liệu tài xế để quét.")
    return df

with gr.Blocks(title="Smart Logistics - Driver Management", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 👨‍✈️ Smart Logistics — Quản lý & Chuẩn hóa Tài xế\n\n**Input → Semantic Mapping → Làm sạch thông tin nhân sự → Validate → Export Excel/JSON**\n\n> 🔒 Cột nào không có trong file, bạn có thể để trống hoặc chọn `-- Không sử dụng --` mà không lo gián đoạn.")

    mode = gr.Radio(["Nhập tay", "Upload file"], value="Nhập tay", label="Cách nhập dữ liệu tài xế")

    with gr.Column() as manual_box:
        gr.Markdown("### ✍️ Nhập trực tiếp danh mục tài xế")
        manual = gr.Dataframe(value=empty_table(), headers=["Mã tài xế", "Họ và tên", "Loại bằng", "Kho hoạt động", "Địa chỉ", "Số điện thoại", "Vị trí làm việc"], datatype=["str", "str", "str", "str", "str", "str", "str"], interactive=True, row_count=(2, "dynamic"), wrap=True)

    with gr.Column(visible=False) as file_box:
        gr.Markdown("### 📂 Upload\nHỗ trợ **Excel / CSV / JSON**.")
        file_obj = gr.File(label="File danh mục tài xế", file_types=[".xlsx", ".xls", ".xlsm", ".csv", ".json"])

    mode.change(lambda m: (gr.update(visible=m == "Nhập tay"), gr.update(visible=m == "Upload file")), mode, [manual_box, file_box])

    scan_btn = gr.Button("🔍 Quét & Semantic Mapping", variant="primary")
    scan_msg = gr.Markdown()

    with gr.Column(visible=False) as mapping_box:
        gr.Markdown("### 🔗 Kiểm tra ánh xạ cột tài xế (Cột nào không có chọn '-- Không sử dụng --')")
        d_id_col = gr.Dropdown(choices=[NONE], value=NONE, label="Mã tài xế ← cột nào?")
        d_name_col = gr.Dropdown(choices=[NONE], value=NONE, label="Họ và tên ← cột nào?")
        lic_col = gr.Dropdown(choices=[NONE], value=NONE, label="Loại bằng ← cột nào?")
        wh_col = gr.Dropdown(choices=[NONE], value=NONE, label="Kho hoạt động ← cột nào?")
        addr_col = gr.Dropdown(choices=[NONE], value=NONE, label="Địa chỉ ← cột nào?")
        phone_col = gr.Dropdown(choices=[NONE], value=NONE, label="Số điện thoại ← cột nào?")
        role_col = gr.Dropdown(choices=[NONE], value=NONE, label="Vị trí làm việc ← cột nào?")
        process_btn = gr.Button("🚀 Chuẩn hóa & Xử lý Driver", variant="primary")

    raw_state = gr.State()
    gr.Markdown("### 📊 Kết quả tài xế")
    result = gr.Dataframe(headers=["driver_id", "driver_name", "license_type", "id_warehouse", "address", "phone", "role", "trạng_thái", "kiểm_tra"], interactive=False, wrap=True)
    result_msg = gr.Markdown()

    export_btn = gr.Button("📦 Xuất Excel + JSON", variant="secondary")
    export_msg = gr.Markdown()
    export_files = gr.Files(label="File kết quả tài xế")

    def scan(m, f, man):
        raw = get_input(m, f, man)
        mapping = semantic_mapping(raw)
        choices = [NONE] + [str(c) for c in raw.columns]
        def get_val(key):
            col, score = mapping[key]
            return str(col) if col else NONE, f"Độ tin cậy: {score:.0%}"
        id_v, id_i = get_val("driver_id")
        nm_v, nm_i = get_val("driver_name")
        lc_v, lc_i = get_val("license_type")
        wh_v, wh_i = get_val("warehouse")
        ad_v, ad_i = get_val("address")
        ph_v, ph_i = get_val("phone")
        rl_v, rl_i = get_val("role")
        return raw, f"### 🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**", gr.update(visible=True), gr.update(choices=choices, value=id_v, info=id_i), gr.update(choices=choices, value=nm_v, info=nm_i), gr.update(choices=choices, value=lc_v, info=lc_i), gr.update(choices=choices, value=wh_v, info=wh_i), gr.update(choices=choices, value=ad_v, info=ad_i), gr.update(choices=choices, value=ph_v, info=ph_i), gr.update(choices=choices, value=rl_v, info=rl_i)

    scan_btn.click(scan, [mode, file_obj, manual], [raw_state, scan_msg, mapping_box, d_id_col, d_name_col, lic_col, wh_col, addr_col, phone_col, role_col])

    def process(raw, i_col, n_col, l_col, w_col, a_col, p_col, r_col):
        try:
            if raw is None: raise gr.Error("Hãy quét dữ liệu trước.")
            raw_df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
            rows = []
            for _, row in raw_df.iterrows():
                rows.append(process_driver(
                    row.get(i_col) if i_col and i_col != NONE else "",
                    row.get(n_col) if n_col and n_col != NONE else "",
                    row.get(l_col) if l_col and l_col != NONE else "",
                    row.get(w_col) if w_col and w_col != NONE else "",
                    row.get(a_col) if a_col and a_col != NONE else "",
                    row.get(p_col) if p_col and p_col != NONE else "",
                    row.get(r_col) if r_col and r_col != NONE else ""
                ))
            out = validate_output(pd.DataFrame(rows))
            success = int(out["kiểm_tra"].astype(str).str.startswith("✅").sum())
            msg = f"### 🧭 Hoàn tất chuẩn hóa nhân sự tài xế\n- Tổng số nhân sự: **{len(out)}**\n- Dòng hợp lệ: **{success}/{len(out)}**"
            return out, msg
        except Exception as e:
            return pd.DataFrame({"Lỗi hệ thống": [str(e)]}), f"❌ **Lỗi chi tiết:** `{str(e)}`"

    process_btn.click(process, [raw_state, d_id_col, d_name_col, lic_col, wh_col, addr_col, phone_col, role_col], [result, result_msg])

    def export_result(df):
        if df is None or len(pd.DataFrame(df)) == 0: raise gr.Error("Chưa có kết quả để xuất.")
        checked = validate_output(pd.DataFrame(df))

        # Lưu trực tiếp vào thư mục nguồn ./output_driver trong Colab Workspace
        out_dir = os.path.join(os.getcwd(), "output_driver")
        os.makedirs(out_dir, exist_ok=True)
        xlsx = os.path.join(out_dir, "DIM_DRIVER.xlsx")
        js = os.path.join(out_dir, "DIM_DRIVER.json")

        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            checked.to_excel(writer, sheet_name="DIM_DRIVER", index=False)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(checked.to_dict("records"), f, ensure_ascii=False, indent=2, default=str)

        return f"### 💾 Xuất thành công danh mục tài xế!\n📁 File đã được lưu thẳng vào thư mục nguồn `./output_driver` trong Colab của cậu ✨", [xlsx, js]

    export_btn.click(export_result, [result], [export_msg, export_files])

if __name__ == "__main__":
    app.launch(share=True)

# =====================================================
# Block thông tin các đơn hàng trong ngày
# =====================================================

import importlib.util, json, os, re, subprocess, sys, warnings
from collections import Counter

REQUIRED = {"gradio": "gradio>=5.0", "pandas": "pandas", "numpy": "numpy", "openpyxl": "openpyxl",
            "rapidfuzz": "rapidfuzz", "unidecode": "unidecode"}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])

import numpy as np, pandas as pd, gradio as gr
from rapidfuzz import fuzz, process as rf_process
from unidecode import unidecode

NONE = "-- Không sử dụng --"
MAP_THRESHOLD = 0.38

# ==========================================================
# 1. TIỆN ÍCH CƠ BẢN
# ==========================================================
def is_blank(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except Exception: pass
    return str(v).strip() == ""

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

def read_any(f):
    path = f if isinstance(f, str) else f.name
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")): df = pd.read_excel(path)
    elif low.endswith(".json"): df = pd.read_json(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding="cp1258")
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

# ==========================================================
# 2. TỪ ĐIỂN TRƯỜNG (alias tên cột) — thêm bớt thoải mái
# ==========================================================
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

def semantic_mapping(df):
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

def validate_output(df):
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

# ==========================================================
# 7. DỮ LIỆU MẪU + NHẬP LIỆU
# ==========================================================
def empty_table():
    return pd.DataFrame({
        "Mã đơn": ["ORD_001"], "Mã khách": ["CUS_01"], "Tên khách": ["Nguyễn Văn A"], "Số lượng": [2],
        "Mặt hàng": ["Ghế sofa, Bàn trà"], "Tổng trọng lượng (kg)": [45.5], "Tổng thể tích (m3)": [0.8],
        "Địa chỉ khách": ["Số 88 - Đường Cổ Linh - Long Biên - Hà Nội"], "Tình trạng đơn": ["Đang xử lý"],
        "Loại đơn": ["Individual"], "Tình trạng Alert": ["Normal"]})

def get_input(mode, file_obj, manual):
    df = read_any(file_obj) if mode == "Upload file" else pd.DataFrame(manual)
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty: raise gr.Error("Chưa có dữ liệu đơn hàng để quét.")
    return df

def conf_icon(s):
    return "🟢" if s >= 0.75 else ("🟡" if s >= 0.55 else "🟠")

# ==========================================================
# 8. GIAO DIỆN GRADIO
# ==========================================================
with gr.Blocks(title="Smart Logistics - Order Management", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📦 Smart Logistics — Quản lý & Chuẩn hóa Đơn hàng\n\n"
                "**Input → Semantic Mapping (cột + giá trị) → Làm sạch → Validate → Export Excel/JSON**\n\n"
                "> 🔒 Tự nhận diện cột theo *tên + nội dung*, và hiểu giá trị như `Individual`→B2C, `Business`→B2B, "
                "`Urgent`→Alert, `Shipping`→Đang giao, `2 tấn`→2000 kg...")

    mode = gr.Radio(["Nhập tay", "Upload file"], value="Nhập tay", label="Cách nhập dữ liệu đơn hàng")

    with gr.Column() as manual_box:
        gr.Markdown("### ✍️ Nhập trực tiếp danh sách đơn hàng")
        manual = gr.Dataframe(value=empty_table(), interactive=True, row_count=(1, "dynamic"), wrap=True)

    with gr.Column(visible=False) as file_box:
        gr.Markdown("### 📂 Upload\nHỗ trợ **Excel / CSV / JSON**.")
        file_obj = gr.File(label="File danh sách đơn hàng", file_types=[".xlsx", ".xls", ".xlsm", ".csv", ".json"])

    mode.change(lambda m: (gr.update(visible=m == "Nhập tay"), gr.update(visible=m == "Upload file")),
                mode, [manual_box, file_box])

    scan_btn = gr.Button("🔍 Quét & Semantic Mapping Đơn hàng", variant="primary")
    scan_msg = gr.Markdown()

    with gr.Column(visible=False) as mapping_box:
        gr.Markdown("### 🔗 Kiểm tra ánh xạ cột (cột nào không có chọn '-- Không sử dụng --')")
        raw_preview = gr.Dataframe(label="Xem trước dữ liệu gốc (8 dòng đầu)", interactive=False, wrap=True)
        dd = {}
        for i in range(0, len(FIELDS), 2):
            with gr.Row():
                for f in FIELDS[i:i + 2]:
                    dd[f] = gr.Dropdown(choices=[NONE], value=NONE, label=f"{ORDER_FIELDS[f]['label']} ← cột nào?")
        process_btn = gr.Button("🚀 Chuẩn hóa & Xử lý Đơn hàng", variant="primary")

    raw_state = gr.State()

    gr.Markdown("### 📊 Kết quả đơn hàng đã chuẩn hóa")
    result = gr.Dataframe(interactive=False, wrap=True)
    result_msg = gr.Markdown()
    gr.Markdown("### 🔁 Bảng ánh xạ giá trị (giá trị gốc → giá trị chuẩn)")
    value_report = gr.Dataframe(interactive=False, wrap=True)

    export_btn = gr.Button("📦 Xuất Excel + JSON", variant="secondary")
    export_msg = gr.Markdown()
    export_files = gr.Files(label="File kết quả đơn hàng")

    # ---------- Quét ----------
    def scan(m, f, man):
        raw = get_input(m, f, man)
        mapping = semantic_mapping(raw)
        choices = [NONE] + [str(c) for c in raw.columns]

        lines = ["| Trường | Cột được chọn | Độ tin cậy | Căn cứ |", "|---|---|---|---|"]
        updates = []
        for fld in FIELDS:
            col, score, why = mapping[fld]
            label = ORDER_FIELDS[fld]["label"]
            if col is None:
                lines.append(f"| {label} | _(không tìm thấy)_ | – | – |")
                updates.append(gr.update(choices=choices, value=NONE, info="Không tìm thấy cột phù hợp"))
            else:
                lines.append(f"| {label} | `{col}` | {conf_icon(score)} {score:.0%} | {why} |")
                updates.append(gr.update(choices=choices, value=str(col), info=f"{conf_icon(score)} Độ tin cậy {score:.0%} ({why})"))
        msg = (f"### 🔍 Đã quét **{len(raw)} dòng × {len(raw.columns)} cột**\n\n" + "\n".join(lines) +
               "\n\n> 🟢 chắc chắn · 🟡 nên kiểm tra · 🟠 độ tin cậy thấp — bạn có thể đổi trong các ô bên dưới.")
        return (raw, msg, gr.update(visible=True), raw.head(8), *updates)

    scan_btn.click(scan, [mode, file_obj, manual], [raw_state, scan_msg, mapping_box, raw_preview, *[dd[f] for f in FIELDS]])

    # ---------- Chuẩn hóa ----------
    def process(raw, *cols):
        try:
            if raw is None: raise gr.Error("Hãy quét dữ liệu trước.")
            raw_df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            lookup = {str(c): c for c in raw_df.columns}
            colmap = {f: lookup.get(c) for f, c in zip(FIELDS, cols) if c and c != NONE and c in lookup}
            reports, rows = Counter(), []
            for _, row in raw_df.iterrows():
                rows.append(process_order({f: row.get(c) for f, c in colmap.items()}, reports))
            out = validate_output(pd.DataFrame(rows))
            n_ok = int((~out["kiểm_tra"].astype(str).str.startswith("❌")).sum())
            n_warn = int(out["kiểm_tra"].astype(str).str.startswith("⚠️").sum())
            n_b2b = int((out["order_type"] == "B2B").sum())
            n_alert = int((out["alert_status"] == "Alert").sum())
            msg = (f"### 🧭 Hoàn tất chuẩn hóa đơn hàng\n- Tổng số đơn: **{len(out)}**\n"
                   f"- Không có lỗi: **{n_ok}/{len(out)}** (trong đó {n_warn} dòng có cảnh báo ⚠️)\n"
                   f"- B2B: **{n_b2b}** · B2C: **{len(out) - n_b2b}** · Alert: **{n_alert}**")
            rep = pd.DataFrame([{"Trường": ORDER_FIELDS[f]["label"], "Giá trị gốc": o, "Chuẩn hóa thành": s,
                                 "Cách nhận diện": h, "Số dòng": n} for (f, o, s, h), n in reports.most_common()],
                               columns=["Trường", "Giá trị gốc", "Chuẩn hóa thành", "Cách nhận diện", "Số dòng"])
            return out, msg, rep
        except gr.Error:
            raise
        except Exception as e:
            return pd.DataFrame({"Lỗi hệ thống": [str(e)]}), f"❌ **Lỗi chi tiết:** `{e}`", pd.DataFrame()

    process_btn.click(process, [raw_state, *[dd[f] for f in FIELDS]], [result, result_msg, value_report])

    # ---------- Xuất ----------
    def export_result(df, rep):
        if df is None or len(pd.DataFrame(df)) == 0: raise gr.Error("Chưa có kết quả để xuất.")
        checked = validate_output(pd.DataFrame(df).drop(columns=["kiểm_tra"], errors="ignore"))
        out_dir = os.path.join(os.getcwd(), "output_orders")
        os.makedirs(out_dir, exist_ok=True)
        xlsx, js = os.path.join(out_dir, "DIM_ORDERS.xlsx"), os.path.join(out_dir, "DIM_ORDERS.json")
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            checked.to_excel(writer, sheet_name="DIM_ORDERS", index=False)       # sheet đầu: pipeline đọc sheet này
            rep_df = pd.DataFrame(rep) if rep is not None else pd.DataFrame()
            if len(rep_df): rep_df.to_excel(writer, sheet_name="VALUE_MAPPING", index=False)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(checked.to_dict("records"), f, ensure_ascii=False, indent=2, default=str)
        return ("### 💾 Xuất thành công danh mục đơn hàng!\n"
                "📁 Đã lưu vào `./output_orders` (sheet `DIM_ORDERS` + `VALUE_MAPPING`) ✨"), [xlsx, js]

    export_btn.click(export_result, [result, value_report], [export_msg, export_files])

if __name__ == "__main__":
    app.launch(share=True)

# ====================================================
# Block list đơn hàng theo ngày
# ====================================================
import os
import json
import pandas as pd
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter

# =========================================================
# 1. ĐỌC DỮ LIỆU ĐƠN HÀNG TỪ THƯ MỤC output_orders
# =========================================================
orders_dir = "output_orders"
order_file_path = os.path.join(orders_dir, "DIM_ORDERS.xlsx")

print(f"📂 Đang đọc dữ liệu đơn hàng từ: {order_file_path}")
if not os.path.exists(order_file_path):
    raise FileNotFoundError(f"Không tìm thấy file đơn hàng tại {order_file_path}. Cậu hãy chạy module tạo đơn trước nhé!")

df_orders = pd.read_excel(order_file_path)

# Trích xuất danh sách khách hàng độc lập từ đơn hàng (lấy các cột cần thiết cho geocoding khách hàng)
# Giả sử trong DIM_ORDERS có các cột: customer_id, customer_name, address
cols_to_extract = [c for c in ["customer_id", "customer_name", "address"] if c in df_orders.columns]
if not cols_to_extract:
    # Fallback nếu tên cột khác
    cols_to_extract = df_orders.columns[:3]

df_cust = df_orders[cols_to_extract].drop_duplicates(subset=["customer_id"] if "customer_id" in df_orders.columns else None).reset_index(drop=True)

# =========================================================
# 2. KHỞI TẠO BỘ GEOCODER ARCGIS & RATE LIMITER
# =========================================================
geolocator = ArcGIS(
    user_agent="smart_logistics_customer_geocoder/1.0",
    timeout=15
)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=0.5,
    swallow_exceptions=True
)

cache = {}

lat_list = []
lng_list = []
matched_address_list = []
status_list = []

print(f"🌍 Đang geocode {len(df_cust)} khách hàng lấy từ {order_file_path}...")

for _, row in df_cust.iterrows():
    addr = str(row.get("address", "")).strip()

    if not addr or addr.lower() == "nan":
        lat_list.append(None)
        lng_list.append(None)
        matched_address_list.append("")
        status_list.append("❌ Địa chỉ trống")
        continue

    if addr in cache:
        lat, lng, matched_address, status = cache[addr]
    else:
        query = addr
        if "việt nam" not in query.lower() and "vietnam" not in query.lower():
            query = f"{query}, Việt Nam"

        try:
            loc = geocode(query)
            if loc:
                lat = float(loc.latitude)
                lng = float(loc.longitude)
                matched_address = str(loc.address)

                # Kiểm tra phạm vi lãnh thổ Việt Nam
                if 8.0 <= lat <= 24.5 and 102.0 <= lng <= 110.0:
                    status = "✅ Geocode hợp lệ"
                else:
                    lat = None
                    lng = None
                    status = "❌ Tọa độ ngoài Việt Nam"
            else:
                lat = None
                lng = None
                matched_address = ""
                status = "⚠️ Không tìm thấy địa chỉ"
        except Exception as e:
            lat = None
            lng = None
            matched_address = ""
            status = f"❌ Lỗi: {str(e)}"

        cache[addr] = (lat, lng, matched_address, status)

    lat_list.append(lat)
    lng_list.append(lng)
    matched_address_list.append(matched_address)
    status_list.append(status)

# =========================================================
# 3. GHI KẾT QUẢ VÀO DATAFRAME KHÁCH HÀNG
# =========================================================
df_cust["lat"] = lat_list
df_cust["lng"] = lng_list
df_cust["matched_address"] = matched_address_list
df_cust["trạng_thái_geocode"] = status_list

# =========================================================
# 4. XUẤT FILE RA THƯ MỤC output_customer
# =========================================================
out_dir = "output_customer"
os.makedirs(out_dir, exist_ok=True)

xlsx_path = os.path.join(out_dir, "DATASET_CUSTOMER.xlsx")
json_path = os.path.join(out_dir, "DATASET_CUSTOMER.json")

# Xuất file Excel
df_cust.to_excel(
    xlsx_path,
    index=False,
    sheet_name="DATASET_CUSTOMER"
)

# Xuất file JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        df_cust.to_dict("records"),
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )

# =========================================================
# 5. THỐNG KÊ KẾT QUẢ GEOCODING
# =========================================================
success = (df_cust["trạng_thái_geocode"] == "✅ Geocode hợp lệ").sum()
failed = len(df_cust) - success

print("\n" + "=" * 50)
print("✨ GEOCODING HOÀN TẤT (LẤY TỪ output_orders)")
print("=" * 50)
print(f"👥 Tổng khách hàng : {len(df_cust)}")
print(f"✅ Hợp lệ          : {success}")
print(f"⚠️ Chưa xác định   : {failed}")
print("\n📁 File kết quả đã xuất:")
print(f" - {xlsx_path}")
print(f" - {json_path}")

# ============================================
# Block quét khách hàng và lập ma trận
# ============================================
import os
import json
import pandas as pd
import requests
import numpy as np
import math

# =========================================================
# 1. ĐỌC FILE DATASET KHÁCH HÀNG TỪ THƯ MỤC output_customer
# =========================================================
out_customer_dir = "output_customer"
input_path = os.path.join(out_customer_dir, "DATASET_CUSTOMER.xlsx")

# Fallback tự động dò tìm nếu file có tên khác 
if not os.path.exists(input_path):
    # Tìm file excel bất kỳ trong thư mục output_customer
    if os.path.exists(out_customer_dir):
        excel_files = [f for f in os.listdir(out_customer_dir) if f.endswith(".xlsx")]
        if excel_files:
            input_path = os.path.join(out_customer_dir, excel_files[0])
            print(f"ℹ️ Không tìm thấy DATASET_CUSTOMER.xlsx, hệ thống tự động nhận file: {input_path}")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu khách hàng trong thư mục '{out_customer_dir}'! Hãy chắc chắn bạn đã chạy module geocoding trước.")

print(f"📂 Đang đọc dữ liệu từ: {input_path}")
df_cust = pd.read_excel(input_path)

# Lọc các khách hàng có tọa độ (lat, lng) hợp lệ
df_valid = df_cust.dropna(subset=["lat", "lng"]).reset_index(drop=True)
print(f"📍 Số lượng khách hàng có tọa độ hợp lệ: {len(df_valid)}")

if len(df_valid) < 2:
    raise ValueError("Cần ít nhất 2 khách hàng có tọa độ Lat/Lon để tính ma trận OSRM!")

# =========================================================
# 2. GỌI OSRM TABLE API HOẶC FALLBACK HAVERSINE
# =========================================================
cust_ids = df_valid["customer_id"].tolist()
coords = [f"{row['lng']},{row['lat']}" for _, row in df_valid.iterrows()]
coords_str = ";".join(coords)

url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"

print(f"🌐 Đang gửi request lấy ma trận OSRM thực tế từ server...")
success_osrm = False

try:
    response = requests.get(url, timeout=20)
    if response.status_code == 200:
        data = response.json()
        if data.get("code") == "Ok":
            distances = data.get("distances") # mét
            durations = data.get("durations") # giây

            # Chuyển đổi sang DataFrame (Khoảng cách tính bằng km, thời gian tính bằng phút)
            dist_km_matrix = pd.DataFrame([[d / 1000.0 for d in row] for row in distances], index=cust_ids, columns=cust_ids)
            dur_min_matrix = pd.DataFrame([[t / 60.0 for t in row] for row in durations], index=cust_ids, columns=cust_ids)
            success_osrm = True
            print("✅ Kết nối OSRM API thành công!")
        else:
            print(f"⚠️ OSRM API trả về mã lỗi: {data.get('code')}")
    else:
        print(f"⚠️ HTTP Error Status: {response.status_code}")
except Exception as e:
    print(f"⚠️ Không gọi được OSRM API trực tuyến ({str(e)})")

# Nếu OSRM lỗi hoặc mất mạng -> Tự động chuyển sang phương án dự phòng Haversine
if not success_osrm:
    print("🔄 Hệ thống tự động chuyển sang tính toán theo khoảng cách đường bộ thực tế (Haversine nhân hệ số 1.2)...")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        return R * 2 * math.asin(math.sqrt(a))

    n = len(cust_ids)
    dist_matrix = np.zeros((n, n))
    dur_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(df_valid.loc[i, "lat"], df_valid.loc[i, "lng"], df_valid.loc[j, "lat"], df_valid.loc[j, "lng"]) * 1.2
                dist_matrix[i][j] = round(d, 2)
                dur_matrix[i][j] = round(d / 40.0 * 60.0, 2) # Giả định tốc độ trung bình 40 km/h

    dist_km_matrix = pd.DataFrame(dist_matrix, index=cust_ids, columns=cust_ids)
    dur_min_matrix = pd.DataFrame(dur_matrix, index=cust_ids, columns=cust_ids)

# =========================================================
# 3. LƯU KẾT QUẢ VÀO THƯ MỤC output_matrix
# =========================================================
out_dir = "output_matrix"
os.makedirs(out_dir, exist_ok=True)

dist_path = os.path.join(out_dir, "DISTANCE_MATRIX_KM.xlsx")
dur_path = os.path.join(out_dir, "DURATION_MATRIX_MIN.xlsx")
json_path = os.path.join(out_dir, "OSRM_MATRIX_RESULT.json")

dist_km_matrix.to_excel(dist_path)
dur_min_matrix.to_excel(dur_path)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "customers": cust_ids,
        "distance_matrix_km": dist_km_matrix.to_dict(),
        "duration_matrix_min": dur_min_matrix.to_dict()
    }, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("✨ XUẤT MA TRẬN THÀNH CÔNG VÀO THƯ MỤC 'output_matrix':")
print("=" * 50)
print(f" 📁 Ma trận khoảng cách (km) : {dist_path}")
print(f" 📁 Ma trận thời gian (phút) : {dur_path}")
print(f" 📁 Dữ liệu JSON tổng hợp    : {json_path}")

# ==============================================================
# Block tính Savings chuẩn bị cho Clarke Wright
# ==============================================================
import os
import math
import pandas as pd
import numpy as np

# =========================================================
# 1. ĐỌC DỮ LIỆU TỪ THƯ MỤC output_matrix & output_customer
# =========================================================
matrix_path = os.path.join("output_matrix", "DISTANCE_MATRIX_KM.xlsx")
if not os.path.exists(matrix_path):
    raise FileNotFoundError(f"Không tìm thấy file {matrix_path} trong thư mục output_matrix!")

df_dist = pd.read_excel(matrix_path, index_col=0)

cust_path = os.path.join("output_customer", "DATASET_CUSTOMER.xlsx")
if not os.path.exists(cust_path):
    cust_dir = "output_customer"
    excel_files = [f for f in os.listdir(cust_dir) if f.endswith(".xlsx")]
    if excel_files:
        cust_path = os.path.join(cust_dir, excel_files[0])
    else:
        raise FileNotFoundError(f"Không tìm thấy file khách hàng trong thư mục {cust_dir}!")

df_cust = pd.read_excel(cust_path)
print("📂 Đã tải thành công ma trận khoảng cách và dữ liệu khách hàng!")

# =========================================================
# 2. ĐỊNH NGHĨA KHO TỔNG (DEPOT) & TÍNH KHOẢNG CÁCH C0I
# =========================================================
depot_lat, depot_lng = 21.0285, 105.8542

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.asin(math.sqrt(a))

customers = df_dist.index.tolist()
c0i = {}

for _, row in df_cust.iterrows():
    c_id = row["customer_id"]
    if c_id in customers:
        lat, lng = row["lat"], row["lng"]
        if abs(lat - depot_lat) < 1e-4 and abs(lng - depot_lng) < 1e-4:
            d = 0.0
        else:
            d = haversine(depot_lat, depot_lng, lat, lng) * 1.2
        c0i[c_id] = round(d, 2)

# =========================================================
# 3. TÍNH TOÁN VÀ SẮP XẾP MỨC TIẾT KIỆM (SAVINGS) GIẢM DẦN
# =========================================================
savings = []
for i in range(len(customers)):
    for j in range(i + 1, len(customers)):
        cust_i = customers[i]
        cust_j = customers[j]

        c_ij = float(df_dist.loc[cust_i, cust_j])
        s_ij = c0i.get(cust_i, 0.0) + c0i.get(cust_j, 0.0) - c_ij

        savings.append({
            "pair": (cust_i, cust_j),
            "savings_km": round(s_ij, 2),
            "c_0i": c0i.get(cust_i, 0.0),
            "c_0j": c0i.get(cust_j, 0.0),
            "c_ij": c_ij
        })

# Sắp xếp giảm dần theo mức tiết kiệm (Savings)
savings_sorted = sorted(savings, key=lambda x: x["savings_km"], reverse=True)

df_savings = pd.DataFrame([{
    "Cặp khách hàng": f"{s['pair'][0]} - {s['pair'][1]}",
    "Khách hàng 1": s['pair'][0],
    "Khách hàng 2": s['pair'][1],
    "Mức tiết kiệm (km)": s['savings_km'],
    "Kho -> Khách 1 (c0i)": s['c_0i'],
    "Kho -> Khách 2 (c0j)": s['c_0j'],
    "Khách 1 <-> Khách 2 (cij)": s['c_ij']
} for s in savings_sorted])

# =========================================================
# 4. XUẤT FILE DATASET_SORT_SAVING.xlsx RA THƯ MỤC output_matrix
# =========================================================
out_dir = "output_matrix"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "DATASET_SORT_SAVING.xlsx")

df_savings.to_excel(out_path, index=False, sheet_name="SORT_SAVING")

print("\n" + "="*50)
print("✨ XUẤT FILE SORT SAVING THÀNH CÔNG:")
print("="*50)
print(f" 📁 Đường dẫn file: {out_path}")
print(df_savings.to_string(index=False))

# =============================================================
# Block chạy gán hàm tổng cho constraints về trọng tải giới hạn
# =============================================================
import os
import pandas as pd
import numpy as np

def run_master_logistics_optimizer(order_file="output_orders/DIM_ORDERS.xlsx",
                                     vehicle_file="output_fleet/DIM_VEHICLE.xlsx",
                                     matrix_file="output_matrix/DISTANCE_MATRIX_KM.xlsx"):
    """
    HÀM TỔNG MASTER: Quét đơn hàng -> Lọc đội xe -> Tách đơn quá cỡ -> Đọc ma trận -> Sẵn sàng chạy Clarke-Wright.
    """
    print("🔮 [MASTER PIPELINE] Đang khởi động hệ thống logistics...")

    # 1. Kiểm tra và đọc file dữ liệu
    df_orders = pd.read_excel(order_file)
    df_vehicles = pd.read_excel(vehicle_file)
    df_dist = pd.read_excel(matrix_file, index_col=0)

    # 2. Quét & Lọc ràng buộc tải trọng (Screening)
    max_w = df_vehicles["max_weight_kg"].max()
    max_v = df_vehicles["max_volume_m3"].max()

    valid_orders, oversized_orders = [], []
    for _, row in df_orders.iterrows():
        w = float(row.get("total_weight_kg", 0.0))
        v = float(row.get("total_volume_m3", 0.0))
        record = row.to_dict()

        if w > max_w or v > max_v:
            oversized_orders.append(record)
        else:
            record["eligible_vehicles"] = df_vehicles[
                (df_vehicles["max_weight_kg"] >= w) &
                (df_vehicles["max_volume_m3"] >= v)
            ]["vehicle_id"].tolist()
            valid_orders.append(record)

    print(f"✅ Quét xong: {len(valid_orders)} đơn hợp lệ | ❌ {len(oversized_orders)} đơn quá cỡ (tách riêng)")

    # 3. Trả về toàn bộ package dữ liệu sẵn sàng "bơm" thẳng vào cell chạy mô hình tuyến
    return {
        "valid_orders": valid_orders,
        "oversized_orders": oversized_orders,
        "distance_matrix": df_dist,
        "fleet_max_specs": {"max_weight": max_w, "max_volume": max_v}
    }

# ====================================================
# Block chạy gán hàm tổng cho constraints về thời gian
# ====================================================
import pandas as pd

def evaluate_route_time_constraint(route_data, service_time_rules=None):
    """
    Hàm xử lý constraint thời gian tuyến (Time Windows <= 8 giờ):
    - route_data: Dict chứa thông tin tuyến đường (danh sách khách hàng, danh sách đơn hàng, tổng quãng đường, loại đơn B2C/B2B, xe vận chuyển).
    - service_time_rules: Quy định thời gian bốc/dỡ hàng (mặc định B2C: 25' load + 35' unload; B2B: 45' load + 60' unload).
    """
    if service_time_rules is None:
        service_time_rules = {
            "B2C": {"loading": 25, "unloading": 35}, # Tổng 60 phút = 1 giờ
            "B2B": {"loading": 45, "unloading": 60}  # Tổng 105 phút = 1.75 giờ
        }

    # 1. Lấy thông tin từ tuyến
    orders_in_route = route_data.get("orders", [])
    total_distance_km = route_data.get("total_distance_km", 0.0)
    vehicle_speed_kmh = route_data.get("vehicle_speed_kmh", 40.0) # Vận tốc của xe được gán từ DIM_VEHICLE
    current_date = route_data.get("current_date", "2026-04-03") # Ngày hiện tại của đơn

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
        # CASE 1: Đạt yêu cầu (<= 8h)
        result_status = {
            "status": "APPROVED",
            "message": "✅ Đạt yêu cầu thời gian tuyến (<= 8h)",
            "total_hours": round(total_route_duration_hours, 2),
            "route": route_data.get("route", [])
        }
    else:
        # Vượt quá 8h -> Phân tách 2 kịch bản phụ theo yêu cầu
        # Kịch bản phụ A: Đẩy đơn/khách vi phạm quay trở lại pool hàng để thuật toán Clarke-Wright tiếp tục gom nhóm lại vào tuyến khác.
        # Kịch bản phụ B: Backlog sang ngày hôm sau (tính số ngày backlog, lưu vết nguồn gốc ngày ban đầu) và chạy ngầm sang pool ngày hôm sau.

        # Ở đây ta đánh dấu cờ backlog và ghi nhận nguồn gốc ngày
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
            "repool_orders": orders_in_route, # Đẩy lại vào pool cho Clarke-Wright
            "backlog_orders_next_day": backlog_orders # Backlog chạy ngầm sang ngày mai kèm đếm số ngày backlog
        }

    return result_status

# ====================================================
# Block chạy clarke wright toàn cục
# =====================================================
"""
LOGISTICS ROUTING DASHBOARD - GENERALIZED & STREAMLINED
========================================================================================
"""
from __future__ import annotations
import math
import datetime as dt
from dataclasses import dataclass, field
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

@dataclass
class Config:
    order_file: str = "output_orders/DIM_ORDERS.xlsx"
    vehicle_file: str = "output_fleet/DIM_VEHICLE.xlsx"
    driver_file: str = "output_driver/DIM_DRIVER.xlsx"
    matrix_file: str = "output_matrix/DISTANCE_MATRIX_KM.xlsx"
    cust_file: str = "output_customer/DATASET_CUSTOMER.xlsx"
    output_daily_file: str = "output_customer/DIM_CUSTOMER.xlsx"
    start_time: str = "08:30"
    max_route_hours: float = 8.0
    detour_factor: float = 1.2
    service_min: dict = field(default_factory=lambda: {"B2B": 105, "B2C": 60})
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

def load_data(cfg: Config = CFG) -> dict:
    df_orders = pd.read_excel(cfg.order_file)
    df_veh = pd.read_excel(cfg.vehicle_file)
    df_driver = pd.read_excel(cfg.driver_file)
    df_dist = pd.read_excel(cfg.matrix_file, index_col=0)
    df_cust = pd.read_excel(cfg.cust_file)

    # Trích xuất linh hoạt cột từ dữ liệu xe
    veh_id_col = first_col(df_veh, "vehicle_id", "VEHICLE_ID", "id")
    plate_col = first_col(df_veh, "license_plate", "bien_so", "plate", "vehicle_id")
    type_col = first_col(df_veh, "vehicle_type", "type", "vehicle_class", "vehicle_name")
    wh_col = first_col(df_veh, "wh_id", "warehouse_id", "depot_id")
    speed_col = first_col(df_veh, "speed_kmh", "speed", "vận_tốc")
    w_col = first_col(df_veh, "max_weight_kg", "weight_capacity", "max_weight", "capacity_kg")
    v_col = first_col(df_veh, "max_volume_m3", "volume_capacity", "max_volume", "capacity_m3")

    df_veh["vehicle_id"] = df_veh[veh_id_col].astype(str) if veh_id_col else [f"VEH_{i}" for i in range(len(df_veh))]
    df_veh["license_plate"] = df_veh[plate_col].astype(str) if plate_col else df_veh["vehicle_id"]
    df_veh["vehicle_type"] = df_veh[type_col].astype(str).str.strip() if type_col else "Truck"
    df_veh["wh_id"] = df_veh[wh_col].astype(str).str.strip() if wh_col else "WH_DEFAULT"
    df_veh["speed_kmh"] = pd.to_numeric(df_veh[speed_col], errors="coerce").fillna(35.0) if speed_col else 35.0
    df_veh["max_weight_kg"] = pd.to_numeric(df_veh[w_col], errors="coerce").fillna(1000.0) if w_col else 1000.0
    df_veh["max_volume_m3"] = pd.to_numeric(df_veh[v_col], errors="coerce").fillna(5.0) if v_col else 5.0

    fx_col = cfg.fixed_cost_col or first_col(df_veh, "fixed_cost", "fixed_cost_per_day")
    vr_col = cfg.variable_cost_col or first_col(df_veh, "variable_cost_per_km", "cost_per_km")
    df_veh["fixed_cost"] = pd.to_numeric(df_veh[fx_col], errors="coerce").fillna(300_000.0) if fx_col else 300_000.0
    df_veh["variable_cost_per_km"] = pd.to_numeric(df_veh[vr_col], errors="coerce").fillna(8_000.0) if vr_col else 8_000.0

    # Trích xuất danh sách kho động hoàn toàn từ dữ liệu xe
    warehouses = {}
    for wh in df_veh["wh_id"].unique():
        warehouses[wh] = {"name": f"Kho {wh}", "lat": 21.0285, "lon": 105.8542}

    # Trích xuất tài xế động theo kho
    d_name_col = first_col(df_driver, "driver_name", "name", "full_name", "TÊN", "driver_id")
    d_role_col = first_col(df_driver, "role", "position", "VAI_TRÒ", "type")
    d_wh_col = first_col(df_driver, "wh_id", "warehouse_id", "depot_id", "KHO")
    df_driver["driver_name"] = df_driver[d_name_col].astype(str) if d_name_col else "Tài xế"
    df_driver["role"] = df_driver[d_role_col].astype(str).str.strip().str.capitalize() if d_role_col else "Chính"
    df_driver["wh_id"] = df_driver[d_wh_col].astype(str).str.strip() if d_wh_col else list(warehouses.keys())[0]

    drivers_by_wh = {}
    for wh in warehouses:
        sub = df_driver[df_driver["wh_id"] == wh]
        chính = sub[sub["role"].str.contains("Chính|Primary|Driver", case=False, na=False)]["driver_name"].tolist()
        phụ = sub[sub["role"].str.contains("Phụ|Assistant|Helper", case=False, na=False)]["driver_name"].tolist()
        if not chính: chính = sub["driver_name"].tolist() or ["Tài xế chính"]
        if not phụ: phụ = ["Phụ xe hỗ trợ"]
        drivers_by_wh[wh] = {"chính": chính, "phụ": phụ}

    # Trích xuất khách hàng động
    addr_col = first_col(df_cust, "address", "Location", "ADDRESS")
    lat_col = first_col(df_cust, "lat", "LATITUDE", "latitude")
    lon_col = first_col(df_cust, "lng", "lon", "LONGITUDE", "longitude")
    cid_col = first_col(df_cust, "customer_id", "CUSTOMER_ID", "id")

    cust = {}
    for _, r in df_cust.iterrows():
        cid = r[cid_col] if cid_col else _
        cust[cid] = {
            "lat": float(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else 21.0,
            "lon": float(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else 105.8,
            "address": str(r[addr_col]) if addr_col and pd.notna(r[addr_col]) else ""
        }

    # Trích xuất đơn hàng động
    oid_c = first_col(df_orders, "order_id", "ORDER_ID")
    date_c = first_col(df_orders, "order_date", "delivery_date", "date", "NGÀY")
    wait_c = first_col(df_orders, "waiting_date", "WAITING_DATE")
    
    orders = []
    for i, r in df_orders.iterrows():
        cid = r.get("customer_id")
        dt_val = pd.to_datetime(r[date_c]) if date_c and pd.notna(r.get(date_c)) else pd.Timestamp.today()
        orders.append({
            "ORDER_ID": str(r[oid_c]) if oid_c and pd.notna(r.get(oid_c)) else f"OR{i}",
            "customer_id": cid,
            "weight": float(r.get("total_weight_kg", 0.0)),
            "volume": float(r.get("total_volume_m3", 0.0)),
            "order_type": str(r.get("order_type", "B2C")),
            "date": str(dt_val.date()),
            "WAITING_DATE": str(pd.to_datetime(r[wait_c]).date()) if wait_c and pd.notna(r[wait_c]) else None,
            "Location": cust.get(cid, {}).get("address", ""),
        })
    return {"orders": orders, "vehicles": df_veh, "drivers": drivers_by_wh, "dist": df_dist, "cust": cust, "warehouses": warehouses}

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
        if self.catalog.empty: return "Truck"
        ok = self.catalog[(self.catalog["w"] >= w) & (self.catalog["v"] >= v)]
        return ok.index[0] if not ok.empty else self.catalog.index[-1]

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
        for o in day_orders:
            if o["customer_id"] not in self.cust:
                exc("MEDIUM", o, "THIẾU TỌA ĐỘ", "Không tìm thấy khách hàng", "Bỏ qua", "Bổ sung tọa độ")
                res["carried_orders"].append(o)
            elif o["weight"] > self.max_w or o["volume"] > self.max_v:
                exc("CRITICAL", o, "ĐƠN QUÁ CỠ", f"{o['weight']:.0f}kg vượt xe lớn nhất", "Tách riêng", "Thuê xe lớn")
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
        cands = [(i, r) for i, r in pool.items() if r["wh_id"] == wh_id and r["max_weight_kg"] >= m["w"] and r["max_volume_m3"] >= m["v"]]
        if cands:
            idx, vrow = min(cands, key=lambda t: (t[1]["max_weight_kg"], t[1]["max_volume_m3"]))
            pool.pop(idx)
            external = False
            vid, plate, vtype = vrow["vehicle_id"], vrow["license_plate"], vrow["vehicle_type"]
            speed, fx, vr = vrow["speed_kmh"], vrow["fixed_cost"], vrow["variable_cost_per_km"]
        else:
            vt = m["vtype"] if m["vtype"] in self.catalog.index else (self.catalog.index[-1] if not self.catalog.empty else "Truck")
            external = True
            vid, plate, vtype = f"3PL-{vt}", "Thuê ngoài 3PL (Hết xe nhà)", vt
            speed = self.catalog.loc[vt, "speed"] if vt in self.catalog.index else 35.0
            fx = self.catalog.loc[vt, "fixed"] if vt in self.catalog.index else 300_000
            vr = self.catalog.loc[vt, "var"] if vt in self.catalog.index else 8_000

        primary, assistant, backup = assign_driver(wh_id)
        km, hours = m["km"], m["hours"]
        start = dt.datetime.combine(pd.to_datetime(date_str).date(), dt.datetime.strptime(cfg.start_time, "%H:%M").time())
        max_w_val = self.catalog.loc[vtype, "w"] if vtype in self.catalog.index else m["w"]
        load_f = m["w"] / max_w_val if max_w_val > 0 else 0.5
        return {
            "kind": "NORMAL", "wh_id": wh_id, "orders": [o for c in route for o in by_cust[c]],
            "vehicle_id": vid, "license_plate": plate, "vehicle_type": vtype, "external": external,
            "driver_primary": primary, "driver_assistant": assistant, "is_backup_driver": backup,
            "km": round(km, 1), "speed_kmh": speed, "load_factor": min(max(load_f, 0.15), 1.0),
            "fixed_cost": fx, "variable_cost": vr * km, "overnight_cost": 0.0,
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

TABLE_CSS = """
<style>
.custom-route-table{width:100%;border-collapse:separate;border-spacing:0;font-family:'Segoe UI',Tahoma,sans-serif;
  margin:10px 0 25px;box-shadow:0 4px 12px rgba(0,0,0,.05);border-radius:10px;overflow:hidden;background:#fff}
.custom-route-table th{background:#f8fafc;color:#475569;font-weight:700;font-size:.85em;text-transform:uppercase;
  padding:12px 10px;border-bottom:2px solid #e2e8f0;text-align:left}
.custom-route-table td{padding:12px 10px;border-bottom:1px solid #f1f5f9;font-size:.9em;vertical-align:middle}
.custom-route-table tr:hover{background:#f8fafc;transition:all .2s}
</style>
"""

def pill(text, fg, bg, bold=True, size="0.8em"):
    return f"<span style='background:{bg};color:{fg};padding:3px 8px;border-radius:12px;font-weight:{'bold' if bold else 'normal'};font-size:{size};'>{text}</span>"

def code_badge(text, fg="#334155", bg="#f1f5f9"):
    return f"<span style='background:{bg};color:{fg};padding:2px 6px;border-radius:4px;font-family:monospace;font-weight:bold;'>{text}</span>"

def to_table(rows):
    return pd.DataFrame(rows).to_html(escape=False, index=False, classes="custom-route-table")

def drivers_html(r):
    badge = "<br><span style='background:#fef2f2;color:#dc2626;font-size:0.75em;padding:2px 6px;border-radius:4px;'>⚠️ Tài xế dự phòng</span>" if r["is_backup_driver"] else ""
    return f"<div style='font-size:.85em;'>👤 <b>Chính:</b> {r['driver_primary']}<br>🤝 <b>Phụ:</b> {r['driver_assistant']}{badge}</div>"

def schedule_html(r):
    return f"⏰ <b>{r['start']:%H:%M}</b> ➔ <b>{r['end']:%H:%M}</b>"

def route_total(r):
    return r["fixed_cost"] + r["variable_cost"] + r["overnight_cost"] + r.get("driver_cost", 0)

def kpi_card(title, value, unit, sub="", color="#10b981", vcolor="#1e293b", bg="white", flex=1):
    return f"""<div style='flex:{flex};background:{bg};padding:15px;border-radius:10px;border-left:5px solid {color};box-shadow:0 2px 8px rgba(0,0,0,.05);'>
      <div style='color:#64748b;font-size:.75em;font-weight:bold;text-transform:uppercase;'>{title}</div>
      <div style='font-size:1.5em;font-weight:bold;color:{vcolor};margin-top:5px;'>{value} <span style='font-size:.6em;color:#64748b;'>{unit}</span></div>
      <div style='font-size:.75em;color:#64748b;margin-top:3px;'>{sub}</div></div>"""

def render_summary(date_str, day, kpis, total_cost):
    k = kpis
    routes = day["routes"] + day["overdue_routes"]
    n_ext = sum(r["external"] for r in routes)
    n_total = len(routes)
    return f"""
<div style="font-family:'Segoe UI',Tahoma,sans-serif;margin:15px 0 20px;">
  <div style="background:linear-gradient(135deg,#1e3a8a,#3b82f6);padding:18px 25px;border-radius:12px;color:#fff;box-shadow:0 4px 15px rgba(59,130,246,.3);display:flex;justify-content:space-between;align-items:center;">
    <div>
      <h2 style="margin:0;font-size:1.5em;font-weight:700;">🗓️ DASHBOARD ĐIỀU PHỐI ĐỘNG - {date_str}</h2>
      <p style="margin:5px 0 0;opacity:.85;font-size:.9em;">Mô hình tối ưu hóa tuyến đường động hoàn toàn theo Input</p>
    </div>
    <div style="background:rgba(255,255,255,.2);padding:8px 15px;border-radius:8px;font-weight:bold;">🎯 Vi phạm: {k['violations']}</div>
  </div>
  <div style="display:flex;gap:12px;margin-top:15px;">
    <div style="flex:1.2;background:linear-gradient(135deg,#fffbe3,#fff3c4);padding:15px;border-radius:10px;border-left:5px solid #f59e0b;box-shadow:0 2px 8px rgba(245,158,11,.15);">
      <div style="color:#b45309;font-size:.75em;font-weight:800;text-transform:uppercase;">👑 TỔNG CHI PHÍ</div>
      <div style="font-size:1.5em;font-weight:800;color:#78350f;margin-top:5px;">{money(total_cost)} <span style="font-size:.6em;color:#92400e;">VNĐ</span></div>
      <div style="font-size:.75em;color:#92400e;margin-top:3px;">Vận hành {money(k['operating_cost'])} · Phạt {money(k['penalty_cost'])}</div>
    </div>
    {kpi_card('Chi phí ngày', money(day['day_cost']), 'VNĐ', f"Phạt {money(day['penalty_cost'])}", '#10b981', '#047857')}
    {kpi_card('Tổng chuyến', n_total, 'Tuyến', f"Thuê ngoài: {n_ext}", '#3b82f6')}
    {kpi_card('Lấp đầy TB', f"{k['avg_load_factor']*100:.1f}%", 'Tải', 'Tối ưu hóa', '#8b5cf6')}
  </div>
</div>"""

def render_main_table(routes, wh_info):
    rows = []
    for idx, r in enumerate(routes, 1):
        wh = wh_info.get(r["wh_id"], {"name": r["wh_id"]})
        orders = "<br>".join(code_badge(o["ORDER_ID"]) for o in r["orders"])
        if r["external"]:
            veh = f"<span style='color:#7e22ce;font-weight:bold;'>🟣 Xe thuê ngoài 3PL</span><br><small style='color:#6b21a8;'>({r['vehicle_type']})</small>"
        else:
            veh = f"🚚 <b>{r['vehicle_type']}</b><br><small style='color:#2563eb;font-weight:bold;'>Biển số: {r['license_plate']}</small>"
        locs = ", ".join(sorted({o["Location"].split(",")[-1].strip() for o in r["orders"] if o["Location"]}))
        rows.append({
            "MÃ TUYẾN": f"<strong style='color:#1e3a8a;'>Tuyến {wh['name']} #{idx}</strong><br><small style='color:#64748b;'>📍 {locs}</small>",
            "ĐƠN GIAO": orders,
            "TÀI XẾ": drivers_html(r),
            "PHÂN BỔ XE": veh,
            "QUÃNG ĐƯỜNG": f"<b>{r['km']} km</b><br><small style='color:#64748b;'>Lấp đầy: <b>{r['load_factor']*100:.0f}%</b></small>",
            "TỔNG CHI PHÍ": f"<span style='color:#047857;font-weight:bold;font-size:1.05em;'>{money(route_total(r))} đ</span>",
            "LỊCH TRÌNH": schedule_html(r),
            "TRẠNG THÁI": pill("✅ Đã tối ưu", "#15803d", "#dcfce7", bold=False),
        })
    return to_table(rows)

def launch_dashboard(cfg: Config = CFG):
    print("🔮 Đang khởi tạo Dashboard động hoàn toàn từ dữ liệu input...")
    data = load_data(cfg)
    planner = RoutePlanner(data, cfg)
    by_date = {}
    for o in data["orders"]: by_date.setdefault(o["date"], []).append(o)
    days = {d: planner.plan_day(d, by_date[d]) for d in sorted(by_date)}
    _, _, kpis, total_cost = simulate_all(data, cfg)
    out = widgets.Output()
    dates = list(days)
    if not dates:
        print("⚠️ Không tìm thấy đơn hàng nào trong dữ liệu!")
        return planner, days
    picker = widgets.Dropdown(options=dates, value=dates[0], description="📅 Ngày:", layout=widgets.Layout(width="260px"))
    def show(date_str):
        with out:
            clear_output(wait=True)
            day = days[date_str]
            display(HTML(render_summary(date_str, day, kpis, total_cost) + TABLE_CSS + render_main_table(day["routes"] + day["overdue_routes"], planner.wh)))
    picker.observe(lambda ch: show(ch["new"]) if ch["name"] == "value" else None, names="value")
    display(picker, out)
    show(picker.value)
    return planner, days

if __name__ == "__main__":
    launch_dashboard()


