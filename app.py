import os
import io
import math
import json
import datetime as dt
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter

# ============================================================================
# 🗺️ TAB 6: ĐỊNH TUYẾN CLARKE-WRIGHT (Đồng bộ biến & Tư duy: Tab A / Tab B)
# ============================================================================
OUT_CUSTOMER = os.path.join(BASE_DIR, "output_customer")
OUT_MATRIX = os.path.join(BASE_DIR, "output_matrix")
OUT_SCREEN = os.path.join(BASE_DIR, "output_screening")
DEFAULT_DEPOT = (21.0285, 105.8542)

SCHEMA_CONTRACT = {
    "output_fleet/DIM_VEHICLE.xlsx": list(VEHICLE_FIELDS),
    "output_warehouse/WAREHOUSE_WITH_COORDINATES.xlsx": ["id_warehouse", "address", "lat", "lng"],
    "output_product/DIM_PRODUCT.xlsx": list(PRODUCT_FIELDS),
    "output_driver/DIM_DRIVER.xlsx": list(DRIVER_FIELDS),
    "output_orders/DIM_ORDERS.xlsx": FIELDS,
}

# ----------------------------------------------------------------------------
# MÀN LỌC THUÊ NGOÀI + TAB A (trọng tải/hạm đội/Max_Distance) + TAB B (thời gian tuyến)
# ----------------------------------------------------------------------------
OC_DEFAULTS = {
    "full_price": 2_500_000.0, "saving_1_price": 800_000.0, "saving_2_price": 300_000.0,
    "full_vehicle_name": "Xe tải thuê ngoài", "full_vehicle_max_weight_kg": 10_000.0, "full_vehicle_max_volume_m3": 40.0,
    "tier2_max_w": 20.0, "tier2_max_v": 1.0, # loại 2: < 20 kg VÀ < 1 m3
    "tier1_max_w": 100.0, "tier1_max_v": 5.0, # loại 1: 20-100 kg / 1-5 m3
}


def _full_cls(w, v, oc):
    fw, fv = max(float(oc["full_vehicle_max_weight_kg"]), 1e-9), max(float(oc["full_vehicle_max_volume_m3"]), 1e-9)
    trips = max(1, math.ceil(w / fw - 1e-9), math.ceil(v / fv - 1e-9))
    return {"type": f"Full — {oc['full_vehicle_name']} × {trips} chuyến", "tier": "FULL", "cost": trips * float(oc["full_price"]),
            "trips": trips, "vehicle": oc["full_vehicle_name"], "cap_w": trips * fw}


def classify_outsourcing(excess_w, excess_v, oc=None, force_full=False) -> dict:
    """MÀN LỌC: phân loại phần bị cắt (Max_Distance / 8h) hoặc phần dư vượt xe lớn nhất.
    < 20kg và < 1m3 -> loại 2 | trong 20-100kg / 1-5m3 -> loại 1 | lớn hơn -> Full theo xe thuê ngoài."""
    oc = {**OC_DEFAULTS, **(oc or {})}
    w, v = max(float(excess_w), 0.0), max(float(excess_v), 0.0)
    if not force_full:
        if w < oc["tier2_max_w"] and v < oc["tier2_max_v"]:
            return {"type": "Tiết kiệm loại 2", "tier": "SAVING_2", "cost": float(oc["saving_2_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 2", "cap_w": float(oc["tier2_max_w"])}
        if w <= oc["tier1_max_w"] and v <= oc["tier1_max_v"]:
            return {"type": "Tiết kiệm loại 1", "tier": "SAVING_1", "cost": float(oc["saving_1_price"]), "trips": 1,
                    "vehicle": "Tiết kiệm loại 1", "cap_w": float(oc["tier1_max_w"])}
    return _full_cls(w, v, oc)


def evaluate_transportation_constraints(route_or_order, df_vehicles, vehicle_available_counts, outsourcing_config):
    """Tab A: trọng tải, hạm đội, Max_Distance."""
    oc = {**OC_DEFAULTS, **(outsourcing_config or {})}
    total_w = float(route_or_order.get("total_weight_kg", 0.0))
    total_v = float(route_or_order.get("total_volume_m3", 0.0))
    total_dist = float(route_or_order.get("total_distance_km", 0.0))

    def outsourced(action, cls, message):
        return {"status": "OUTSOURCED", "action_type": action, "assigned_vehicle": None, "outsourcing_type": cls["type"],
                "cost": cls["cost"], "cls": cls, "message": message}

    if df_vehicles.empty:
        return outsourced("NO_FLEET", _full_cls(total_w, total_v, oc), "⚠️ Hạm đội trống -> Thuê ngoài Full.")
    max_fleet_w, max_fleet_v = df_vehicles["max_weight_kg"].max(), df_vehicles["max_volume_m3"].max()
    sum_fleet_w, sum_fleet_v = df_vehicles["max_weight_kg"].sum(), df_vehicles["max_volume_m3"].sum()
    avail = df_vehicles[df_vehicles["vehicle_id"].map(lambda x: vehicle_available_counts.get(x, 1) > 0)]

    if total_w > sum_fleet_w or total_v > sum_fleet_v:
        return outsourced("OUTSOURCE_FULL", _full_cls(total_w, total_v, oc),
                        "🚨 Vượt quá tổng sức chứa toàn bộ hạm đội nhà -> Thuê ngoài loại Full theo chuyến.")
    if total_w > max_fleet_w or total_v > max_fleet_v:
        if avail.empty:
            return outsourced("OUTSOURCE_NO_VEHICLE_AVAILABLE", classify_outsourcing(total_w, total_v, oc),
                            "⚠️ Hết xe khả dụng trong ngày -> thuê ngoài toàn bộ qua màn lọc.")
        big = avail.loc[avail["max_weight_kg"].idxmax()]
        v_id = big["vehicle_id"]
        vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
        cls = classify_outsourcing(total_w - big["max_weight_kg"], total_v - big["max_volume_m3"], oc)
        return {"status": "PARTIAL_SPLIT", "action_type": "MAX_VEHICLE_PLUS_OUTSOURCE", "assigned_vehicle": v_id,
                "outsourcing_type": cls["type"], "cost": cls["cost"], "cls": cls,
                "message": f"⚠️ Vượt xe lớn nhất nhà ({v_id}). Cắt full-fill xe này (trừ 1 xe), phần dư thuê ngoài {cls['type']}."}

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
                "message": f"⚠️ Tuyến vượt Max_Distance ({total_dist:.1f}km > {limit:.0f}km) của xe {v_id}. Cắt đơn, phần cắt qua màn lọc thuê ngoài."}
    vehicle_available_counts[v_id] = vehicle_available_counts.get(v_id, 1) - 1
    return {"status": "APPROVED", "action_type": "IN_HOUSE_SUCCESS", "assigned_vehicle": v_id, "outsourcing_type": None,
            "cost": 0.0, "cls": None, "message": f"✅ Thỏa mãn trọng tải! Giao xe NHỎ NHẤT khả thi: {v_id} (Đã trừ 1 xe trong ngày)."}


def evaluate_time_constraints(route_or_order, chosen_vehicle=None, service_time_rules=None, max_hours=8.0):
    """Tab B: thời gian tuyến <= max_hours, vượt -> CUT_TIME_EXCEEDED."""
    if service_time_rules is None:
        service_time_rules = {"B2C": {"loading": 25, "unloading": 35}, "B2B": {"loading": 45, "unloading": 60}}
    orders = route_or_order.get("orders", [])
    dist = float(route_or_order.get("total_distance_km", 0.0))
    speed = float(chosen_vehicle.get("average_speed_kmh", 40.0)) if chosen_vehicle is not None else 40.0
    travel = dist / speed if speed > 0 else 0.0
    rule = lambda o: service_time_rules.
