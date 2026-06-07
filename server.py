from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import geoglows
import joblib
import datetime
import rasterio
import time
from pyproj import Transformer

app = FastAPI(title="Katingan Real-Time Time-Slider API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

print("🚀 Loading AI Surrogate Matrix into RAM...")
MODEL = joblib.load("./flood_model.joblib")
SPATIAL_INFO = joblib.load("./spatial_meta.joblib")

# ==========================================
# ⚡ THE STATIC COORDINATE CACHE ENGINE
# ==========================================
print("🌍 Pre-calculating Geographic Coordinates...")
transformer = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
transform = SPATIAL_INFO['meta']['transform']
valid_mask = SPATIAL_INFO['mask']

# 1. Get the rows/cols of all valid mesh pixels
all_rows, all_cols = np.where(valid_mask)

# 2. Apply downsampling rule ONCE (Skip every 3rd pixel)
downsample_mask = (all_rows % 3 == 0) & (all_cols % 3 == 0)
final_rows = all_rows[downsample_mask]
final_cols = all_cols[downsample_mask]

# 3. Translate to Lat/Lng ONCE and hold in Server RAM
xs, ys = rasterio.transform.xy(transform, final_rows, final_cols)
STATIC_LNGS, STATIC_LATS = transformer.transform(xs, ys)

print(f"✅ Cached {len(STATIC_LATS)} permanent coordinate points.")
# ==========================================

FORECAST_CACHE = None
TIMESTAMPS, FLOW_VALUES = [], []

def fetch_and_cache_forecast():
    global FORECAST_CACHE, TIMESTAMPS, FLOW_VALUES
    for days_back in range(4):
        target_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)).strftime('%Y%m%d')
        try:
            df = geoglows.data.forecast(520315071, date=target_date)
            if df.empty: continue
            FORECAST_CACHE = df
            TIMESTAMPS = [ts.strftime('%Y-%m-%d %H:%M') for ts in df.index]
            FLOW_VALUES = [float(v) for v in df['flow_uncertainty_upper'].values]
            return
        except Exception:
            pass
    TIMESTAMPS, FLOW_VALUES = ["Error"], [0.0]

fetch_and_cache_forecast()

@app.get("/api/init")
def get_timeline_metadata():
    return {
        "status": "success", "steps_count": len(TIMESTAMPS),
        "timeline": TIMESTAMPS, "flows": FLOW_VALUES
    }

@app.get("/api/map")
async def get_map_layer(flow: float): # 🚀 'async' prevents Thread Thrashing!
    t_start = time.time()

    # ==========================================
    # ⚡ THE PURE NUMPY AI BYPASS
    # ==========================================
    # 1. Ask the AI ONLY to find the 3 closest maps (Takes ~0.001s)
    distances, indices = MODEL.kneighbors([[flow]])
    
    # 2. Extract those 3 raw maps directly from the model's memory
    neighbor_maps = MODEL._y[indices[0]] 
    
    # 3. Manually apply the distance weights in pure Numpy (Takes ~0.005s)
    # (We add 1e-10 to prevent division-by-zero if the flow exactly matches a map)
    weights = 1.0 / (distances[0] + 1e-10) 
    predicted_depths = np.average(neighbor_maps, axis=0, weights=weights)
    # ==========================================

    t_ai = time.time()

    downsampled_depths = predicted_depths[downsample_mask]
    water_mask = downsampled_depths > 0.1

    flood_points = [
        {"lat": lat, "lng": lng, "depth": round(float(d), 2)}
        for lat, lng, d in zip(STATIC_LATS[water_mask], STATIC_LNGS[water_mask], downsampled_depths[water_mask])
    ]
    t_loop = time.time()

    print(f"⏱️ SPEED | AI Math: {t_ai - t_start:.4f}s | JSON Assembly: {t_loop - t_ai:.4f}s | Total: {t_loop - t_start:.4f}s")

    return {
        "requested_flow_m3s": flow,
        "total_flooded_pixels": len(flood_points),
        "data": flood_points
    }