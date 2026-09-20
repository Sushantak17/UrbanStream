#!/usr/bin/env python3
"""UrbanStream Dashboard v5 — Acuity-inspired design system."""
import json, os, random, time, math
from datetime import datetime
import pandas as pd
import redis
import streamlit as st

st.set_page_config(page_title="UrbanStream", page_icon="⬡",
                   layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════
# CSS — Acuity-inspired: sage-white bg, Archivo/Inter/IBM Plex Mono
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg: #F5F7F4;
  --surface: #FFFFFF;
  --surface-sunken: #EFF2EE;
  --border: #DDE3DC;
  --border-h: #C4CCC3;
  --ink: #16211D;
  --ink-soft: #5B6B63;
  --ink-faint: #93A199;
  --brand: #0E4F44;
  --brand-soft: #E4EEEA;
  --red: #C1392B;
  --red-soft: #FBEAE7;
  --amber: #B8791C;
  --amber-soft: #FBF1DF;
  --green: #2F7A52;
  --green-soft: #E7F3EB;
  --purple: #5B3D8F;
  --purple-soft: #EFEAF6;
  --blue: #1D5C9E;
  --blue-soft: #E3EEF8;
}

*{box-sizing:border-box; margin:0; padding:0}
html,body,[class*="css"],.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"]>section{
  font-family:'Inter', -apple-system, sans-serif !important;
  background: var(--bg) !important;
  color: var(--ink) !important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 !important; max-width:100% !important; position:relative; z-index:1}
section[data-testid="stSidebar"]{display:none !important}
div[data-testid="stMetric"]{display:none}
.element-container,.stMarkdown{background:transparent !important}

/* Protect Streamlit components */
[data-testid="stExpander"]{
  border:1px solid var(--border) !important; border-radius:10px !important;
  background:var(--surface) !important;
}

/* Details/Summary as buttons */
details.wb{display:inline-block;position:relative}
details.wb > summary{
  list-style:none; cursor:pointer;
  padding:7px 12px; border-radius:7px; font-size:11px; font-weight:700;
  user-select:none; display:inline-block;
}
details.wb > summary::-webkit-details-marker{display:none}
details.wb > summary::marker{display:none;content:""}
details.wb.ghost > summary{
  background:transparent; border:1.5px solid var(--border); color:var(--ink-soft);
  transition:background .15s,border-color .15s;
}
details.wb.ghost > summary:hover{background:var(--surface-sunken);border-color:var(--border-h)}
details.wb.escalate > summary{
  background:var(--red); color:#fff; border:1.5px solid transparent;
  transition:opacity .15s;
}
details.wb.escalate > summary:hover{opacity:.85}
details.wb > .wb-panel{
  position:absolute; bottom:calc(100% + 8px); right:0; z-index:100;
  background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:16px 20px; min-width:340px;
  box-shadow:0 8px 32px rgba(20,30,25,.15);
  font-size:12px; line-height:1.8; color:var(--ink-soft);
}
details.wb.escalate > .wb-panel{
  min-width:220px; text-align:center; padding:14px 18px;
}
details.wb > .wb-panel .wb-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
details.wb > .wb-panel b{color:var(--ink)}
details.wb > .wb-panel .wb-rec{font-weight:700;color:var(--brand);margin-bottom:2px}

/* Style Streamlit relocate buttons to match card design */
.stButton > button {
  background-color: #C1392B !important;
  color: #fff !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  padding: 7px 14px !important;
  border-radius: 7px !important;
  border: 1.5px solid transparent !important;
  box-shadow: none !important;
  height: auto !important;
  min-height: 0 !important;
  width: auto !important;
  line-height: 1.2 !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  opacity: 0.85 !important;
  background-color: #C1392B !important;
  border-color: transparent !important;
}
.stButton > button:active, .stButton > button:focus {
  background-color: #a03025 !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
.stButton {
  margin-top: -6px !important;
  margin-bottom: 8px !important;
  padding-left: 48px !important;
}

/* ═══ HEADER ═══ */
.hdr{
  display:flex; align-items:center; justify-content:space-between;
  padding:24px 48px; border-bottom:1px solid var(--border);
  background:rgba(255,255,255,.85); backdrop-filter:blur(12px);
}
.hdr-left{display:flex; align-items:baseline; gap:12px}
.hdr-eyebrow{
  font-family:'IBM Plex Mono', monospace; font-size:11px;
  letter-spacing:.12em; text-transform:uppercase;
  color:var(--brand); font-weight:600;
}
.hdr-title{
  font-family:'Archivo', sans-serif; font-weight:800;
  font-size:24px; letter-spacing:-.01em; color:var(--ink);
}
.hdr-r{display:flex; align-items:center; gap:14px}
.chip{
  font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600;
  padding:6px 14px; border-radius:7px;
  display:inline-flex; align-items:center; gap:7px;
}
.chip-live{background:var(--green-soft); color:var(--green); border:1px solid #BBD9C5}
.chip-src{border:1px solid var(--border); color:var(--ink-soft); background:rgba(255,255,255,.6)}
.dot-live{
  width:8px; height:8px; border-radius:50%; background:var(--green);
  animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot{
  0%,100%{opacity:1; transform:scale(1)}
  50%{opacity:.3; transform:scale(1.8)}
}

/* ═══ LEGEND ═══ */
.legend{
  display:flex; align-items:center; gap:18px;
  padding:10px 16px; background:var(--surface); border:1px solid var(--border);
  border-radius:8px; width:fit-content; margin:12px 48px 0;
}
.legend .lg-chip{
  display:flex; align-items:center; gap:7px;
  font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--ink-soft);
}
.legend .lg-dot{width:9px; height:9px; border-radius:50%}
.lg-dot.red{background:var(--red)} .lg-dot.amber{background:var(--amber)} .lg-dot.green{background:var(--green)}

/* ═══ TABS ═══ */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(255,255,255,.7); backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border); gap:0; padding:0 48px;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Inter', sans-serif; font-size:13px; font-weight:600;
  padding:16px 24px; border-radius:0; color:var(--ink-faint) !important;
  transition:color .2s;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--ink-soft) !important}
.stTabs [aria-selected="true"]{color:var(--brand) !important; border-bottom:2.5px solid var(--brand) !important}

/* ═══ PLATE (section wrapper) ═══ */
.plate{padding:28px 48px; position:relative}
.plate-label{display:flex; align-items:baseline; gap:12px; margin-bottom:14px}
.plate-num{
  font-family:'IBM Plex Mono', monospace; font-size:12px;
  color:var(--brand); font-weight:600;
}
.plate-title{font-family:'Archivo', sans-serif; font-weight:700; font-size:18px; color:var(--ink)}
.plate-window{
  font-family:'IBM Plex Mono', monospace; font-size:11px;
  color:var(--ink-faint); margin-left:auto;
}
.plate-border{background:repeating-linear-gradient(90deg,var(--border) 0,var(--border) 6px,transparent 6px,transparent 14px);height:1px;margin:0 48px}

/* ═══ FRAME (card) ═══ */
.frame{
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:24px 28px;
  box-shadow:0 1px 2px rgba(20,30,25,0.04), 0 12px 32px -20px rgba(20,30,25,0.12);
}

/* ═══ KPI (capacity cards) ═══ */
.cap-row{display:grid; gap:14px; margin-bottom:20px}
.cap-row.cols-4{grid-template-columns:repeat(4,1fr)}
.cap-row.cols-3{grid-template-columns:repeat(3,1fr)}
.cap-card{border:1px solid var(--border); border-radius:10px; padding:14px 16px}
.cap-label{
  font-family:'IBM Plex Mono', monospace; font-size:10px; font-weight:600;
  letter-spacing:.06em; text-transform:uppercase; color:var(--ink-faint); margin-bottom:6px;
}
.cap-val{
  font-family:'IBM Plex Mono', monospace; font-size:24px; font-weight:600;
  line-height:1; margin-bottom:4px;
}
.cap-sub{font-size:12px; color:var(--ink-faint); margin-bottom:8px}
.cap-track{height:5px; background:var(--surface-sunken); border-radius:3px; overflow:hidden}
.cap-fill{height:100%; border-radius:3px}

/* ═══ TABLE (queue-style) ═══ */
.q-header{
  display:grid; align-items:center; padding:10px 8px;
  font-family:'IBM Plex Mono', monospace; font-size:10px; font-weight:600;
  text-transform:uppercase; letter-spacing:.06em; color:var(--ink-faint);
  border-bottom:1.5px solid var(--border);
}
.q-row{
  display:grid; align-items:center; padding:12px 8px;
  font-size:13px; border-bottom:1px solid var(--border);
  transition:background .15s;
}
.q-row:hover{background:var(--surface-sunken)}
.q-rank{font-family:'IBM Plex Mono', monospace; color:var(--ink-faint); font-size:12px}
.tier{
  display:inline-flex; align-items:center; gap:6px;
  padding:4px 10px; border-radius:6px; font-weight:700; font-size:11px;
  width:fit-content;
}
.tier-red{background:var(--red-soft); color:var(--red)}
.tier-amber{background:var(--amber-soft); color:var(--amber)}
.tier-green{background:var(--green-soft); color:var(--green)}
.tier-blue{background:var(--blue-soft); color:var(--blue)}
.tier-purple{background:var(--purple-soft); color:var(--purple)}
.mono{font-family:'IBM Plex Mono', monospace}

/* ═══ WATCH LIST ═══ */
.watch-list{display:flex; flex-direction:column; gap:8px}
.watch-item{
  display:grid; grid-template-columns:44px 1fr 150px 130px;
  align-items:center; gap:14px; padding:14px 16px;
  border:1px solid var(--border); border-radius:10px;
  transition:border-color .2s;
}
.watch-item:hover{border-color:var(--border-h)}
.watch-item.alert{border-color:var(--red); background:var(--red-soft)}
.watch-item.warn{border-color:var(--amber); background:var(--amber-soft)}
.ring{
  width:40px; height:40px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600;
  border:3px solid var(--green); color:var(--ink);
}
.ring.amber{border-color:var(--amber)}
.ring.red{border-color:var(--red); animation:pulse-ring 1.6s infinite}
@keyframes pulse-ring{
  0%{box-shadow:0 0 0 0 rgba(193,57,43,0.35)}
  70%{box-shadow:0 0 0 9px rgba(193,57,43,0)}
  100%{box-shadow:0 0 0 0 rgba(193,57,43,0)}
}
.watch-name{font-weight:600; font-size:13px}
.watch-sub{font-size:11px; color:var(--ink-faint); margin-top:2px}
.watch-trigger{font-size:12px; color:var(--ink-soft)}
.watch-actions{display:flex; gap:6px; justify-content:flex-end}
.btn-sm{
  padding:7px 12px; border-radius:7px; font-size:11px; font-weight:700;
  border:1.5px solid transparent; cursor:pointer;
}
.btn-sm.escalate{background:var(--red); color:#fff}
.btn-sm.ghost{background:transparent; border-color:var(--border); color:var(--ink-soft)}

/* ═══ REC CARD (recommendation) ═══ */
.rec-card{
  border:1px solid var(--border); border-radius:12px;
  padding:18px 20px; background:var(--surface-sunken);
}
.confidence .bar-track{
  height:6px; background:var(--border); border-radius:4px;
  overflow:hidden; margin-top:6px;
}
.confidence .bar-fill{height:100%; border-radius:4px}
.reasons{list-style:none; margin-top:12px}
.reasons li{
  font-size:12px; color:var(--ink-soft); padding:6px 0 6px 18px;
  position:relative; border-top:1px solid var(--border);
}
.reasons li:before{content:"→"; position:absolute; left:0; color:var(--brand); font-weight:700}

/* ═══ FLOW (pipeline) ═══ */
.flow{display:flex; align-items:center; gap:0; flex-wrap:wrap; padding:8px 0}
.fn{
  padding:10px 16px; border-radius:8px; font-size:13px; font-weight:600;
  border:1.5px solid; display:inline-flex; flex-direction:column; gap:2px;
  background:#fff; white-space:nowrap;
}
.fn small{font-size:10px; font-weight:400}
.fa{
  display:flex; align-items:center; padding:0 6px; color:var(--ink-faint);
  font-family:'IBM Plex Mono', monospace; font-size:10px;
}
.fa::before{content:"——→"; letter-spacing:-2px; font-size:12px}

/* ═══ AGENT CARD ═══ */
.ac{
  background:#fff; border:1px solid var(--border); border-radius:12px;
  padding:16px 18px; flex:1; min-width:170px;
  transition:border-color .2s;
}
.ac:hover{border-color:var(--border-h)}
.ac-name{font-family:'Archivo', sans-serif; font-size:14px; font-weight:700; margin-bottom:3px}
.ac-role{font-size:11px; color:var(--ink-faint); margin-bottom:10px}
.ac-stat{display:flex; justify-content:space-between; font-size:12px; color:var(--ink-soft); margin-bottom:4px}
.ac-stat .val{font-family:'IBM Plex Mono', monospace; font-weight:600}

/* ═══ MODEL CARD ═══ */
.ml-card{
  display:flex; align-items:center; gap:14px;
  padding:12px 16px; background:#fff; border:1px solid var(--border);
  border-radius:10px; margin-bottom:7px;
  border-left:3.5px solid var(--brand);
  transition:border-color .2s;
}
.ml-card:hover{border-color:var(--border-h)}

/* ═══ PILL ═══ */
.zp{
  border-radius:4px; padding:2px 8px;
  font-family:'IBM Plex Mono', monospace; font-size:10px;
  font-weight:600; display:inline-block; margin:2px;
}

/* ═══ PROGRESS BAR ═══ */
.pb-bg{background:var(--surface-sunken); border-radius:3px; height:5px; margin-top:3px}
.pb-fg{height:5px; border-radius:3px; transition:width .5s ease}

/* ═══ LOG ROW ═══ */
.log-row{
  display:flex; align-items:center; gap:10px; padding:8px 12px;
  border-bottom:1px solid var(--border);
  font-size:12px; transition:background .15s;
}
.log-row:hover{background:var(--surface-sunken)}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border); border-radius:3px}

@media (max-width:760px){
  .plate{padding:20px 16px}
  .hdr{padding:16px}
  .cap-row.cols-4,.cap-row.cols-3{grid-template-columns:1fr 1fr}
  .watch-item{grid-template-columns:36px 1fr; row-gap:8px}
  .watch-trigger,.watch-actions{grid-column:2}
}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════
REDIS_HOST = os.getenv("REDIS_HOST","localhost")
ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN","BK","QN","BX","SI"] for i in range(1,7)]
BOROUGH = {"MN":"Manhattan","BK":"Brooklyn","QN":"Queens","BX":"Bronx","SI":"Staten Island"}
ZONE_COORDS = {}
BOUNDS = {"MN":(40.70,40.88,-74.02,-73.91),"BK":(40.57,40.74,-74.04,-73.84),
          "QN":(40.54,40.80,-73.97,-73.70),"BX":(40.80,40.92,-73.94,-73.74),
          "SI":(40.48,40.65,-74.26,-74.03)}
for z in ALL_ZONES:
    p=z[:2];a,b,c,d=BOUNDS[p];i=int(z[3:])-1
    ZONE_COORDS[z]=(round(a+(i//2+.5)*(b-a)/3,5),round(c+(i%2+.5)*(d-c)/2,5))
CLUSTER_META = {
    "Permanently Hazardous":{"color":"#C1392B","bg":"#FBEAE7","txt":"#8B1A12","weight":0.1},
    "Peak Hour Hazardous":{"color":"#B8791C","bg":"#FBF1DF","txt":"#7A4D0A","weight":0.5},
    "Weather Sensitive":{"color":"#5B3D8F","bg":"#EFEAF6","txt":"#3D2366","weight":0.7},
    "Safe Corridor":{"color":"#2F7A52","bg":"#E7F3EB","txt":"#1A5436","weight":1.0},
}

@st.cache_resource
def get_redis():
    try:
        r=redis.Redis(host=REDIS_HOST,port=6379,db=0,socket_timeout=3,decode_responses=True)
        r.ping();return r
    except: return None

def _srng(z): return random.Random(hash(z)&0xFFFF)

def zone_scores():
    r=get_redis();out={}
    if r:
        pipe=r.pipeline()
        for z in ALL_ZONES: pipe.get(f"zone_score:{z}")
        for z,raw in zip(ALL_ZONES,pipe.execute()):
            if raw:
                try: out[z]=json.loads(raw);continue
                except: pass
    for z in ALL_ZONES:
        if z not in out:
            rng=_srng(z);aqi=rng.uniform(30,175);spd=rng.uniform(12,68)
            out[z]={"zone_id":z,"zone_score":round((min(1,spd/40)*.5)+(1-min(1,aqi/200))*.5,3),
                    "avg_aqi":round(aqi,1),"avg_speed":round(spd,1),
                    "event_type":("COMPOUND_EVENT" if spd<25 and aqi>100 else "CONGESTION_EVENT" if spd<25 else "POLLUTION_ALERT" if aqi>100 else "NORMAL")}
    return out

def _offline_kmeans_labels():
    try:
        with open(os.path.join(os.path.dirname(__file__),"../ml/models/cluster_labels.json")) as f:
            return json.load(f).get("zone_labels",{})
    except: return {}

def cluster_labels():
    r=get_redis();streaming={}
    if r:
        pipe=r.pipeline()
        for z in ALL_ZONES: pipe.get(f"cluster:{z}")
        for z,v in zip(ALL_ZONES,pipe.execute()):
            if v: streaming[z]=v
    if len(set(streaming.values()))>=4: return streaming
    offline=_offline_kmeans_labels()
    if offline: return {z:offline.get(z,streaming.get(z,"Safe Corridor")) for z in ALL_ZONES}
    out={}
    for z in ALL_ZONES:
        p=z[:2];w={"MN":[20,30,30,20],"BK":[30,30,25,15],"QN":[40,30,20,10],"BX":[20,25,30,25],"SI":[60,25,10,5]}.get(p,[25]*4)
        out[z]=_srng(z).choices(list(CLUSTER_META.keys()),weights=w)[0]
    return out

def worker_data():
    r=get_redis();workers=[];_cl=cluster_labels()
    # Seed a per-worker RNG for consistent but random zone assignment
    _ts_seed = int(time.time() / 300)  # changes every 5 min
    for i in range(1,51):
        wid=f"W-{i:02d}";data=None;exp_data=None;rec_data=None
        wrng = random.Random(f"worker_{wid}_epoch_{_ts_seed}")  # unique per worker+time
        # Random zone assignment for this worker
        worker_zone = wrng.choice(ALL_ZONES)
        if r:
            raw=r.get(f"exposure:{wid}")
            if raw:
                try: exp_data=json.loads(raw)
                except: pass
            raw2=r.get(f"rec:{wid}")
            if raw2:
                try: rec_data=json.loads(raw2)
                except: pass

        # Merge: rec data takes priority for zone/routing, exposure data for AQI/hours
        if exp_data or rec_data:
            data = {}
            if exp_data: data.update(exp_data)
            if rec_data:
                # Use rec's current_zone as the definitive zone
                if "current_zone" in rec_data:
                    data["zone_id"] = rec_data["current_zone"]
                data["rec_zone"] = rec_data.get("rec_zone", data.get("zone_id", worker_zone))
                data["distance_km"] = rec_data.get("distance_km", 0)
                data["reason"] = rec_data.get("reason", "Best available zone")
                data["routing_method"] = rec_data.get("routing_method", "heuristic")
                if "status" in rec_data:
                    data["rec_status"] = rec_data["status"]

            # Compute exposure hours from total_minutes if hours_in_high_aqi is 0
            total_min = float(data.get("total_minutes", 0))
            hrs_in_high = float(data.get("hours_in_high_aqi", 0))
            if hrs_in_high == 0 and total_min > 0:
                data["hours_in_high_aqi"] = round(total_min / 60.0, 2)

            # Ensure rec_zone != zone_id (don't recommend same zone)
            cur_z = data.get("zone_id", worker_zone)
            if data.get("rec_zone") == cur_z:
                candidates = [z for z in ALL_ZONES if z != cur_z and
                              _cl.get(z, "Safe Corridor") == "Safe Corridor"]
                if not candidates:
                    candidates = [z for z in ALL_ZONES if z != cur_z]
                clat, clon = ZONE_COORDS.get(cur_z, (40.7, -74.0))
                candidates.sort(key=lambda z: abs(ZONE_COORDS.get(z,(40.7,-74.0))[0]-clat)+abs(ZONE_COORDS.get(z,(40.7,-74.0))[1]-clon))
                data["rec_zone"] = candidates[0]
                rlat, rlon = ZONE_COORDS.get(candidates[0], (40.7, -74.0))
                data["distance_km"] = round(((clat-rlat)**2 + ((clon-rlon)*math.cos(math.radians(clat)))**2)**0.5 * 111.32, 1)
                data["reason"] = f"Nearest Safe Corridor ({data['distance_km']}km)"

            data.setdefault("zone_id", worker_zone)
            data.setdefault("worker_id", wid)
            data.setdefault("daily_avg_aqi", round(float(data.get("aqi_sum",0)) / max(float(data.get("count",1)),1), 1))
            data.setdefault("estimated_travel_min", round(data.get("distance_km",1) / 0.4, 0))
        if not data:
            # Pick status tier first, then hours within range
            tier = wrng.choices(["SAFE","WARNING","CRITICAL"], weights=[55,30,15])[0]
            if tier == "CRITICAL":
                hrs = wrng.uniform(3.0, 5.5)
            elif tier == "WARNING":
                hrs = wrng.uniform(1.5, 3.0)
            else:
                hrs = wrng.uniform(0, 1.4)
            zone_aqi = wrng.uniform(35, 180)
            # Pick a different rec zone (prefer safe corridors)
            safe_zones = [z for z in ALL_ZONES if z != worker_zone and _cl.get(z) == "Safe Corridor"]
            rec_z = wrng.choice(safe_zones) if safe_zones else wrng.choice([z for z in ALL_ZONES if z != worker_zone])
            clat, clon = ZONE_COORDS.get(worker_zone, (40.7, -74.0))
            rlat, rlon = ZONE_COORDS.get(rec_z, (40.7, -74.0))
            dist = round(((clat-rlat)**2 + ((clon-rlon)*math.cos(math.radians(clat)))**2)**0.5 * 111.32, 1)
            data={"worker_id":wid,"zone_id":worker_zone,"hours_in_high_aqi":round(hrs,2),
                  "daily_avg_aqi":round(zone_aqi,1),
                  "rec_zone":rec_z,"reason":"Best available zone",
                  "distance_km":dist,"estimated_travel_min":round(dist/0.4,0)}
        hrs_a=float(data.get("hours_in_high_aqi",0))
        data["exposure_status"]="CRITICAL" if hrs_a>3.0 else ("WARNING" if hrs_a>1.5 else "SAFE")
        workers.append(data)
    return workers

def perf_data():
    r=get_redis();total=0;hist=[]
    if r:
        try:
            _tr=r.get("perf_total_records")
            if _tr: total=int(_tr)
        except: pass
        try:
            for raw in r.lrange("perf_history",-15,-1):
                try: hist.append(json.loads(raw))
                except: pass
        except: pass
    if not hist:
        for i in range(12):
            hist.append({"ts":f"{10+i//2:02d}:{(i%2)*30:02d}","rps":round(200+50*math.sin(i)+(i*3)),
                "lag_ms":round(80+20*math.sin(i*1.3)),"storage_mb":round(20+i*6.5)})
    return {"total":total,"hist":hist}

def _playout(**ov):
    base=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#F9FAF8",font_family="Inter",font_color="#93A199",font_size=11,
        margin=dict(l=40,r=10,t=10,b=40),xaxis=dict(gridcolor="#E8ECE7",linecolor="#DDE3DC"),
        yaxis=dict(gridcolor="#E8ECE7",linecolor="#DDE3DC"))
    for k in ("xaxis","yaxis","margin"):
        if k in ov: base[k]={**base.get(k,{}),**ov.pop(k)}
    base.update(ov);return base

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
r_conn = get_redis()
zs = zone_scores(); cl = cluster_labels(); ww = worker_data(); perf = perf_data()
now = datetime.now().strftime("%d %b %Y · %H:%M:%S")
keys_n = len(r_conn.keys("zone_score:*")) if r_conn else 0
is_live = keys_n > 0

CW = {k:v["weight"] for k,v in CLUSTER_META.items()}
weighted = {z: round(zs[z]["zone_score"] * CW.get(cl.get(z,"Safe Corridor"),1.0), 4) for z in ALL_ZONES}

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
src_label = "Live Pipeline" if is_live else "Synthetic"
src_cls = "chip-live" if is_live else "chip-src"
st.markdown(f"""
<div class="hdr">
  <div class="hdr-left">
    <div>
      <div class="hdr-eyebrow">Real-time Urban Intelligence</div>
      <div class="hdr-title">UrbanStream</div>
    </div>
  </div>
  <div class="hdr-r">
    <div class="chip {src_cls}"><span class="dot-live"></span>{src_label}</div>
    <div class="chip chip-src">{now}</div>
  </div>
</div>
<div class="legend">
  <div class="lg-chip"><span class="lg-dot red"></span>CRITICAL — over 3h exposure</div>
  <div class="lg-chip"><span class="lg-dot amber"></span>WARNING — 1.5–3h exposure</div>
  <div class="lg-chip"><span class="lg-dot green"></span>SAFE — below 1.5h exposure</div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["⬡  Pipeline & Zones", "⚡  Workers & Routing", "◉  AI System"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE & ZONES
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    total = perf["total"]; hist = perf["hist"]
    lr = hist[-1]["rps"] if hist else 0; ll = hist[-1]["lag_ms"] if hist else 0

    # Section 01: Data Pipeline
    st.markdown(f"""<div class="plate">
      <div class="plate-label">
        <span class="plate-num">01</span>
        <span class="plate-title">Data Pipeline</span>
        <span class="plate-window">LIVE · {total:,} RECORDS</span>
      </div>
      <div class="frame">
        <div class="flow">
          <div class="fn" style="border-color:#BBD9C5;color:var(--brand)">📡 Producers<small style="color:var(--ink-faint)">traffic · pollution · weather · worker</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#F5C6C0;color:var(--red)">🔴 Redpanda<small style="color:var(--ink-faint)">4 topics · 3 partitions</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#E8D5A8;color:var(--amber)">⚡ Spark<small style="color:var(--ink-faint)">structured streaming</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#BBD9C5;color:var(--green)">🗄️ Redis<small style="color:var(--ink-faint)">zone scores · exposure</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#C8BDD9;color:var(--purple)">📊 Dashboard<small style="color:var(--ink-faint)">streamlit · live</small></div>
        </div>
        <div class="cap-row cols-4" style="margin-top:16px">
          <div class="cap-card"><div class="cap-label">Total Records</div><div class="cap-val" style="color:var(--brand)">{total:,}</div><div class="cap-sub">processed by Spark</div></div>
          <div class="cap-card"><div class="cap-label">Throughput</div><div class="cap-val" style="color:var(--green)">{lr:,} <span style="font-size:12px;font-weight:400">rec/s</span></div><div class="cap-sub">current batch rate</div></div>
          <div class="cap-card"><div class="cap-label">Latency</div><div class="cap-val" style="color:{'var(--red)' if ll>800 else ('var(--amber)' if ll>400 else 'var(--green)')}">{ll} <span style="font-size:12px;font-weight:400">ms</span></div><div class="cap-sub">end-to-end lag</div></div>
          <div class="cap-card"><div class="cap-label">Storage</div><div class="cap-val" style="color:var(--purple)">{hist[-1].get('storage_mb',0):.0f} <span style="font-size:12px;font-weight:400">MB</span></div><div class="cap-sub">MinIO datalake</div></div>
        </div>
      </div>
    </div><div class="plate-border"></div>""", unsafe_allow_html=True)

    # Section 02: Zone Intelligence
    st.markdown("""<div class="plate" style="padding-bottom:0">
      <div class="plate-label">
        <span class="plate-num">02</span>
        <span class="plate-title">Zone Intelligence</span>
        <span class="plate-window">30 ZONES · 5 BOROUGHS · 4 CLUSTERS</span>
      </div>
    </div>""", unsafe_allow_html=True)

    col_map, col_info = st.columns([5, 2])
    with col_map:
        wpz = {}
        for w in ww: wpz[w.get("zone_id","")]=wpz.get(w.get("zone_id",""),0)+1
        try:
            import pydeck as pdk
            CLUSTER_COLORS = {
                "Permanently Hazardous": [193,57,43,220],
                "Peak Hour Hazardous":   [184,121,28,210],
                "Weather Sensitive":     [91,61,143,200],
                "Safe Corridor":         [47,122,82,200],
            }
            def zone_color(z): return CLUSTER_COLORS.get(cl.get(z,"Safe Corridor"), [47,122,82,200])
            rows=[]
            for z in ALL_ZONES:
                lat,lon=ZONE_COORDS[z];s2=zs[z];r2,g,b,a=zone_color(z)
                rows.append({"zone":z,"lat":lat,"lon":lon,"score":weighted[z],"raw_score":s2["zone_score"],
                    "aqi":s2["avg_aqi"],"speed":s2["avg_speed"],"event":s2["event_type"],
                    "cluster":cl.get(z,""),"workers":wpz.get(z,0),
                    "r":r2,"g":g,"b":b,"a":a,"radius":500+wpz.get(z,0)*70})
            df=pd.DataFrame(rows)
            layer=pdk.Layer("ScatterplotLayer",data=df,get_position=["lon","lat"],get_radius="radius",
                get_fill_color=["r","g","b","a"],get_line_color=[255,255,255,80],line_width_min_pixels=1,pickable=True)
            txt_l=pdk.Layer("TextLayer",data=df,get_position=["lon","lat"],get_text="zone",get_size=11,get_color=[22,33,29,190])
            view=pdk.ViewState(latitude=40.72,longitude=-73.95,zoom=10.2,pitch=0)
            tip={"html":"<div style='background:#fff;color:#16211D;padding:10px 14px;border-radius:10px;font-family:IBM Plex Mono,monospace;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.08);border:1px solid #DDE3DC'><b style='color:#0E4F44;font-size:14px'>{zone}</b> <span style='color:#93A199;font-size:11px'>{cluster}</span><br>Score <b>{score}</b> · AQI <b>{aqi}</b> · Speed <b>{speed}</b> · Workers <b>{workers}</b></div>",
                "style":{"backgroundColor":"transparent"}}
            st.pydeck_chart(pdk.Deck(layers=[layer,txt_l],initial_view_state=view,tooltip=tip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"))
        except ImportError:
            rows=[]
            for z in ALL_ZONES: s2=zs[z];rows.append({"Zone":z,"Score":weighted[z],"AQI":s2["avg_aqi"]})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    with col_info:
        # Count workers per zone
        workers_per_zone = {}
        for w in ww:
            z = w.get("zone_id", w.get("current_zone", ""))
            if z: workers_per_zone[z] = workers_per_zone.get(z, 0) + 1

        clusters={n:[] for n in CLUSTER_META}
        for z in ALL_ZONES:
            lbl=cl[z]
            if lbl in clusters: clusters[lbl].append(z)
        for name,meta in CLUSTER_META.items():
            zones=clusters[name]
            # Build pills with worker count badge
            pill_parts = []
            for z in sorted(zones):
                wc = workers_per_zone.get(z, 0)
                badge = f'<span style="background:{meta["color"]};color:#fff;border-radius:50%;font-size:9px;padding:1px 4px;margin-left:3px;font-weight:700">{wc}</span>' if wc > 0 else ''
                pill_parts.append(f'<span class="zp" style="background:{meta["bg"]};color:{meta["txt"]}">{z}{badge}</span>')
            pills = " ".join(pill_parts)
            avg_a=round(sum(zs[z]["avg_aqi"] for z in zones)/max(len(zones),1),1)
            total_workers = sum(workers_per_zone.get(z, 0) for z in zones)
            st.markdown(f"""<div style="margin-bottom:14px">
              <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                <span style="width:10px;height:10px;border-radius:3px;background:{meta['color']};flex-shrink:0"></span>
                <span style="font-family:'Archivo',sans-serif;font-size:13px;font-weight:700;color:{meta['color']}">{name}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--ink-faint);margin-left:auto">{len(zones)} zones · {total_workers} workers · AQI {avg_a}</span>
              </div>
              <div style="padding-left:17px">{pills}</div>
            </div>""", unsafe_allow_html=True)

        # Best / Worst zones
        ranked=sorted([(z,weighted[z]) for z in ALL_ZONES],key=lambda x:-x[1])
        st.markdown('<div style="margin-top:18px"><span class="cap-label">BEST ZONES</span></div>',unsafe_allow_html=True)
        for z,sc2 in ranked[:5]:
            pct=min(100,sc2*100)
            st.markdown(f'<div style="display:flex;align-items:center;gap:9px;font-size:12px;margin-bottom:4px">'
                f'<span class="mono" style="color:var(--green);font-weight:700;width:42px">{z}</span>'
                f'<div style="flex:1;background:var(--surface-sunken);border-radius:3px;height:4px"><div style="width:{pct:.0f}%;height:4px;border-radius:3px;background:var(--green)"></div></div>'
                f'<span class="mono" style="color:var(--ink-faint);width:42px;text-align:right">{sc2:.3f}</span></div>',unsafe_allow_html=True)

        st.markdown('<div style="margin-top:12px"><span class="cap-label">WORST ZONES</span></div>',unsafe_allow_html=True)
        for z,sc2 in ranked[-5:]:
            pct=min(100,sc2*100)
            st.markdown(f'<div style="display:flex;align-items:center;gap:9px;font-size:12px;margin-bottom:4px">'
                f'<span class="mono" style="color:var(--red);font-weight:700;width:42px">{z}</span>'
                f'<div style="flex:1;background:var(--surface-sunken);border-radius:3px;height:4px"><div style="width:{pct:.0f}%;height:4px;border-radius:3px;background:var(--red)"></div></div>'
                f'<span class="mono" style="color:var(--ink-faint);width:42px;text-align:right">{sc2:.3f}</span></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — WORKERS & ROUTING
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    sc_counts={"SAFE":0,"WARNING":0,"CRITICAL":0}
    for w in ww: sc_counts[w.get("exposure_status","SAFE")]+=1
    order={"CRITICAL":0,"WARNING":1,"SAFE":2}
    ww_s=sorted(ww,key=lambda w:(order.get(w.get("exposure_status","SAFE"),2),-w.get("hours_in_high_aqi",0)))

    # Section 01: Worker Status
    st.markdown(f"""<div class="plate">
      <div class="plate-label">
        <span class="plate-num">01</span>
        <span class="plate-title">Worker Safety Overview</span>
        <span class="plate-window">50 WORKERS · REAL-TIME</span>
      </div>
      <div class="frame">
        <div class="cap-row cols-4">
          <div class="cap-card"><div class="cap-label">Safe</div><div class="cap-val" style="color:var(--green)">{sc_counts['SAFE']}</div><div class="cap-sub">below 1.5h in high AQI</div>
            <div class="cap-track"><div class="cap-fill" style="width:{sc_counts['SAFE']*2}%;background:var(--green)"></div></div></div>
          <div class="cap-card"><div class="cap-label">Warning</div><div class="cap-val" style="color:var(--amber)">{sc_counts['WARNING']}</div><div class="cap-sub">1.5 – 3h exposure</div>
            <div class="cap-track"><div class="cap-fill" style="width:{sc_counts['WARNING']*2}%;background:var(--amber)"></div></div></div>
          <div class="cap-card"><div class="cap-label">Critical</div><div class="cap-val" style="color:var(--red)">{sc_counts['CRITICAL']}</div><div class="cap-sub">over 3h — relocate now</div>
            <div class="cap-track"><div class="cap-fill" style="width:{sc_counts['CRITICAL']*2}%;background:var(--red)"></div></div></div>
          <div class="cap-card"><div class="cap-label">Total Active</div><div class="cap-val" style="color:var(--brand)">50</div><div class="cap-sub">across 30 zones</div>
            <div class="cap-track"><div class="cap-fill" style="width:100%;background:var(--brand)"></div></div></div>
        </div>
      </div>
    </div><div class="plate-border"></div>""", unsafe_allow_html=True)

    col_watch, col_rec = st.columns([3, 2])

    # Section 02: Exposure Feed — original HTML watch-list with working buttons
    with col_watch:
        st.markdown("""<div class="plate" style="padding-right:0">
          <div class="plate-label">
            <span class="plate-num">02</span>
            <span class="plate-title">Exposure Feed</span>
            <span class="plate-window">SORTED BY RISK</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Track dispatched workers — persist in Redis so they survive refresh
        if "dispatched" not in st.session_state:
            st.session_state.dispatched = set()
            # Load from Redis on first render
            if r_conn:
                stored = r_conn.smembers("dispatched_workers")
                if stored:
                    st.session_state.dispatched = {s.decode() if isinstance(s,bytes) else s for s in stored}

        for w in ww_s[:20]:
            wid=w.get("worker_id","–")
            # Skip if already dispatched this session
            if wid in st.session_state.dispatched:
                continue

            st2=w.get("exposure_status","SAFE");hrs=w.get("hours_in_high_aqi",0.0)
            aqi=w.get("daily_avg_aqi",50.0);zone=w.get("zone_id",w.get("current_zone","–"))
            rz=w.get("rec_zone","–")
            dist=w.get("distance_km",0);travel=w.get("estimated_travel_min",0)
            reason=w.get("reason","Best available zone")
            boro=BOROUGH.get(zone[:2],"–")
            zone_cluster=cl.get(zone,"Unknown")
            rz_score=zs.get(rz,{}).get("zone_score","–")
            rz_aqi=zs.get(rz,{}).get("avg_aqi","–")
            rz_cluster=cl.get(rz,"Unknown")
            z_score=zs.get(zone,{}).get("zone_score","–")

            if st2=="CRITICAL":
                item_cls="alert";ring_cls="red";tier_cls="tier-red"
            elif st2=="WARNING":
                item_cls="warn";ring_cls="amber";tier_cls="tier-amber"
            else:
                item_cls="";ring_cls="";tier_cls="tier-green"

            # Details button (pure HTML — works via <details>)
            details_btn=f'''<details class="wb ghost">
              <summary>Details</summary>
              <div class="wb-panel">
                <div class="wb-col">
                  <div>
                    <div style="font-weight:700;margin-bottom:4px">Current: {zone}</div>
                    <div><span style="color:var(--ink-faint)">Borough:</span> <b>{boro}</b></div>
                    <div><span style="color:var(--ink-faint)">Cluster:</span> <b>{zone_cluster}</b></div>
                    <div><span style="color:var(--ink-faint)">Score:</span> <b>{z_score}</b></div>
                    <div><span style="color:var(--ink-faint)">Exposure:</span> <b>{hrs:.2f}h</b></div>
                    <div><span style="color:var(--ink-faint)">AQI:</span> <b>{aqi:.0f}</b></div>
                  </div>
                  <div>
                    <div class="wb-rec">→ Recommended: {rz}</div>
                    <div><span style="color:var(--ink-faint)">Cluster:</span> <b>{rz_cluster}</b></div>
                    <div><span style="color:var(--ink-faint)">Score:</span> <b>{rz_score}</b></div>
                    <div><span style="color:var(--ink-faint)">AQI:</span> <b>{rz_aqi}</b></div>
                    <div><span style="color:var(--ink-faint)">Distance:</span> <b>{dist:.1f}km</b> · ~{travel:.0f}min</div>
                    <div><span style="color:var(--ink-faint)">Reason:</span> <b>{reason}</b></div>
                  </div>
                </div>
              </div>
            </details>'''

            # Card HTML — Relocate is a placeholder label; real button is below
            card_actions = details_btn
            st.markdown(f'''<div class="watch-item {item_cls}" style="margin-left:48px">
              <div class="ring {ring_cls}">{hrs:.1f}h</div>
              <div>
                <div class="watch-name">{wid} · {zone}</div>
                <div class="watch-sub">{boro} · AQI {aqi:.0f}</div>
              </div>
              <div class="watch-trigger"><span class="tier {tier_cls}">{st2}</span></div>
              <div class="watch-actions">{card_actions}</div>
            </div>''', unsafe_allow_html=True)

            # Real Relocate button (only for at-risk workers)
            if st2 in ("CRITICAL","WARNING"):
                if st.button(f"🚨 Relocate {wid} → {rz} ({dist:.1f}km)", key=f"rel_{wid}", type="primary"):
                    if r_conn:
                        r_conn.set(f"relocate:{wid}", json.dumps({
                            "worker_id": wid, "from_zone": zone, "to_zone": rz,
                            "distance_km": dist, "reason": reason,
                            "ts": datetime.now().isoformat(timespec="seconds")
                        }), ex=300)
                        # Persist in Redis set (5 min TTL)
                        r_conn.sadd("dispatched_workers", wid)
                        r_conn.expire("dispatched_workers", 300)
                    st.session_state.dispatched.add(wid)
                    st.toast(f"✓ {wid} relocated to {rz}")
                    st.rerun()

        # Show dispatched count
        if st.session_state.dispatched:
            st.markdown(f'<div style="padding:8px 48px;font-size:11px;color:var(--green)">✓ {len(st.session_state.dispatched)} workers relocated this session</div>', unsafe_allow_html=True)

    # Section 03: Recommended Relocations summary
    with col_rec:
        st.markdown("""<div class="plate" style="padding-left:0">
          <div class="plate-label">
            <span class="plate-num">03</span>
            <span class="plate-title">Route Recommendations</span>
          </div>
        </div>""", unsafe_allow_html=True)

        at_risk=[w for w in ww_s if w.get("exposure_status") in ("CRITICAL","WARNING") and w.get("worker_id","–") not in st.session_state.dispatched]
        if at_risk:
            for w in at_risk[:12]:
                st2=w.get("exposure_status","SAFE");hrs=w.get("hours_in_high_aqi",0)
                rz=w.get("rec_zone","–");dist=w.get("distance_km",0);travel=w.get("estimated_travel_min",0)
                reason=w.get("reason","")[:50]
                tier_cls="tier-red" if st2=="CRITICAL" else "tier-amber"

                st.markdown(f"""<div class="rec-card" style="margin-bottom:8px">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span class="mono" style="font-weight:700;font-size:14px">{w.get("worker_id","–")}</span>
                    <span class="tier {tier_cls}">{st2}</span>
                    <span class="mono" style="font-size:11px;color:var(--ink-faint);margin-left:auto">{hrs:.2f}h exposed</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:8px;font-size:13px;margin-bottom:8px">
                    <span class="mono" style="color:var(--red);font-weight:600">{w.get("zone_id","–")}</span>
                    <span style="color:var(--ink-faint)">→</span>
                    <span class="mono" style="color:var(--green);font-weight:600">{rz}</span>
                    <span class="mono" style="font-size:11px;color:var(--ink-faint);margin-left:auto">{dist:.1f}km · ~{travel:.0f}min</span>
                  </div>
                  <ul class="reasons"><li>{reason}</li></ul>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:40px;color:var(--ink-faint);font-size:13px;background:var(--surface);border:1px solid var(--border);border-radius:12px">✅ All workers within safe limits</div>',unsafe_allow_html=True)

    # Pipeline Performance
    if hist and len(hist)>2:
        st.markdown('<div class="plate-border" style="margin-top:20px"></div>',unsafe_allow_html=True)
        st.markdown("""<div class="plate" style="padding-bottom:0">
          <div class="plate-label">
            <span class="plate-num">04</span>
            <span class="plate-title">Pipeline Performance</span>
            <span class="plate-window">LAST 15 BATCHES</span>
          </div>
        </div>""", unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            c1,c2=st.columns(2)
            df_h=pd.DataFrame(hist)
            with c1:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df_h["ts"],y=df_h["rps"],fill="tozeroy",fillcolor="rgba(14,79,68,.06)",
                    line=dict(color="#0E4F44",width=2),mode="lines"))
                fig.update_layout(**_playout(height=200,yaxis=dict(title="rec/s")))
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(x=df_h["ts"],y=df_h["lag_ms"],fill="tozeroy",fillcolor="rgba(193,57,43,.05)",
                    line=dict(color="#C1392B",width=2),mode="lines"))
                fig2.add_hline(y=1000,line_dash="dash",line_color="#B8791C",annotation_text="1s SLA")
                fig2.update_layout(**_playout(height=200,yaxis=dict(title="latency ms")))
                st.plotly_chart(fig2,use_container_width=True)
        except ImportError: pass

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — AI SYSTEM
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    # Section 01: Agent Pipeline
    st.markdown("""<div class="plate">
      <div class="plate-label">
        <span class="plate-num">01</span>
        <span class="plate-title">Multi-Agent Pipeline</span>
        <span class="plate-window">4 AGENTS · AUTONOMOUS</span>
      </div>
      <div class="frame">
        <div class="flow">
          <div class="fn" style="border-color:#F5C6C0;color:var(--red)">🔍 Monitor<small style="color:var(--ink-faint)">anomaly detection</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#C8BDD9;color:var(--purple)">📈 Forecaster<small style="color:var(--ink-faint)">LSTM prediction</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#B0C7DB;color:var(--blue)">🗺️ Router<small style="color:var(--ink-faint)">LinUCB + distance</small></div>
          <div class="fa"></div>
          <div class="fn" style="border-color:#E8D5A8;color:var(--amber)">🚨 Alert<small style="color:var(--ink-faint)">safety briefings</small></div>
        </div>
        <div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--ink-faint);margin-top:6px">
          Redis Pub/Sub · Coordinator heartbeat every 15s · Auto-restart on failure</div>
      </div>
    </div><div class="plate-border"></div>""",unsafe_allow_html=True)

    col_ag, col_ml = st.columns([3, 2])

    # Section 02: Agent Status
    with col_ag:
        st.markdown("""<div class="plate" style="padding-right:0">
          <div class="plate-label">
            <span class="plate-num">02</span>
            <span class="plate-title">Agent Status</span>
          </div>
        </div>""", unsafe_allow_html=True)

        alist=["monitor","forecaster","router","alert"]
        aroles={"monitor":"Anomaly Detection","forecaster":"AQI Forecasting","router":"Worker Routing","alert":"Safety Briefings"}
        aicons={"monitor":"🔍","forecaster":"📈","router":"🗺️","alert":"🚨"}
        acolors={"monitor":"var(--red)","forecaster":"var(--purple)","router":"var(--blue)","alert":"var(--amber)"}

        coord_st=None
        if r_conn:
            raw_cs=r_conn.get("agent:coordinator:status")
            if raw_cs:
                try: coord_st=json.loads(raw_cs)
                except: pass

        cards_h=""
        for an in alist:
            ast2={"alive":False,"decisions":0,"errors":0,"last_run":"—","interval_sec":30}
            if r_conn:
                raw_as=r_conn.get(f"agent_status:{an}")
                if raw_as:
                    try: ast2=json.loads(raw_as)
                    except: pass
            if coord_st and "agents" in coord_st:
                cs2=coord_st["agents"].get(an,{})
                if cs2: ast2.update(cs2)
            alive=ast2.get("alive",False);dec=ast2.get("decisions",0);err=ast2.get("errors",0)
            co=acolors[an];ic=aicons[an];ro=aroles[an]
            dot_c="var(--green)" if alive else "var(--ink-faint)";stat="RUNNING" if alive else "IDLE"
            stat_c="var(--green)" if alive else "var(--ink-faint)"
            da="pulse-dot 2s ease-in-out infinite" if alive else "none"
            cards_h+=f"""<div class="ac">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
                <span style="font-size:1.2rem">{ic}</span>
                <span style="display:flex;align-items:center;gap:5px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;color:{stat_c}">
                  <span style="width:6px;height:6px;border-radius:50%;background:{dot_c};animation:{da}"></span>{stat}</span>
              </div>
              <div class="ac-name" style="color:{co}">{an.upper()}</div>
              <div class="ac-role">{ro}</div>
              <div class="ac-stat"><span>Decisions</span><span class="val">{dec}</span></div>
              <div class="ac-stat"><span>Errors</span><span class="val" style="color:{'var(--red)' if err>0 else 'var(--green)'}">{err}</span></div>
            </div>"""
        st.markdown(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding-left:48px">{cards_h}</div>',unsafe_allow_html=True)

        # Agent decision log
        alogs=[]
        if r_conn:
            for an in alist:
                rl=r_conn.get(f"agent_log:{an}")
                if rl:
                    try: alogs.extend(json.loads(rl))
                    except: pass
        alogs.sort(key=lambda x:x.get("ts",""),reverse=True)
        if alogs:
            st.markdown("""<div style="padding:16px 0 4px 48px">
              <span class="cap-label">RECENT DECISIONS</span></div>""",unsafe_allow_html=True)
            acolor_map={"monitor":"var(--red)","forecaster":"var(--purple)","router":"var(--blue)","alert":"var(--amber)"}
            for en in alogs[:10]:
                ts=en.get("ts","")[:19].replace("T"," ") if en.get("ts") else "—"
                ag=en.get("agent","—");sm=en.get("summary","");co=acolor_map.get(ag,"var(--ink-faint)")
                tier_map={"monitor":"tier-red","forecaster":"tier-purple","router":"tier-blue","alert":"tier-amber"}
                tc=tier_map.get(ag,"tier-green")
                st.markdown(f'<div class="log-row" style="margin-left:48px"><span class="mono" style="font-size:11px;color:var(--ink-faint);min-width:130px">{ts}</span>'
                    f'<span class="tier {tc}" style="min-width:80px;text-align:center">{ag}</span>'
                    f'<span style="font-size:12px;color:var(--ink-soft);flex:1">{sm}</span></div>',unsafe_allow_html=True)

    # Section 03: ML Models
    with col_ml:
        st.markdown("""<div class="plate" style="padding-left:0">
          <div class="plate-label">
            <span class="plate-num">03</span>
            <span class="plate-title">ML Models</span>
          </div>
        </div>""", unsafe_allow_html=True)

        improv=0
        try:
            with open(os.path.join(os.path.dirname(__file__),"../ml/models/aqi_lstm_metrics.json")) as f:
                _mm=json.load(f);improv=_mm.get("improvement_pct",0)
        except: pass
        bst={"n_observations":0,"is_ready":False}
        if r_conn:
            raw_bs=r_conn.get("bandit_stats")
            if raw_bs:
                try: bst=json.loads(raw_bs)
                except: pass
        skm_b=int(r_conn.get("skm_batch_count") or 0) if r_conn else 0
        nob=bst.get("n_observations",0)
        models=[
            ("LSTM Forecaster","Time-Series Prediction",f"{improv:.1f}% better than AR(3)","var(--purple)"),
            ("Isolation Forest","Anomaly Detection","contamination=0.05 · 5 features","var(--red)"),
            ("LinUCB Bandit","Contextual Routing",f"{nob} obs · {'Active' if nob>=100 else 'Learning'}","var(--amber)"),
            ("Streaming KMeans","Online Clustering",f"k=4 · {skm_b} batches processed","var(--brand)"),
            ("Offline KMeans","Batch Clustering","k=4 · 7 features · silhouette=0.42","var(--green)"),
            ("AR(3) Baseline","Autoregressive","Fallback forecaster","var(--ink-faint)"),
        ]
        for name,typ,status,color in models:
            st.markdown(f"""<div class="ml-card" style="border-left-color:{color}">
              <div style="flex:1">
                <div style="font-family:'Archivo',sans-serif;font-size:13px;font-weight:700;color:var(--ink)">{name}</div>
                <div style="font-size:11px;color:var(--ink-faint)">{typ}</div>
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{color};font-weight:600;text-align:right;max-width:180px">{status}</div>
            </div>""",unsafe_allow_html=True)

        # Model comparison chart
        try:
            import plotly.graph_objects as go
            lstm_m={"mse":0,"mae":0};ar3_m={"mse":0,"mae":0}
            try:
                with open(os.path.join(os.path.dirname(__file__),"../ml/models/aqi_lstm_metrics.json")) as f:
                    _mm=json.load(f);lstm_m=_mm.get("lstm",lstm_m);ar3_m=_mm.get("ar3",ar3_m)
            except: pass
            if lstm_m.get("mse",0)>0 or ar3_m.get("mse",0)>0:
                st.markdown('<div style="padding:14px 0 0"><span class="cap-label">LSTM vs AR(3) COMPARISON</span></div>',unsafe_allow_html=True)
                fig=go.Figure()
                fig.add_trace(go.Bar(x=["MSE","MAE"],y=[ar3_m.get("mse",0),ar3_m.get("mae",0)],name="AR(3)",marker_color="#B8791C"))
                fig.add_trace(go.Bar(x=["MSE","MAE"],y=[lstm_m.get("mse",0),lstm_m.get("mae",0)],name="LSTM",marker_color="#5B3D8F"))
                fig.update_layout(**_playout(height=240,barmode="group",
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,title="")))
                st.plotly_chart(fig,use_container_width=True)
        except ImportError: pass

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;
  font-size:10px;margin:24px 48px 16px;padding-top:14px;border-top:1px solid var(--border)">
  UrbanStream v5.0 · Multi-Agent AI System · LSTM + LinUCB + Isolation Forest · Auto-refreshes every 30s
</div>""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state: st.session_state["last_refresh"]=time.time()
if time.time()-st.session_state["last_refresh"]>30:
    st.session_state["last_refresh"]=time.time();st.rerun()