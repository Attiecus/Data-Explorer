"""
Data Explorer — Adaptive Analytics Dashboard (v2)
Run with: streamlit run data_explorer.py
Requires: pip install streamlit pandas openpyxl plotly scipy scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
import io, re
from datetime import datetime

st.set_page_config(page_title="Data Explorer", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#f0ede8;}
.block-container{padding-top:1.2rem!important;}
.banner{background:#1a1a2e;padding:18px 28px;border-radius:14px;margin-bottom:18px;display:flex;align-items:center;gap:16px;}
.banner h1{margin:0;font-size:21px;color:#fff;font-weight:600;}
.banner p{margin:3px 0 0;font-size:13px;color:#8888aa;}
.banner .badge{background:rgba(201,168,76,0.18);color:#c9a84c;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;border:1px solid rgba(201,168,76,0.35);margin-left:auto;}
.kpi-card{background:#fff;border-radius:12px;padding:16px 20px;border:1px solid #e5e0d8;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#c9a84c,#8b5e2a);}
.kpi-label{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;}
.kpi-value{font-size:26px;font-weight:600;color:#1a1a2e;margin:4px 0 2px;line-height:1;}
.kpi-delta{font-size:12px;color:#666;}
.insight{background:#fff;border-radius:10px;padding:12px 16px 12px 20px;border:1px solid #e5e0d8;border-left:4px solid #c9a84c;font-size:13px;color:#333;line-height:1.55;margin-bottom:8px;}
.insight .tag{display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:1px 6px;border-radius:3px;margin-right:6px;}
.tag-warn{background:#fef3cd;color:#856404;}.tag-info{background:#d1ecf1;color:#0c5460;}.tag-good{background:#d4edda;color:#155724;}.tag-bad{background:#f8d7da;color:#721c24;}
.section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;color:#999;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.section-title::after{content:'';flex:1;height:1px;background:#e5e0d8;}
.col-pill{display:inline-flex;align-items:center;gap:5px;background:#fff;border:1px solid #e5e0d8;border-radius:20px;padding:3px 10px;font-size:12px;color:#444;margin:2px;}
.col-pill .dot{width:8px;height:8px;border-radius:50%;}
.dot-num{background:#c9a84c;}.dot-cat{background:#4a6fa5;}.dot-date{background:#5a8a6a;}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0ede8;font-size:13px;}
.stat-row span:first-child{color:#888;}.stat-row span:last-child{font-weight:500;color:#1a1a2e;}
.compute-card{background:#fff;border-radius:12px;padding:16px 18px;border:1px solid #e5e0d8;margin-bottom:10px;}
.seg-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;}
[data-testid="stSidebar"]{background:#1a1a2e!important;}
[data-testid="stSidebar"] *{color:#e8e4dc!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#c9a84c!important;}
[data-testid="stSidebar"] .stMarkdown p{color:#aaa!important;font-size:12px!important;}
section[data-testid="stSidebar"] .stButton>button{background:rgba(201,168,76,0.15)!important;color:#c9a84c!important;border:1px solid rgba(201,168,76,0.3)!important;width:100%;border-radius:8px!important;font-size:12px!important;margin-bottom:2px;}
.stTabs [data-baseweb="tab-list"]{gap:0;background:#e5e0d8;border-radius:10px;padding:3px;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;font-size:12px!important;font-weight:500!important;padding:6px 14px!important;color:#666!important;}
.stTabs [aria-selected="true"]{background:#1a1a2e!important;color:#c9a84c!important;}
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PAL     = ["#1a1a2e","#c9a84c","#4a6fa5","#e07b54","#5a8a6a","#9b6b9b","#e8c97a","#7a9fc0","#d4a5a5","#a5d4c8"]
PAL_SEQ = [[0,"#f0ede8"],[0.5,"#c9a84c"],[1,"#1a1a2e"]]
CLUSTER_COLORS = ["#c9a84c","#4a6fa5","#e07b54","#5a8a6a","#9b6b9b"]

def plo(fig, title="", height=380):
    fig.update_layout(plot_bgcolor="#fafaf8",paper_bgcolor="#fafaf8",font_family="DM Sans",font_color="#333",
        title=dict(text=title,font_size=14,font_color="#1a1a2e",x=0),
        margin=dict(t=44 if title else 20,b=20,l=10,r=10),height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)",borderwidth=0,font_size=12),
        xaxis=dict(gridcolor="#eeebe5",linecolor="#ddd"),yaxis=dict(gridcolor="#eeebe5",linecolor="#ddd"))
    return fig

def stat_row(label, val):
    st.markdown(f'<div class="stat-row"><span>{label}</span><span>{val}</span></div>', unsafe_allow_html=True)

for k,v in [("sheets",{}),("file_name",None),("active_sheet",None),("computed_cols",{})]:
    if k not in st.session_state: st.session_state[k]=v

# ── Profiler ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def profile_df(key):
    df = st.session_state.sheets[key].copy()
    for col in df.columns:
        if re.search(r'date|time|dt|year|month|_at|_on',col,re.I):
            try: df[col]=pd.to_datetime(df[col],errors="coerce")
            except: pass
    p={}
    p["nulls"]      = df.isnull().sum().to_dict()
    p["null_pct"]   = (df.isnull().sum()/max(len(df),1)*100).round(1).to_dict()
    p["nunique"]    = df.nunique().to_dict()
    p["dups"]       = int(df.duplicated().sum())
    p["num_cols"]   = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    p["cat_cols"]   = [c for c in df.columns if df[c].dtype==object or str(df[c].dtype)=="category"]
    p["date_cols"]  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    p["quality_score"] = round(max(0,100-(df.isnull().sum().sum()/max(df.size,1)*100)-min(p["dups"]/max(len(df),1)*20,20)),1)
    p["num_stats"]={}
    for c in p["num_cols"]:
        s=df[c].dropna()
        if len(s)>1:
            q1,q3=s.quantile(0.25),s.quantile(0.75); iqr=q3-q1
            out=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
            try: norm_p=float(stats.shapiro(s.sample(min(len(s),5000),random_state=0))[1])
            except: norm_p=None
            p["num_stats"][c]={"min":round(float(s.min()),3),"max":round(float(s.max()),3),
                "mean":round(float(s.mean()),3),"median":round(float(s.median()),3),
                "std":round(float(s.std()),3),"sum":round(float(s.sum()),3),
                "skew":round(float(s.skew()),2),"kurt":round(float(s.kurtosis()),2),
                "p25":round(float(s.quantile(0.25)),3),"p75":round(float(s.quantile(0.75)),3),
                "p95":round(float(s.quantile(0.95)),3),
                "outliers":out,"outlier_pct":round(out/len(s)*100,1),
                "norm_p":round(norm_p,4) if norm_p else None}
    p["cat_stats"]={}
    for c in p["cat_cols"]:
        vc=df[c].value_counts()
        p["cat_stats"][c]={"top":str(vc.index[0]) if len(vc)>0 else "","top_pct":round(vc.iloc[0]/max(len(df),1)*100,1) if len(vc)>0 else 0,"nunique":int(df[c].nunique())}
    insights=[]
    high_null=[(c,v) for c,v in p["null_pct"].items() if v>20]
    if high_null:
        worst=max(high_null,key=lambda x:x[1])
        insights.append(("warn",f"<b>{worst[0]}</b> is {worst[1]}% missing — may need imputation"))
    if p["dups"]>0: insights.append(("bad",f"<b>{p['dups']:,} duplicate rows</b> ({round(p['dups']/max(len(df),1)*100,1)}%)"))
    for c,st_ in p["num_stats"].items():
        if st_["outlier_pct"]>10: insights.append(("warn",f"<b>{c}</b> has {st_['outlier_pct']}% outliers (IQR)"))
        if abs(st_["skew"])>2: insights.append(("info",f"<b>{c}</b> {'right' if st_['skew']>0 else 'left'}-skewed (skew={st_['skew']}) — consider log transform"))
        if st_["norm_p"] and st_["norm_p"]>0.05: insights.append(("good",f"<b>{c}</b> passes normality test (Shapiro p={st_['norm_p']})"))
    for c,st_ in p["cat_stats"].items():
        if st_["top_pct"]>80: insights.append(("info",f"<b>{st_['top']}</b> dominates <b>{c}</b> at {st_['top_pct']}%"))
        if st_["nunique"]==len(df): insights.append(("info",f"<b>{c}</b> looks like a unique ID column"))
    if p["date_cols"]: insights.append(("good",f"Time-series detected in <b>{p['date_cols'][0]}</b>"))
    if p["quality_score"]>=90: insights.append(("good",f"Quality score <b>{p['quality_score']}%</b> — clean dataset"))
    p["insights"]=insights[:8]
    p["df_json"]=df.to_json(orient="split",date_format="iso")
    return p

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✦ Data Explorer v2")
    st.markdown("Upload Excel or CSV — fully adaptive.")
    st.divider()
    uploaded=st.file_uploader("Upload file",type=["xlsx","xls","csv"],label_visibility="collapsed")
    if uploaded:
        if uploaded.name.lower().endswith(".csv"): raw={"Data":pd.read_csv(uploaded)}
        else: raw=pd.read_excel(uploaded,sheet_name=None,engine="openpyxl")
        st.session_state.sheets=raw; st.session_state.file_name=uploaded.name
        st.session_state.computed_cols={}
        if st.session_state.active_sheet not in raw: st.session_state.active_sheet=list(raw.keys())[0]
        profile_df.clear()
    if st.session_state.sheets:
        st.markdown("### Sheets")
        for name,df in st.session_state.sheets.items():
            m="✦  " if name==st.session_state.active_sheet else "      "
            if st.button(f"{m}{name}  ({len(df):,}×{len(df.columns)})",key=f"s_{name}"):
                st.session_state.active_sheet=name; st.rerun()

if not st.session_state.sheets:
    st.markdown("""<div class="banner"><div style="width:42px;height:42px;background:linear-gradient(135deg,#c9a84c,#8b5e2a);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;color:#1a1a2e;">✦</div><div><h1>Data Explorer v2</h1><p>Upload any Excel or CSV — no API key needed</p></div></div>""",unsafe_allow_html=True)
    for icon,title,desc in [("🧠","Smart Dashboard","Auto-KPIs, smart chart selection, insights"),("🔬","Quality Audit","Nulls, outliers, skew, normality tests"),("🎨","Chart Studio","20+ chart types: treemap, sankey, radar, 3D scatter, funnel..."),("🔢","Compute","Create columns, bin, rank, normalize, rolling stats"),("🧩","Segment","K-means clustering, PCA, top/bottom N, cohort analysis"),("📈","Relationships","Correlation, scatter matrix, regression, partial corr"),("🕐","Time Series","Trend, seasonality, rolling averages, YoY comparison"),("🔍","Filter & Export","Multi-filter, pivot export, Excel & CSV download")]:
        pass
    cols=st.columns(4)
    items=[("🧠","Smart Dashboard","Auto-KPIs, smart chart selection, insights"),("🎨","Chart Studio","20+ chart types: treemap, radar, funnel, sankey, 3D..."),("🔢","Compute","Create columns, bin, rank, normalize, rolling stats"),("🧩","Segment","K-means, PCA, top/bottom N, cohort analysis")]
    for i,(icon,title,desc) in enumerate(items):
        with cols[i]: st.markdown(f'<div style="background:#fff;border-radius:12px;padding:18px;border:1px solid #e5e0d8;"><div style="font-size:22px;margin-bottom:8px;">{icon}</div><div style="font-weight:600;font-size:14px;color:#1a1a2e;">{title}</div><div style="font-size:12px;color:#888;margin-top:4px;">{desc}</div></div>',unsafe_allow_html=True)
    st.stop()

# ── Load & profile ────────────────────────────────────────────────
sname=st.session_state.active_sheet
with st.spinner("Profiling data..."):
    P=profile_df(sname)

df_raw=pd.read_json(io.StringIO(P["df_json"]),orient="split")
for col in df_raw.columns:
    if re.search(r'date|time|dt|year|month|_at|_on',col,re.I):
        try: df_raw[col]=pd.to_datetime(df_raw[col],errors="coerce")
        except: pass

# Merge computed cols
for cname,cdata in st.session_state.computed_cols.items():
    df_raw[cname]=cdata

num_cols=[c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
cat_cols=[c for c in df_raw.columns if df_raw[c].dtype==object or str(df_raw[c].dtype)=="category"]
date_cols=[c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]

# ── Banner ────────────────────────────────────────────────────────
pills="".join([f'<span class="col-pill"><span class="dot dot-num"></span>{c}</span>' for c in num_cols[:4]]+[f'<span class="col-pill"><span class="dot dot-cat"></span>{c}</span>' for c in cat_cols[:3]]+([f'<span class="col-pill"><span class="dot dot-date"></span>{date_cols[0]}</span>'] if date_cols else []))
if len(df_raw.columns)>7: pills+=f'<span style="font-size:12px;color:#aaa;padding:4px;">+{len(df_raw.columns)-7} more</span>'
q=P["quality_score"]; qc="#3a8a5a" if q>=90 else "#c9a84c" if q>=70 else "#c0392b"
st.markdown(f"""<div class="banner"><div style="width:42px;height:42px;background:linear-gradient(135deg,#c9a84c,#8b5e2a);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;color:#1a1a2e;">✦</div><div><h1>{st.session_state.file_name} · {sname}</h1><p>{len(df_raw):,} rows · {len(df_raw.columns)} cols · {len(num_cols)} numeric · {len(cat_cols)} categorical{f" · {len(date_cols)} date" if date_cols else ""}</p></div><div class="badge" style="color:{qc};border-color:{qc}40;background:{qc}18;">Quality {q}%</div></div><div style="margin-bottom:14px;">{pills}</div>""",unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab_names=["🧠 Dashboard","🔬 Quality","📊 Distributions","🎨 Chart Studio","🔢 Compute","🧩 Segment","📈 Relationships","🔍 Filter & Export","🔮 Embeddings","🚨 Smart Alerts"]
if date_cols: tab_names.insert(4,"🕐 Time Series")
tabs=st.tabs(tab_names)
idx=0
t_dash=tabs[idx]; idx+=1
t_qual=tabs[idx]; idx+=1
t_dist=tabs[idx]; idx+=1
t_chart=tabs[idx]; idx+=1
t_time=tabs[idx] if date_cols else None
if date_cols: idx+=1
t_comp=tabs[idx]; idx+=1
t_seg=tabs[idx]; idx+=1
t_rel=tabs[idx]; idx+=1
t_filt=tabs[idx]; idx+=1
t_emb=tabs[idx]; idx+=1
t_alerts=tabs[idx]

# ════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════
with t_dash:
    kpi_cands=sorted(num_cols,key=lambda c:df_raw[c].sum() if df_raw[c].notna().any() else 0,reverse=True)
    if kpi_cands:
        kcs=st.columns(min(len(kpi_cands),5))
        for i,col in enumerate(kpi_cands[:5]):
            s=df_raw[col].dropna(); total=s.sum(); mean=s.mean()
            fmt=lambda v: f"{v:,.0f}" if abs(v)>=1 else f"{v:.3f}"
            with kcs[i]: st.markdown(f'<div class="kpi-card"><div class="kpi-label">{col}</div><div class="kpi-value">{fmt(total)}</div><div class="kpi-delta">avg {fmt(mean)} · {len(s):,} records</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    r1l,r1r=st.columns([3,2])
    with r1l:
        if cat_cols and num_cols:
            bc=min(cat_cols,key=lambda c:df_raw[c].nunique() if df_raw[c].nunique()>1 else 999)
            bn=kpi_cands[0] if kpi_cands else num_cols[0]
            grp=df_raw.groupby(bc)[bn].sum().nlargest(min(df_raw[bc].nunique(),15)).reset_index()
            fig=px.bar(grp,x=bn,y=bc,orientation="h",color=bn,color_continuous_scale=PAL_SEQ,title=f"Total {bn} by {bc}")
            fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
            plo(fig,height=360); st.plotly_chart(fig,use_container_width=True)
        elif num_cols:
            fig=px.histogram(df_raw,x=num_cols[0],nbins=30,color_discrete_sequence=[PAL[1]],title=f"Distribution — {num_cols[0]}")
            plo(fig,height=360); st.plotly_chart(fig,use_container_width=True)
    with r1r:
        if cat_cols:
            bc2=cat_cols[0]; vc=df_raw[bc2].value_counts().head(8).reset_index(); vc.columns=[bc2,"count"]
            fig=px.pie(vc,names=bc2,values="count",hole=0.52,color_discrete_sequence=PAL,title=f"Breakdown: {bc2}")
            fig.update_traces(textposition="outside",textinfo="percent+label",textfont_size=11)
            plo(fig,height=360); fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
        elif len(num_cols)>=2:
            fig=px.scatter(df_raw.head(500),x=num_cols[0],y=num_cols[1],opacity=0.6,color_discrete_sequence=[PAL[1]],title=f"{num_cols[1]} vs {num_cols[0]}")
            plo(fig,height=360); st.plotly_chart(fig,use_container_width=True)
    if date_cols and num_cols:
        dc2=date_cols[0]; nc2=kpi_cands[0] if kpi_cands else num_cols[0]
        trend=df_raw[[dc2,nc2]].dropna().sort_values(dc2)
        if len(trend)>500: trend=trend.set_index(dc2).resample("W").sum().reset_index()
        fig=px.area(trend,x=dc2,y=nc2,color_discrete_sequence=[PAL[1]],title=f"{nc2} over time")
        fig.update_traces(fill="tozeroy",line_width=2,fillcolor="rgba(201,168,76,0.15)")
        plo(fig,height=240); st.plotly_chart(fig,use_container_width=True)
    elif len(num_cols)>=3:
        rcs=st.columns(min(len(num_cols),5))
        for i,nc in enumerate(num_cols[:5]):
            with rcs[i]:
                fig=go.Figure(go.Histogram(x=df_raw[nc].dropna(),nbinsx=20,marker_color=PAL[i%len(PAL)],marker_line_width=0))
                plo(fig,nc,160); fig.update_layout(margin=dict(t=28,b=8,l=8,r=8),showlegend=False)
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")
    st.markdown("---")
    st.markdown('<div class="section-title">Auto-generated insights</div>',unsafe_allow_html=True)
    tm={"warn":"tag-warn","info":"tag-info","good":"tag-good","bad":"tag-bad"}
    lm={"warn":"⚠ Watch","info":"ℹ Note","good":"✓ Good","bad":"✗ Issue"}
    ic=st.columns(2)
    for i,(level,text) in enumerate(P["insights"]):
        with ic[i%2]: st.markdown(f'<div class="insight"><span class="tag {tm[level]}">{lm[level]}</span>{text}</div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# QUALITY
# ════════════════════════════════════════════════════════════════
with t_qual:
    color="#3a8a5a" if q>=90 else "#c9a84c" if q>=70 else "#c0392b"
    fig_g=go.Figure(go.Indicator(mode="gauge+number",value=q,
        title={"text":"Data Quality Score","font":{"size":14,"color":"#666","family":"DM Sans"}},
        number={"suffix":"%","font":{"size":40,"color":color,"family":"DM Sans"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"family":"DM Sans","size":11}},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"#f0ede8","borderwidth":0,
               "steps":[{"range":[0,60],"color":"#fde8e8"},{"range":[60,80],"color":"#fef3cd"},{"range":[80,100],"color":"#d4edda"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":q}}))
    fig_g.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=210,margin=dict(t=30,b=0,l=30,r=30))
    gc,ic2=st.columns([1,2])
    with gc: st.plotly_chart(fig_g,use_container_width=True)
    with ic2:
        st.markdown("<br>",unsafe_allow_html=True)
        m1,m2,m3,m4,m5=st.columns(5)
        tn=df_raw.isnull().sum().sum(); np2=round(tn/max(df_raw.size,1)*100,1)
        m1.metric("Total Nulls",f"{tn:,}"); m2.metric("Null %",f"{np2}%")
        m3.metric("Duplicates",f"{P['dups']:,}")
        to=sum(s.get("outliers",0) for s in P["num_stats"].values()); m4.metric("Outliers",f"{to:,}")
        m5.metric("Columns",f"{len(df_raw.columns)}")
    st.markdown("---")
    ql,qr=st.columns(2)
    with ql:
        st.markdown('<div class="section-title">Missing values</div>',unsafe_allow_html=True)
        nd=pd.DataFrame({"Column":list(P["null_pct"].keys()),"Missing %":list(P["null_pct"].values()),"Count":list(P["nulls"].values())}).sort_values("Missing %",ascending=False)
        nd=nd[nd["Missing %"]>0]
        if nd.empty: st.success("✓ No missing values")
        else:
            fig=px.bar(nd.head(20),x="Missing %",y="Column",orientation="h",color="Missing %",color_continuous_scale=PAL_SEQ,text=nd.head(20)["Missing %"].apply(lambda x:f"{x}%"))
            fig.update_traces(textposition="outside"); fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
            plo(fig,height=max(200,len(nd)*26+60)); st.plotly_chart(fig,use_container_width=True)
    with qr:
        st.markdown('<div class="section-title">Outliers & skew</div>',unsafe_allow_html=True)
        if P["num_stats"]:
            od=pd.DataFrame([{"Column":c,"Outliers":s["outliers"],"Outlier %":s["outlier_pct"],"Skew":s["skew"],"Kurt":s["kurt"]} for c,s in P["num_stats"].items()]).sort_values("Outlier %",ascending=False)
            ow=od[od["Outliers"]>0]
            if ow.empty: st.success("✓ No outliers detected")
            else:
                fig=px.bar(ow,x="Outlier %",y="Column",orientation="h",color="Outlier %",color_continuous_scale=PAL_SEQ,text=ow["Outlier %"].apply(lambda x:f"{x}%"))
                fig.update_traces(textposition="outside"); fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
                plo(fig,height=max(200,len(ow)*26+60)); st.plotly_chart(fig,use_container_width=True)
    st.markdown("---")
    # Null heatmap
    if tn>0:
        st.markdown('<div class="section-title">Null pattern heatmap</div>',unsafe_allow_html=True)
        null_map=df_raw.isnull().astype(int)
        sample=null_map.sample(min(100,len(null_map)),random_state=0)
        fig=px.imshow(sample.T,color_continuous_scale=[[0,"#f0ede8"],[1,"#c0392b"]],aspect="auto",title="Missing values (red=missing, 100-row sample)")
        fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=max(150,len(df_raw.columns)*18+60),coloraxis_showscale=False,margin=dict(t=44,b=10,l=10,r=10))
        try:
            st.plotly_chart(fig,use_container_width=True)
        except Exception as _chart_err:
            st.warning(f"⚠️ Chart error: {_chart_err}")
    st.markdown("---")
    st.markdown('<div class="section-title">Full schema + statistics</div>',unsafe_allow_html=True)
    schema=[]
    for col in df_raw.columns:
        dtype="numeric" if col in num_cols else "date" if col in date_cols else "categorical"
        row={"Column":col,"Type":dtype,"Non-null":int(df_raw[col].count()),"Null %":P["null_pct"].get(col,0),"Unique":P["nunique"].get(col,0)}
        if col in P["num_stats"]:
            s=P["num_stats"][col]; row.update({"Min":s["min"],"Max":s["max"],"Mean":s["mean"],"Std":s["std"],"Skew":s["skew"],"Outliers":s["outliers"]})
            row["Normal?"]="Yes" if s["norm_p"] and s["norm_p"]>0.05 else "No" if s["norm_p"] else "—"
        elif col in P["cat_stats"]:
            s=P["cat_stats"][col]; row["Top Value"]=f"{s['top']} ({s['top_pct']}%)"
        schema.append(row)
    st.dataframe(pd.DataFrame(schema),use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════
with t_dist:
    mode=st.radio("View",["Histogram","Categorical breakdown","Box / Violin / Strip","Q-Q Plot","All numerics grid"],horizontal=True,key="dist_mode")
    if mode=="Histogram":
        if not num_cols: st.info("No numeric columns.")
        else:
            sel=st.selectbox("Column",num_cols,key="dist_col")
            c1,c2=st.columns([3,1])
            with c1:
                cb=st.selectbox("Split by",["None"]+cat_cols,key="dist_color")
                bins=st.slider("Bins",10,100,40,key="dist_bins")
                fig=px.histogram(df_raw,x=sel,nbins=bins,color=None if cb=="None" else cb,color_discrete_sequence=PAL,marginal="box",opacity=0.85,title=f"Distribution — {sel}")
                plo(fig,height=400); st.plotly_chart(fig,use_container_width=True)
            with c2:
                st.markdown("<br><br>",unsafe_allow_html=True)
                if sel in P["num_stats"]:
                    ns=P["num_stats"][sel]
                    for label,val in [("Min",ns["min"]),("Max",ns["max"]),("Mean",ns["mean"]),("Median",ns["median"]),("Std dev",ns["std"]),("Skew",ns["skew"]),("Kurtosis",ns["kurt"]),("P25",ns["p25"]),("P75",ns["p75"]),("P95",ns["p95"]),("Outliers",f"{ns['outliers']} ({ns['outlier_pct']}%)"),("Normal?","Yes" if ns["norm_p"] and ns["norm_p"]>0.05 else f"No (p={ns['norm_p']})" if ns["norm_p"] else "—")]:
                        stat_row(label,val)
    elif mode=="Categorical breakdown":
        if not cat_cols: st.info("No categorical columns.")
        else:
            sel=st.selectbox("Column",cat_cols,key="cat_dist_col"); top_n=st.slider("Top N",5,50,15,key="cat_top_n")
            vc=df_raw[sel].value_counts().head(top_n).reset_index(); vc.columns=[sel,"count"]; vc["pct"]=(vc["count"]/len(df_raw)*100).round(1)
            c1,c2=st.columns(2)
            with c1:
                fig=px.bar(vc,x="count",y=sel,orientation="h",color="count",color_continuous_scale=PAL_SEQ,text=vc["pct"].apply(lambda x:f"{x}%"),title=f"Top {top_n} in {sel}")
                fig.update_traces(textposition="outside"); fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
                plo(fig,height=max(300,top_n*24+60)); st.plotly_chart(fig,use_container_width=True)
            with c2:
                fig2=px.pie(vc.head(10),names=sel,values="count",hole=0.5,color_discrete_sequence=PAL,title=f"{sel} share")
                fig2.update_traces(textposition="outside",textinfo="percent+label",textfont_size=11)
                plo(fig2,height=max(300,top_n*24+60)); fig2.update_layout(showlegend=False); st.plotly_chart(fig2,use_container_width=True)
    elif mode=="Box / Violin / Strip":
        if not num_cols: st.info("No numeric columns.")
        else:
            yc=st.selectbox("Numeric",num_cols,key="bv_y"); xc=st.selectbox("Group by",["None"]+cat_cols,key="bv_x")
            vtype=st.radio("Type",["Violin","Box","Strip"],horizontal=True,key="bv_type")
            kwargs=dict(y=yc,x=None if xc=="None" else xc,color=None if xc=="None" else xc,color_discrete_sequence=PAL,title=f"{yc}" + (f" by {xc}" if xc!="None" else ""))
            if vtype=="Violin": fig=px.violin(df_raw,box=True,points="outliers",**kwargs)
            elif vtype=="Box": fig=px.box(df_raw,points="outliers",**kwargs)
            else: fig=px.strip(df_raw,**kwargs)
            plo(fig,height=440); fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
    elif mode=="Q-Q Plot":
        if not num_cols: st.info("No numeric columns.")
        else:
            sel=st.selectbox("Column",num_cols,key="qq_col")
            s=df_raw[sel].dropna().sample(min(len(df_raw[sel].dropna()),2000),random_state=0)
            theoretical=stats.norm.ppf(np.linspace(0.01,0.99,len(s)))
            observed=np.sort(s.values)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=theoretical,y=observed,mode="markers",marker=dict(color=PAL[1],size=4,opacity=0.7),name="Data"))
            mn,mx=theoretical.min(),theoretical.max()
            fig.add_trace(go.Scatter(x=[mn,mx],y=[mn*s.std()+s.mean(),mx*s.std()+s.mean()],mode="lines",line=dict(color=PAL[0],width=2,dash="dash"),name="Normal line"))
            plo(fig,f"Q-Q Plot — {sel}",420)
            fig.update_layout(xaxis_title="Theoretical quantiles",yaxis_title="Sample quantiles")
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")
            ns=P["num_stats"].get(sel,{})
            if ns.get("norm_p"):
                col_n="#3a8a5a" if ns["norm_p"]>0.05 else "#c0392b"
                verdict="Likely normal" if ns["norm_p"]>0.05 else "Not normal"
                st.markdown(f'<div style="background:#fff;border-radius:8px;padding:10px 16px;border:1px solid #e5e0d8;display:inline-block;font-size:13px;">Shapiro-Wilk p = <b style="color:{col_n};">{ns["norm_p"]}</b> &nbsp;|&nbsp; <b style="color:{col_n};">{verdict}</b> (α=0.05)</div>',unsafe_allow_html=True)
    else:
        if not num_cols: st.info("No numeric columns.")
        else:
            n=len(num_cols); nc2=min(n,3); nr=(n+nc2-1)//nc2
            fig=make_subplots(rows=nr,cols=nc2,subplot_titles=num_cols)
            for i,col in enumerate(num_cols):
                r,c=i//nc2+1,i%nc2+1
                fig.add_trace(go.Histogram(x=df_raw[col].dropna(),nbinsx=25,marker_color=PAL[i%len(PAL)],marker_line_width=0,showlegend=False,name=col),row=r,col=c)
            fig.update_layout(paper_bgcolor="#fafaf8",plot_bgcolor="#fafaf8",font_family="DM Sans",height=nr*190+60,margin=dict(t=50,b=20,l=20,r=20),showlegend=False)
            fig.update_xaxes(gridcolor="#eeebe5"); fig.update_yaxes(gridcolor="#eeebe5")
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

# ════════════════════════════════════════════════════════════════
# CHART STUDIO
# ════════════════════════════════════════════════════════════════
with t_chart:
    chart_types=["Treemap","Sunburst","Funnel","Waterfall","Radar / Spider","Bubble","Sankey","Parallel Coordinates","3D Scatter","Heatmap (value matrix)","Candlestick (OHLC)","Dot / Lollipop","Area (stacked)","Histogram 2D","Animated Bar Race"]
    ct=st.selectbox("Chart type",chart_types,key="studio_ct")

    if ct=="Treemap":
        if len(cat_cols)<1 or not num_cols: st.warning("Need at least 1 category and 1 numeric column.")
        else:
            c1,c2=st.columns(2)
            path_cols=c1.multiselect("Hierarchy (order matters)",cat_cols,default=cat_cols[:min(2,len(cat_cols))],key="tm_path")
            val_col=c2.selectbox("Value",num_cols,key="tm_val")
            if path_cols:
                fig=px.treemap(df_raw,path=path_cols,values=val_col,color=val_col,color_continuous_scale=PAL_SEQ,title=f"Treemap — {val_col}")
                fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=500,margin=dict(t=44,b=10,l=10,r=10))
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Sunburst":
        if len(cat_cols)<1 or not num_cols: st.warning("Need at least 1 category and 1 numeric column.")
        else:
            c1,c2=st.columns(2)
            path_cols=c1.multiselect("Hierarchy",cat_cols,default=cat_cols[:min(2,len(cat_cols))],key="sb_path")
            val_col=c2.selectbox("Value",num_cols,key="sb_val")
            if path_cols:
                fig=px.sunburst(df_raw,path=path_cols,values=val_col,color=val_col,color_continuous_scale=PAL_SEQ,title="Sunburst")
                fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=500,margin=dict(t=44,b=10,l=10,r=10))
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Funnel":
        if not cat_cols or not num_cols: st.warning("Need a category and numeric column.")
        else:
            c1,c2=st.columns(2)
            stage_col=c1.selectbox("Stage column",cat_cols,key="fn_stage")
            val_col=c2.selectbox("Value",num_cols,key="fn_val")
            fd=df_raw.groupby(stage_col)[val_col].sum().reset_index().sort_values(val_col,ascending=False)
            fig=px.funnel(fd,x=val_col,y=stage_col,color_discrete_sequence=PAL,title=f"Funnel — {val_col} by {stage_col}")
            plo(fig,height=420); st.plotly_chart(fig,use_container_width=True)

    elif ct=="Waterfall":
        if not num_cols: st.warning("Need a numeric column.")
        else:
            c1,c2=st.columns(2)
            val_col=c1.selectbox("Value column",num_cols,key="wf_val")
            label_col=c2.selectbox("Label column",cat_cols+[None],key="wf_label") if cat_cols else None
            top_n=st.slider("Top N items",3,30,10,key="wf_n")
            if label_col:
                wd=df_raw.groupby(label_col)[val_col].sum().nlargest(top_n).reset_index()
                labels=wd[label_col].tolist(); values=wd[val_col].tolist()
            else:
                labels=[str(i) for i in range(min(top_n,len(df_raw)))]
                values=df_raw[val_col].head(top_n).tolist()
            # Plotly Waterfall: measure="relative" for each bar, add a "total" at the end
            measures=["relative"]*len(values)+["total"]
            labels_wf=labels+["Total"]
            values_wf=values+[0]
            inc_color="#3a8a5a"; dec_color="#c0392b"; tot_color="#1a1a2e"
            fig=go.Figure(go.Waterfall(
                x=labels_wf, y=values_wf, measure=measures,
                increasing=dict(marker_color=inc_color),
                decreasing=dict(marker_color=dec_color),
                totals=dict(marker_color=tot_color),
                connector=dict(line=dict(color="#ccc", width=1)),
                text=[f"{v:,.0f}" for v in values]+[f"{sum(values):,.0f}"],
                textposition="outside",
            ))
            plo(fig,f"Waterfall — {val_col}",420); st.plotly_chart(fig,use_container_width=True)

    elif ct=="Radar / Spider":
        if len(num_cols)<3: st.warning("Need at least 3 numeric columns for radar chart.")
        else:
            axes=st.multiselect("Axes (numeric columns)",num_cols,default=num_cols[:min(6,len(num_cols))],key="rd_axes")
            split_col=st.selectbox("Group by (optional)",["None"]+cat_cols,key="rd_split")
            if axes:
                if split_col!="None":
                    groups=df_raw[split_col].value_counts().head(5).index.tolist()
                    fig=go.Figure()
                    for i,g in enumerate(groups):
                        vals=df_raw[df_raw[split_col]==g][axes].mean().tolist()
                        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=axes+[axes[0]],fill="toself",name=str(g),line_color=PAL[i%len(PAL)]))
                else:
                    vals=df_raw[axes].mean().tolist()
                    fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=axes+[axes[0]],fill="toself",line_color=PAL[1]))
                fig.update_layout(polar=dict(bgcolor="#fafaf8",radialaxis=dict(visible=True,gridcolor="#ddd"),angularaxis=dict(gridcolor="#ddd")),paper_bgcolor="#fafaf8",font_family="DM Sans",height=460,showlegend=split_col!="None",title="Radar chart — mean values",margin=dict(t=60,b=40))
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Bubble":
        if len(num_cols)<3: st.warning("Need at least 3 numeric columns.")
        else:
            c1,c2,c3,c4=st.columns(4)
            xc=c1.selectbox("X",num_cols,key="bb_x"); yc=c2.selectbox("Y",num_cols,index=min(1,len(num_cols)-1),key="bb_y")
            szc=c3.selectbox("Size",num_cols,index=min(2,len(num_cols)-1),key="bb_sz")
            cc=c4.selectbox("Colour",["None"]+cat_cols,key="bb_col")
            fig=px.scatter(df_raw.head(500),x=xc,y=yc,size=szc,color=None if cc=="None" else cc,color_discrete_sequence=PAL,size_max=50,opacity=0.7,title=f"Bubble — {yc} vs {xc}, size={szc}")
            plo(fig,height=460); st.plotly_chart(fig,use_container_width=True)

    elif ct=="Sankey":
        if len(cat_cols)<2 or not num_cols: st.warning("Need at least 2 category columns and 1 numeric.")
        else:
            c1,c2,c3=st.columns(3)
            src=c1.selectbox("Source",cat_cols,key="sk_src"); tgt=c2.selectbox("Target",cat_cols,index=min(1,len(cat_cols)-1),key="sk_tgt")
            val=c3.selectbox("Value",num_cols,key="sk_val")
            flow=df_raw.groupby([src,tgt])[val].sum().reset_index().nlargest(30,val)
            all_nodes=list(set(flow[src].tolist()+flow[tgt].tolist()))
            node_idx={n:i for i,n in enumerate(all_nodes)}
            fig=go.Figure(go.Sankey(
                node=dict(label=all_nodes,color=PAL[1],pad=15,thickness=20),
                link=dict(source=flow[src].map(node_idx).tolist(),target=flow[tgt].map(node_idx).tolist(),value=flow[val].tolist(),color="rgba(201,168,76,0.3)")))
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=480,title=f"Sankey — {src} → {tgt}",title_font_size=14,margin=dict(t=50,b=20))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Parallel Coordinates":
        if len(num_cols)<2: st.warning("Need at least 2 numeric columns.")
        else:
            sel_cols=st.multiselect("Columns",num_cols,default=num_cols[:min(5,len(num_cols))],key="pc_cols")
            cc=st.selectbox("Colour by",["None"]+num_cols,key="pc_col")
            if sel_cols:
                fig=px.parallel_coordinates(df_raw.head(1000),dimensions=sel_cols,color=None if cc=="None" else cc,color_continuous_scale=PAL_SEQ,title="Parallel Coordinates")
                fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=460,margin=dict(t=50,b=20))
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="3D Scatter":
        if len(num_cols)<3: st.warning("Need at least 3 numeric columns.")
        else:
            c1,c2,c3,c4=st.columns(4)
            xc=c1.selectbox("X",num_cols,key="3d_x"); yc=c2.selectbox("Y",num_cols,index=min(1,len(num_cols)-1),key="3d_y")
            zc=c3.selectbox("Z",num_cols,index=min(2,len(num_cols)-1),key="3d_z"); cc=c4.selectbox("Colour",["None"]+cat_cols,key="3d_col")
            fig=px.scatter_3d(df_raw.head(800),x=xc,y=yc,z=zc,color=None if cc=="None" else cc,color_discrete_sequence=PAL,opacity=0.7,title=f"3D Scatter — {xc}, {yc}, {zc}")
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=550,margin=dict(t=50,b=20,l=0,r=0))
            fig.update_traces(marker_size=3)
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Heatmap (value matrix)":
        if len(cat_cols)<2 or not num_cols: st.warning("Need 2 category columns and 1 numeric.")
        else:
            c1,c2,c3=st.columns(3)
            rc=c1.selectbox("Rows",cat_cols,key="hm_row"); cc2=c2.selectbox("Columns",cat_cols,index=min(1,len(cat_cols)-1),key="hm_col")
            vc2=c3.selectbox("Values",num_cols,key="hm_val"); agg=st.selectbox("Aggregation",["sum","mean","count","max","min"],key="hm_agg")
            pivot=pd.pivot_table(df_raw,values=vc2,index=rc,columns=cc2,aggfunc=agg,fill_value=0)
            fig=px.imshow(pivot,color_continuous_scale=PAL_SEQ,aspect="auto",title=f"{agg}({vc2}) — {rc} × {cc2}",text_auto=".1f")
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=max(300,len(pivot)*30+100),margin=dict(t=50,b=20))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Candlestick (OHLC)":
        if not date_cols: st.warning("Need a date column.")
        elif len(num_cols)<4: st.warning("Need 4 numeric columns (Open, High, Low, Close).")
        else:
            c1,c2,c3,c4,c5=st.columns(5)
            dc3=c1.selectbox("Date",date_cols,key="cs_date"); oc=c2.selectbox("Open",num_cols,key="cs_o")
            hc=c3.selectbox("High",num_cols,key="cs_h"); lc=c4.selectbox("Low",num_cols,key="cs_l"); cc3=c5.selectbox("Close",num_cols,key="cs_c")
            df_cs=df_raw[[dc3,oc,hc,lc,cc3]].dropna().sort_values(dc3)
            fig=go.Figure(go.Candlestick(x=df_cs[dc3],open=df_cs[oc],high=df_cs[hc],low=df_cs[lc],close=df_cs[cc3],increasing_line_color="#3a8a5a",decreasing_line_color="#c0392b"))
            plo(fig,"Candlestick Chart",440); st.plotly_chart(fig,use_container_width=True)

    elif ct=="Dot / Lollipop":
        if not cat_cols or not num_cols: st.warning("Need a category and numeric column.")
        else:
            c1,c2=st.columns(2)
            yc=c1.selectbox("Category",cat_cols,key="ll_y"); xc=c2.selectbox("Value",num_cols,key="ll_x")
            top_n=st.slider("Top N",5,40,20,key="ll_n")
            ld=df_raw.groupby(yc)[xc].sum().nlargest(top_n).reset_index()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ld[xc],y=ld[yc],mode="markers",marker=dict(color=PAL[1],size=12),name=""))
            for _,row in ld.iterrows():
                fig.add_shape(type="line",x0=0,x1=row[xc],y0=row[yc],y1=row[yc],line=dict(color="#ddd",width=2))
            plo(fig,f"Top {top_n} {yc} by {xc}",max(300,top_n*24+60)); fig.update_layout(yaxis=dict(autorange="reversed"),showlegend=False)
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Area (stacked)":
        if not cat_cols or not date_cols or not num_cols: st.warning("Need a date, category, and numeric column.")
        else:
            c1,c2,c3=st.columns(3); dc4=c1.selectbox("Date",date_cols,key="as_date"); cc4=c2.selectbox("Category",cat_cols,key="as_cat"); vc4=c3.selectbox("Value",num_cols,key="as_val")
            freq_map={"Weekly":"W","Monthly":"ME","Quarterly":"QE"}; fl=st.select_slider("Resample",list(freq_map.keys()),value="Monthly",key="as_freq")
            top_cats=df_raw[cc4].value_counts().head(8).index.tolist()
            df_as=df_raw[df_raw[cc4].isin(top_cats)].groupby([pd.Grouper(key=dc4,freq=freq_map[fl]),cc4])[vc4].sum().reset_index()
            fig=px.area(df_as,x=dc4,y=vc4,color=cc4,color_discrete_sequence=PAL,title=f"Stacked area — {vc4} by {cc4}")
            plo(fig,height=400); st.plotly_chart(fig,use_container_width=True)

    elif ct=="Histogram 2D":
        if len(num_cols)<2: st.warning("Need 2 numeric columns.")
        else:
            c1,c2=st.columns(2); xc=c1.selectbox("X",num_cols,key="h2_x"); yc=c2.selectbox("Y",num_cols,index=min(1,len(num_cols)-1),key="h2_y")
            fig=px.density_heatmap(df_raw,x=xc,y=yc,nbinsx=30,nbinsy=30,color_continuous_scale=PAL_SEQ,marginal_x="histogram",marginal_y="histogram",title=f"2D Density — {xc} vs {yc}")
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=480,margin=dict(t=50,b=20))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif ct=="Animated Bar Race":
        if not date_cols or not cat_cols or not num_cols: st.warning("Need date, category, and numeric columns.")
        else:
            c1,c2,c3=st.columns(3); dc5=c1.selectbox("Date",date_cols,key="br_date"); cc5=c2.selectbox("Category",cat_cols,key="br_cat"); vc5=c3.selectbox("Value",num_cols,key="br_val")
            df_br=df_raw[[dc5,cc5,vc5]].dropna()
            df_br["period"]=df_br[dc5].dt.to_period("M").astype(str)
            df_br=df_br.groupby(["period",cc5])[vc5].sum().reset_index().sort_values([vc5],ascending=False)
            top_cats=df_br.groupby(cc5)[vc5].sum().nlargest(10).index.tolist()
            df_br=df_br[df_br[cc5].isin(top_cats)]
            fig=px.bar(df_br,x=vc5,y=cc5,animation_frame="period",orientation="h",color=cc5,color_discrete_sequence=PAL,range_x=[0,df_br[vc5].max()*1.1],title=f"Bar race — {vc5} by {cc5}")
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=480,yaxis=dict(autorange="reversed"),showlegend=False,margin=dict(t=50,b=20))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

# ════════════════════════════════════════════════════════════════
# TIME SERIES
# ════════════════════════════════════════════════════════════════
if t_time is not None:
    with t_time:
        dc=st.selectbox("Date column",date_cols,key="ts_date")
        ys=st.multiselect("Metrics",num_cols,default=num_cols[:2],key="ts_metrics")
        if not ys: st.info("Select at least one metric.")
        else:
            freq_map={"Raw":"raw","Daily":"D","Weekly":"W","Monthly":"ME","Quarterly":"QE"}
            fl=st.select_slider("Resample",list(freq_map.keys()),value="Monthly",key="ts_freq")
            freq=freq_map[fl]
            df_ts=df_raw[[dc]+ys].dropna(subset=[dc]).sort_values(dc)
            if freq!="raw": df_ts=df_ts.set_index(dc)[ys].resample(freq).sum().reset_index()
            at=st.radio("Type",["Line","Area","Bar"],horizontal=True,key="ts_type")
            if at=="Line": fig=px.line(df_ts,x=dc,y=ys,color_discrete_sequence=PAL)
            elif at=="Area": fig=px.area(df_ts,x=dc,y=ys,color_discrete_sequence=PAL)
            else: fig=px.bar(df_ts,x=dc,y=ys,color_discrete_sequence=PAL,barmode="group")
            plo(fig,f"{', '.join(ys)} — {fl}",380); st.plotly_chart(fig,use_container_width=True)
            # Rolling + YoY
            if freq in("ME","QE","W") and len(df_ts)>3:
                st.markdown("---")
                tl,tr=st.columns(2)
                with tl:
                    st.markdown('<div class="section-title">Rolling averages</div>',unsafe_allow_html=True)
                    dr=df_ts.copy()
                    for y in ys:
                        dr[f"{y} (3-period)"]=dr[y].rolling(3).mean()
                        dr[f"{y} (7-period)"]=dr[y].rolling(7).mean()
                    rc=[f"{y} (3-period)" for y in ys]+[f"{y} (7-period)" for y in ys]
                    fig2=px.line(dr,x=dc,y=rc,color_discrete_sequence=PAL)
                    plo(fig2,height=260); st.plotly_chart(fig2,use_container_width=True)
                with tr:
                    if freq=="ME" and len(ys)>0:
                        st.markdown('<div class="section-title">Month-over-month % change</div>',unsafe_allow_html=True)
                        dm=df_ts.copy(); dm[f"{ys[0]} MoM%"]=(dm[ys[0]].pct_change()*100).round(1)
                        fig3=px.bar(dm,x=dc,y=f"{ys[0]} MoM%",color=f"{ys[0]} MoM%",color_continuous_scale=[[0,"#c0392b"],[0.5,"#f0ede8"],[1,"#3a8a5a"]],title=f"{ys[0]} MoM %")
                        fig3.update_layout(coloraxis_showscale=False); plo(fig3,height=260); st.plotly_chart(fig3,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# COMPUTE
# ════════════════════════════════════════════════════════════════
with t_comp:
    st.markdown('<div class="section-title">Column operations — results are added to your dataset for use in other tabs</div>',unsafe_allow_html=True)
    op=st.radio("Operation",["Formula column","Bin numeric","Rank column","Normalize / Standardize","Rolling window","% of total","Lag column"],horizontal=True,key="comp_op")

    if op=="Formula column":
        st.markdown('<div class="compute-card">',unsafe_allow_html=True)
        new_name=st.text_input("New column name",value="new_col",key="fc_name")
        expr=st.text_input("Formula (pandas eval expression)",placeholder="e.g.  Revenue - Cost   or   Revenue / Units",key="fc_expr")
        st.caption("Available columns: "+", ".join(f"`{c}`" for c in num_cols))
        if st.button("➕ Create column",key="fc_btn"):
            try:
                result=df_raw.eval(expr)
                st.session_state.computed_cols[new_name]=result.values
                st.success(f"✓ Column '{new_name}' created"); st.rerun()
            except Exception as e: st.error(f"Error: {e}")
        st.markdown('</div>',unsafe_allow_html=True)

    elif op=="Bin numeric":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2,c3=st.columns(3)
            bin_col=c1.selectbox("Column to bin",num_cols,key="bn_col")
            n_bins=c2.slider("Number of bins",2,20,5,key="bn_n")
            bin_label=c3.text_input("New column name",value=f"{bin_col}_bin",key="bn_name")
            strategy=st.radio("Strategy",["Equal width","Equal frequency (quantile)","Custom edges"],horizontal=True,key="bn_strat")
            if strategy=="Custom edges":
                edges_str=st.text_input("Bin edges (comma-separated)",placeholder="e.g. 0,100,500,1000",key="bn_edges")
            if st.button("➕ Bin column",key="bn_btn"):
                try:
                    if strategy=="Equal width": result=pd.cut(df_raw[bin_col],bins=n_bins,labels=False)
                    elif strategy=="Equal frequency (quantile)": result=pd.qcut(df_raw[bin_col],q=n_bins,labels=False,duplicates="drop")
                    else:
                        edges=[float(x.strip()) for x in edges_str.split(",")]
                        result=pd.cut(df_raw[bin_col],bins=edges,labels=False)
                    st.session_state.computed_cols[bin_label]=result.values
                    # preview
                    preview=pd.cut(df_raw[bin_col],bins=n_bins).value_counts().sort_index().reset_index()
                    preview.columns=["Bin range","Count"]
                    fig=px.bar(preview,x="Bin range",y="Count",color_discrete_sequence=[PAL[1]],title=f"Bin distribution — {bin_col}")
                    plo(fig,height=300); st.plotly_chart(fig,use_container_width=True)
                    st.success(f"✓ Created '{bin_label}'"); st.rerun()
                except Exception as e: st.error(f"Error: {e}")

    elif op=="Rank column":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2,c3=st.columns(3)
            rc=c1.selectbox("Column",num_cols,key="rk_col"); method=c2.selectbox("Method",["average","min","max","dense","ordinal"],key="rk_method")
            rname=c3.text_input("New column name",value=f"{num_cols[0]}_rank",key="rk_name")
            asc=st.checkbox("Ascending rank",value=False,key="rk_asc")
            if st.button("➕ Rank column",key="rk_btn"):
                result=df_raw[rc].rank(method=method,ascending=asc)
                st.session_state.computed_cols[rname]=result.values
                st.success(f"✓ Created '{rname}'"); st.rerun()

    elif op=="Normalize / Standardize":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2=st.columns(2)
            nc3=c1.selectbox("Column",num_cols,key="nm_col"); method=c2.selectbox("Method",["Min-max (0–1)","Z-score (mean=0, std=1)","Log transform","Square root"],key="nm_method")
            nname=st.text_input("New column name",value=f"{num_cols[0]}_norm",key="nm_name")
            if st.button("➕ Apply",key="nm_btn"):
                s=df_raw[nc3]
                if method=="Min-max (0–1)": result=(s-s.min())/(s.max()-s.min())
                elif method=="Z-score (mean=0, std=1)": result=(s-s.mean())/s.std()
                elif method=="Log transform": result=np.log1p(s.clip(lower=0))
                else: result=np.sqrt(s.clip(lower=0))
                st.session_state.computed_cols[nname]=result.values
                before,after=st.columns(2)
                with before:
                    fig=px.histogram(df_raw,x=nc3,nbins=30,color_discrete_sequence=[PAL[0]],title="Before")
                    plo(fig,height=240); st.plotly_chart(fig,use_container_width=True)
                with after:
                    fig=px.histogram(x=result,nbins=30,color_discrete_sequence=[PAL[1]],title="After")
                    plo(fig,height=240); st.plotly_chart(fig,use_container_width=True)
                st.success(f"✓ Created '{nname}'"); st.rerun()

    elif op=="Rolling window":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2,c3=st.columns(3)
            rc2=c1.selectbox("Column",num_cols,key="rw_col"); window=c2.slider("Window size",2,30,7,key="rw_win")
            fn=c3.selectbox("Function",["mean","sum","std","min","max"],key="rw_fn")
            rw_name=st.text_input("New column name",value=f"{num_cols[0]}_rolling_{fn}",key="rw_name")
            if st.button("➕ Apply rolling",key="rw_btn"):
                result=getattr(df_raw[rc2].rolling(window),fn)()
                st.session_state.computed_cols[rw_name]=result.values
                st.success(f"✓ Created '{rw_name}'"); st.rerun()

    elif op=="% of total":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2=st.columns(2)
            pc=c1.selectbox("Column",num_cols,key="pt_col")
            pname=c2.text_input("New column name",value=f"{num_cols[0]}_pct",key="pt_name")
            group_col=st.selectbox("Group % within (optional)",["None"]+cat_cols,key="pt_grp")
            if st.button("➕ Create % column",key="pt_btn"):
                if group_col=="None": result=df_raw[pc]/df_raw[pc].sum()*100
                else: result=df_raw.groupby(group_col)[pc].transform(lambda x: x/x.sum()*100)
                st.session_state.computed_cols[pname]=result.values
                st.success(f"✓ Created '{pname}'"); st.rerun()

    elif op=="Lag column":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2,c3=st.columns(3)
            lc=c1.selectbox("Column",num_cols,key="lg_col"); periods=c2.slider("Lag periods",1,12,1,key="lg_per")
            lname=c3.text_input("New column name",value=f"{num_cols[0]}_lag{1}",key="lg_name")
            if st.button("➕ Create lag",key="lg_btn"):
                result=df_raw[lc].shift(periods)
                st.session_state.computed_cols[lname]=result.values
                st.success(f"✓ Created '{lname}'"); st.rerun()

    # Show computed columns
    if st.session_state.computed_cols:
        st.markdown("---")
        st.markdown('<div class="section-title">Computed columns in this session</div>',unsafe_allow_html=True)
        cc_df=pd.DataFrame({c:v for c,v in st.session_state.computed_cols.items()})
        st.dataframe(cc_df.head(20),use_container_width=True,hide_index=True)
        if st.button("🗑 Clear all computed columns",key="clear_cc"):
            st.session_state.computed_cols={}; st.rerun()

# ════════════════════════════════════════════════════════════════
# SEGMENT
# ════════════════════════════════════════════════════════════════
with t_seg:
    seg_mode=st.radio("Analysis",["K-Means Clustering","PCA Projection","Top / Bottom N","Cohort Analysis","Percentile Bands"],horizontal=True,key="seg_mode")

    if seg_mode=="K-Means Clustering":
        if len(num_cols)<2: st.info("Need at least 2 numeric columns.")
        else:
            c1,c2=st.columns(2)
            feat_cols=c1.multiselect("Features for clustering",num_cols,default=num_cols[:min(4,len(num_cols))],key="km_feats")
            n_clusters=c2.slider("Number of clusters",2,8,3,key="km_k")
            # Clear stale result if feature selection changed
            if "km_feats" in st.session_state and st.session_state["km_feats"] != feat_cols:
                for k in ["km_result","km_feats"]: st.session_state.pop(k,None)
            if feat_cols and st.button("▶ Run K-Means",key="km_run"):
                try:
                    data=df_raw[feat_cols].dropna()
                    scaler=StandardScaler(); scaled=scaler.fit_transform(data)
                    km=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
                    labels=km.fit_predict(scaled)
                    df_clustered=data.copy(); df_clustered["Cluster"]=labels.astype(str)
                    st.session_state["km_result"]=df_clustered
                    st.session_state["km_feats"]=list(feat_cols)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")
            if "km_result" in st.session_state:
                df_c=st.session_state["km_result"]; fc=st.session_state["km_feats"]
                # Guard: make sure stored features still exist in the result
                valid_fc=[c for c in fc if c in df_c.columns]
                if len(valid_fc)<2:
                    st.info("Feature selection changed — click ▶ Run K-Means again.")
                    for k in ["km_result","km_feats"]: st.session_state.pop(k,None)
                else:
                    fc=valid_fc
                cl,cr=st.columns(2)
                if "km_result" in st.session_state:
                  with cl:
                    try:
                        fig=px.scatter(df_c,x=fc[0],y=fc[1],color="Cluster",color_discrete_sequence=CLUSTER_COLORS,opacity=0.75,title=f"Clusters — {fc[0]} vs {fc[1]}")
                        plo(fig,height=400); st.plotly_chart(fig,use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart error: {e}")
                with cr:
                    summary=df_c.groupby("Cluster")[fc].mean().round(2).reset_index()
                    st.markdown("**Cluster means:**")
                    st.dataframe(summary,use_container_width=True,hide_index=True)
                    counts=df_c["Cluster"].value_counts().reset_index(); counts.columns=["Cluster","Count"]
                    fig2=px.pie(counts,names="Cluster",values="Count",hole=0.5,color_discrete_sequence=CLUSTER_COLORS,title="Cluster sizes")
                    plo(fig2,height=280); fig2.update_layout(showlegend=False); st.plotly_chart(fig2,use_container_width=True)
                # Radar per cluster
                fig3=go.Figure()
                for i,row in summary.iterrows():
                    vals=row[fc].tolist()
                    fig3.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=fc+[fc[0]],fill="toself",name=f"Cluster {row['Cluster']}",line_color=CLUSTER_COLORS[int(row["Cluster"])%len(CLUSTER_COLORS)]))
                fig3.update_layout(polar=dict(bgcolor="#fafaf8",radialaxis=dict(visible=True),angularaxis=dict(gridcolor="#ddd")),paper_bgcolor="#fafaf8",font_family="DM Sans",height=380,title="Cluster profiles (radar)",showlegend=True)
                try:
                    st.plotly_chart(fig3,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")

    elif seg_mode=="PCA Projection":
        if len(num_cols)<3: st.info("Need at least 3 numeric columns.")
        else:
            feat_cols=st.multiselect("Features",num_cols,default=num_cols[:min(6,len(num_cols))],key="pca_feats")
            cc2=st.selectbox("Colour by",["None"]+cat_cols,key="pca_col")
            n_comp=st.radio("Components",["2D","3D"],horizontal=True,key="pca_comp")
            if feat_cols and st.button("▶ Run PCA",key="pca_run"):
                data=df_raw[feat_cols].dropna()
                scaler=StandardScaler(); scaled=scaler.fit_transform(data)
                n=3 if n_comp=="3D" else 2
                pca=PCA(n_components=n,random_state=42)
                components=pca.fit_transform(scaled)
                df_pca=pd.DataFrame(components,columns=[f"PC{i+1}" for i in range(n)])
                if cc2!="None": df_pca[cc2]=df_raw.loc[data.index,cc2].values
                ev=pca.explained_variance_ratio_*100
                pl,pr=st.columns([2,1])
                with pl:
                    if n_comp=="2D":
                        fig=px.scatter(df_pca,x="PC1",y="PC2",color=None if cc2=="None" else cc2,color_discrete_sequence=PAL,opacity=0.7,title=f"PCA — PC1 ({ev[0]:.1f}%) vs PC2 ({ev[1]:.1f}%)")
                        plo(fig,height=440); st.plotly_chart(fig,use_container_width=True)
                    else:
                        fig=px.scatter_3d(df_pca,x="PC1",y="PC2",z="PC3",color=None if cc2=="None" else cc2,color_discrete_sequence=PAL,opacity=0.7)
                        fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=500); fig.update_traces(marker_size=3)
                        try:
                            st.plotly_chart(fig,use_container_width=True)
                        except Exception as _chart_err:
                            st.warning(f"⚠️ Chart error: {_chart_err}")
                with pr:
                    st.markdown("**Explained variance:**")
                    for i,e in enumerate(ev): stat_row(f"PC{i+1}",f"{e:.1f}%")
                    stat_row("Total",f"{sum(ev):.1f}%")
                    fig2=px.bar(x=[f"PC{i+1}" for i in range(len(ev))],y=ev,color_discrete_sequence=[PAL[1]],title="Variance explained")
                    plo(fig2,height=220); fig2.update_layout(margin=dict(t=40,b=10,l=10,r=10),showlegend=False)
                    try:
                        st.plotly_chart(fig2,use_container_width=True)
                    except Exception as _chart_err:
                        st.warning(f"⚠️ Chart error: {_chart_err}")

    elif seg_mode=="Top / Bottom N":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2,c3=st.columns(3)
            sort_col=c1.selectbox("Sort by",num_cols,key="tb_col"); n=c2.slider("N rows",5,50,10,key="tb_n")
            label_col=c3.selectbox("Label",cat_cols+num_cols,key="tb_label") if (cat_cols or num_cols) else None
            tl,tr=st.columns(2)
            with tl:
                st.markdown(f"**Top {n}**")
                top=df_raw.nlargest(n,sort_col)
                if label_col:
                    fig=px.bar(top,x=sort_col,y=label_col,orientation="h",color=sort_col,color_continuous_scale=PAL_SEQ,title=f"Top {n} by {sort_col}")
                    fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
                    plo(fig,height=max(200,n*26+60)); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(top.reset_index(drop=True),use_container_width=True,hide_index=True)
            with tr:
                st.markdown(f"**Bottom {n}**")
                bot=df_raw.nsmallest(n,sort_col)
                if label_col:
                    fig=px.bar(bot,x=sort_col,y=label_col,orientation="h",color=sort_col,color_continuous_scale=PAL_SEQ,title=f"Bottom {n} by {sort_col}")
                    fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
                    plo(fig,height=max(200,n*26+60)); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(bot.reset_index(drop=True),use_container_width=True,hide_index=True)

    elif seg_mode=="Cohort Analysis":
        if not date_cols or not cat_cols or not num_cols: st.info("Need date, category, and numeric columns.")
        else:
            c1,c2,c3=st.columns(3); dc6=c1.selectbox("Date",date_cols,key="ch_date"); cc6=c2.selectbox("Cohort (category)",cat_cols,key="ch_cat"); vc6=c3.selectbox("Value",num_cols,key="ch_val")
            period=st.selectbox("Period",["Month","Quarter","Year"],key="ch_period")
            pfmt={"Month":"M","Quarter":"Q","Year":"Y"}[period]
            df_ch=df_raw[[dc6,cc6,vc6]].dropna()
            df_ch["period"]=df_ch[dc6].dt.to_period(pfmt).astype(str)
            pivot=pd.pivot_table(df_ch,values=vc6,index=cc6,columns="period",aggfunc="sum",fill_value=0)
            pivot=pivot[sorted(pivot.columns)]
            fig=px.imshow(pivot,color_continuous_scale=PAL_SEQ,aspect="auto",title=f"Cohort — {vc6} by {cc6} × {period}",text_auto=".0f")
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=max(300,len(pivot)*22+100),margin=dict(t=50,b=20))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")

    elif seg_mode=="Percentile Bands":
        if not num_cols: st.info("No numeric columns.")
        else:
            c1,c2=st.columns(2); pc2=c1.selectbox("Column",num_cols,key="pb_col"); band_col=c2.text_input("New band column name",value=f"{num_cols[0]}_band",key="pb_name")
            labels=["Bottom 10%","10–25%","25–75%","75–90%","Top 10%"]
            bands=pd.cut(df_raw[pc2],bins=[df_raw[pc2].quantile(q) for q in [0,0.1,0.25,0.75,0.9,1.0]],labels=labels,include_lowest=True)
            vc3=bands.value_counts().reindex(labels).reset_index(); vc3.columns=["Band","Count"]
            c1b,c2b=st.columns(2)
            with c1b:
                fig=px.bar(vc3,x="Band",y="Count",color="Band",color_discrete_sequence=PAL,title=f"Percentile bands — {pc2}")
                plo(fig,height=320); fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
            with c2b:
                for label in labels:
                    s2=df_raw.loc[bands==label,pc2]
                    stat_row(label,f"{s2.min():.2f} – {s2.max():.2f}  (n={len(s2):,})")
            if st.button("➕ Save bands as column",key="pb_save"):
                st.session_state.computed_cols[band_col]=bands.values
                st.success(f"✓ Created '{band_col}'"); st.rerun()

# ════════════════════════════════════════════════════════════════
# RELATIONSHIPS
# ════════════════════════════════════════════════════════════════
with t_rel:
    if len(num_cols)<2: st.info("Need at least 2 numeric columns.")
    else:
        mode_r=st.radio("View",["Correlation heatmap","Scatter explorer","Scatter matrix","Regression analysis","Numeric vs Category","Partial correlation"],horizontal=True,key="rel_mode")
        if mode_r=="Correlation heatmap":
            corr=df_raw[num_cols].corr().round(2)
            fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),colorscale=[[0,"#4a6fa5"],[0.5,"#f0ede8"],[1,"#c9a84c"]],text=corr.values,texttemplate="%{text:.2f}",textfont_size=11,zmin=-1,zmax=1,zmid=0))
            fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",title="Correlation Matrix",height=max(350,len(num_cols)*50+100),margin=dict(t=50,b=20,l=10,r=10))
            try:
                st.plotly_chart(fig,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")
            pairs=[(corr.columns[i],corr.columns[j],corr.iloc[i,j]) for i in range(len(corr)) for j in range(i+1,len(corr))]
            pairs.sort(key=lambda x:abs(x[2]),reverse=True)
            cs2=st.columns(min(3,len(pairs[:6])))
            for i,(a,b,r) in enumerate(pairs[:6]):
                c3=("#3a8a5a" if r>0.5 else "#c0392b" if r<-0.5 else "#c9a84c")
                with cs2[i%3]: st.markdown(f'<div style="background:#fff;border-radius:8px;padding:10px 14px;border:1px solid #e5e0d8;border-left:3px solid {c3};margin-bottom:6px;"><div style="font-size:12px;color:#888;">{a} × {b}</div><div style="font-size:22px;font-weight:600;color:{c3};">{r:+.2f}</div></div>',unsafe_allow_html=True)
        elif mode_r=="Scatter explorer":
            c1,c2,c3,c4=st.columns(4)
            xc=c1.selectbox("X",num_cols,key="sc2_x"); yc=c2.selectbox("Y",num_cols,index=min(1,len(num_cols)-1),key="sc2_y")
            szc=c3.selectbox("Size",["None"]+num_cols,key="sc2_sz"); cc3=c4.selectbox("Colour",["None"]+cat_cols,key="sc2_col")
            tr=st.checkbox("Trendline",value=True,key="sc2_tr")
            fig=px.scatter(df_raw.head(2000),x=xc,y=yc,size=None if szc=="None" else szc,color=None if cc3=="None" else cc3,color_discrete_sequence=PAL,trendline="ols" if tr and cc3=="None" else None,opacity=0.65,title=f"{yc} vs {xc}")
            plo(fig,height=450); st.plotly_chart(fig,use_container_width=True)
            if tr and cc3=="None":
                s2=df_raw[[xc,yc]].dropna()
                if len(s2)>2:
                    rv,pv=stats.pearsonr(s2[xc],s2[yc]); rc2=("#3a8a5a" if abs(rv)>0.5 else "#c9a84c")
                    sig_label="<b style='color:#3a8a5a;'>Significant</b>" if pv<0.05 else "<b style='color:#aaa;'>Not significant</b>"
                    st.markdown(f'<div style="background:#fff;border-radius:8px;padding:10px 16px;border:1px solid #e5e0d8;display:inline-block;font-size:13px;">Pearson r = <b style="color:{rc2};">{rv:.3f}</b> &nbsp;|&nbsp; p-value = <b>{pv:.4f}</b> &nbsp;|&nbsp; {sig_label} (α=0.05)</div>',unsafe_allow_html=True)
        elif mode_r=="Scatter matrix":
            sm=st.multiselect("Columns",num_cols,default=num_cols[:min(4,len(num_cols))],key="smat_cols"); smcc=st.selectbox("Colour",["None"]+cat_cols,key="smat_col")
            if sm:
                fig=px.scatter_matrix(df_raw.head(1000),dimensions=sm,color=None if smcc=="None" else smcc,color_discrete_sequence=PAL,opacity=0.6)
                fig.update_traces(diagonal_visible=False,marker_size=3)
                fig.update_layout(paper_bgcolor="#fafaf8",font_family="DM Sans",height=600,margin=dict(t=30,b=20,l=20,r=20))
                try:
                    st.plotly_chart(fig,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")
        elif mode_r=="Regression analysis":
            c1,c2=st.columns(2); yc2=c1.selectbox("Target (Y)",num_cols,key="reg_y"); xcs=c2.multiselect("Features (X)",num_cols,default=num_cols[:min(3,len(num_cols))],key="reg_x")
            if xcs and st.button("▶ Run regression",key="reg_run"):
                data=df_raw[[yc2]+xcs].dropna()
                X=data[xcs].values; y=data[yc2].values
                reg=LinearRegression().fit(X,y); pred=reg.predict(X)
                r2=reg.score(X,y); residuals=y-pred
                rl,rr=st.columns(2)
                with rl:
                    fig=px.scatter(x=y,y=pred,opacity=0.6,color_discrete_sequence=[PAL[1]],title=f"Actual vs Predicted — {yc2}")
                    mn2,mx2=float(min(y.min(),pred.min())),float(max(y.max(),pred.max()))
                    fig.add_trace(go.Scatter(x=[mn2,mx2],y=[mn2,mx2],mode="lines",line=dict(color=PAL[0],dash="dash"),name="Perfect fit"))
                    plo(fig,height=360); fig.update_layout(xaxis_title="Actual",yaxis_title="Predicted")
                    try:
                        st.plotly_chart(fig,use_container_width=True)
                    except Exception as _chart_err:
                        st.warning(f"⚠️ Chart error: {_chart_err}")
                with rr:
                    fig2=px.histogram(x=residuals,nbins=30,color_discrete_sequence=[PAL[1]],title="Residual distribution")
                    plo(fig2,height=360); st.plotly_chart(fig2,use_container_width=True)
                stat_row("R² score",f"{r2:.4f}"); stat_row("RMSE",f"{np.sqrt(np.mean(residuals**2)):.4f}"); stat_row("MAE",f"{np.mean(np.abs(residuals)):.4f}")
                coef_df=pd.DataFrame({"Feature":xcs,"Coefficient":reg.coef_.round(4),"Importance %":np.abs(reg.coef_)/np.abs(reg.coef_).sum()*100}).sort_values("Importance %",ascending=False)
                fig3=px.bar(coef_df,x="Importance %",y="Feature",orientation="h",color="Importance %",color_continuous_scale=PAL_SEQ,title="Feature importance (absolute coeff %)")
                fig3.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed")); plo(fig3,height=max(200,len(xcs)*30+80))
                try:
                    st.plotly_chart(fig3,use_container_width=True)
                except Exception as _chart_err:
                    st.warning(f"⚠️ Chart error: {_chart_err}")
        elif mode_r=="Numeric vs Category":
            if not cat_cols: st.info("No categorical columns.")
            else:
                c1,c2=st.columns(2); nc3=c1.selectbox("Numeric",num_cols,key="nvc_num"); cc4=c2.selectbox("Category",cat_cols,key="nvc_cat")
                tp=st.slider("Max categories",3,20,8,key="nvc_top")
                tv=df_raw[cc4].value_counts().head(tp).index.tolist(); ds=df_raw[df_raw[cc4].isin(tv)]
                c1b,c2b=st.columns(2)
                with c1b:
                    fig=px.violin(ds,x=cc4,y=nc3,color=cc4,color_discrete_sequence=PAL,box=True,points="outliers",title=f"{nc3} by {cc4}")
                    plo(fig,height=400); fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
                with c2b:
                    gs=ds.groupby(cc4)[nc3].agg(["mean","median","std","count"]).round(2).reset_index()
                    fig2=px.bar(gs,x=cc4,y="mean",error_y="std",color="mean",color_continuous_scale=PAL_SEQ,title=f"Mean ±std — {nc3} by {cc4}")
                    fig2.update_layout(coloraxis_showscale=False); plo(fig2,height=400); st.plotly_chart(fig2,use_container_width=True)
                st.dataframe(gs,use_container_width=True,hide_index=True)
                # ANOVA
                if len(tv)>1:
                    groups=[ds[ds[cc4]==g][nc3].dropna().values for g in tv]
                    f_stat,p_val=stats.f_oneway(*groups)
                    sig="Significant difference" if p_val<0.05 else "No significant difference"
                    col_a="#3a8a5a" if p_val<0.05 else "#aaa"
                    st.markdown(f'<div style="background:#fff;border-radius:8px;padding:10px 16px;border:1px solid #e5e0d8;display:inline-block;font-size:13px;">One-way ANOVA: F={f_stat:.3f}, p={p_val:.4f} &nbsp;|&nbsp; <b style="color:{col_a};">{sig}</b> (α=0.05)</div>',unsafe_allow_html=True)
        elif mode_r=="Partial correlation":
            if len(num_cols)<3: st.info("Need at least 3 numeric columns for partial correlation.")
            else:
                c1,c2,c3=st.columns(3)
                xc2=c1.selectbox("Variable X",num_cols,key="pc_x"); yc3=c2.selectbox("Variable Y",num_cols,index=1,key="pc_y")
                ctrl=c3.multiselect("Control for",num_cols,default=[num_cols[2]] if len(num_cols)>2 else [],key="pc_ctrl")
                if ctrl and st.button("▶ Compute partial correlation",key="pc_run"):
                    data=df_raw[[xc2,yc3]+ctrl].dropna()
                    def partial_corr(data,x,y,covar):
                        from sklearn.linear_model import LinearRegression
                        def residuals(target,predictors):
                            m=LinearRegression().fit(predictors,target); return target-m.predict(predictors)
                        rx=residuals(data[x].values,data[covar].values)
                        ry=residuals(data[y].values,data[covar].values)
                        return stats.pearsonr(rx,ry)
                    pr,pp=partial_corr(data,xc2,yc3,ctrl)
                    raw_r,raw_p=stats.pearsonr(data[xc2],data[yc3])
                    cc5=("#3a8a5a" if abs(pr)>0.3 else "#c9a84c")
                    st.markdown(f"""<div style="background:#fff;border-radius:10px;padding:16px 20px;border:1px solid #e5e0d8;display:inline-block;font-size:14px;line-height:2;">
                    <b>Raw correlation</b> ({xc2} × {yc3}): <b>{raw_r:+.3f}</b> (p={raw_p:.4f})<br>
                    <b>Partial correlation</b> controlling for [{', '.join(ctrl)}]: <b style="color:{cc5};">{pr:+.3f}</b> (p={pp:.4f})
                    </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FILTER & EXPORT
# ════════════════════════════════════════════════════════════════
with t_filt:
    st.markdown('<div class="section-title">Build your query</div>',unsafe_allow_html=True)
    df_f=df_raw.copy(); af=[]
    fc1,fc2,fc3=st.columns(3)
    with fc1:
        st.markdown("**Numeric filters**")
        nfs=st.multiselect("Columns",num_cols,key="nf_sel")
        for col in nfs:
            mn,mx=float(df_raw[col].min()),float(df_raw[col].max())
            if mn<mx:
                lo,hi=st.slider(col,mn,mx,(mn,mx),key=f"nf_{col}",format="%.2f")
                df_f=df_f[(df_f[col]>=lo)&(df_f[col]<=hi)]
                if lo>mn or hi<mx: af.append(f"{col} [{lo:.2f}–{hi:.2f}]")
    with fc2:
        st.markdown("**Category filters**")
        cfs=st.multiselect("Columns",cat_cols,key="cf_sel")
        for col in cfs:
            opts=sorted(df_raw[col].dropna().unique().tolist())
            chosen=st.multiselect(col,opts,default=opts,key=f"cf_{col}")
            df_f=df_f[df_f[col].isin(chosen)]
            if len(chosen)<len(opts): af.append(f"{col}: {len(chosen)}/{len(opts)} selected")
    with fc3:
        st.markdown("**Date & search**")
        if date_cols:
            dcol=st.selectbox("Date column",date_cols,key="df_dcol")
            d1=st.date_input("From",value=df_raw[dcol].min(),key="df_d1")
            d2=st.date_input("To",value=df_raw[dcol].max(),key="df_d2")
            df_f=df_f[(df_f[dcol]>=pd.Timestamp(d1))&(df_f[dcol]<=pd.Timestamp(d2))]
        if cat_cols:
            scol=st.selectbox("Search in",cat_cols,key="df_scol"); sterm=st.text_input("Contains",key="df_sterm")
            if sterm: df_f=df_f[df_f[scol].astype(str).str.contains(sterm,case=False,na=False)]; af.append(f"'{sterm}' in {scol}")
    if af:
        st.markdown("**Active:** "+" ".join([f'<span class="col-pill">{f}</span>' for f in af]),unsafe_allow_html=True)
    st.markdown("---")
    s1,s2,s3,s4=st.columns(4)
    mp=round(len(df_f)/max(len(df_raw),1)*100,1)
    s1.metric("Matching rows",f"{len(df_f):,}",f"{mp}% of data")
    s2.metric("Filtered out",f"{len(df_raw)-len(df_f):,}")
    s3.metric("Columns",len(df_f.columns))
    if num_cols: s4.metric(f"Sum of {num_cols[0]}",f"{df_f[num_cols[0]].sum():,.0f}")
    # Aggregation on filtered
    if num_cols:
        st.markdown("**Quick aggregate on filtered data:**")
        ac1,ac2,ac3=st.columns(3)
        agg_col=ac1.selectbox("Column",num_cols,key="flt_agg_col"); agg_fn=ac2.selectbox("Function",["sum","mean","count","max","min","std"],key="flt_agg_fn")
        if cat_cols: agg_grp=ac3.selectbox("Group by",["None"]+cat_cols,key="flt_agg_grp")
        else: agg_grp="None"
        if agg_grp=="None":
            result=getattr(df_f[agg_col],agg_fn)()
            st.metric(f"{agg_fn}({agg_col})",f"{result:,.3f}")
        else:
            ag_res=df_f.groupby(agg_grp)[agg_col].agg(agg_fn).reset_index().sort_values(agg_col,ascending=False)
            fc_agg=px.bar(ag_res.head(20),x=agg_col,y=agg_grp,orientation="h",color=agg_col,color_continuous_scale=PAL_SEQ)
            fc_agg.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed")); plo(fc_agg,height=max(200,min(len(ag_res),20)*24+60))
            try:
                st.plotly_chart(fc_agg,use_container_width=True)
            except Exception as _chart_err:
                st.warning(f"⚠️ Chart error: {_chart_err}")
    col_sel=st.multiselect("Columns to export",df_raw.columns.tolist(),default=df_raw.columns.tolist(),key="filt_cols")
    df_exp=df_f[col_sel] if col_sel else df_f
    st.dataframe(df_exp.head(500),use_container_width=True,height=320)
    if len(df_exp)>500: st.caption(f"Showing 500 of {len(df_exp):,} rows")
    dl1,dl2,dl3,_=st.columns([1,1,1,2])
    with dl1:
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w: df_exp.to_excel(w,index=False)
        st.download_button("⬇ Excel",data=buf.getvalue(),file_name=f"filtered_{st.session_state.file_name or 'data.xlsx'}",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with dl2:
        st.download_button("⬇ CSV",data=df_exp.to_csv(index=False),file_name=f"filtered_{st.session_state.file_name or 'data'}.csv",mime="text/csv")
    with dl3:
        # Pivot export
        if cat_cols and num_cols:
            buf2=io.BytesIO()
            with pd.ExcelWriter(buf2,engine="openpyxl") as w:
                df_exp.to_excel(w,sheet_name="Filtered",index=False)
                if len(cat_cols)>0 and len(num_cols)>0:
                    pv=pd.pivot_table(df_exp,values=num_cols[0],index=cat_cols[0],aggfunc="sum") if cat_cols else pd.DataFrame()
                    if not pv.empty: pv.to_excel(w,sheet_name="Pivot")
            st.download_button("⬇ Excel + Pivot",data=buf2.getvalue(),file_name=f"export_pivot_{st.session_state.file_name or 'data.xlsx'}",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ════════════════════════════════════════════════════════════════
with t_emb:
    st.markdown('<div class="section-title">Embedding engine — fully local, no API calls</div>', unsafe_allow_html=True)

    emb_mode = st.radio(
        "What to embed",
        ["📝 Text columns — semantic search & similarity",
         "🔢 Numeric columns — similarity search & anomaly detection"],
        horizontal=True, key="emb_mode"
    )

    # ── TEXT EMBEDDINGS ──────────────────────────────────────────
    if emb_mode.startswith("📝"):
        if not cat_cols:
            st.info("No text columns found in this dataset.")
        else:
            from sentence_transformers import SentenceTransformer
            import hashlib

            text_col = st.selectbox("Text column to embed", cat_cols, key="emb_text_col")
            model_choice = st.selectbox("Model (local, downloads once)",
                ["all-MiniLM-L6-v2  (~22 MB, fast)", "all-mpnet-base-v2  (~420 MB, accurate)"],
                key="emb_model")
            model_name = model_choice.split()[0]

            color_col = st.selectbox("Colour points by (optional)", ["None"] + cat_cols + num_cols, key="emb_color")

            _emb_max = max(2, min(2000, len(df_raw)))
            _emb_min = min(2, _emb_max)
            _emb_def = min(max(2, min(500, len(df_raw))), _emb_max)
            sample_n = st.slider("Max rows to embed (larger = slower)", _emb_min, _emb_max, _emb_def, max(1, _emb_max//20), key="emb_sample") if _emb_max > _emb_min else st.number_input("Rows to embed", value=_emb_max, disabled=True, key="emb_sample_n") or _emb_max

            emb_key = hashlib.md5(f"{sname}{text_col}{model_name}{sample_n}".encode()).hexdigest()

            if st.button("▶ Compute embeddings", key="emb_run_text"):
                with st.spinner(f"Loading {model_name} and embedding {sample_n} rows..."):
                    try:
                        model = SentenceTransformer(model_name)
                        df_sample = df_raw[[text_col] + ([color_col] if color_col != "None" else [])].dropna(subset=[text_col]).head(sample_n).reset_index(drop=True)
                        texts = df_sample[text_col].astype(str).tolist()
                        embs = model.encode(texts, show_progress_bar=False, batch_size=64)
                        st.session_state["emb_matrix"] = embs
                        st.session_state["emb_texts"]  = texts
                        st.session_state["emb_df"]     = df_sample
                        st.session_state["emb_key"]    = emb_key
                        st.success(f"✓ Embedded {len(texts)} rows into {embs.shape[1]}-dim vectors")
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")

            if st.session_state.get("emb_matrix") is not None:
                embs     = st.session_state["emb_matrix"]
                texts    = st.session_state["emb_texts"]
                df_emb   = st.session_state["emb_df"]

                text_sub = st.radio("Explore",
                    ["Semantic search", "Embedding space (2D)", "Near-duplicate finder", "Semantic clusters"],
                    horizontal=True, key="emb_text_sub")

                # ── Semantic search
                if text_sub == "Semantic search":
                    query = st.text_input("🔍 Search query (natural language)", placeholder="e.g. 'high radiation near equator'", key="emb_query")
                    _tk_max = max(1, min(20, len(df_raw)))
                    top_k = st.slider("Top K results", 1, _tk_max, min(5, _tk_max), key="emb_topk")
                    if query.strip():
                        with st.spinner("Searching..."):
                            try:
                                model = SentenceTransformer(model_name)
                                q_emb = model.encode([query])
                                sims  = cosine_similarity(q_emb, embs)[0]
                                top_idx = np.argsort(sims)[::-1][:top_k]
                                results = df_emb.iloc[top_idx].copy()
                                results.insert(0, "Similarity", sims[top_idx].round(3))
                                search_header = f"Top {top_k} results for: {query}"
                                st.markdown(f"**{search_header}**")
                                st.dataframe(results, use_container_width=True, hide_index=True)
                                fig = px.bar(x=sims[top_idx], y=[texts[i][:60]+"…" for i in top_idx],
                                    orientation="h", color=sims[top_idx],
                                    color_continuous_scale=PAL_SEQ,
                                    title="Cosine similarity scores")
                                fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                                plo(fig, height=max(200, top_k*32+60))
                                try: st.plotly_chart(fig, use_container_width=True)
                                except Exception as _e: st.warning(f"Chart error: {_e}")
                            except Exception as e:
                                st.error(f"Search failed: {e}")

                # ── Embedding space
                elif text_sub == "Embedding space (2D)":
                    proj_method = st.radio("Projection method",
                        ["UMAP (non-linear, better clusters)", "PCA (linear, faster)"] if UMAP_AVAILABLE else ["PCA (linear, faster)"],
                        horizontal=True, key="emb_proj")
                    with st.spinner("Projecting to 2D..."):
                        try:
                            if "UMAP" in proj_method and UMAP_AVAILABLE:
                                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                                coords  = reducer.fit_transform(embs)
                            else:
                                coords = PCA(n_components=2, random_state=42).fit_transform(embs)
                            df_proj = pd.DataFrame(coords, columns=["Dim 1", "Dim 2"])
                            df_proj["text_preview"] = [t[:80] for t in texts]
                            if color_col != "None" and color_col in df_emb.columns:
                                df_proj[color_col] = df_emb[color_col].values
                                color_arg = color_col
                            else:
                                color_arg = None
                            fig = px.scatter(df_proj, x="Dim 1", y="Dim 2",
                                color=color_arg, hover_data=["text_preview"],
                                color_discrete_sequence=PAL, opacity=0.75,
                                title=f"Embedding space — {text_col}")
                            plo(fig, height=500)
                            try: st.plotly_chart(fig, use_container_width=True)
                            except Exception as _e: st.warning(f"Chart error: {_e}")
                        except Exception as e:
                            st.error(f"Projection failed: {e}")

                # ── Near-duplicate finder
                elif text_sub == "Near-duplicate finder":
                    threshold = st.slider("Similarity threshold (higher = stricter)", 0.80, 0.99, 0.92, 0.01, key="emb_thresh")
                    with st.spinner("Finding near-duplicates..."):
                        try:
                            sim_matrix = cosine_similarity(embs)
                            pairs = []
                            for i in range(len(texts)):
                                for j in range(i+1, len(texts)):
                                    if sim_matrix[i,j] >= threshold:
                                        pairs.append({"Row A": i, "Text A": texts[i][:80],
                                                      "Row B": j, "Text B": texts[j][:80],
                                                      "Similarity": round(float(sim_matrix[i,j]), 4)})
                            if pairs:
                                pairs_df = pd.DataFrame(pairs).sort_values("Similarity", ascending=False)
                                st.markdown(f"**Found {len(pairs_df)} near-duplicate pairs** (similarity ≥ {threshold})")
                                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                                fig = px.histogram(pairs_df, x="Similarity", nbins=20,
                                    color_discrete_sequence=[PAL[1]], title="Similarity distribution of near-duplicate pairs")
                                plo(fig, height=260)
                                try: st.plotly_chart(fig, use_container_width=True)
                                except Exception as _e: st.warning(f"Chart error: {_e}")
                            else:
                                st.success(f"✓ No near-duplicates found above {threshold} similarity threshold.")
                        except Exception as e:
                            st.error(f"Duplicate search failed: {e}")

                # ── Semantic clusters
                elif text_sub == "Semantic clusters":
                    n_clusters = st.slider("Number of clusters", 2, 10, 4, key="emb_n_clust")
                    with st.spinner("Clustering embeddings..."):
                        try:
                            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            cluster_labels = km.fit_predict(embs)

                            # Project to 2D for viz
                            if UMAP_AVAILABLE:
                                coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embs)
                            else:
                                coords = PCA(n_components=2, random_state=42).fit_transform(embs)

                            df_clust = pd.DataFrame(coords, columns=["Dim 1","Dim 2"])
                            df_clust["Cluster"] = cluster_labels.astype(str)
                            df_clust["text_preview"] = [t[:80] for t in texts]

                            fig = px.scatter(df_clust, x="Dim 1", y="Dim 2", color="Cluster",
                                hover_data=["text_preview"], color_discrete_sequence=CLUSTER_COLORS,
                                opacity=0.75, title=f"Semantic clusters — {text_col}")
                            plo(fig, height=480)
                            try: st.plotly_chart(fig, use_container_width=True)
                            except Exception as _e: st.warning(f"Chart error: {_e}")

                            # Show top examples per cluster
                            st.markdown("**Representative samples per cluster:**")
                            cluster_cols = st.columns(min(n_clusters, 4))
                            for c in range(n_clusters):
                                with cluster_cols[c % 4]:
                                    idxs = np.where(cluster_labels == c)[0]
                                    # Pick the row closest to centroid
                                    centroid = km.cluster_centers_[c]
                                    dists = np.linalg.norm(embs[idxs] - centroid, axis=1)
                                    top_ex = [texts[idxs[i]][:100] for i in np.argsort(dists)[:3]]
                                    st.markdown(f'<div class="compute-card"><b style="color:#c9a84c;">Cluster {c}</b> ({len(idxs)} rows)<br><br>'+"<br><br>".join([f'<span style="font-size:12px;color:#555;">• {t}</span>' for t in top_ex])+"</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Clustering failed: {e}")

    # ── NUMERIC EMBEDDINGS ───────────────────────────────────────
    else:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for numeric embeddings.")
        else:
            feat_cols_emb = st.multiselect("Features to use as embedding vectors",
                num_cols, default=num_cols[:min(6, len(num_cols))], key="emb_num_feats")
            color_col_n = st.selectbox("Colour by (optional)", ["None"] + cat_cols, key="emb_num_color")

            if feat_cols_emb:
                df_num = df_raw[feat_cols_emb].dropna().reset_index(drop=True)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_num)

                num_sub = st.radio("Analysis",
                    ["Embedding space (2D/3D)", "Similarity search", "Anomaly detection", "Distance matrix"],
                    horizontal=True, key="emb_num_sub")

                # ── Embedding space
                if num_sub == "Embedding space (2D/3D)":
                    dim = st.radio("Dimensions", ["2D","3D"], horizontal=True, key="emb_num_dim")
                    proj_method = st.radio("Method",
                        ["UMAP", "PCA"] if UMAP_AVAILABLE else ["PCA"],
                        horizontal=True, key="emb_num_proj")
                    n_dim = 3 if dim == "3D" else 2
                    with st.spinner("Projecting..."):
                        try:
                            if "UMAP" in proj_method and UMAP_AVAILABLE:
                                coords = umap.UMAP(n_components=n_dim, random_state=42).fit_transform(X_scaled)
                            else:
                                coords = PCA(n_components=n_dim, random_state=42).fit_transform(X_scaled)

                            df_proj = pd.DataFrame(coords, columns=[f"Dim {i+1}" for i in range(n_dim)])
                            if color_col_n != "None" and color_col_n in df_raw.columns:
                                df_proj[color_col_n] = df_raw[color_col_n].iloc[df_num.index].values

                            if dim == "2D":
                                fig = px.scatter(df_proj, x="Dim 1", y="Dim 2",
                                    color=None if color_col_n=="None" else color_col_n,
                                    color_discrete_sequence=PAL, opacity=0.7,
                                    title=f"Numeric embedding space ({proj_method})")
                                plo(fig, height=500)
                                try: st.plotly_chart(fig, use_container_width=True)
                                except Exception as _e: st.warning(f"Chart error: {_e}")
                            else:
                                fig = px.scatter_3d(df_proj, x="Dim 1", y="Dim 2", z="Dim 3",
                                    color=None if color_col_n=="None" else color_col_n,
                                    color_discrete_sequence=PAL, opacity=0.7)
                                fig.update_traces(marker_size=3)
                                fig.update_layout(paper_bgcolor="#fafaf8", font_family="DM Sans", height=540)
                                try: st.plotly_chart(fig, use_container_width=True)
                                except Exception as _e: st.warning(f"Chart error: {_e}")
                        except Exception as e:
                            st.error(f"Projection failed: {e}")

                # ── Similarity search
                elif num_sub == "Similarity search":
                    st.markdown("Select a row and find the most similar rows in the dataset.")
                    row_idx = st.number_input("Row index to query", min_value=0, max_value=len(df_num)-1, value=0, key="emb_row_idx")
                    _sk_max = max(1, min(20, len(df_num)-1))
                    top_k   = st.slider("Top K similar rows", 1, _sk_max, min(5, _sk_max), key="emb_sim_k")
                    metric  = st.radio("Distance metric", ["Cosine similarity", "Euclidean distance"], horizontal=True, key="emb_metric")

                    try:
                        query_vec = X_scaled[row_idx].reshape(1, -1)
                        if metric == "Cosine similarity":
                            scores = cosine_similarity(query_vec, X_scaled)[0]
                            scores[row_idx] = -1  # exclude self
                            top_idx = np.argsort(scores)[::-1][:top_k]
                            score_label = "Cosine similarity"
                            score_vals  = scores[top_idx]
                        else:
                            dists = euclidean_distances(query_vec, X_scaled)[0]
                            dists[row_idx] = np.inf
                            top_idx = np.argsort(dists)[:top_k]
                            score_label = "Euclidean distance"
                            score_vals  = dists[top_idx]

                        st.markdown(f"**Query row {row_idx}:**")
                        st.dataframe(df_num.iloc[[row_idx]], use_container_width=True, hide_index=True)
                        st.markdown(f"**Top {top_k} most similar rows:**")
                        result = df_raw.iloc[top_idx].copy().reset_index(drop=True)
                        result.insert(0, score_label, score_vals.round(4))
                        st.dataframe(result, use_container_width=True, hide_index=True)

                        # Radar comparison
                        if len(feat_cols_emb) >= 3:
                            fig = go.Figure()
                            query_vals = scaler.inverse_transform(query_vec)[0].tolist()
                            fig.add_trace(go.Scatterpolar(r=query_vals+[query_vals[0]],
                                theta=feat_cols_emb+[feat_cols_emb[0]], fill="toself",
                                name=f"Query (row {row_idx})", line_color=PAL[0]))
                            for rank, idx in enumerate(top_idx[:3]):
                                vals = scaler.inverse_transform(X_scaled[idx].reshape(1,-1))[0].tolist()
                                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],
                                    theta=feat_cols_emb+[feat_cols_emb[0]], fill="toself",
                                    name=f"Match {rank+1} (row {idx})", line_color=PAL[rank+1]))
                            fig.update_layout(polar=dict(bgcolor="#fafaf8",
                                radialaxis=dict(visible=True, gridcolor="#ddd"),
                                angularaxis=dict(gridcolor="#ddd")),
                                paper_bgcolor="#fafaf8", font_family="DM Sans",
                                height=420, title="Query vs top matches (radar)", showlegend=True)
                            try: st.plotly_chart(fig, use_container_width=True)
                            except Exception as _e: st.warning(f"Chart error: {_e}")
                    except Exception as e:
                        st.error(f"Similarity search failed: {e}")

                # ── Anomaly detection
                elif num_sub == "Anomaly detection":
                    contamination = st.slider("Expected anomaly % (contamination)", 1, 20, 5, key="emb_contam") / 100
                    n_estimators  = st.slider("Isolation Forest trees", 50, 300, 100, 50, key="emb_trees")

                    if st.button("▶ Detect anomalies", key="emb_anom_run"):
                        with st.spinner("Running Isolation Forest..."):
                            try:
                                iso = IsolationForest(contamination=contamination,
                                    n_estimators=n_estimators, random_state=42)
                                preds  = iso.fit_predict(X_scaled)
                                scores = iso.decision_function(X_scaled)

                                df_anom = df_raw.iloc[df_num.index].copy().reset_index(drop=True)
                                df_anom["anomaly_score"] = scores
                                df_anom["is_anomaly"]    = (preds == -1)

                                n_anom = (preds == -1).sum()
                                al, ar = st.columns([1,3])
                                with al:
                                    st.metric("Anomalies found", f"{n_anom:,}", f"{round(n_anom/len(preds)*100,1)}% of rows")
                                    st.metric("Normal rows", f"{(preds==1).sum():,}")

                                with ar:
                                    # Project to 2D and colour by anomaly
                                    if UMAP_AVAILABLE:
                                        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
                                    else:
                                        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
                                    df_viz = pd.DataFrame(coords, columns=["Dim 1","Dim 2"])
                                    df_viz["Status"] = ["🔴 Anomaly" if p==-1 else "🔵 Normal" for p in preds]
                                    df_viz["Score"]  = scores.round(4)
                                    fig = px.scatter(df_viz, x="Dim 1", y="Dim 2", color="Status",
                                        color_discrete_map={"🔴 Anomaly":"#c0392b","🔵 Normal":"#4a6fa5"},
                                        hover_data=["Score"], opacity=0.75,
                                        title="Anomaly detection — embedding space")
                                    plo(fig, height=420)
                                    try: st.plotly_chart(fig, use_container_width=True)
                                    except Exception as _e: st.warning(f"Chart error: {_e}")

                                # Score distribution
                                fig2 = px.histogram(df_anom, x="anomaly_score", nbins=40,
                                    color="is_anomaly",
                                    color_discrete_map={True:"#c0392b", False:"#4a6fa5"},
                                    title="Anomaly score distribution (lower = more anomalous)",
                                    barmode="overlay", opacity=0.75)
                                plo(fig2, height=280)
                                try: st.plotly_chart(fig2, use_container_width=True)
                                except Exception as _e: st.warning(f"Chart error: {_e}")

                                st.markdown("**Top anomalous rows:**")
                                top_anom = df_anom[df_anom["is_anomaly"]].nsmallest(20, "anomaly_score")
                                st.dataframe(top_anom.drop(columns=["is_anomaly"]), use_container_width=True, hide_index=True)

                                # Download anomalies
                                buf = io.BytesIO()
                                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                                    top_anom.to_excel(w, index=False)
                                st.download_button("⬇ Download anomalies", data=buf.getvalue(),
                                    file_name="anomalies.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            except Exception as e:
                                st.error(f"Anomaly detection failed: {e}")

                # ── Distance matrix
                elif num_sub == "Distance matrix":
                    _dm_max = max(2, min(200, len(df_num)))
                    _dm_def = min(max(2, min(50, len(df_num))), _dm_max)
                    max_rows = st.slider("Rows to include (sample)", min(2, _dm_max), _dm_max, _dm_def, max(1, _dm_max//10), key="emb_dist_n") if _dm_max > 2 else _dm_max
                    label_col_d = st.selectbox("Row labels", ["Row index"] + cat_cols, key="emb_dist_label")

                    try:
                        sample = X_scaled[:max_rows]
                        if label_col_d != "Row index" and label_col_d in df_raw.columns:
                            labels_d = df_raw[label_col_d].iloc[:max_rows].astype(str).tolist()
                        else:
                            labels_d = [str(i) for i in range(max_rows)]

                        dist_mat = cosine_similarity(sample)
                        fig = go.Figure(go.Heatmap(
                            z=dist_mat, x=labels_d, y=labels_d,
                            colorscale=PAL_SEQ, zmin=0, zmax=1,
                            hovertemplate="Row A: %{y}<br>Row B: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
                        ))
                        fig.update_layout(paper_bgcolor="#fafaf8", font_family="DM Sans",
                            title=f"Cosine similarity matrix (first {max_rows} rows)",
                            height=max(400, max_rows*14+100),
                            margin=dict(t=50, b=20, l=10, r=10))
                        try: st.plotly_chart(fig, use_container_width=True)
                        except Exception as _e: st.warning(f"Chart error: {_e}")

                        # Most similar & most dissimilar pairs
                        pairs = [(labels_d[i], labels_d[j], float(dist_mat[i,j]))
                            for i in range(len(labels_d)) for j in range(i+1, len(labels_d))]
                        pairs_df = pd.DataFrame(pairs, columns=["Row A","Row B","Similarity"])
                        pl, pr = st.columns(2)
                        with pl:
                            st.markdown("**Most similar pairs:**")
                            st.dataframe(pairs_df.nlargest(10, "Similarity").round(4), use_container_width=True, hide_index=True)
                        with pr:
                            st.markdown("**Most dissimilar pairs:**")
                            st.dataframe(pairs_df.nsmallest(10, "Similarity").round(4), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Distance matrix failed: {e}")

# ════════════════════════════════════════════════════════════════
# SMART ALERTS
# ════════════════════════════════════════════════════════════════
with t_alerts:

    st.markdown("""
    <style>
    .alert-card{border-radius:10px;padding:14px 18px;margin-bottom:10px;border:1px solid;}
    .alert-crit{background:#fff5f5;border-color:#e74c3c;border-left:5px solid #e74c3c;}
    .alert-warn{background:#fffbf0;border-color:#f39c12;border-left:5px solid #f39c12;}
    .alert-info{background:#f0f8ff;border-color:#3498db;border-left:5px solid #3498db;}
    .alert-good{background:#f0fff4;border-color:#27ae60;border-left:5px solid #27ae60;}
    .alert-title{font-weight:600;font-size:14px;margin-bottom:4px;}
    .alert-body{font-size:13px;color:#444;line-height:1.55;}
    .alert-rec{font-size:12px;color:#666;margin-top:6px;padding-top:6px;
               border-top:1px solid rgba(0,0,0,0.07);font-style:italic;}
    .alert-meta{font-size:11px;color:#999;margin-top:4px;}
    .sev-badge{display:inline-block;padding:2px 8px;border-radius:12px;
               font-size:10px;font-weight:700;text-transform:uppercase;
               letter-spacing:0.06em;margin-right:6px;}
    .sev-CRITICAL{background:#e74c3c;color:#fff;}
    .sev-WARNING{background:#f39c12;color:#fff;}
    .sev-INFO{background:#3498db;color:#fff;}
    .sev-GOOD{background:#27ae60;color:#fff;}
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def run_alert_engine(df_json, sheet_key):
        df = pd.read_json(io.StringIO(df_json), orient="split")
        for col in df.columns:
            if re.search(r'date|time|dt|_at|_on', col, re.I):
                try: df[col] = pd.to_datetime(df[col], errors="coerce")
                except: pass

        num_cols_a  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols_a  = [c for c in df.columns if df[c].dtype == object]
        date_cols_a = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        alerts = []

        def add(severity, category, title, body, recommendation, col=None, value=None):
            alerts.append({"severity": severity, "category": category,
                           "title": title, "body": body,
                           "recommendation": recommendation,
                           "col": col, "value": value})

        sev_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "GOOD": 3}

        # ── 1. Data quality ──────────────────────────────────────
        total_nulls = df.isnull().sum().sum()
        null_pct_total = total_nulls / max(df.size, 1) * 100
        if null_pct_total > 20:
            add("CRITICAL", "Data Quality",
                f"{null_pct_total:.1f}% of all values are missing",
                f"Your dataset has {total_nulls:,} missing values across {df.shape[0]:,} rows and {df.shape[1]} columns. "
                f"This is a high proportion that will bias any analysis or model trained on this data.",
                "Consider imputing missing values (mean/median for numeric, mode for categorical) "
                "or dropping columns with >50% missing before modelling.")
        elif null_pct_total > 5:
            add("WARNING", "Data Quality",
                f"{null_pct_total:.1f}% of values are missing",
                f"{total_nulls:,} missing values detected. Most analyses will drop these rows automatically.",
                "Review which columns have the most nulls and decide on an imputation strategy.")

        worst_null_col = df.isnull().sum().idxmax()
        worst_null_pct = df[worst_null_col].isnull().mean() * 100
        if worst_null_pct > 30:
            add("WARNING", "Data Quality",
                f"Column '{worst_null_col}' is {worst_null_pct:.0f}% empty",
                f"{worst_null_pct:.0f}% of rows have no value for '{worst_null_col}'. "
                f"This column may not be reliable for analysis.",
                f"Consider dropping '{worst_null_col}' from models or imputing with the column median/mode.")

        dups = df.duplicated().sum()
        if dups > 0:
            dup_pct = dups / len(df) * 100
            add("WARNING" if dup_pct > 5 else "INFO", "Data Quality",
                f"{dups:,} duplicate rows detected ({dup_pct:.1f}%)",
                f"Exact duplicate rows can inflate metrics and skew model training. "
                f"{dups} rows are identical to at least one other row.",
                "Run df.drop_duplicates() before modelling. Investigate whether duplicates "
                "are genuine repeated events or data entry errors.")

        # ── 2. Outlier alerts ────────────────────────────────────
        for col in num_cols_a:
            s = df[col].dropna()
            if len(s) < 10: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            extreme = s[(s < q1 - 4*iqr) | (s > q3 + 4*iqr)]
            moderate = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]
            if len(extreme) > 0:
                max_val = s.max(); min_val = s.min()
                add("CRITICAL", "Outliers",
                    f"'{col}' has {len(extreme)} extreme outlier(s)",
                    f"{len(extreme)} value(s) in '{col}' are more than 4× the IQR from the median. "
                    f"Range: {min_val:,.2f} – {max_val:,.2f}. Median: {s.median():,.2f}. "
                    f"These values are likely errors or exceptional events.",
                    f"Inspect these {len(extreme)} row(s) manually. If they are errors, remove or cap them. "
                    f"If genuine, flag them separately before modelling.", col=col, value=len(extreme))
            elif len(moderate) > len(s) * 0.10:
                add("WARNING", "Outliers",
                    f"'{col}' has {len(moderate)} outliers ({len(moderate)/len(s)*100:.0f}% of rows)",
                    f"More than 10% of values in '{col}' fall outside the normal IQR range. "
                    f"This suggests either a heavy-tailed distribution or systematic data quality issues.",
                    f"Plot a histogram of '{col}' to check the distribution. "
                    f"Consider log-transforming if right-skewed.", col=col)

        # ── 3. Skew / distribution alerts ────────────────────────
        for col in num_cols_a:
            s = df[col].dropna()
            if len(s) < 30: continue
            skew = float(s.skew())
            if abs(skew) > 3:
                direction = "right" if skew > 0 else "left"
                add("WARNING", "Distribution",
                    f"'{col}' is heavily {direction}-skewed (skew = {skew:.2f})",
                    f"A skew of {skew:.2f} means the distribution of '{col}' is far from normal. "
                    f"{'Most values are low with a long tail of very high values.' if skew > 0 else 'Most values are high with a long tail of very low values.'} "
                    f"This will reduce the effectiveness of linear models.",
                    f"Apply a {'log' if skew > 0 else 'square root'} transform to '{col}' "
                    f"before using it in regression or distance-based models. "
                    f"Use the Compute tab → Normalize to do this.", col=col, value=round(skew,2))

        # ── 4. Correlation alerts ────────────────────────────────
        if len(num_cols_a) >= 2:
            corr = df[num_cols_a].corr()
            for i in range(len(num_cols_a)):
                for j in range(i+1, len(num_cols_a)):
                    r = corr.iloc[i,j]
                    if abs(r) > 0.90:
                        ca, cb = num_cols_a[i], num_cols_a[j]
                        add("WARNING", "Correlation",
                            f"'{ca}' and '{cb}' are nearly perfectly correlated (r = {r:.2f})",
                            f"A correlation of {r:.2f} means these two columns contain almost identical information. "
                            f"Including both in a model causes multicollinearity, inflating standard errors "
                            f"and making coefficients unreliable.",
                            f"Drop one of these columns before modelling. "
                            f"Keep the one that is more directly interpretable.")
                    elif abs(r) > 0.75:
                        ca, cb = num_cols_a[i], num_cols_a[j]
                        add("INFO", "Correlation",
                            f"Strong correlation between '{ca}' and '{cb}' (r = {r:.2f})",
                            f"These columns move together strongly. This may be a useful predictive relationship "
                            f"or a sign of redundancy depending on your use case.",
                            f"Explore this relationship in the Relationships tab → Scatter explorer.")

        # ── 5. Trend alerts (time series) ────────────────────────
        for dc in date_cols_a:
            for nc in num_cols_a[:3]:
                try:
                    ts = df[[dc, nc]].dropna().sort_values(dc)
                    if len(ts) < 20: continue
                    ts["t"] = (ts[dc] - ts[dc].min()).dt.days
                    slope, intercept, r, p, _ = stats.linregress(ts["t"], ts[nc])
                    if p < 0.05 and abs(r) > 0.4:
                        total_change = slope * ts["t"].max()
                        pct_change   = total_change / max(abs(ts[nc].mean()), 1e-9) * 100
                        direction    = "increasing" if slope > 0 else "declining"
                        sev = "WARNING" if abs(pct_change) > 30 else "INFO"
                        add(sev, "Trend",
                            f"'{nc}' is {direction} over time ({pct_change:+.0f}% trend)",
                            f"Linear regression on '{nc}' vs '{dc}' shows a statistically significant "
                            f"{'upward' if slope > 0 else 'downward'} trend "
                            f"(r = {r:.2f}, p = {p:.4f}). "
                            f"Over the full time range, '{nc}' {'grew' if slope > 0 else 'fell'} "
                            f"by approximately {abs(pct_change):.0f}%.",
                            f"Investigate what drove this {'growth' if slope > 0 else 'decline'}. "
                            f"Use the Time Series tab to explore further.", col=nc)
                except: pass

        # ── 6. Low-variance / constant column alerts ─────────────
        for col in num_cols_a:
            s = df[col].dropna()
            if len(s) < 10: continue
            cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
            if cv < 0.01 and s.nunique() > 1:
                add("INFO", "Data Quality",
                    f"'{col}' has near-zero variance (CV = {cv:.4f})",
                    f"'{col}' barely changes across rows. It will contribute almost no predictive "
                    f"signal to a model and may slow down training.",
                    f"Consider dropping '{col}' from model features. "
                    f"Check if it should be a constant/config value rather than a data column.")

        # ── 7. Category imbalance alerts ─────────────────────────
        for col in cat_cols_a:
            vc = df[col].value_counts(normalize=True)
            if len(vc) < 2: continue
            if vc.iloc[0] > 0.80:
                add("INFO", "Imbalance",
                    f"'{col}' is dominated by '{vc.index[0]}' ({vc.iloc[0]*100:.0f}%)",
                    f"One category makes up {vc.iloc[0]*100:.0f}% of all values in '{col}'. "
                    f"If used as a model target, this class imbalance will cause the model "
                    f"to predict the majority class almost always.",
                    f"If '{col}' is a classification target, use SMOTE, class weights, "
                    f"or stratified sampling to balance the classes.")

        # ── 8. Isolation Forest — ML anomalies ───────────────────
        if len(num_cols_a) >= 2:
            try:
                feat_data = df[num_cols_a].dropna()
                if len(feat_data) >= 20:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.ensemble import IsolationForest
                    sc  = StandardScaler()
                    X   = sc.fit_transform(feat_data)
                    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
                    preds = iso.fit_predict(X)
                    n_anom = (preds == -1).sum()
                    pct    = n_anom / len(feat_data) * 100
                    if n_anom > 0:
                        add("WARNING" if pct > 3 else "INFO", "ML Anomaly",
                            f"Isolation Forest flagged {n_anom} anomalous rows ({pct:.1f}%)",
                            f"An Isolation Forest model trained on all {len(num_cols_a)} numeric columns "
                            f"identified {n_anom} rows ({pct:.1f}%) as statistically unusual — "
                            f"they are isolated quickly in the feature space, suggesting they differ "
                            f"significantly from the rest of the data in multiple dimensions simultaneously.",
                            f"Inspect these rows in the Embeddings tab → Anomaly Detection for full details. "
                            f"They may represent fraud, sensor faults, data entry errors, or genuine outlier events.")
            except: pass

        # ── 9. Positive signals ──────────────────────────────────
        null_pct_cols = df.isnull().sum() / len(df) * 100
        clean_cols = (null_pct_cols == 0).sum()
        if clean_cols == len(df.columns):
            add("GOOD", "Data Quality",
                "No missing values — dataset is complete",
                "Every column in this dataset is fully populated. "
                "This is excellent data quality and means no imputation is required.",
                "Proceed directly to modelling or analysis.")

        if dups == 0:
            add("GOOD", "Data Quality",
                "No duplicate rows detected",
                "All rows in the dataset are unique. "
                "There is no risk of double-counting inflating your metrics.",
                "Good to go — no deduplication needed.")

        alerts.sort(key=lambda x: (sev_order.get(x["severity"], 99),))
        return alerts

    # Run engine
    with st.spinner("Scanning dataset for issues, patterns and anomalies..."):
        try:
            alerts = run_alert_engine(P["df_json"], sname)
        except Exception as e:
            st.error(f"Alert engine error: {e}")
            alerts = []

    # ── Summary row ──────────────────────────────────────────────
    counts = {"CRITICAL":0,"WARNING":0,"INFO":0,"GOOD":0}
    for a in alerts: counts[a["severity"]] = counts.get(a["severity"],0)+1

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total alerts", len(alerts))
    m2.metric("🔴 Critical",  counts["CRITICAL"])
    m3.metric("🟡 Warning",   counts["WARNING"])
    m4.metric("🔵 Info",      counts["INFO"])
    m5.metric("✅ Good",      counts["GOOD"])

    st.markdown("---")

    # ── Filter controls ──────────────────────────────────────────
    fc1, fc2, _ = st.columns([1,1,2])
    sev_filter = fc1.multiselect("Filter by severity",
        ["CRITICAL","WARNING","INFO","GOOD"],
        default=["CRITICAL","WARNING","INFO","GOOD"], key="alert_sev")
    cat_filter = fc2.multiselect("Filter by category",
        sorted(set(a["category"] for a in alerts)),
        default=sorted(set(a["category"] for a in alerts)), key="alert_cat")

    filtered = [a for a in alerts
                if a["severity"] in sev_filter and a["category"] in cat_filter]

    if not filtered:
        st.info("No alerts match your current filters.")
    else:
        st.markdown(f"**Showing {len(filtered)} alert(s)**")
        st.markdown("")

        style_map = {
            "CRITICAL": ("alert-crit","#e74c3c"),
            "WARNING":  ("alert-warn","#f39c12"),
            "INFO":     ("alert-info","#3498db"),
            "GOOD":     ("alert-good","#27ae60"),
        }
        icon_map = {
            "CRITICAL":"🔴","WARNING":"🟡","INFO":"🔵","GOOD":"✅"
        }

        for a in filtered:
            css_class, color = style_map.get(a["severity"], ("alert-info","#3498db"))
            icon = icon_map.get(a["severity"],"ℹ️")
            cat_pill = f'<span style="background:{color}22;color:{color};font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;margin-left:6px;">{a["category"]}</span>'
            st.markdown(f"""
            <div class="alert-card {css_class}">
              <div class="alert-title">
                <span class="sev-badge sev-{a["severity"]}">{icon} {a["severity"]}</span>
                {cat_pill}
                {a["title"]}
              </div>
              <div class="alert-body">{a["body"]}</div>
              <div class="alert-rec">💡 <b>Recommendation:</b> {a["recommendation"]}</div>
              <div class="alert-meta">Sheet: {sname} · Auto-detected by alert engine</div>
            </div>""", unsafe_allow_html=True)

    # ── Export report ────────────────────────────────────────────
    st.markdown("---")
    el, er = st.columns([1,3])
    with el:
        if alerts:
            report_df = pd.DataFrame([{
                "Severity":       a["severity"],
                "Category":       a["category"],
                "Alert":          a["title"],
                "Detail":         a["body"],
                "Recommendation": a["recommendation"],
                "Column":         a.get("col",""),
                "Sheet":          sname,
            } for a in alerts])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                report_df.to_excel(w, index=False, sheet_name="Alerts")
            st.download_button("⬇ Download alert report",
                data=buf.getvalue(),
                file_name=f"alerts_{sname}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with er:
        st.caption(
            "Alert engine checks: missing values · duplicates · extreme outliers (4×IQR) · "
            "moderate outliers (1.5×IQR) · heavy skew · near-perfect correlations · "
            "time-series trends (linear regression p<0.05) · low variance columns · "
            "category imbalance · multi-dimensional anomalies (Isolation Forest)"
        )