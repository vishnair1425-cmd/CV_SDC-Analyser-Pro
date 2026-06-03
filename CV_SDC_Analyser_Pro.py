"""
Cyclic Voltammetry (CV) Charge Integration App — multi-file
===========================================================
Upload MULTIPLE Excel files (name them "point 1" … "point 10"). For each file:
  - Combined CV plot (Potential vs WE(1).Current (A)) coloured by scan.
  - Peak current / voltage tables, ΔEp, draw/drag-baseline per-scan integration
    (Q = ∫ I dt over time, in coulombs), and trend-vs-scan plots.
  - Peak current vs time for anode & cathode, each fitted to
        I(t) = C + A*exp(-k*t)
    where k is the rate constant for surface change. C, A, k and R² are tabled.
Select which file to view from a dropdown. A final cross-file section gives a
point-wise summary table, six comparison curves, and a fit-parameter table.

Expected columns:  Potential , WE(1).Current (A) , scan , and a time column.

Run with:  streamlit run cv_app.py
Requirements: streamlit, pandas, numpy, plotly, openpyxl, scipy
"""

import io
import re
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import curve_fit

st.set_page_config(page_title="CV Charge Integration", layout="wide")

PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Dark24
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def guess_column(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for low, orig in cols_lower.items():
            if low == cand.lower():
                return orig
    for cand in candidates:
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None


def point_sort_key(name):
    """Sort filenames by the first integer found (so 'point 2' < 'point 10')."""
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 10**9, name)


def point_label(name):
    """Human label 'Point N' from a filename containing a number, else the name."""
    m = re.search(r"(\d+)", name)
    return f"Point {m.group(1)}" if m else name


def safe_key(text):
    """Alphanumeric-only key safe for HTML ids and Streamlit element keys."""
    return re.sub(r"[^0-9a-zA-Z]", "", str(text))


@st.cache_data(show_spinner=False)
def load_excel(file_bytes, sheet_name):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def trapz_signed(xvar, current, baseline):
    """Integrate (current - baseline) over xvar (TIME, in seconds); split into
    positive / negative areas. Zero-crossings within a segment are split at the
    crossing for accuracy. With xvar = time, the result is charge in coulombs."""
    X = np.asarray(xvar, dtype=float)
    diff = np.asarray(current, dtype=float) - np.asarray(baseline, dtype=float)
    pos = neg = 0.0
    for i in range(len(X) - 1):
        dt = X[i + 1] - X[i]
        y0, y1 = diff[i], diff[i + 1]
        if y0 >= 0 and y1 >= 0:
            pos += 0.5 * (y0 + y1) * dt
        elif y0 <= 0 and y1 <= 0:
            neg += 0.5 * (y0 + y1) * dt
        else:
            t = y0 / (y0 - y1) if y1 != y0 else 0.5
            a1 = 0.5 * y0 * dt * t
            a2 = 0.5 * y1 * dt * (1 - t)
            for a in (a1, a2):
                if a >= 0:
                    pos += a
                else:
                    neg += a
    return pos, neg


# ----------------------------------------------------------------------------
# Exponential decay fit:  I(t) = C + A*exp(-k*t)
# t is measured from the first peak (t = 0) for numerical stability; this
# leaves C and k unchanged and only rescales A.
# ----------------------------------------------------------------------------

def exp_model(t, C, A, k):
    return C + A * np.exp(-k * t)


def fit_exp_decay(t, y):
    """Fit I(t)=C+A*exp(-k*t). Returns dict with C, A, k, R2, and the t-origin,
    or None if there are too few points / the fit fails."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 4:
        return None
    t_min = t.min()
    t0 = t - t_min                       # t = 0 at first peak
    span = t0.max() if t0.max() > 0 else 1.0
    C0 = y[-1]
    A0 = y[0] - y[-1]
    k0 = 1.0 / span
    try:
        popt, _ = curve_fit(exp_model, t0, y, p0=[C0, A0, k0], maxfev=20000)
        C, A, k = (float(v) for v in popt)
        yhat = exp_model(t0, C, A, k)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"C": C, "A": A, "k": k, "R2": r2, "t_min": float(t_min)}
    except Exception:
        return None


def peak_time_series(work):
    """Per-scan peak times and peak currents for anode & cathode.
    Returns (anodic_t, anodic_I, cathodic_t, cathodic_I) as numpy arrays,
    ordered by scan."""
    scans = list(work["scan"].unique())
    try:
        scans = sorted(scans, key=lambda s: float(s))
    except (ValueError, TypeError):
        scans = sorted(scans)
    at, ai, ct, ci = [], [], [], []
    for s in scans:
        sub = work[work["scan"] == s]
        if sub.empty:
            continue
        i_max = sub["current"].idxmax()
        i_min = sub["current"].idxmin()
        at.append(float(sub.loc[i_max, "time"]))
        ai.append(float(sub.loc[i_max, "current"]))
        ct.append(float(sub.loc[i_min, "time"]))
        ci.append(float(sub.loc[i_min, "current"]))
    return (np.array(at), np.array(ai), np.array(ct), np.array(ci))


def fit_plot(t, y, fit, title, color, key):
    """Scatter of peak current vs (time since first peak) with fitted curve."""
    f = go.Figure()
    if len(t):
        t_rel = t - t.min()
        f.add_trace(go.Scatter(
            x=t_rel, y=y, mode="markers", name="Peak current",
            marker=dict(size=9, color=color),
            hovertemplate="t=%{x:.1f} s<br>I=%{y:.4g} A<extra></extra>",
        ))
        if fit is not None:
            tt = np.linspace(t_rel.min(), t_rel.max(), 200)
            yy = exp_model(tt, fit["C"], fit["A"], fit["k"])
            f.add_trace(go.Scatter(
                x=tt, y=yy, mode="lines", name="C + A·exp(−k·t)",
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
            ))
    f.update_layout(
        xaxis_title="Time since first peak (s)", yaxis_title="Peak current (A)",
        height=360, margin=dict(l=70, r=20, t=30, b=45),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.markdown(f"**{title}**")
    st.plotly_chart(f, use_container_width=True, key=key)


def fit_params_table(anod_fit, cath_fit):
    """Build a small dataframe of fit parameters for anodic & cathodic."""
    def row(label, fit):
        if fit is None:
            return {"Region": label, "C": np.nan, "A": np.nan,
                    "k (rate constant, 1/s)": np.nan, "R²": np.nan}
        return {"Region": label, "C": fit["C"], "A": fit["A"],
                "k (rate constant, 1/s)": fit["k"], "R²": fit["R2"]}
    return pd.DataFrame([row("Anodic", anod_fit), row("Cathodic", cath_fit)])


# ----------------------------------------------------------------------------
# Draw/drag-baseline component (HTML + Plotly.js)
# ----------------------------------------------------------------------------

def draggable_cv_component(potential, current, time, color, x_title, y_title,
                           default_x1, default_y1, default_x2, default_y2,
                           dom_key, key_height=440):
    """Render a Plotly chart (Potential X vs Current Y) with a draw/drag linear
    baseline. Charge is integrated over TIME -> coulombs, live in the browser."""
    P = [float(v) for v in potential]
    I = [float(v) for v in current]
    T = [float(v) for v in time]

    pmin, pmax = min(P), max(P)
    imin, imax = min(I), max(I)
    imin = min(imin, 0.0)
    imax = max(imax, 0.0)
    px_pad = (pmax - pmin) * 0.05 or 0.01
    iy_pad = (imax - imin) * 0.10 or abs(imax) * 0.1 or 1e-9
    x_range = [pmin - px_pad, pmax + px_pad]
    y_range = [imin - iy_pad, imax + iy_pad]

    payload = json.dumps({
        "P": P, "I": I, "T": T, "color": color,
        "xTitle": x_title, "yTitle": y_title,
        "x1": default_x1, "y1": default_y1,
        "x2": default_x2, "y2": default_y2,
        "xRange": x_range, "yRange": y_range,
        "k": dom_key,
    })

    html = """
<div id="toolbar_K">
  <button id="btnDraw_K" class="tbtn">\u270F\uFE0F Draw baseline</button>
  <button id="btnReset_K" class="tbtn">\u21BA Reset</button>
  <span class="hint">Tip: click "Draw baseline", then press &amp; drag on the plot. Or drag either red endpoint directly.</span>
</div>
<div id="root_K"></div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function(){
const D = __PAYLOAD__;
const K = D.k;
const P = D.P, I = D.I, T = D.T;
let A1 = {x: D.x1, y: D.y1};
let A2 = {x: D.x2, y: D.y2};
const unit = "C";

function baselineAt(x){
  if (A2.x === A1.x) return A1.y;
  const m = (A2.y - A1.y)/(A2.x - A1.x);
  const b = A1.y - m*A1.x;
  return m*x + b;
}
function integrate(){
  let pos=0, neg=0;
  for(let i=0;i<P.length-1;i++){
    const dt = T[i+1]-T[i];
    const y0 = I[i]   - baselineAt(P[i]);
    const y1 = I[i+1] - baselineAt(P[i+1]);
    if(y0>=0 && y1>=0){ pos += 0.5*(y0+y1)*dt; }
    else if(y0<=0 && y1<=0){ neg += 0.5*(y0+y1)*dt; }
    else {
      const t = (y1!==y0) ? y0/(y0-y1) : 0.5;
      const a1 = 0.5*y0*dt*t, a2 = 0.5*y1*dt*(1-t);
      [a1,a2].forEach(a => { if(a>=0) pos+=a; else neg+=a; });
    }
  }
  return {pos, neg};
}
function baselineYs(){ return P.map(baselineAt); }
function fmt(v){ return v.toExponential(4); }

function buildData(){
  const ys = baselineYs();
  const posCurve = I.map((v,i) => Math.max(v, ys[i]));
  const negCurve = I.map((v,i) => Math.min(v, ys[i]));
  const curve = {x:P, y:I, mode:"lines", name:"CV",
    line:{color:D.color, width:2},
    hovertemplate:"E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>"};
  const base = {x:P, y:ys, mode:"lines", name:"Baseline",
    line:{color:"black", width:2, dash:"dash"}, hoverinfo:"skip"};
  const fillPos = {x: P.concat(P.slice().reverse()),
    y: posCurve.concat(ys.slice().reverse()),
    fill:"toself", fillcolor:"rgba(214,39,40,0.20)",
    line:{color:"rgba(0,0,0,0)"}, hoverinfo:"skip", name:"Positive area"};
  const fillNeg = {x: P.concat(P.slice().reverse()),
    y: negCurve.concat(ys.slice().reverse()),
    fill:"toself", fillcolor:"rgba(31,119,180,0.20)",
    line:{color:"rgba(0,0,0,0)"}, hoverinfo:"skip", name:"Negative area"};
  const anchors = {x:[A1.x, A2.x], y:[A1.y, A2.y], mode:"markers",
    name:"Drag me", marker:{color:"red", size:14, symbol:"circle",
      line:{color:"white", width:2}},
    hovertemplate:"drag<br>E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>"};
  return [fillPos, fillNeg, curve, base, anchors];
}

const layout = {
  height: __HEIGHT__,
  margin:{l:70, r:20, t:10, b:50},
  xaxis:{title:{text:D.xTitle}, zeroline:false, range:D.xRange.slice(), autorange:false},
  yaxis:{title:{text:D.yTitle}, zeroline:true, range:D.yRange.slice(), autorange:false},
  template:"plotly_white",
  legend:{orientation:"h", yanchor:"bottom", y:1.02, xanchor:"right", x:1},
  dragmode:false
};

const DEF1 = {x: D.x1, y: D.y1};
const DEF2 = {x: D.x2, y: D.y2};

const gd = document.getElementById("root_"+K);
Plotly.newPlot(gd, buildData(), layout,
  {displayModeBar:true, responsive:true,
   modeBarButtonsToRemove:["lasso2d","select2d"]});

function refresh(){
  Plotly.react(gd, buildData(), gd.layout || layout);
  const r = integrate();
  document.getElementById("out_"+K).innerHTML =
    '<span class="pos">\u25A0 Positive charge: '+fmt(r.pos)+' '+unit+'</span>'+
    '<span class="neg">\u25A0 Negative charge: '+fmt(r.neg)+' '+unit+'</span>'+
    '<span class="net">Net: '+fmt(r.pos+r.neg)+' '+unit+
    '  &middot;  |Total|: '+fmt(Math.abs(r.pos)+Math.abs(r.neg))+' '+unit+'</span>';
}

function pixelToData(e){
  const bb = gd.getBoundingClientRect();
  const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
  const px = e.clientX - bb.left - gd._fullLayout.margin.l;
  const py = e.clientY - bb.top  - gd._fullLayout.margin.t;
  return {x: xa.p2d(px), y: ya.p2d(py)};
}
function nearestWithin(e, tol){
  const d = pixelToData(e);
  const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
  function dist(A){
    const dx = xa.d2p(A.x) - xa.d2p(d.x);
    const dy = ya.d2p(A.y) - ya.d2p(d.y);
    return Math.sqrt(dx*dx+dy*dy);
  }
  const d1 = dist(A1), d2 = dist(A2);
  const which = d1 <= d2 ? 1 : 2;
  return (Math.min(d1,d2) < tol) ? which : null;
}

let drawMode = false, dragging = null, drawingNew = false;
const btnDraw  = document.getElementById("btnDraw_"+K);
const btnReset = document.getElementById("btnReset_"+K);

function setDrawMode(on){
  drawMode = on;
  btnDraw.classList.toggle("active", on);
  btnDraw.textContent = on ? "\u270F\uFE0F Drawing… (click & drag)" : "\u270F\uFE0F Draw baseline";
  gd.style.cursor = on ? "crosshair" : "default";
}
btnDraw.addEventListener("click", () => setDrawMode(!drawMode));
btnReset.addEventListener("click", () => {
  A1 = {x: DEF1.x, y: DEF1.y};
  A2 = {x: DEF2.x, y: DEF2.y};
  setDrawMode(false);
  refresh();
});

gd.addEventListener("mousedown", e => {
  if(drawMode){
    const d = pixelToData(e);
    A1 = {x: d.x, y: d.y};
    A2 = {x: d.x, y: d.y};
    drawingNew = true;
    e.preventDefault();
    refresh();
  } else {
    const which = nearestWithin(e, 25);
    if(which){ dragging = which; e.preventDefault(); }
  }
});
window.addEventListener("mousemove", e => {
  if(drawingNew){
    A2 = pixelToData(e);
    refresh();
  } else if(dragging){
    const d = pixelToData(e);
    if(dragging===1){ A1 = d; } else { A2 = d; }
    refresh();
  }
});
window.addEventListener("mouseup", () => {
  if(drawingNew){ drawingNew = false; setDrawMode(false); }
  dragging = null;
});

refresh();
})();
</script>
<style>
  #toolbar_K{font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin-bottom:6px;
           display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
  .tbtn{font-size:14px; padding:6px 12px; border:1px solid #c7c7c7; border-radius:6px;
        background:#f6f6f6; cursor:pointer;}
  .tbtn:hover{background:#ececec;}
  .tbtn.active{background:#1f77b4; color:#fff; border-color:#1f77b4;}
  #toolbar_K .hint{font-size:12px; color:#888;}
  #out_K{font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin-top:10px;
       display:flex; gap:18px; flex-wrap:wrap; font-size:15px;}
  #out_K .pos{color:#d62728; font-weight:600;}
  #out_K .neg{color:#1f77b4; font-weight:600;}
  #out_K .net{color:#444;}
</style>
<div id="out_K"></div>
"""
    html = (html.replace("__PAYLOAD__", payload)
                .replace("__HEIGHT__", str(key_height))
                .replace("_K", "_" + dom_key))
    components.html(html, height=key_height + 130, scrolling=False)


# ----------------------------------------------------------------------------
# Per-file analysis (all the original features for one file)
# ----------------------------------------------------------------------------

def analyse_file(work, pot_col, cur_col, view_key):
    scans = list(work["scan"].unique())
    try:
        scans = sorted(scans, key=lambda s: float(s))
    except (ValueError, TypeError):
        scans = sorted(scans)
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scans)}
    unit_label = "C"

    # Combined plot
    st.subheader("Combined CV — all scans")
    fig = go.Figure()
    for s in scans:
        sub = work[work["scan"] == s]
        fig.add_trace(go.Scatter(
            x=sub["potential"], y=sub["current"], mode="lines",
            name=f"Scan {s}", line=dict(color=color_map[s], width=1.5),
            hovertemplate="Scan " + s + "<br>E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=pot_col, yaxis_title=cur_col, height=560,
        legend=dict(title="Scan"), margin=dict(l=60, r=20, t=20, b=50),
        template="plotly_white", hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"combined_{view_key}")

    # Peak tables (also captures peak times for the exp-fit section)
    st.subheader("Peak current / voltage per scan")
    st.caption(
        "Positive region = maximum (anodic) current; Negative region = minimum "
        "(cathodic) current. Peak Voltage is the potential at that peak current. "
        "ΔEp = anodic peak voltage − cathodic peak voltage."
    )
    peak_metrics = {}
    pos_rows, neg_rows = [], []
    for s in scans:
        sub = work[work["scan"] == s]
        if sub.empty:
            continue
        i_max = sub["current"].idxmax()
        i_min = sub["current"].idxmin()
        peak_metrics[s] = {
            "v_anodic": float(sub.loc[i_max, "potential"]),
            "i_anodic": float(sub.loc[i_max, "current"]),
            "t_anodic": float(sub.loc[i_max, "time"]),
            "v_cathodic": float(sub.loc[i_min, "potential"]),
            "i_cathodic": float(sub.loc[i_min, "current"]),
            "t_cathodic": float(sub.loc[i_min, "time"]),
        }
        peak_metrics[s]["dEp"] = peak_metrics[s]["v_anodic"] - peak_metrics[s]["v_cathodic"]
        pos_rows.append({"Scan": s, "Peak Current (A)": peak_metrics[s]["i_anodic"],
                         "Peak Voltage (V)": peak_metrics[s]["v_anodic"]})
        neg_rows.append({"Scan": s, "Peak Current (A)": peak_metrics[s]["i_cathodic"],
                         "Peak Voltage (V)": peak_metrics[s]["v_cathodic"]})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Positive region (anodic peak)**")
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Negative region (cathodic peak)**")
        st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)

    # Per-scan draw/drag integration
    st.subheader("Per-scan integration — draw or drag the baseline")
    st.caption(
        "Default baseline is horizontal at Y = 0 (the current axis). Area above the "
        "baseline is shaded **red** (positive), below it **blue** (negative). "
        "Charge is integrated as ∫ I dt over time (coulombs). "
        "Click **Draw baseline** then press-and-drag, or drag either red endpoint. "
        "**Reset** returns to Y = 0."
    )
    summary_rows = []
    for s in scans:
        sub = work[work["scan"] == s].reset_index(drop=True)
        if len(sub) < 2:
            continue
        x1 = float(sub["potential"].min()); y1 = 0.0
        x2 = float(sub["potential"].max()); y2 = 0.0
        dom_key = safe_key(f"{view_key}{s}")
        with st.expander(f"Scan {s}", expanded=(s == scans[0])):
            st.markdown(f"#### Scan {s}")
            draggable_cv_component(
                potential=sub["potential"].tolist(),
                current=sub["current"].tolist(),
                time=sub["time"].tolist(),
                color=color_map[s], x_title=pot_col, y_title=cur_col,
                default_x1=x1, default_y1=y1, default_x2=x2, default_y2=y2,
                dom_key=dom_key, key_height=440,
            )
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1); b = y1 - m * x1
            bl = m * sub["potential"].values + b
        else:
            bl = np.full(len(sub), y1)
        pos_area, neg_area = trapz_signed(sub["time"].values,
                                          sub["current"].values, bl)
        pm = peak_metrics.get(s, {})
        ratio = (neg_area / pos_area) if pos_area != 0 else float("nan")
        summary_rows.append({
            "Scan": s,
            "Anodic peak V (V)": pm.get("v_anodic", float("nan")),
            "Cathodic peak V (V)": pm.get("v_cathodic", float("nan")),
            "ΔEp (V)": pm.get("dEp", float("nan")),
            "Anodic peak I (A)": pm.get("i_anodic", float("nan")),
            "Cathodic peak I (A)": pm.get("i_cathodic", float("nan")),
            f"Positive charge ({unit_label})": pos_area,
            f"Negative charge ({unit_label})": neg_area,
            f"Net charge ({unit_label})": pos_area + neg_area,
            "Neg/Pos charge ratio": ratio,
        })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary

    # Summary + download
    st.subheader("Charge summary — all scans (default baseline)")
    st.caption("Charge = ∫ I dt over the time column, default Y = 0 baseline (coulombs).")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download charge summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="cv_charge_summary.csv", mime="text/csv",
        key=f"dl_{view_key}",
    )

    # Trend-vs-scan plots
    st.subheader("Trends vs scan number")
    try:
        x_scan = [float(s) for s in summary["Scan"]]
    except (ValueError, TypeError):
        x_scan = list(range(1, len(summary) + 1))
    pos_q_col = f"Positive charge ({unit_label})"
    neg_q_col = f"Negative charge ({unit_label})"

    def trend_chart(y, y_title, color):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=x_scan, y=list(y), mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7, color=color),
            hovertemplate="Scan %{x}<br>%{y:.4g}<extra></extra>",
        ))
        f.update_layout(xaxis_title="Scan", yaxis_title=y_title, height=320,
                        margin=dict(l=70, r=20, t=30, b=45),
                        template="plotly_white", showlegend=False)
        return f

    trends = [
        ("a) Positive (anodic) peak voltage vs scan", summary["Anodic peak V (V)"],   "Anodic peak V (V)",   "#d62728"),
        ("b) Negative (cathodic) peak voltage vs scan", summary["Cathodic peak V (V)"], "Cathodic peak V (V)", "#1f77b4"),
        ("c) Positive (anodic) peak current vs scan", summary["Anodic peak I (A)"],   "Anodic peak I (A)",   "#d62728"),
        ("d) Negative (cathodic) peak current vs scan", summary["Cathodic peak I (A)"], "Cathodic peak I (A)", "#1f77b4"),
        ("e) Peak voltage difference ΔEp vs scan", summary["ΔEp (V)"],                "ΔEp (V)",             "#2ca02c"),
        ("f) Positive charge vs scan", summary[pos_q_col],                            pos_q_col,             "#d62728"),
        ("g) Negative charge vs scan", summary[neg_q_col],                            neg_q_col,             "#1f77b4"),
        ("h) |Neg/Pos| charge ratio vs scan", summary["Neg/Pos charge ratio"].abs(), "|Neg/Pos| charge ratio","#9467bd"),
    ]
    for i in range(0, len(trends), 2):
        ccols = st.columns(2)
        for j, (col, (title, ydata, ytitle, color)) in enumerate(zip(ccols, trends[i:i + 2])):
            with col:
                st.markdown(f"**{title}**")
                st.plotly_chart(trend_chart(ydata, ytitle, color),
                                use_container_width=True,
                                key=f"trend_{view_key}_{i}_{j}")

    # Peak current vs time with exponential fit  I(t) = C + A*exp(-k*t)
    st.subheader("Peak current vs time — exponential fit  I(t) = C + A·exp(−k·t)")
    st.caption(
        "k is the **rate constant for surface change**. Time is measured from "
        "the first peak (t = 0). Fit by non-linear least squares."
    )
    at, ai, ct, ci = peak_time_series(work)
    anod_fit = fit_exp_decay(at, ai)
    cath_fit = fit_exp_decay(ct, ci)
    fc1, fc2 = st.columns(2)
    with fc1:
        fit_plot(at, ai, anod_fit, "Anodic peak current vs time", "#d62728",
                 key=f"fit_anod_{view_key}")
    with fc2:
        fit_plot(ct, ci, cath_fit, "Cathodic peak current vs time", "#1f77b4",
                 key=f"fit_cath_{view_key}")
    ptbl = fit_params_table(anod_fit, cath_fit)
    st.markdown("**Fitted parameters**")
    st.dataframe(ptbl, use_container_width=True, hide_index=True)
    if anod_fit is None or cath_fit is None:
        st.caption("Note: a fit is shown only when a region has at least 4 scans "
                   "and the optimisation converges.")

    return summary


def per_scan_summary_silent(work):
    """Per-scan metrics for a file WITHOUT rendering (default Y = 0 baseline)."""
    scans = list(work["scan"].unique())
    try:
        scans = sorted(scans, key=lambda s: float(s))
    except (ValueError, TypeError):
        scans = sorted(scans)
    rows = []
    for s in scans:
        sub = work[work["scan"] == s].reset_index(drop=True)
        if len(sub) < 2:
            continue
        i_max = sub["current"].idxmax()
        i_min = sub["current"].idxmin()
        bl = np.full(len(sub), 0.0)
        pos_area, neg_area = trapz_signed(sub["time"].values,
                                          sub["current"].values, bl)
        rows.append({
            "Anodic peak I (A)": float(sub.loc[i_max, "current"]),
            "Cathodic peak I (A)": float(sub.loc[i_min, "current"]),
            "Positive charge (C)": pos_area,
            "Negative charge (C)": neg_area,
        })
    return pd.DataFrame(rows)


def representative_metrics(summary, mode):
    if summary.empty:
        return None
    pos_col = [c for c in summary.columns if c.startswith("Positive charge")][0]
    neg_col = [c for c in summary.columns if c.startswith("Negative charge")][0]
    if mode == "First scan":
        row = summary.iloc[0]
    elif mode == "Mean across scans":
        row = summary.mean(numeric_only=True)
    else:
        row = summary.iloc[-1]
    anodic_I = float(row["Anodic peak I (A)"])
    cathodic_I = float(row["Cathodic peak I (A)"])
    anodic_Q = float(row[pos_col])
    cathodic_Q = float(row[neg_col])
    ratio = abs(cathodic_Q / anodic_Q) if anodic_Q != 0 else float("nan")
    return {
        "Peak Anodic Current (A)": anodic_I,
        "abs(Peak Cathodic Current) (A)": abs(cathodic_I),
        "Anodic charge (C)": anodic_Q,
        "Cathodic charge (C)": cathodic_Q,
        "Net charge (C)": anodic_Q + cathodic_Q,
        "abs(Cathodic/Anodic charge) ratio": ratio,
    }


# ----------------------------------------------------------------------------
# Sidebar - input & configuration
# ----------------------------------------------------------------------------

st.title("Cyclic Voltammetry — Charge Integration (multi-file)")
st.caption(
    "Upload several Excel files (name them 'point 1' … 'point 10'). View each "
    "file's full analysis from the dropdown; see the point-wise comparison below."
)

with st.sidebar:
    st.header("1 · Data")
    uploads = st.file_uploader(
        "Excel files (.xlsx / .xls) — name them 'point N'",
        type=["xlsx", "xls"], accept_multiple_files=True,
    )

if not uploads:
    st.info("⬅️ Upload one or more Excel files to begin. Expected columns: "
            "`Potential`, `WE(1).Current (A)`, `scan`, and a time column. "
            "Name the files 'point 1' … 'point 10' for identification.")
    st.stop()

files = sorted(uploads, key=lambda f: point_sort_key(f.name))
file_names = [f.name for f in files]
file_by_name = {f.name: f for f in files}

with st.sidebar:
    st.header("2 · View")
    selected_name = st.selectbox("Select file to analyse", file_names, index=0)

    st.header("3 · Columns")
    sample_bytes = file_by_name[selected_name].getvalue()
    sheets = list_sheets(sample_bytes)
    sheet = st.selectbox("Sheet (applied to all files)", sheets, index=0)
    cols = list(load_excel(sample_bytes, sheet).columns)
    pot_guess = guess_column(["Potential"], cols) or cols[0]
    cur_guess = guess_column(["WE(1).Current (A)", "Current (A)"], cols) or cols[0]
    scan_guess = guess_column(["scan", "cycle"], cols) or cols[0]
    time_guess = guess_column(["Corrected time", "Time (s)", "Time", "t (s)", "time"], cols) or cols[0]
    pot_col = st.selectbox("Potential (X)", cols, index=cols.index(pot_guess))
    cur_col = st.selectbox("Current (Y)", cols, index=cols.index(cur_guess))
    scan_col = st.selectbox("Scan / Cycle", cols, index=cols.index(scan_guess))
    time_col = st.selectbox("Time (s) — for charge integration & fits", cols,
                            index=cols.index(time_guess))

    st.header("4 · Point-wise representative scan")
    rep_mode = st.radio(
        "Value used per point in the comparison",
        ["Last scan (steady state)", "First scan", "Mean across scans"],
        index=0,
    )


def prepare(file_obj):
    df = load_excel(file_obj.getvalue(), sheet)
    if not all(c in df.columns for c in (scan_col, pot_col, cur_col, time_col)):
        return None
    w = df[[scan_col, pot_col, cur_col, time_col]].copy()
    w.columns = ["scan", "potential", "current", "time"]
    w = w.dropna(subset=["potential", "current", "time"])
    w["scan"] = w["scan"].astype(str)
    return w


# ----------------------------------------------------------------------------
# Selected-file analysis
# ----------------------------------------------------------------------------

st.markdown(f"## Analysis: {selected_name}")
view_key = safe_key(selected_name)

work = prepare(file_by_name[selected_name])
if work is None or work.empty:
    st.error(f"'{selected_name}' is missing one of the required columns "
             f"({pot_col}, {cur_col}, {scan_col}, {time_col}) or has no valid rows.")
else:
    analyse_file(work, pot_col, cur_col, view_key)

# ----------------------------------------------------------------------------
# Point-wise comparison across all files
# ----------------------------------------------------------------------------

st.markdown("---")
st.header("Point-wise comparison")
st.caption(
    f"One representative value per file using: **{rep_mode}**. "
    "Anodic = positive charge, Cathodic = negative charge. "
    "Charges use the default Y = 0 baseline."
)

rows = []
fit_rows = []
for name in file_names:
    w = prepare(file_by_name[name])
    if w is None or w.empty:
        continue
    label = point_label(name)
    # representative metrics row
    s_df = per_scan_summary_silent(w)
    rep = representative_metrics(s_df, rep_mode)
    if rep is not None:
        rep_row = {"Point": label}
        rep_row.update(rep)
        rows.append(rep_row)
    # exponential fit parameters (per point, anodic & cathodic)
    at, ai, ct, ci = peak_time_series(w)
    af = fit_exp_decay(at, ai)
    cf = fit_exp_decay(ct, ci)
    fit_rows.append({
        "Point": label,
        "Anodic C": af["C"] if af else np.nan,
        "Anodic A": af["A"] if af else np.nan,
        "Anodic k (1/s)": af["k"] if af else np.nan,
        "Anodic R²": af["R2"] if af else np.nan,
        "Cathodic C": cf["C"] if cf else np.nan,
        "Cathodic A": cf["A"] if cf else np.nan,
        "Cathodic k (1/s)": cf["k"] if cf else np.nan,
        "Cathodic R²": cf["R2"] if cf else np.nan,
    })

if not rows:
    st.warning("No files produced valid metrics. Check the column selections.")
else:
    pw = pd.DataFrame(rows)
    st.dataframe(pw, use_container_width=True, hide_index=True)
    st.download_button(
        "Download point-wise comparison (CSV)",
        pw.to_csv(index=False).encode("utf-8"),
        file_name="point_wise_comparison.csv", mime="text/csv",
        key="dl_pointwise",
    )

    x_pts = pw["Point"].tolist()

    def pw_chart(col, color):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=x_pts, y=pw[col].tolist(), mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8, color=color),
            hovertemplate="%{x}<br>%{y:.4g}<extra></extra>",
        ))
        f.update_layout(xaxis_title="Point", yaxis_title=col, height=330,
                        margin=dict(l=70, r=20, t=30, b=45),
                        template="plotly_white", showlegend=False)
        return f

    pw_curves = [
        ("Peak Anodic Current (A)",            "#d62728"),
        ("abs(Peak Cathodic Current) (A)",     "#1f77b4"),
        ("Anodic charge (C)",                  "#d62728"),
        ("Cathodic charge (C)",                "#1f77b4"),
        ("Net charge (C)",                     "#2ca02c"),
        ("abs(Cathodic/Anodic charge) ratio",  "#9467bd"),
    ]
    for i in range(0, len(pw_curves), 2):
        ccols = st.columns(2)
        for j, (col, (metric, color)) in enumerate(zip(ccols, pw_curves[i:i + 2])):
            with col:
                st.markdown(f"**{metric} vs point**")
                st.plotly_chart(pw_chart(metric, color),
                                use_container_width=True, key=f"pw_{i}_{j}")

# Cross-point exponential-fit parameter table
if fit_rows:
    st.subheader("Exponential fit parameters per point  —  I(t) = C + A·exp(−k·t)")
    st.caption(
        "k = rate constant for surface change (1/s). Time measured from each "
        "file's first peak. Anodic uses anodic peak current vs time; cathodic "
        "uses cathodic peak current vs time."
    )
    fit_df = pd.DataFrame(fit_rows)
    st.dataframe(fit_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download fit parameters (CSV)",
        fit_df.to_csv(index=False).encode("utf-8"),
        file_name="exp_fit_parameters.csv", mime="text/csv",
        key="dl_fitparams",
    )

    # k vs point comparison (the headline quantity)
    if "Anodic k (1/s)" in fit_df:
        kfig = go.Figure()
        kfig.add_trace(go.Scatter(
            x=fit_df["Point"], y=fit_df["Anodic k (1/s)"], mode="lines+markers",
            name="Anodic k", line=dict(color="#d62728", width=2),
            marker=dict(size=8)))
        kfig.add_trace(go.Scatter(
            x=fit_df["Point"], y=fit_df["Cathodic k (1/s)"], mode="lines+markers",
            name="Cathodic k", line=dict(color="#1f77b4", width=2),
            marker=dict(size=8)))
        kfig.update_layout(
            xaxis_title="Point", yaxis_title="k (rate constant, 1/s)",
            height=360, margin=dict(l=70, r=20, t=30, b=45),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.markdown("**Rate constant k vs point**")
        st.plotly_chart(kfig, use_container_width=True, key="k_vs_point")
