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
  - Anodic / Cathodic CHARGE vs time (one charge per cycle plotted against the
    time that marks the beginning of that cycle's anodic / cathodic scan), with a
    user-selectable empirical fit:
        Option 1:  Q(t) = Qbase + A(1−exp(−k1·t)) − B(1−exp(−k2·t))
                   k1 = growth rate constant, k2 = decay rate constant
        Option 2:  Q(t) = C + A·exp(−k·t)
    Fitted parameters and R² are tabled and the fitted curve is plotted, both per
    file and as a per-point cross-comparison table.

Across files:
  - Overlay of WE(1).Current (A) vs time. Tick which points to include; each
    point is drawn as ONE thin line in ONE colour (all its scans joined), with a
    legend entry per point.
  - Point-wise comparison tables/plots and fit-parameter tables.

Scan 1 handling (keep or remove) is selectable in the sidebar and applies to all
trend graphs (a–h), the exponential fits, and the charge-vs-time fits/tables.
The raw peak tables, per-scan integration, and the current-vs-time overlay always
show every scan.

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

# Empirical models offered for the Charge-vs-time fit (dropdown labels).
MODEL_GROWTH_DECAY = "Option 1:  Q(t) = Qbase + A(1−exp(−k₁·t)) − B(1−exp(−k₂·t))"
MODEL_SINGLE_EXP = "Option 2:  Q(t) = C + A·exp(−k·t)"

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


def sorted_scans(work):
    """Unique scan labels in numeric order where possible."""
    scans = list(work["scan"].unique())
    try:
        scans = sorted(scans, key=lambda s: float(s))
    except (ValueError, TypeError):
        scans = sorted(scans)
    return scans


@st.cache_data(show_spinner=False)
def load_excel(file_bytes, sheet_name):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def trapz_signed(xvar, current, baseline):
    """Integrate (current - baseline) over xvar (TIME, in seconds); split into
    positive / negative areas. With xvar = time, the result is charge in coulombs."""
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
# t measured from the first (used) peak; this leaves C and k unchanged.
# ----------------------------------------------------------------------------

def exp_model(t, C, A, k):
    return C + A * np.exp(-k * t)


def fit_exp_decay(t, y):
    """Fit I(t)=C+A*exp(-k*t). Returns dict with C, A, k, R2, or None."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 4:
        return None
    t0 = t - t.min()
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
        return {"C": C, "A": A, "k": k, "R2": r2}
    except Exception:
        return None


def peak_time_series(work, skip_first=False):
    """Per-scan peak times & peak currents for anode & cathode, ordered by scan.
    If skip_first, the first scan (scan 1) is excluded."""
    scans = sorted_scans(work)
    if skip_first and len(scans) > 1:
        scans = scans[1:]
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
        xaxis_title="Time since first used peak (s)", yaxis_title="Peak current (A)",
        height=360, margin=dict(l=70, r=20, t=30, b=45),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.markdown(f"**{title}**")
    st.plotly_chart(f, use_container_width=True, key=key)


def fit_params_table(anod_fit, cath_fit):
    def row(label, fit):
        if fit is None:
            return {"Region": label, "C": np.nan, "A": np.nan,
                    "k (rate constant, 1/s)": np.nan, "R²": np.nan}
        return {"Region": label, "C": fit["C"], "A": fit["A"],
                "k (rate constant, 1/s)": fit["k"], "R²": fit["R2"]}
    return pd.DataFrame([row("Anodic", anod_fit), row("Cathodic", cath_fit)])


# ----------------------------------------------------------------------------
# Charge-vs-time analysis
# ----------------------------------------------------------------------------

def growth_decay_model(t, Qbase, A, B, k1, k2):
    return Qbase + A * (1.0 - np.exp(-k1 * t)) - B * (1.0 - np.exp(-k2 * t))


def fit_growth_decay(t, Q):
    """Fit Q(t)=Qbase+A(1-exp(-k1 t))-B(1-exp(-k2 t)). 5 free parameters, so at
    least 5 cycles are needed. Rate constants are constrained to be ≥ 0."""
    t = np.asarray(t, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if len(t) < 5:
        return None
    t0 = t - t.min()
    span = t0.max() if t0.max() > 0 else 1.0
    Qbase0 = float(Q[0])
    total = float(Q[-1] - Q[0])
    amp = float(np.max(Q) - np.min(Q)) or 1.0
    A0 = amp
    B0 = amp - total
    k1_0 = 3.0 / span
    k2_0 = 1.0 / span
    lower = [-np.inf, -np.inf, -np.inf, 0.0, 0.0]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf]
    try:
        popt, _ = curve_fit(
            growth_decay_model, t0, Q,
            p0=[Qbase0, A0, B0, k1_0, k2_0],
            bounds=(lower, upper), max_nfev=40000,
        )
        Qbase, A, B, k1, k2 = (float(v) for v in popt)
        yhat = growth_decay_model(t0, Qbase, A, B, k1, k2)
        ss_res = float(np.sum((Q - yhat) ** 2))
        ss_tot = float(np.sum((Q - np.mean(Q)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"Qbase": Qbase, "A": A, "B": B, "k1": k1, "k2": k2, "R2": r2}
    except Exception:
        return None


def charge_time_series(work, skip_first=True):
    """Per-cycle charge vs cycle-start-time."""
    scans = sorted_scans(work)
    if skip_first and len(scans) > 1:
        scans = scans[1:]
    at, aq, ct, cq = [], [], [], []
    for s in scans:
        sub = work[work["scan"] == s]
        if len(sub) < 2:
            continue
        bl = np.zeros(len(sub))
        pos_area, neg_area = trapz_signed(sub["time"].values,
                                          sub["current"].values, bl)
        at.append(float(sub.loc[sub["potential"].idxmin(), "time"]))
        aq.append(pos_area)
        ct.append(float(sub.loc[sub["potential"].idxmax(), "time"]))
        cq.append(neg_area)
    return (np.array(at), np.array(aq), np.array(ct), np.array(cq))


def charge_fit_plot(t, Q, model_label, fit, title, color, key):
    f = go.Figure()
    if len(t):
        t_rel = t - t.min()
        f.add_trace(go.Scatter(
            x=t_rel, y=Q, mode="markers", name="Charge per cycle",
            marker=dict(size=9, color=color),
            hovertemplate="t=%{x:.1f} s<br>Q=%{y:.4g} C<extra></extra>",
        ))
        if fit is not None:
            tt = np.linspace(t_rel.min(), t_rel.max(), 300)
            if model_label == MODEL_GROWTH_DECAY:
                yy = growth_decay_model(tt, fit["Qbase"], fit["A"], fit["B"],
                                        fit["k1"], fit["k2"])
                fit_name = "Qbase + A(1−e^(−k₁t)) − B(1−e^(−k₂t))"
            else:
                yy = exp_model(tt, fit["C"], fit["A"], fit["k"])
                fit_name = "C + A·e^(−k·t)"
            f.add_trace(go.Scatter(
                x=tt, y=yy, mode="lines", name=fit_name,
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
            ))
    f.update_layout(
        xaxis_title="Time since first used cycle (s)",
        yaxis_title="Charge Q (C)", height=360,
        margin=dict(l=70, r=20, t=30, b=45), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.markdown(f"**{title}**")
    st.plotly_chart(f, use_container_width=True, key=key)


def charge_fit_table(model_label, anod_fit, cath_fit):
    if model_label == MODEL_GROWTH_DECAY:
        def row(region, fit):
            if fit is None:
                return {"Region": region, "Qbase (C)": np.nan, "A (C)": np.nan,
                        "B (C)": np.nan,
                        "k₁ — growth rate constant (1/s)": np.nan,
                        "k₂ — decay rate constant (1/s)": np.nan, "R²": np.nan}
            return {"Region": region, "Qbase (C)": fit["Qbase"], "A (C)": fit["A"],
                    "B (C)": fit["B"],
                    "k₁ — growth rate constant (1/s)": fit["k1"],
                    "k₂ — decay rate constant (1/s)": fit["k2"], "R²": fit["R2"]}
    else:
        def row(region, fit):
            if fit is None:
                return {"Region": region, "C (C)": np.nan, "A (C)": np.nan,
                        "k — rate constant (1/s)": np.nan, "R²": np.nan}
            return {"Region": region, "C (C)": fit["C"], "A (C)": fit["A"],
                    "k — rate constant (1/s)": fit["k"], "R²": fit["R2"]}
    return pd.DataFrame([row("Anodic", anod_fit), row("Cathodic", cath_fit)])


def charge_fit_row(label, model_label, anod_fit, cath_fit):
    if model_label == MODEL_GROWTH_DECAY:
        return {
            "Point": label,
            "Anodic Qbase (C)": anod_fit["Qbase"] if anod_fit else np.nan,
            "Anodic A (C)": anod_fit["A"] if anod_fit else np.nan,
            "Anodic B (C)": anod_fit["B"] if anod_fit else np.nan,
            "Anodic k₁ (1/s)": anod_fit["k1"] if anod_fit else np.nan,
            "Anodic k₂ (1/s)": anod_fit["k2"] if anod_fit else np.nan,
            "Anodic R²": anod_fit["R2"] if anod_fit else np.nan,
            "Cathodic Qbase (C)": cath_fit["Qbase"] if cath_fit else np.nan,
            "Cathodic A (C)": cath_fit["A"] if cath_fit else np.nan,
            "Cathodic B (C)": cath_fit["B"] if cath_fit else np.nan,
            "Cathodic k₁ (1/s)": cath_fit["k1"] if cath_fit else np.nan,
            "Cathodic k₂ (1/s)": cath_fit["k2"] if cath_fit else np.nan,
            "Cathodic R²": cath_fit["R2"] if cath_fit else np.nan,
        }
    return {
        "Point": label,
        "Anodic C (C)": anod_fit["C"] if anod_fit else np.nan,
        "Anodic A (C)": anod_fit["A"] if anod_fit else np.nan,
        "Anodic k (1/s)": anod_fit["k"] if anod_fit else np.nan,
        "Anodic R²": anod_fit["R2"] if anod_fit else np.nan,
        "Cathodic C (C)": cath_fit["C"] if cath_fit else np.nan,
        "Cathodic A (C)": cath_fit["A"] if cath_fit else np.nan,
        "Cathodic k (1/s)": cath_fit["k"] if cath_fit else np.nan,
        "Cathodic R²": cath_fit["R2"] if cath_fit else np.nan,
    }


# ----------------------------------------------------------------------------
# Draw/drag-baseline component (HTML + Plotly.js)
# ----------------------------------------------------------------------------

def draggable_cv_component(potential, current, time, color, x_title, y_title,
                           default_x1, default_y1, default_x2, default_y2,
                           dom_key, key_height=440):
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
# Per-file analysis
# ----------------------------------------------------------------------------

def analyse_file(work, pot_col, cur_col, view_key, skip_first):
    scans = sorted_scans(work)
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scans)}
    unit_label = "C"
    first_scan = scans[0] if scans else None

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
            "v_cathodic": float(sub.loc[i_min, "potential"]),
            "i_cathodic": float(sub.loc[i_min, "current"]),
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

    st.subheader("Charge summary — all scans (default baseline)")
    st.caption("Charge = ∫ I dt over the time column, default Y = 0 baseline (coulombs).")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download charge summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="cv_charge_summary.csv", mime="text/csv",
        key=f"dl_{view_key}",
    )

    if skip_first:
        st.subheader("Trends vs scan number (scan 1 excluded)")
        st.caption("Scan 1 is treated as a transient and omitted; trends start at scan 2.")
        summary_t = summary[summary["Scan"] != first_scan].reset_index(drop=True)
        fallback_start = 2
    else:
        st.subheader("Trends vs scan number (all scans)")
        st.caption("All scans included; trends start at scan 1.")
        summary_t = summary.reset_index(drop=True)
        fallback_start = 1
    if summary_t.empty:
        st.info("No scans available to plot trends with the current scan-1 setting.")
    else:
        try:
            x_scan = [float(s) for s in summary_t["Scan"]]
        except (ValueError, TypeError):
            x_scan = list(range(fallback_start, len(summary_t) + fallback_start))
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
            ("a) Positive (anodic) peak voltage vs scan", summary_t["Anodic peak V (V)"],   "Anodic peak V (V)",   "#d62728"),
            ("b) Negative (cathodic) peak voltage vs scan", summary_t["Cathodic peak V (V)"], "Cathodic peak V (V)", "#1f77b4"),
            ("c) Positive (anodic) peak current vs scan", summary_t["Anodic peak I (A)"],   "Anodic peak I (A)",   "#d62728"),
            ("d) Negative (cathodic) peak current vs scan", summary_t["Cathodic peak I (A)"], "Cathodic peak I (A)", "#1f77b4"),
            ("e) Peak voltage difference ΔEp vs scan", summary_t["ΔEp (V)"],                "ΔEp (V)",             "#2ca02c"),
            ("f) Positive charge vs scan", summary_t[pos_q_col],                            pos_q_col,             "#d62728"),
            ("g) Negative charge vs scan", summary_t[neg_q_col],                            neg_q_col,             "#1f77b4"),
            ("h) |Neg/Pos| charge ratio vs scan", summary_t["Neg/Pos charge ratio"].abs(), "|Neg/Pos| charge ratio","#9467bd"),
        ]
        for i in range(0, len(trends), 2):
            ccols = st.columns(2)
            for j, (col, (title, ydata, ytitle, color)) in enumerate(zip(ccols, trends[i:i + 2])):
                with col:
                    st.markdown(f"**{title}**")
                    st.plotly_chart(trend_chart(ydata, ytitle, color),
                                    use_container_width=True,
                                    key=f"trend_{view_key}_{i}_{j}")

    ref_label = "first used scan (scan 2)" if skip_first else "first scan (scan 1)"

    st.subheader("Peak current vs time — exponential fit  I(t) = C + A·exp(−k·t)")
    st.caption(
        f"{'Scan 1 excluded. ' if skip_first else 'All scans included. '}"
        f"k is the **rate constant for surface change**. Time is measured from the "
        f"{ref_label} (t = 0). Fit by non-linear least squares."
    )
    at, ai, ct, ci = peak_time_series(work, skip_first=skip_first)
    anod_fit = fit_exp_decay(at, ai)
    cath_fit = fit_exp_decay(ct, ci)
    fc1, fc2 = st.columns(2)
    with fc1:
        fit_plot(at, ai, anod_fit, "Anodic peak current vs time", "#d62728",
                 key=f"fit_anod_{view_key}")
    with fc2:
        fit_plot(ct, ci, cath_fit, "Cathodic peak current vs time", "#1f77b4",
                 key=f"fit_cath_{view_key}")
    st.markdown("**Fitted parameters**")
    st.dataframe(fit_params_table(anod_fit, cath_fit),
                 use_container_width=True, hide_index=True)
    if anod_fit is None or cath_fit is None:
        st.caption("Note: a fit is shown only when a region has at least 4 scans "
                   "(with the current scan-1 setting) and the optimisation converges.")

    st.subheader("Charge vs time — empirical rate-constant fit")
    st.caption(
        f"{'Scan 1 excluded. ' if skip_first else 'All scans included. '}"
        "One charge per cycle (default Y = 0 baseline, Q = ∫ I dt). The **anodic** "
        "charge is plotted against the time at the beginning of that cycle's anodic "
        "scan (the lower-potential vertex where oxidation starts); the **cathodic** "
        "charge against the time at the beginning of that cycle's cathodic scan "
        f"(the upper-potential vertex). Time is measured from the {ref_label} (t = 0)."
    )
    qt_model = st.selectbox(
        "Empirical model to fit Q vs t (this file)",
        [MODEL_GROWTH_DECAY, MODEL_SINGLE_EXP],
        index=0, key=f"qtmodel_{view_key}",
    )
    if qt_model == MODEL_GROWTH_DECAY:
        st.markdown(
            "**Legend** &nbsp;·&nbsp; "
            "**k₁** = growth rate constant (1/s) &nbsp;·&nbsp; "
            "**k₂** = decay rate constant (1/s) &nbsp;·&nbsp; "
            "Qbase = baseline charge, A = growth amplitude, B = decay amplitude."
        )
    else:
        st.markdown(
            "**Legend** &nbsp;·&nbsp; "
            "**k** = rate constant (1/s) &nbsp;·&nbsp; "
            "C = offset, A = amplitude."
        )

    aqt, aq, cqt, cq = charge_time_series(work, skip_first=skip_first)
    if qt_model == MODEL_GROWTH_DECAY:
        anod_qfit = fit_growth_decay(aqt, aq)
        cath_qfit = fit_growth_decay(cqt, cq)
    else:
        anod_qfit = fit_exp_decay(aqt, aq)
        cath_qfit = fit_exp_decay(cqt, cq)

    qc1, qc2 = st.columns(2)
    with qc1:
        charge_fit_plot(aqt, aq, qt_model, anod_qfit,
                        "Anodic charge vs time", "#d62728",
                        key=f"qfit_anod_{view_key}")
    with qc2:
        charge_fit_plot(cqt, cq, qt_model, cath_qfit,
                        "Cathodic charge vs time", "#1f77b4",
                        key=f"qfit_cath_{view_key}")
    st.markdown("**Fitted parameters**")
    st.dataframe(charge_fit_table(qt_model, anod_qfit, cath_qfit),
                 use_container_width=True, hide_index=True)
    if anod_qfit is None or cath_qfit is None:
        need = "5" if qt_model == MODEL_GROWTH_DECAY else "4"
        st.caption(
            f"Note: a fit is shown only when a region has at least {need} cycles "
            "(with the current scan-1 setting) and the optimisation converges. The "
            "growth–decay model has 5 free parameters, so more cycles give a more "
            "reliable R² and better-separated k₁ / k₂."
        )

    return summary


def per_scan_summary_silent(work):
    scans = sorted_scans(work)
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
            "Scan": s,
            "Anodic peak I (A)": float(sub.loc[i_max, "current"]),
            "Cathodic peak I (A)": float(sub.loc[i_min, "current"]),
            "Positive charge (C)": pos_area,
            "Negative charge (C)": neg_area,
        })
    return pd.DataFrame(rows)


def representative_metrics(summary, mode, skip_first):
    if summary.empty:
        return None
    pos_col = [c for c in summary.columns if c.startswith("Positive charge")][0]
    neg_col = [c for c in summary.columns if c.startswith("Negative charge")][0]
    start = 1 if (skip_first and len(summary) > 1) else 0
    if mode == "First used scan":
        row = summary.iloc[start]
    elif mode == "Mean across used scans":
        sub = summary.iloc[start:] if len(summary) > start else summary
        row = sub.mean(numeric_only=True)
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
    "file's full analysis from the dropdown; overlay current vs time across "
    "points; see the point-wise comparison below."
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

    st.header("4 · Scan 1 handling")
    scan1_mode = st.radio(
        "Scan 1 in trend / fit plots",
        ["Exclude scan 1 (transient)", "Include scan 1"],
        index=0,
    )
    skip_first = scan1_mode.startswith("Exclude")

    st.header("5 · Point-wise representative scan")
    rep_mode = st.radio(
        "Value used per point in the comparison",
        ["Last scan (steady state)", "First used scan", "Mean across used scans"],
        index=0,
    )

    st.header("6 · Charge-vs-time fit (point-wise table)")
    pw_qt_model = st.selectbox(
        "Model for the per-point Q-vs-t table",
        [MODEL_GROWTH_DECAY, MODEL_SINGLE_EXP],
        index=0,
    )

    st.header("7 · Overlay appearance")
    overlay_width = st.slider("Overlay line thickness", 0.2, 2.0, 0.7, 0.1)


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
    analyse_file(work, pot_col, cur_col, view_key, skip_first)

# ----------------------------------------------------------------------------
# Overlay: current vs time across selected points
# ----------------------------------------------------------------------------

st.markdown("---")
st.header("Overlay — current vs time")
st.caption(
    "Tick the points to include. Each point is drawn as a single thin line in one "
    "colour (all of its scans joined in time order); the legend names the point. "
    "All scans are included here regardless of the scan-1 setting."
)

# Fixed colour per point, so a point keeps its colour whatever the selection is.
overlay_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(file_names)}

sel_all_c1, sel_all_c2, _ = st.columns([1, 1, 4])
with sel_all_c1:
    if st.button("Select all", key="ovl_all"):
        for name in file_names:
            st.session_state[f"ovl_{safe_key(name)}"] = True
with sel_all_c2:
    if st.button("Clear all", key="ovl_none"):
        for name in file_names:
            st.session_state[f"ovl_{safe_key(name)}"] = False

n_cb_cols = 4
cb_cols = st.columns(n_cb_cols)
selected_overlay = []
for i, name in enumerate(file_names):
    with cb_cols[i % n_cb_cols]:
        checked = st.checkbox(point_label(name), value=True,
                              key=f"ovl_{safe_key(name)}")
    if checked:
        selected_overlay.append(name)

if not selected_overlay:
    st.info("Tick at least one point to build the overlay.")
else:
    ofig = go.Figure()
    for name in selected_overlay:
        w = prepare(file_by_name[name])
        if w is None or w.empty:
            continue
        wt = w.sort_values("time")
        label = point_label(name)
        ofig.add_trace(go.Scatter(
            x=wt["time"], y=wt["current"], mode="lines", name=label,
            line=dict(color=overlay_colors[name], width=overlay_width),
            hovertemplate=label + "<br>t=%{x:.1f} s<br>I=%{y:.3e} A<extra></extra>",
        ))
    ofig.update_layout(
        xaxis_title=time_col, yaxis_title=cur_col, height=580,
        margin=dict(l=70, r=20, t=30, b=50), template="plotly_white",
        hovermode="closest",
        legend=dict(title="Point", itemsizing="constant"),
    )
    st.plotly_chart(ofig, use_container_width=True, key="overlay_current_time")

# ----------------------------------------------------------------------------
# Point-wise comparison across all files
# ----------------------------------------------------------------------------

st.markdown("---")
st.header("Point-wise comparison")
st.caption(
    f"One representative value per file using: **{rep_mode}**. "
    f"{'Scan 1 excluded from fits/representative selection. ' if skip_first else 'Scan 1 included. '}"
    "Anodic = positive charge, Cathodic = negative charge. "
    "Charges use the default Y = 0 baseline."
)

rows = []
fit_rows = []
qt_fit_rows = []
for name in file_names:
    w = prepare(file_by_name[name])
    if w is None or w.empty:
        continue
    label = point_label(name)
    s_df = per_scan_summary_silent(w)
    rep = representative_metrics(s_df, rep_mode, skip_first)
    if rep is not None:
        rep_row = {"Point": label}
        rep_row.update(rep)
        rows.append(rep_row)

    at, ai, ct, ci = peak_time_series(w, skip_first=skip_first)
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

    aqt, aq, cqt, cq = charge_time_series(w, skip_first=skip_first)
    if pw_qt_model == MODEL_GROWTH_DECAY:
        aqf = fit_growth_decay(aqt, aq)
        cqf = fit_growth_decay(cqt, cq)
    else:
        aqf = fit_exp_decay(aqt, aq)
        cqf = fit_exp_decay(cqt, cq)
    qt_fit_rows.append(charge_fit_row(label, pw_qt_model, aqf, cqf))

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

if fit_rows:
    st.subheader("Peak-current exponential fit parameters per point  —  I(t) = C + A·exp(−k·t)")
    st.caption(
        f"{'Scan 1 excluded. ' if skip_first else 'All scans included. '}"
        "k = rate constant for surface change (1/s). Time measured from each "
        "file's first used peak."
    )
    fit_df = pd.DataFrame(fit_rows)
    st.dataframe(fit_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download peak-current fit parameters (CSV)",
        fit_df.to_csv(index=False).encode("utf-8"),
        file_name="exp_fit_parameters.csv", mime="text/csv",
        key="dl_fitparams",
    )

    if "Anodic k (1/s)" in fit_df:
        kfig = go.Figure()
        kfig.add_trace(go.Scatter(
            x=fit_df["Point"], y=fit_df["Anodic k (1/s)"], mode="lines+markers",
            name="Anodic k", line=dict(color="#d62728", width=2), marker=dict(size=8)))
        kfig.add_trace(go.Scatter(
            x=fit_df["Point"], y=fit_df["Cathodic k (1/s)"], mode="lines+markers",
            name="Cathodic k", line=dict(color="#1f77b4", width=2), marker=dict(size=8)))
        kfig.update_layout(
            xaxis_title="Point", yaxis_title="k (rate constant, 1/s)",
            height=360, margin=dict(l=70, r=20, t=30, b=45),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.markdown("**Peak-current rate constant k vs point**")
        st.plotly_chart(kfig, use_container_width=True, key="k_vs_point")

if qt_fit_rows:
    if pw_qt_model == MODEL_GROWTH_DECAY:
        st.subheader(
            "Charge-vs-time fit parameters per point  —  "
            "Q(t) = Qbase + A(1−exp(−k₁·t)) − B(1−exp(−k₂·t))"
        )
        st.caption(
            f"{'Scan 1 excluded. ' if skip_first else 'All scans included. '}"
            "**k₁** = growth rate constant (1/s), **k₂** = decay rate constant "
            "(1/s). Anodic = positive charge, Cathodic = negative charge. Time "
            "measured from each file's first used cycle."
        )
    else:
        st.subheader(
            "Charge-vs-time fit parameters per point  —  Q(t) = C + A·exp(−k·t)"
        )
        st.caption(
            f"{'Scan 1 excluded. ' if skip_first else 'All scans included. '}"
            "**k** = rate constant (1/s). Anodic = positive charge, Cathodic = "
            "negative charge. Time measured from each file's first used cycle."
        )
    qt_df = pd.DataFrame(qt_fit_rows)
    st.dataframe(qt_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download charge-vs-time fit parameters (CSV)",
        qt_df.to_csv(index=False).encode("utf-8"),
        file_name="charge_time_fit_parameters.csv", mime="text/csv",
        key="dl_qtfitparams",
    )

    def k_vs_point_chart(anod_col, cath_col, y_title, title, key):
        f = go.Figure()
        if anod_col in qt_df:
            f.add_trace(go.Scatter(
                x=qt_df["Point"], y=qt_df[anod_col], mode="lines+markers",
                name="Anodic", line=dict(color="#d62728", width=2),
                marker=dict(size=8)))
        if cath_col in qt_df:
            f.add_trace(go.Scatter(
                x=qt_df["Point"], y=qt_df[cath_col], mode="lines+markers",
                name="Cathodic", line=dict(color="#1f77b4", width=2),
                marker=dict(size=8)))
        f.update_layout(
            xaxis_title="Point", yaxis_title=y_title, height=340,
            margin=dict(l=70, r=20, t=30, b=45), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1))
        st.markdown(f"**{title}**")
        st.plotly_chart(f, use_container_width=True, key=key)

    if pw_qt_model == MODEL_GROWTH_DECAY:
        kc1, kc2 = st.columns(2)
        with kc1:
            k_vs_point_chart("Anodic k₁ (1/s)", "Cathodic k₁ (1/s)",
                             "k₁ — growth rate constant (1/s)",
                             "Growth rate constant k₁ vs point", "qt_k1_vs_point")
        with kc2:
            k_vs_point_chart("Anodic k₂ (1/s)", "Cathodic k₂ (1/s)",
                             "k₂ — decay rate constant (1/s)",
                             "Decay rate constant k₂ vs point", "qt_k2_vs_point")
    else:
        k_vs_point_chart("Anodic k (1/s)", "Cathodic k (1/s)",
                         "k — rate constant (1/s)",
                         "Charge-vs-time rate constant k vs point", "qt_k_vs_point")
