"""Plotly figures for the app: the cell layout view and the run time series.

No Streamlit here. Unit conversion is passed in as `cv(quantity, si_value)`
and labels as `ul(quantity)`, so the figures work in SI or Imperial.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# red (load-in temperature) -> purple -> blue (target core temperature)
COLORSCALE = [[0.0, "#1f5fbf"], [0.5, "#8b4fd6"], [1.0, "#d7263d"]]
ARROW_COLOR = "#2b3a4a"
ACCENT = "#6633ff"                    # Rebound logo purple (matches the Streamlit theme)
ACCENT_LIGHT = "rgba(102, 51, 255, 0.15)"
ACCENT_EDGE = "rgba(102, 51, 255, 0.45)"
LOOP_OFF = "#d0d4d8"                  # greyed-out (inactive) boost coil lines and outline
TEXT_OFF = "#c3c7cc"
BOX_FILL, BOX_FILL_OFF = "#eceef0", "#f6f7f8"

FACE = {  # face -> (axis, index of the face's end: 0 = low side, 1 = high side)
    "front": ("depth", 0), "back": ("depth", 1), "left": ("width", 0),
    "right": ("width", 1), "bottom": ("height", 0), "top": ("height", 1),
}
GUI_FACES = ("left", "right", "bottom", "top")      # inlet faces offered in the GUI
MAX_FRAMES = 240                                     # time steps sent to the browser


def series_position(face: str, idx: dict, n: dict) -> int:
    """0-based position along the flow for a pallet at cell indices idx."""
    axis, end = FACE[face]
    return idx[axis] if end == 0 else n[axis] - 1 - idx[axis]


def _fill(T: float, T_cold: float, T_hot: float) -> str:
    x = 0.5 if T_hot == T_cold else (T - T_cold) / (T_hot - T_cold)
    return sample_colorscale(COLORSCALE, [float(np.clip(x, 0.0, 1.0))])[0]


def _fmt_T(cv, T) -> str:
    return f"{cv('temp', T) + 0.0:.1f}".replace("-0.0", "0.0")


def cell_figure(n_deep: int, n_wide: int, n_high: int, inlet_face: str,
                T_core_frames, times: Sequence[float], T_return_frames: Sequence,
                T_hot: float, T_cold: float, cv: Callable, ul: Callable, T_supply_frames: Sequence,
                flow_text: str | Sequence[str] = "", height: int = 560, max_frames: int = MAX_FRAMES,
                SST_frames: Sequence | None = None, boost_frames: Sequence[bool] | None = None,
                SST_boost_frames: Sequence | None = None) -> go.Figure:
    """Front view of the cell (width across, height up, depth stacked behind) with its air loop.

    T_core_frames: array (n_times, n_series) of core temperatures by series position.
    T_supply_frames / T_return_frames: cell supply / return air per time (None = unknown).
    SST_frames: main coil SST per time (None / NaN = unknown), written inside the coil box.
    flow_text: supply airflow label, one string or one per time (it changes when the boost stops).
    The main coil sits above and to the left of the pallets; the boost coil is drawn
    above it as a parallel branch: black while boost_frames is True, greyed out otherwise,
    with SST_boost_frames written inside.
    With more than one time, the figure carries animation frames and its own time
    slider (plus play/pause), so stepping through the solution runs in the browser.
    """
    if inlet_face not in GUI_FACES:
        raise ValueError(f"Cell view supports inlet faces {GUI_FACES}, not '{inlet_face}'.")
    T_core_frames = np.atleast_2d(np.asarray(T_core_frames, dtype=float))
    times = list(times)
    keep = np.unique(np.linspace(0, len(times) - 1, min(len(times), max_frames)).round().astype(int))
    n = {"depth": n_deep, "width": n_wide, "height": n_high}
    nh, nv, nd = n_wide, n_high, n_deep

    size = 0.78
    off = min(0.14, 0.42 / max(1, nd - 1))                 # offset per depth step (up-right)
    pitch = size + (nd - 1) * off + 0.28
    font = int(np.clip(150 / max(nh, nv) / max(1.0, pitch), 8, 15))

    # pallets, far layer first so near layers draw on top
    pallets = []
    for d in reversed(range(nd)):
        for i in range(nh):
            for j in range(nv):
                idx = {"width": i, "height": j, "depth": d}
                pallets.append((i * pitch + d * off, j * pitch + d * off, d,
                                series_position(inlet_face, idx, n), idx))
    front = [p for p in pallets if p[2] == 0]

    def colors(T):
        return [_fill(T[s], T_cold, T_hot) for *_, s, _ in pallets]

    def labels(T):
        return [_fmt_T(cv, T[s]) for *_, s, _ in front]

    def hovers(T, t):
        out = []
        for x0, y0, _, _, idx in front:
            lines = [f"t = {t:.1f} h"]
            for dd in range(nd):
                k = dict(idx, depth=dd)
                sk = series_position(inlet_face, k, n)
                lines.append(f"depth {dd + 1}: P{sk + 1} core {_fmt_T(cv, T[sk])} {ul('temp')}")
            out.append("<br>".join(lines))
        return out

    W = (nh - 1) * pitch + size + (nd - 1) * off
    H = (nv - 1) * pitch + size + (nd - 1) * off
    u = max(1.0, max(W, H) / 5.0)                   # scale for the loop, boxes and gaps
    L = _loop(inlet_face, W, H, u)
    nt = len(times)
    SST_frames = [None] * nt if SST_frames is None else list(SST_frames)
    SST_boost_frames = [None] * nt if SST_boost_frames is None else list(SST_boost_frames)
    boost_frames = [False] * nt if boost_frames is None else [bool(b) for b in boost_frames]
    flow_frames = [flow_text] * nt if isinstance(flow_text, str) else list(flow_text)
    fT = lambda T: "—" if T is None or not np.isfinite(T) else f"{_fmt_T(cv, T)} {ul('temp')}"

    def air_labels(T_supply, T_return, flow):
        return ["Supply air " + fT(T_supply) + (f"<br>{flow}" if flow else ""),
                "Return air " + fT(T_return)]

    def boost_style(on, SST):
        """Frame data for the boost traces: 2 lines, 2 heads, box, text (name + SST)."""
        c = ARROW_COLOR if on else LOOP_OFF
        ink = ARROW_COLOR if on else TEXT_OFF
        line = dict(type="scatter", line=dict(color=c, width=4))
        head = dict(type="scatter", fillcolor=c, line=dict(color=c, width=1))
        box = dict(type="scatter", fillcolor=BOX_FILL if on else BOX_FILL_OFF, line=dict(color=c, width=2))
        text = dict(type="scatter", text=["Boost Coil", f"SST:<br>{fT(SST) if on else '—'}"],
                    textfont=dict(size=[14, 11], color=ink))
        return [line, line, head, head, box, text]

    def sst_label(SST):
        return [f"SST:<br>{fT(SST)}"]

    fig = go.Figure()
    # air loop: main (cell <-> main coil) and the boost branch, lines under everything else
    # Lines stop inside the base of their arrowhead; heads are triangles in drawing units, so
    # they scale with the drawing and the line can never poke out past the tip.
    head_len, head_half = 0.4 * u, 0.2 * u

    def add_line(path, color):
        xs, ys = zip(*_trim(path, 0.75 * head_len))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=4),
                                 hoverinfo="skip", showlegend=False))

    def add_head(tip, d, color):
        (tx, ty), (dx, dy) = tip, d
        bx_, by_ = tx - head_len * dx, ty - head_len * dy
        fig.add_trace(go.Scatter(x=[tx, bx_ - head_half * dy, bx_ + head_half * dy, tx],
                                 y=[ty, by_ + head_half * dx, by_ - head_half * dx, ty],
                                 mode="lines", fill="toself", fillcolor=color, line=dict(color=color, width=1),
                                 hoverinfo="skip", showlegend=False, name="arrowhead"))

    # boost branch first (animated: black while running, grey when off)
    b0 = keep[0]
    st0 = boost_style(boost_frames[b0], SST_boost_frames[b0])
    i_boost0 = len(fig.data)
    add_line(L["boost_in"], st0[0]["line"]["color"])
    add_line(L["boost_out"], st0[1]["line"]["color"])
    for tip, d in L["boost_heads"]:
        add_head(tip, d, st0[2]["fillcolor"])
    x, y, w, h = L["boost_box"]
    fig.add_trace(go.Scatter(x=[x - w / 2, x + w / 2, x + w / 2, x - w / 2, x - w / 2],
                             y=[y - h / 2, y - h / 2, y + h / 2, y + h / 2, y - h / 2], mode="lines",
                             fill="toself", fillcolor=st0[4]["fillcolor"], line=st0[4]["line"],
                             hoverinfo="skip", showlegend=False, name="coil box"))
    fig.add_trace(go.Scatter(x=[x, x - w / 2 + 0.08 * u], y=[y + h / 2 + 0.1 * u, y + h / 2 - 0.08 * u],
                             mode="text", text=st0[5]["text"], textposition=["top center", "bottom right"],
                             textfont=st0[5]["textfont"], hoverinfo="skip", showlegend=False))
    i_boost = list(range(i_boost0, len(fig.data)))
    for path in (L["supply"], L["return"]):
        add_line(path, ARROW_COLOR)
    for tip, d in L["heads"]:
        add_head(tip, d, ARROW_COLOR)
    T0 = T_core_frames[keep[0]]
    for (x0, y0, d, s, idx), c in zip(pallets, colors(T0)):
        fig.add_trace(go.Scatter(
            x=[x0, x0 + size, x0 + size, x0, x0], y=[y0, y0, y0 + size, y0 + size, y0],
            mode="lines", fill="toself", fillcolor=c, line=dict(color="#1b2633", width=1.2),
            hoveron="fills", hoverinfo="text", showlegend=False,
            text=f"P{s + 1} · width {idx['width'] + 1}, height {idx['height'] + 1}, depth {d + 1}"))
    i_pal0 = len(fig.data) - len(pallets)
    i_text = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0 + size / 2 for x0, *_ in front], y=[y0 + size / 2 for _, y0, *_ in front],
                             mode="text", text=labels(T0), textfont=dict(color="white", size=font),
                             hovertext=hovers(T0, times[keep[0]]), hoverinfo="text", showlegend=False))
    i_air = len(fig.data)
    (sx, sy, sp), (rx, ry, rp) = L["labels"]
    fig.add_trace(go.Scatter(x=[sx, rx], y=[sy, ry], mode="text",
                             text=air_labels(T_supply_frames[keep[0]], T_return_frames[keep[0]],
                                             flow_frames[keep[0]]),
                             textposition=[sp, rp], textfont=dict(size=11, color=ARROW_COLOR),
                             hoverinfo="skip", showlegend=False))
    i_sst = len(fig.data)
    bx, by, bw, bh = L["main_box"]
    fig.add_trace(go.Scatter(x=[bx - bw / 2 + 0.08 * u], y=[by + bh / 2 - 0.08 * u], mode="text",
                             text=sst_label(SST_frames[keep[0]]), textposition="bottom right",
                             textfont=dict(size=11, color=ARROW_COLOR), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=COLORSCALE, cmin=cv("temp", T_cold), cmax=cv("temp", T_hot), color=[0],
                    showscale=True, colorbar=dict(title=dict(text=f"Core<br>{ul('temp')}"),
                                                  thickness=12, len=0.7))))

    # main coil box and name (always running)
    x, y, w, h = L["main_box"]
    fig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - h / 2, y1=y + h / 2,
                  line=dict(color=ARROW_COLOR, width=2), fillcolor=BOX_FILL, layer="below")
    fig.add_annotation(x=x, y=y - h / 2 - 0.1 * u, yanchor="top", text="Main Coil", showarrow=False,
                       font=dict(size=14, color=ARROW_COLOR))

    (x_lo, x_hi), (y_lo, y_hi) = L["range"]
    # constrain="domain": keep 1:1 by shrinking the plot area, never by widening the ranges.
    # (The default widens the ranges and Plotly writes them back into the figure, so each
    # browser resize compounds and the drawing ratchets smaller.)
    fig.update_xaxes(visible=False, range=[x_lo, x_hi], fixedrange=True, autorange=False, constrain="domain")
    fig.update_yaxes(visible=False, range=[y_lo, y_hi], scaleanchor="x", scaleratio=1, fixedrange=True,
                     autorange=False, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

    if len(keep) > 1:
        idx_traces = list(range(i_pal0, i_pal0 + len(pallets))) + [i_text, i_air, i_sst] + i_boost
        frames = []
        for k in keep:
            T = T_core_frames[k]
            data = [dict(type="scatter", fillcolor=c) for c in colors(T)]
            data.append(dict(type="scatter", text=labels(T), hovertext=hovers(T, times[k])))
            data.append(dict(type="scatter", text=air_labels(T_supply_frames[k], T_return_frames[k],
                                                              flow_frames[k])))
            data.append(dict(type="scatter", text=sst_label(SST_frames[k])))
            data += boost_style(boost_frames[k], SST_boost_frames[k])
            frames.append(go.Frame(name=str(k), data=data, traces=idx_traces))
        fig.frames = frames
        step_args = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
        fig.update_layout(
            sliders=[dict(active=0, x=0.12, len=0.86, y=0, yanchor="top", pad=dict(t=10, b=0),
                          bgcolor=ACCENT_LIGHT, activebgcolor=ACCENT, bordercolor=ACCENT_EDGE,
                          currentvalue=dict(prefix="Model time: ", suffix=" h", font=dict(size=13)),
                          steps=[dict(method="animate", label=f"{times[k]:.1f}", args=[[str(k)], step_args])
                                 for k in keep])],
            updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0, xanchor="left", yanchor="top",
                              pad=dict(t=22), showactive=False, bgcolor=ACCENT_LIGHT,
                              bordercolor=ACCENT_EDGE, font=dict(color=ACCENT), buttons=[
                                  dict(label="▶", method="animate",
                                       args=[None, dict(frame=dict(duration=80, redraw=True),
                                                        fromcurrent=True, transition=dict(duration=0))]),
                                  dict(label="❚❚", method="animate",
                                       args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                          transition=dict(duration=0))])])],
            margin=dict(l=0, r=0, t=10, b=90), height=height + 80)
    return fig


def _trim(path, t):
    """Pull the last point of a polyline back toward the previous one by t."""
    (x0, y0), (x1, y1) = path[-2], path[-1]
    d = max(np.hypot(x1 - x0, y1 - y0), 1e-9)
    return path[:-1] + [(x1 - t * (x1 - x0) / d, y1 - t * (y1 - y0) / d)]


def _direction(a, b):
    d = max(np.hypot(b[0] - a[0], b[1] - a[1]), 1e-9)
    return (b[0] - a[0]) / d, (b[1] - a[1]) / d


def _loop(inlet: str, W: float, H: float, u: float) -> dict:
    """Air loop geometry for the front view, in the pallet-drawing coordinates.

    The cell occupies [0, W] x [0, H]. The main coil sits above and left of it; air
    goes coil -> supply leg -> cell -> return leg -> coil. "A" is the coil's left
    side, "B" its right side. Flow into the low face (left / bottom) leaves the coil
    from A, so air crosses the coil right -> left; the high faces reverse every path.
    The boost coil is a branch in parallel with the main coil, directly above it.
    """
    gap, leg, rise, coil_gap = 0.3 * u, 2.8 * u, 1.5 * u, 0.15 * u
    bw, bh = 1.4 * u, 1.15 * u
    xL = -gap - leg
    xc, yT = 0.5 * xL, H + rise
    yT2 = yT + 1.9 * u
    A, B = xc - bw / 2, xc + bw / 2
    if inlet in ("left", "right"):
        yM, xR = H / 2, W + gap + leg
        main_A = [(A, yT), (xL, yT), (xL, yM), (-gap, yM)]           # coil A -> left face
        main_B = [(W + gap, yM), (xR, yM), (xR, yT), (B, yT)]        # right face -> coil B
        low = inlet == "left"
        lab_low, lab_high = (-gap - 0.1 * u, yM + 0.2 * u, "top left"), (W + gap + 0.1 * u, yM + 0.2 * u, "top right")
        sup_lab, ret_lab = (lab_low, lab_high) if low else (lab_high, lab_low)
        x_range = (xL - 0.5 * u, xR + 0.5 * u)
        y_lo = -0.6 * u
        riser = xR
    else:
        xm, yB = W / 2, -gap - 2.2 * u
        main_A = [(A, yT), (xL, yT), (xL, yB), (xm, yB), (xm, -gap)]  # coil A -> bottom face
        main_B = [(xm, H + gap), (xm, yT), (B, yT)]                  # top face -> coil B
        low = inlet == "bottom"
        lab_low = (xm + 0.2 * u, (-gap + yB) / 2, "middle right")
        lab_high = (xm + 0.2 * u, (H + gap + yT) / 2 + 0.2 * u, "middle right")
        sup_lab, ret_lab = (lab_low, lab_high) if low else (lab_high, lab_low)
        x_range = (xL - 0.5 * u, max(W + 0.5 * u, xm + 3.2 * u))
        y_lo = yB - 0.6 * u
        riser = xm
    xB = (B + riser) / 2
    boost_A = [(xL, yT), (xL, yT2), (A, yT2)]                         # main loop corner -> boost A
    boost_B = [(B, yT2), (xB, yT2), (xB, yT)]                         # boost B -> main loop
    if low:
        supply, ret = main_A, main_B
        b_in, b_out = boost_B[::-1], boost_A[::-1]
    else:
        supply, ret = main_B[::-1], main_A[::-1]
        b_in, b_out = boost_A, boost_B
    ret = _trim(ret, coil_gap)
    b_in, b_out = _trim(b_in, coil_gap), _trim(b_out, 0.2 * u)
    d_leave = _direction(ret[0], ret[1])
    leave_tip = (ret[0][0] + 0.6 * u * d_leave[0], ret[0][1] + 0.6 * u * d_leave[1])
    heads = [(supply[-1], _direction(supply[-2], supply[-1])),       # into the cell
             (leave_tip, d_leave),                                    # leaving the cell
             (ret[-1], _direction(ret[-2], ret[-1]))]                 # into the coil
    boost_heads = [(b_in[-1], _direction(b_in[-2], b_in[-1])), (b_out[-1], _direction(b_out[-2], b_out[-1]))]
    return dict(supply=supply, **{"return": ret}, boost_in=b_in, boost_out=b_out,
                heads=heads, boost_heads=boost_heads, labels=(sup_lab, ret_lab),
                main_box=(xc, yT, bw, bh), boost_box=(xc, yT2, bw, bh),
                range=(x_range, (y_lo, yT2 + bh / 2 + 0.8 * u)))


# ---- time series --------------------------------------------------------------
def extreme_positions(freeze_times_h: Sequence, final_T_core: Sequence[float]) -> tuple[int, int]:
    """(coldest, warmest) 1-based series positions: first / last to freeze."""
    key = [(t if t is not None else np.inf, T) for t, T in zip(freeze_times_h, final_T_core)]
    order = sorted(range(len(key)), key=lambda i: key[i])
    return order[0] + 1, order[-1] + 1


# one colour per quantity; dashed = without boost, solid = with boost
SERIES_COLORS = {"core": "#d7263d", "surface": "#f28e2b", "supply": "#1f77b4", "return": "#17becf",
                 "SST": "#2ca02c", "SST_boost": ACCENT, "coil": "#1f77b4", "boost_coil": ACCENT,
                 "pallet": "#d7263d", "fan": "#f28e2b", "total": "#2b3a4a"}
RUN_DASH = {False: "dash", True: "solid"}
RUN_NAME = {False: "no boost", True: "boost"}


def warmest_position(res) -> int:
    """1-based series position of the warmest pallet (last to reach the target)."""
    P = res.pallets
    final = P[P.time_h == P.time_h.max()].sort_values("position").T_core.tolist()
    return extreme_positions(res.freeze_times_h, final)[1]


def _runs(off, on):
    return [(r, b) for r, b in ((off, False), (on, True)) if r is not None]


def _add(fig, x, y, key, label, boost, width=2.0):
    fig.add_trace(go.Scatter(x=x, y=y, name=f"{label} · {RUN_NAME[boost]}", legendgroup=key,
                             line=dict(color=SERIES_COLORS[key], dash=RUN_DASH[boost], width=width)))


def product_temperature_figure(off, on, cv: Callable, ul: Callable, height: int = 400) -> go.Figure:
    """Warmest pallet core and surface node. off / on: results without and with boost (either may be None)."""
    fig = go.Figure()
    for res, boost in _runs(off, on):
        pos = warmest_position(res)
        g = res.pallets[res.pallets.position == pos]
        prof = res.profiles[pos]
        _add(fig, g.time_h, cv("temp", g.T_core), "core", f"Core, warmest pallet (P{pos})", boost, 2.5)
        _add(fig, prof.index, cv("temp", prof.iloc[:, -1].values), "surface", "Surface node, warmest pallet", boost)
    _finish(fig, ul("temp"), height)
    return fig


def air_temperature_figure(off, on, cv: Callable, ul: Callable, height: int = 400) -> go.Figure:
    """Supply and return air, main SST, and boost SST (boost run only)."""
    fig = go.Figure()
    for res, boost in _runs(off, on):
        C = res.cell
        _add(fig, C.time_h, cv("temp", C.T_supply), "supply", "Supply air", boost)
        _add(fig, C.time_h, cv("temp", C.T_return), "return", "Return air", boost)
        if C.SST.notna().any():
            _add(fig, C.time_h, cv("temp", C.SST), "SST", "SST main coil", boost, 1.5)
        if boost and "SST_boost" in C and C.SST_boost.notna().any():
            _add(fig, C.time_h, cv("temp", C.SST_boost), "SST_boost", "SST boost coil", boost, 1.5)
    _finish(fig, ul("temp"), height)
    return fig


def cooling_bands(C) -> dict:
    """Stacked cooling bands [kW] that add up to the total coil load: fan heat, then the rest (= pallet
    load) split between the main and boost coils in proportion to their coil loads. Each coil's raw load
    includes its share of the fan heat, so stacking raw loads on top of fan heat would count it twice."""
    Q = C.Q_coil_kW.to_numpy(float)
    fan = C.fan_kW.to_numpy(float)
    net = np.clip(Q - fan, 0.0, None)
    share = np.divide(C.Q_boost_kW.to_numpy(float), Q, out=np.zeros_like(Q), where=Q > 0)
    return {"fan": fan, "main": net * (1.0 - share), "boost": net * share}


def cooling_figure(off, on, cv: Callable, ul: Callable, height: int = 400,
                   design_kW: float | None = None) -> go.Figure:
    """Stacked areas for the boost run (fan heat, main coil, boost coil, bottom to top); the run
    without boost as a dashed total-coil-load line."""
    fig = go.Figure()
    if on is not None:
        C = on.cell
        bands = cooling_bands(C)
        for key, label, color in (("fan", "Fan heat", SERIES_COLORS["fan"]),
                                  ("main", "Main coil", SERIES_COLORS["coil"]),
                                  ("boost", "Boost coil", SERIES_COLORS["boost_coil"])):
            fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", bands[key]), name=f"{label} · boost",
                                     legendgroup=key, stackgroup="boost", mode="lines",
                                     line=dict(color=color, width=0.5), fillcolor=_alpha(color, 0.55)))
    if off is not None:
        _add(fig, off.cell.time_h, cv("power", off.cell.Q_coil_kW), "total", "Total coil load", False, 2.0)
    if design_kW is not None:
        fig.add_hline(y=cv("power", design_kW), line_dash="dot", line_color="#888",
                      annotation_text="Main refrigeration design")
    _finish(fig, ul("power"), height)
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def _finish(fig, y_unit, height):
    fig.update_layout(height=height, xaxis_title="Blasting time [h]", yaxis_title=y_unit,
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.22))
