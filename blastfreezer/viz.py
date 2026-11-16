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
                flow_text: str = "", height: int = 560, max_frames: int = MAX_FRAMES,
                SST_frames: Sequence | None = None, boost_on: bool = False) -> go.Figure:
    """Front view of the cell (width across, height up, depth stacked behind) with its air loop.

    T_core_frames: array (n_times, n_series) of core temperatures by series position.
    T_supply_frames / T_return_frames: cell supply / return air per time (None = unknown).
    SST_frames: main coil SST per time (None / NaN = unknown), written inside the coil box.
    The main coil sits above and to the left of the pallets; the boost coil is drawn
    above it as a parallel branch, greyed out unless boost_on.
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
    SST_frames = [None] * len(times) if SST_frames is None else list(SST_frames)
    fT = lambda T: "—" if T is None or not np.isfinite(T) else f"{_fmt_T(cv, T)} {ul('temp')}"

    def air_labels(T_supply, T_return):
        return ["Supply air " + fT(T_supply) + (f"<br>{flow_text}" if flow_text else ""),
                "Return air " + fT(T_return)]

    def sst_label(SST):
        return [f"SST:<br>{fT(SST)}"]

    fig = go.Figure()
    # air loop: main (cell <-> main coil) and the boost branch, lines under everything else
    boost_color = ARROW_COLOR if boost_on else LOOP_OFF
    for path, color in [(L["boost_in"], boost_color), (L["boost_out"], boost_color),
                        (L["supply"], ARROW_COLOR), (L["return"], ARROW_COLOR)]:
        xs, ys = zip(*path)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=4),
                                 hoverinfo="skip", showlegend=False))
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
                             text=air_labels(T_supply_frames[keep[0]], T_return_frames[keep[0]]),
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

    # coil boxes and their names
    for (x, y, w, h), on, name, above in [(L["main_box"], True, "Main Coil", False),
                                          (L["boost_box"], boost_on, "Boost Coil", True)]:
        edge, ink = (ARROW_COLOR, ARROW_COLOR) if on else (LOOP_OFF, TEXT_OFF)
        fig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - h / 2, y1=y + h / 2,
                      line=dict(color=edge, width=2), fillcolor=BOX_FILL if on else BOX_FILL_OFF, layer="below")
        fig.add_annotation(x=x, y=y + h / 2 + 0.1 * u if above else y - h / 2 - 0.1 * u,
                           yanchor="bottom" if above else "top", text=name, showarrow=False,
                           font=dict(size=14, color=ink))
        if name == "Boost Coil" and not boost_on:
            fig.add_annotation(x=x - w / 2 + 0.08 * u, y=y + h / 2 - 0.08 * u, xanchor="left", yanchor="top",
                               align="left", text="SST:<br>—", showarrow=False, font=dict(size=11, color=ink))

    # arrowheads (tip, unit direction)
    heads = [(t, d, ARROW_COLOR) for t, d in L["heads"]] + \
            [(t, d, ARROW_COLOR if boost_on else LOOP_OFF) for t, d in L["boost_heads"]]
    for (x1, y1), (dx, dy), color in heads:
        fig.add_annotation(x=x1, y=y1, ax=x1 - 0.3 * u * dx, ay=y1 - 0.3 * u * dy, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=4,
                           arrowcolor=color, text="")

    (x_lo, x_hi), (y_lo, y_hi) = L["range"]
    fig.update_xaxes(visible=False, range=[x_lo, x_hi], fixedrange=True)
    fig.update_yaxes(visible=False, range=[y_lo, y_hi], scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

    if len(keep) > 1:
        idx_traces = list(range(i_pal0, i_pal0 + len(pallets))) + [i_text, i_air, i_sst]
        frames = []
        for k in keep:
            T = T_core_frames[k]
            data = [dict(type="scatter", fillcolor=c) for c in colors(T)]
            data.append(dict(type="scatter", text=labels(T), hovertext=hovers(T, times[k])))
            data.append(dict(type="scatter", text=air_labels(T_supply_frames[k], T_return_frames[k])))
            data.append(dict(type="scatter", text=sst_label(SST_frames[k])))
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


def temperature_figure(res, cv: Callable, ul: Callable, height: int = 420) -> go.Figure:
    P = res.pallets
    final = P[P.time_h == P.time_h.max()].sort_values("position").T_core.tolist()
    cold, warm = extreme_positions(res.freeze_times_h, final)
    fig = go.Figure()
    styles = {warm: ("#d7263d", "Warmest"), cold: ("#1f5fbf", "Coldest")}
    for pos in dict.fromkeys([warm, cold]):
        color, label = styles[pos]
        g = P[P.position == pos]
        fig.add_trace(go.Scatter(x=g.time_h, y=cv("temp", g.T_core), name=f"{label} (P{pos}) core",
                                 line=dict(color=color, width=2.5)))
        prof = res.profiles[pos]
        fig.add_trace(go.Scatter(x=prof.index, y=cv("temp", prof.iloc[:, -1].values),
                                 name=f"{label} (P{pos}) surface node", line=dict(color=color, dash="dash")))
    C = res.cell
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_supply), name="Supply air (after coil + fans)",
                             line=dict(color="#5a6b7b", dash="dot")))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_return), name="Return air (to coil)",
                             line=dict(color="#2b3a4a", dash="dashdot")))
    if "SST" in C and C.SST.notna().any():
        fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.SST), name="SST",
                                 line=dict(color="#2a9d8f", width=1.5)))
    _finish(fig, ul("temp"), height)
    return fig


def power_figure(res, cv: Callable, ul: Callable, height: int = 420,
                 design_kW: float | None = None) -> go.Figure:
    C = res.cell
    fig = go.Figure()
    coil = C["Q_coil_kW"] if "Q_coil_kW" in C else C.Q_cell_kW
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", coil), name="Cooling power (coil)",
                             line=dict(color="#1f5fbf", width=2.5)))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", C.Q_cell_kW), name="Pallet load",
                             line=dict(color="#8b4fd6", width=1.5, dash="dash")))
    if design_kW is not None:
        fig.add_hline(y=cv("power", design_kW), line_dash="dot", line_color="#888",
                      annotation_text="Refrigeration design")
    fan = C["fan_kW"] if "fan_kW" in C else np.zeros(len(C))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", fan), name="Fan power",
                             line=dict(color="#e08a1e", width=2)))
    _finish(fig, ul("power"), height)
    return fig


def _finish(fig, y_unit, height):
    fig.update_layout(height=height, xaxis_title="Blasting time [h]", yaxis_title=y_unit,
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
