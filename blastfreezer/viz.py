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

# view -> (horizontal axis, vertical axis, depth axis), each (cell axis, reversed?)
# depth index 0 is nearest the viewer.
VIEWS = {
    "front": (("width", False), ("height", False), ("depth", False)),
    "side": (("depth", False), ("height", False), ("width", False)),
    "top": (("width", False), ("depth", True), ("height", True)),
}
FACE = {  # face -> (axis, index of the face's end: 0 = low side, 1 = high side)
    "front": ("depth", 0), "back": ("depth", 1), "left": ("width", 0),
    "right": ("width", 1), "bottom": ("height", 0), "top": ("height", 1),
}


def series_position(face: str, idx: dict, n: dict) -> int:
    """0-based position along the flow for a pallet at cell indices idx."""
    axis, end = FACE[face]
    return idx[axis] if end == 0 else n[axis] - 1 - idx[axis]


def _fill(T: float, T_cold: float, T_hot: float) -> str:
    x = 0.5 if T_hot == T_cold else (T - T_cold) / (T_hot - T_cold)
    return sample_colorscale(COLORSCALE, [float(np.clip(x, 0.0, 1.0))])[0]


def cell_figure(n_deep: int, n_wide: int, n_high: int, inlet_face: str,
                T_core_by_series: Sequence[float], T_hot: float, T_cold: float,
                cv: Callable, ul: Callable, T_air_in: float, T_air_out: float | None,
                flow_text: str = "", view: str = "front", height: int = 560) -> go.Figure:
    """Pseudo-3D view of the cell: pallets coloured by core temperature, airflow arrows."""
    n = {"depth": n_deep, "width": n_wide, "height": n_high}
    (ha, hr), (va, vr), (da, dr) = VIEWS[view]
    nh, nv, nd = n[ha], n[va], n[da]

    size = 0.78
    off = min(0.14, 0.42 / max(1, nd - 1))                 # offset per depth step (up-right)
    pitch = size + (nd - 1) * off + 0.28
    font = int(np.clip(150 / max(nh, nv) / max(1.0, pitch), 8, 15))

    fig = go.Figure()
    hover_x, hover_y, hover_t = [], [], []
    # draw far layers first so near ones cover them
    for d in reversed(range(nd)):
        for i in range(nh):
            for j in range(nv):
                idx = {ha: (nh - 1 - i) if hr else i, va: (nv - 1 - j) if vr else j,
                       da: (nd - 1 - d) if dr else d}
                s = series_position(inlet_face, idx, n)
                T = T_core_by_series[s]
                x0, y0 = i * pitch + d * off, j * pitch + d * off
                fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0 + size, y1=y0 + size,
                              fillcolor=_fill(T, T_cold, T_hot),
                              line=dict(color="#1b2633", width=1.2), layer="above")
                if d == 0:
                    fig.add_annotation(x=x0 + size / 2, y=y0 + size / 2, showarrow=False,
                                       text=f"{cv('temp', T) + 0.0:.1f}".replace("-0.0", "0.0"), font=dict(color="white", size=font))
                    stack = []
                    for dd in range(nd):
                        k = dict(idx)
                        k[da] = (nd - 1 - dd) if dr else dd
                        sk = series_position(inlet_face, k, n)
                        stack.append(f"{da} {dd + 1}: P{sk + 1} core {cv('temp', T_core_by_series[sk]):.1f} {ul('temp')}")
                    hover_x.append(x0 + size / 2)
                    hover_y.append(y0 + size / 2)
                    hover_t.append("<br>".join(stack))

    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=1, opacity=0),
                             hovertext=hover_t, hoverinfo="text", showlegend=False))
    # colour bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=COLORSCALE, cmin=cv("temp", T_cold), cmax=cv("temp", T_hot), color=[0],
                    showscale=True, colorbar=dict(title=dict(text=f"Core<br>{ul('temp')}"),
                                                  thickness=12, len=0.7))))

    W = (nh - 1) * pitch + size + (nd - 1) * off
    H = (nv - 1) * pitch + size + (nd - 1) * off
    _arrows(fig, inlet_face, ha, hr, va, vr, da, dr, W, H, off * (nd - 1), size,
            f"Air in {cv('temp', T_air_in):.1f} {ul('temp')}" + (f"<br>{flow_text}" if flow_text else ""),
            "Air out " + ("—" if T_air_out is None else f"{cv('temp', T_air_out):.1f} {ul('temp')}"))

    pad = 1.3
    fig.update_xaxes(visible=False, range=[-pad, W + pad])
    fig.update_yaxes(visible=False, range=[-pad, H + pad], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")
    return fig


def _arrows(fig, inlet, ha, hr, va, vr, da, dr, W, H, depth_span, size, in_text, out_text):
    """Inlet arrow on the upstream side, outlet arrow on the downstream side."""
    axis, end = FACE[inlet]
    sign = 1 if end == 0 else -1                      # +1: flow towards higher index on that axis
    L = 0.9
    if axis == ha:
        sx = sign * (-1 if hr else 1)
        ym = H / 2
        xin = -0.35 if sx > 0 else W + 0.35
        xout = W + 0.35 if sx > 0 else -0.35
        pts = [((xin - sx * L, ym), (xin, ym), in_text, "above"),
               ((xout, ym), (xout + sx * L, ym), out_text, "above")]
    elif axis == va:
        sy = sign * (-1 if vr else 1)
        xm = W / 2
        yin = -0.35 if sy > 0 else H + 0.35
        yout = H + 0.35 if sy > 0 else -0.35
        pts = [((xm, yin - sy * L), (xm, yin), in_text, "right"),
               ((xm, yout), (xm, yout + sy * L), out_text, "right")]
    else:
        # depth axis: increasing depth points up-right on screen
        sd = sign * (-1 if dr else 1)
        near = (size * 0.35, size * 0.35)                 # on the nearest bottom-left pallet
        far = (W - size * 0.35, H - size * 0.35)          # on the farthest top-right pallet
        v = np.array([0.6, 0.6])
        if sd > 0:   # into the screen
            pts = [(tuple(np.array(near) - v * 0.9), near, in_text, "left"),
                   (far, tuple(np.array(far) + v), out_text, "right")]
        else:        # out of the screen
            pts = [(tuple(np.array(far) + v), far, in_text, "right"),
                   (near, tuple(np.array(near) - v * 0.9), out_text, "left")]
    for (x0, y0), (x1, y1), text, where in pts:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=5,
                           arrowcolor=ARROW_COLOR, text="")
        tx, ty = (x0 + x1) / 2, (y0 + y1) / 2
        place = {"above": dict(yshift=24, xanchor="center"), "right": dict(xshift=14, xanchor="left"),
                 "left": dict(xshift=-14, xanchor="right")}[where]
        fig.add_annotation(x=tx, y=ty, text=text, showarrow=False, xref="x", yref="y",
                           font=dict(size=12, color=ARROW_COLOR), bgcolor="rgba(255,255,255,0.75)",
                           align="left", **place)


# ---- time series --------------------------------------------------------------
def extreme_positions(freeze_times_h: Sequence, final_T_core: Sequence[float]) -> tuple[int, int]:
    """(coldest, warmest) 1-based series positions: first / last to freeze."""
    key = [(t if t is not None else np.inf, T) for t, T in zip(freeze_times_h, final_T_core)]
    order = sorted(range(len(key)), key=lambda i: key[i])
    return order[0] + 1, order[-1] + 1


def temperature_figure(res, T_target: float, cv: Callable, ul: Callable, t_mark: float | None = None,
                       height: int = 420) -> go.Figure:
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
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_air_in), name="Air in",
                             line=dict(color="#5a6b7b", dash="dot")))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_air_out), name="Air out",
                             line=dict(color="#2b3a4a", dash="dashdot")))
    fig.add_hline(y=cv("temp", T_target), line_dash="dot", line_color="#888", annotation_text="Target")
    _finish(fig, ul("temp"), t_mark, height)
    return fig


def power_figure(res, cv: Callable, ul: Callable, t_mark: float | None = None, height: int = 420) -> go.Figure:
    C = res.cell
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", C.Q_cell_kW), name="Cooling power",
                             line=dict(color="#1f5fbf", width=2.5)))
    fan = C["fan_kW"] if "fan_kW" in C else np.zeros(len(C))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", fan), name="Fan power",
                             line=dict(color="#e08a1e", width=2)))
    _finish(fig, ul("power"), t_mark, height)
    return fig


def _finish(fig, y_unit, t_mark, height):
    if t_mark is not None:
        fig.add_vline(x=t_mark, line_color="#888", line_width=1)
    fig.update_layout(height=height, xaxis_title="Blasting time [h]", yaxis_title=y_unit,
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
