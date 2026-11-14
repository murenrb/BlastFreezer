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
                T_core_frames, times: Sequence[float], T_out_frames: Sequence,
                T_hot: float, T_cold: float, cv: Callable, ul: Callable, T_in_frames: Sequence,
                flow_text: str = "", height: int = 560, max_frames: int = MAX_FRAMES) -> go.Figure:
    """Front view of the cell (width across, height up, depth stacked behind).

    T_core_frames: array (n_times, n_series) of core temperatures by series position.
    T_in_frames / T_out_frames: cell supply / return air per time (None = unknown).
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
    in_xy, out_xy, label_pos = _arrows(inlet_face, W, H)
    def air_labels(T_in, T_out):
        f = lambda T: "—" if T is None else f"{_fmt_T(cv, T)} {ul('temp')}"
        return ["Air in " + f(T_in) + (f"<br>{flow_text}" if flow_text else ""), "Air out " + f(T_out)]

    fig = go.Figure()
    T0 = T_core_frames[keep[0]]
    for (x0, y0, d, s, idx), c in zip(pallets, colors(T0)):
        fig.add_trace(go.Scatter(
            x=[x0, x0 + size, x0 + size, x0, x0], y=[y0, y0, y0 + size, y0 + size, y0],
            mode="lines", fill="toself", fillcolor=c, line=dict(color="#1b2633", width=1.2),
            hoveron="fills", hoverinfo="text", showlegend=False,
            text=f"P{s + 1} · width {idx['width'] + 1}, height {idx['height'] + 1}, depth {d + 1}"))
    i_text = len(fig.data)
    fig.add_trace(go.Scatter(x=[x0 + size / 2 for x0, *_ in front], y=[y0 + size / 2 for _, y0, *_ in front],
                             mode="text", text=labels(T0), textfont=dict(color="white", size=font),
                             hovertext=hovers(T0, times[keep[0]]), hoverinfo="text", showlegend=False))
    i_air = len(fig.data)
    fig.add_trace(go.Scatter(x=[label_pos[0][0], label_pos[1][0]], y=[label_pos[0][1], label_pos[1][1]],
                             mode="text", text=air_labels(T_in_frames[keep[0]], T_out_frames[keep[0]]),
                             textposition=[label_pos[0][2], label_pos[1][2]],
                             textfont=dict(size=12, color=ARROW_COLOR), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=COLORSCALE, cmin=cv("temp", T_cold), cmax=cv("temp", T_hot), color=[0],
                    showscale=True, colorbar=dict(title=dict(text=f"Core<br>{ul('temp')}"),
                                                  thickness=12, len=0.7))))
    for (x0, y0), (x1, y1) in (in_xy, out_xy):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=5,
                           arrowcolor=ARROW_COLOR, text="")

    pad = 1.5
    xpad = 3.2 if inlet_face in ("left", "right") else pad      # room for side labels
    fig.update_xaxes(visible=False, range=[-xpad, W + xpad], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-pad, H + pad], scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

    if len(keep) > 1:
        idx_traces = list(range(len(pallets))) + [i_text, i_air]
        frames = []
        for k in keep:
            T = T_core_frames[k]
            data = [dict(type="scatter", fillcolor=c) for c in colors(T)]
            data.append(dict(type="scatter", text=labels(T), hovertext=hovers(T, times[k])))
            data.append(dict(type="scatter", text=air_labels(T_in_frames[k], T_out_frames[k])))
            frames.append(go.Frame(name=str(k), data=data, traces=idx_traces))
        fig.frames = frames
        step_args = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
        fig.update_layout(
            sliders=[dict(active=0, x=0.12, len=0.86, y=0, yanchor="top", pad=dict(t=10, b=0),
                          currentvalue=dict(prefix="Model time: ", suffix=" h", font=dict(size=13)),
                          steps=[dict(method="animate", label=f"{times[k]:.1f}", args=[[str(k)], step_args])
                                 for k in keep])],
            updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0, xanchor="left", yanchor="top",
                              pad=dict(t=22), showactive=False, buttons=[
                                  dict(label="▶", method="animate",
                                       args=[None, dict(frame=dict(duration=80, redraw=True),
                                                        fromcurrent=True, transition=dict(duration=0))]),
                                  dict(label="❚❚", method="animate",
                                       args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                          transition=dict(duration=0))])])],
            margin=dict(l=0, r=0, t=10, b=90), height=height + 80)
    return fig


def _arrows(inlet: str, W: float, H: float):
    """Inlet/outlet arrow end points and label positions (x, y, textposition)."""
    L, gap = 0.9, 0.35
    if inlet in ("left", "right"):
        sx = 1 if inlet == "left" else -1
        xin = -gap if sx > 0 else W + gap
        xout = W + gap if sx > 0 else -gap
        y = H / 2
        in_xy = ((xin - sx * L, y), (xin, y))
        out_xy = ((xout, y), (xout + sx * L, y))
        # anchor at the outer end of each arrow, text extending away from the cell
        outer_in, outer_out = xin - sx * L, xout + sx * L
        lab = [(outer_in, y + 0.25, "top left" if sx > 0 else "top right"),
               (outer_out, y + 0.25, "top right" if sx > 0 else "top left")]
    else:
        sy = 1 if inlet == "bottom" else -1
        yin = -gap if sy > 0 else H + gap
        yout = H + gap if sy > 0 else -gap
        x = W / 2
        in_xy = ((x, yin - sy * L), (x, yin))
        out_xy = ((x, yout), (x, yout + sy * L))
        lab = [(x + 0.2, yin - sy * L / 2, "middle right"), (x + 0.2, yout + sy * L / 2, "middle right")]
    return in_xy, out_xy, lab


# ---- time series --------------------------------------------------------------
def extreme_positions(freeze_times_h: Sequence, final_T_core: Sequence[float]) -> tuple[int, int]:
    """(coldest, warmest) 1-based series positions: first / last to freeze."""
    key = [(t if t is not None else np.inf, T) for t, T in zip(freeze_times_h, final_T_core)]
    order = sorted(range(len(key)), key=lambda i: key[i])
    return order[0] + 1, order[-1] + 1


def temperature_figure(res, T_target: float, cv: Callable, ul: Callable, height: int = 420) -> go.Figure:
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
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_air_in), name="Air in (coil supply)",
                             line=dict(color="#5a6b7b", dash="dot")))
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.T_air_out), name="Air out (coil return)",
                             line=dict(color="#2b3a4a", dash="dashdot")))
    if "SST" in C and C.SST.notna().any():
        fig.add_trace(go.Scatter(x=C.time_h, y=cv("temp", C.SST), name="SST",
                                 line=dict(color="#2a9d8f", width=1.5)))
    fig.add_hline(y=cv("temp", T_target), line_dash="dot", line_color="#888", annotation_text="Target")
    _finish(fig, ul("temp"), height)
    return fig


def power_figure(res, cv: Callable, ul: Callable, height: int = 420,
                 design_kW: float | None = None) -> go.Figure:
    C = res.cell
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=C.time_h, y=cv("power", C.Q_cell_kW), name="Cooling power",
                             line=dict(color="#1f5fbf", width=2.5)))
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
