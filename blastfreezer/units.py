"""Display units. The model is SI internally; this module only converts for the UI.

display_value = si_value * scale + offset
"""

from __future__ import annotations

from dataclasses import dataclass

SI, IP = "SI", "Imperial"


@dataclass(frozen=True)
class Quantity:
    si: str
    ip: str
    scale: float          # IP per SI
    offset: float = 0.0
    step_si: float = 1.0
    step_ip: float = 1.0
    fmt_si: str = "%.2f"
    fmt_ip: str = "%.2f"


Q = {
    "temp":     Quantity("°C", "°F", 9 / 5, 32.0, 0.5, 1.0, "%.1f", "%.1f"),
    "length":   Quantity("m", "ft", 3.280839895, 0, 0.01, 0.05, "%.2f", "%.2f"),
    "small":    Quantity("mm", "in", 1 / 25.4, 0, 0.5, 0.01, "%.1f", "%.3f"),   # SI value is mm
    "mass":     Quantity("kg", "lb", 2.2046226218, 0, 10.0, 10.0, "%.1f", "%.0f"),
    "cp":       Quantity("J/kg·K", "Btu/lb·°F", 1 / 4186.8, 0, 10.0, 0.005, "%.0f", "%.3f"),
    "k":        Quantity("W/m·K", "Btu/h·ft·°F", 1 / 1.730735, 0, 0.01, 0.005, "%.3f", "%.3f"),
    "density":  Quantity("kg/m³", "lb/ft³", 1 / 16.018463, 0, 10.0, 0.5, "%.0f", "%.1f"),
    "flow":     Quantity("m³/s", "CFM", 2118.880003, 0, 0.05, 100.0, "%.2f", "%.0f"),
    "htc":      Quantity("W/m²·K", "Btu/h·ft²·°F", 1 / 5.678263, 0, 1.0, 0.1, "%.1f", "%.2f"),
    "power":    Quantity("kW", "TR", 1 / 3.516853, 0, 1.0, 1.0, "%.1f", "%.1f"),
    "energy":   Quantity("kWh", "ton·h", 1 / 3.516853, 0, 1.0, 1.0, "%.1f", "%.1f"),
    "area":     Quantity("m²", "ft²", 10.76391042, 0, 0.1, 1.0, "%.2f", "%.1f"),
    "volume":   Quantity("m³", "ft³", 35.31466672, 0, 0.1, 1.0, "%.3f", "%.1f"),
    "latent":   Quantity("MJ/m³", "Btu/ft³", 26.839192, 0, 1.0, 1.0, "%.1f", "%.0f"),  # SI value is MJ/m3
}


def unit(q: str, system: str) -> str:
    return Q[q].si if system == SI else Q[q].ip


def to_disp(q: str, v, system: str):
    if system == SI:
        return v
    d = Q[q]
    return v * d.scale + d.offset


def from_disp(q: str, v, system: str):
    if system == SI:
        return v
    d = Q[q]
    return (v - d.offset) / d.scale


def step(q: str, system: str) -> float:
    return Q[q].step_si if system == SI else Q[q].step_ip


def fmt(q: str, system: str) -> str:
    return Q[q].fmt_si if system == SI else Q[q].fmt_ip
