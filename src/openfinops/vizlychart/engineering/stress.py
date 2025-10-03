"""Mechanical engineering plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..charts.line import LineChart


@dataclass
class StressStrainChart(LineChart):
    """Render stress-strain curves with typical annotation defaults."""

    def plot(
        self,
        strain: Sequence[float] | Iterable[float],
        stress: Sequence[float] | Iterable[float],
        *,
        yield_point: tuple[float, float] | None = None,
        ultimate_point: tuple[float, float] | None = None,
        xlabel: str = "Strain",
        ylabel: str = "Stress (MPa)",
    ) -> None:
        super().plot(strain, stress, xlabel=xlabel, ylabel=ylabel, grid=True)
        if yield_point:
            self.axes.scatter(*yield_point, color="#dc2626", label="Yield")
        if ultimate_point:
            self.axes.scatter(*ultimate_point, color="#16a34a", label="Ultimate")
        if yield_point or ultimate_point:
            self.axes.legend()
