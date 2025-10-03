"""Box plot chart helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .base import BaseChart


@dataclass
class BoxChart(BaseChart):
    """Convenience wrapper for box plots."""

    def boxplot(
        self,
        data: List[Sequence[float]],
        labels: List[str] | None = None,
        patch_artist: bool = True,
    ) -> None:
        """Create a box plot."""

        box_plot = self.axes.boxplot(data, labels=labels, patch_artist=patch_artist)

        # Color the boxes
        if patch_artist:
            colors = [
                "lightblue",
                "lightgreen",
                "lightcoral",
                "lightyellow",
                "lightgray",
            ]
            for patch, color in zip(box_plot["boxes"], colors * len(box_plot["boxes"])):
                patch.set_facecolor(color)
