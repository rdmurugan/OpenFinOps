"""Export utilities for web components."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from .components import BaseWebComponent

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base class for exporters."""

    @abstractmethod
    def export(self, component: BaseWebComponent) -> str:
        """Export component to string format."""
        ...


class HTMLExporter(BaseExporter):
    """Export components to HTML."""

    def export(self, component: BaseWebComponent) -> str:
        """Export component to HTML."""
        return component.to_html()


class JSONExporter(BaseExporter):
    """Export components to JSON."""

    def export(self, component: BaseWebComponent) -> str:
        """Export component to JSON."""
        return json.dumps(component.to_json(), indent=2)


class WebComponentExporter(BaseExporter):
    """Export components as web components."""

    def export(self, component: BaseWebComponent) -> str:
        """Export as web component."""
        html = component.to_html()
        json_data = json.dumps(component.to_json())

        return f"""
        <script>
        class VizlyComponent extends HTMLElement {{
            constructor() {{
                super();
                this.attachShadow({{ mode: 'open' }});
                this.data = {json_data};
            }}

            connectedCallback() {{
                this.shadowRoot.innerHTML = `{html}`;
            }}
        }}

        customElements.define('vizly-component', VizlyComponent);
        </script>
        """
