"""Local stdio MCP adapter around the Windows Canon print queue and scanner files.

This package never replaces a driver. It validates PrintGPT artifacts, talks to
the installed Windows queue through injectable adapters, and keeps an audit
trail. Nothing here imports the generation pipeline or Streamlit.
"""

from .config import HardwareConfig
from .service import HardwareService

__all__ = ["HardwareConfig", "HardwareService"]
