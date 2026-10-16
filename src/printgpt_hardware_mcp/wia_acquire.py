"""Subprocess entry point: acquire one A4, 300-DPI page through WIA.

Run only by :meth:`Win32ScannerBackend.acquire`, which enforces a timeout and
kills this process if the Canon driver hangs. Never imported by tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

WIA_FORMAT_PNG = "{B96B3CAF-0728-11D3-9D7B-0000F81EF32E}"
A4_300DPI = (2480, 3508)
INTENT = {"color": 1, "grayscale": 2}


def _set(properties, name: str, value) -> None:
    for index in range(1, properties.Count + 1):
        item = properties.Item(index)
        if item.Name == name:
            item.Value = value
            return
    raise RuntimeError(f"scanner does not expose property {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=sorted(INTENT), default="grayscale")
    args = parser.parse_args(argv)
    if args.output.exists():
        print("output already exists", file=sys.stderr)
        return 2

    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    manager = win32com.client.Dispatch("WIA.DeviceManager")
    info = next((manager.DeviceInfos.Item(i) for i in range(1, manager.DeviceInfos.Count + 1)
                 if manager.DeviceInfos.Item(i).Type == 1
                 and "scangear" in str(manager.DeviceInfos.Item(i).Properties("Name").Value).lower()), None)
    if info is None:
        print("Canon ScanGear WIA device not found", file=sys.stderr)
        return 3
    device = info.Connect()
    item = device.Items.Item(1)
    props = item.Properties
    _set(props, "Current Intent", INTENT[args.mode])
    _set(props, "Horizontal Resolution", 300)
    _set(props, "Vertical Resolution", 300)
    _set(props, "Horizontal Start Position", 0)
    _set(props, "Vertical Start Position", 0)
    _set(props, "Horizontal Extent", A4_300DPI[0])
    _set(props, "Vertical Extent", A4_300DPI[1])
    image = item.Transfer(WIA_FORMAT_PNG)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.SaveFile(str(args.output))
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
