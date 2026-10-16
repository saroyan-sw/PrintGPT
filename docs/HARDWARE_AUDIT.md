# Canon imageRUNNER 2425i — read-only hardware audit

Audited on 2026-10-16 from the PrintGPT development laptop (Windows 11) with
`Get-Printer`, `Get-PrinterPort`, `Get-PrintConfiguration`, pywin32
`DeviceCapabilities`/`GetDeviceCaps` and WIA property reads. No settings, ports,
drivers or firmware were changed, nothing was printed, and no scan was started.

## Print queues

| Queue | Driver | Port |
| --- | --- | --- |
| **Canon Generic Plus UFR II** | Canon Generic Plus UFR II | `10.1.3.175` |
| Microsoft Print to PDF | Microsoft Print To PDF | `PORTPROMPT:` |
| Fax, OneNote (×3) | Microsoft virtual drivers | local |

## Canon queue

| Property | Value |
| --- | --- |
| Exact queue name | `Canon Generic Plus UFR II` (allowlisted) |
| Driver | Canon Generic Plus UFR II, version **3.50.0.0**, Windows x64, print processor `CnXP0PP`, datatype RAW |
| Port | Standard TCP/IP port `10.1.3.175`, host address 10.1.3.175, RAW 9100, SNMP disabled |
| Reachability | ICMP ping OK; TCP 9100 connect OK |
| Spooler state | Normal / idle, not "work offline", 0 queued jobs |
| **Queue defaults** | **Letter**, portrait, **duplex long edge**, monochrome, 100 %, 600 DPI |
| Colour | Monochrome-only device (`DC_COLORDEVICE` = 0) |
| Duplex | Supported |
| Resolutions | 600 × 600 and 1200 × 1200 DPI |
| Copies | up to 9999 (server restricts to 1) |
| Paper sizes | Letter, Tabloid, 11×17, Legal, Statement, Executive, A5, B5, **A4 (id 9)**, B4, A3, A2, envelopes, Oficio/Folio variants, 16K, 8K, custom |
| A4 printable area (driver) | Physical 4960 × 7014 px @ 600 DPI = 209.97 × 296.93 mm; offset 120 px = **5.08 mm** on every side; printable 199.81 × 286.77 mm |

Consequences for PrintGPT: every job must override paper size (Letter → A4) and
duplex (long edge → simplex) explicitly. The driver accepts these through the
public DEVMODE: a DC created with the job DEVMODE reports a portrait A4 sheet.
Ink inside the outer 5.08 mm is clipped by the device; PrintGPT masking already
keeps a 10 mm margin.

Not exposed by Windows or the public DEVMODE: Canon driver-private layout options
(Enlarge Print Region, gutter, booklet, poster), tray paper size, and true
physical scale. Windows also reports "Normal" for a RAW TCP queue when the device
is switched off, so the server adds a TCP probe to 9100.

## Scanner

| Property | Value |
| --- | --- |
| Software | Canon Color Network ScanGear 2 2.3.7 (also NAPS2 8.5.1 installed) |
| WIA | Device **"Color Network ScanGear 2"**, WIA 2.0, driver 2.3.7.0, port "Network" |
| TWAIN | `twain_32\ScanGearIR\SGIR.DS` and `wiatwain.ds` |
| WIA device properties | Optical 300 × 300 DPI, flatbed selected, bed 297 × 420 mm (A3), feeder capability reported |
| Device online | `ScanGear IR Device Online = 0` while the panel was not in Remote Scanner mode |
| WIA scan item | Default 200 DPI colour BMP; resolution, extent and intent are writable |
| Scans folder | `%USERPROFILE%\OneDrive\Documents\Scans` (example `0775_001.pdf`, A4 PDF) |

Conclusion: WIA exposes the scanner noninteractively at the property level, but
acquisition only works after the user selects **Scan → Remote Scanner** on the
printer panel, and it has not been verified with a real page. Direct acquisition
is therefore implemented as an opt-in experimental tool (`acquire_scan`, disabled
by default). The supported path is manual import from the scans folder.
