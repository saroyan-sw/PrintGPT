"""Reproducible marker-free geometry benchmark with independent image measurements.

Corners are supplied ground truth with optional annotation error, NOT automatic
corner detection. This measures the correction/coordinate/export building block.
It does not establish scanner or printer accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .document import DocumentPage
from .geometry import A4CalibrationLayout
from .registration import RegistrationOptions, register_document, transform_points, evaluate_control_points


def run_benchmark(output_dir: str | Path, *, count=30, seed=2425):
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    layout = A4CalibrationLayout(dpi=150)
    w,h = layout.raster_width_px,layout.raster_height_px
    scale = np.array([210/w,297/h])
    corners=np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    points=np.array([(round(x*w),round(y*h)) for x in [.12,.32,.5,.68,.88] for y in [.12,.32,.5,.68,.88]],dtype=float)
    expected_mm=points*scale
    base=np.full((h,w,3),255,np.uint8)
    for x,y in points.astype(int): cv2.circle(base,(x,y),7,(0,0,0),-1)
    results=[]
    for index in range(count):
        rng=np.random.default_rng(seed+index)
        case=root/f"case_{index:02d}"; case.mkdir(exist_ok=True)
        if index % 6 == 5:
            # A known crop; edges cannot be invented by a corner detector.
            x0,y0=60,80
            source=base[y0:h-70,x0:w-50].copy()
            gt=np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])
            # Exact axis spacing matches the raster's physical extent.
            opts=RegistrationOptions(mode="crop",confirmed=True,
                dpi_xy=(w/210*25.4,h/297*25.4),origin_mm=(x0*scale[0],y0*scale[1]))
            label="known crop offsets"
        else:
            quad=(corners*rng.uniform(.8,1.05)+[130,130]+rng.uniform(-65,65,(4,2))).astype(np.float32)
            quad=np.roll(quad,index%4,axis=0)
            gt=cv2.getPerspectiveTransform(corners,quad)
            source=cv2.warpPerspective(base,gt,(1550,2050),borderValue=(110,110,110))
            # Subpixel corner annotation noise is independent of check points.
            annotated=quad+rng.normal(0,.2,(4,2))
            opts=RegistrationOptions(mode="corners",confirmed=True,corners_px=tuple(map(tuple,annotated)))
            label="perspective / borders / quarter-turn / annotated corners"
        source=cv2.GaussianBlur(source,(3,3),.5+(index%3)*.2)
        source=np.clip(source*np.linspace(.72,1,source.shape[1])[None,:,None]+rng.normal(0,1.5,source.shape),0,255).astype(np.uint8)
        cv2.imwrite(str(case/"input.png"),source)
        record={"case":index,"kind":label,"ground_truth_canonical_to_source":gt.tolist(),
                "expected_control_mm":expected_mm.tolist(),"ground_truth_source_controls":transform_points(points,gt).tolist()}
        try:
            warped, report=register_document(DocumentPage(source,{}),layout=layout,options=opts)
            centroid=[]
            for x,y in points.astype(int):
                yy,xx=np.nonzero(cv2.cvtColor(warped[y-12:y+13,x-12:x+13],cv2.COLOR_BGR2GRAY)<80)
                if len(xx)<10: raise ValueError("independent printed dot was not recovered")
                centroid.append([x-12+xx.mean(),y-12+yy.mean()])
            actual=np.array(centroid)
            errors=np.linalg.norm((actual-points)*scale,axis=1)
            corner_error=np.linalg.norm((transform_points(transform_points(corners,gt),report["source_to_canonical"])-corners)*scale,axis=1)
            overlay=warped.copy()
            for expected,recovered in zip(points,actual):
                cv2.drawMarker(overlay,tuple(np.rint(expected).astype(int)),(0,180,0),cv2.MARKER_CROSS,25,2)
                cv2.circle(overlay,tuple(np.rint(recovered).astype(int)),10,(0,0,255),2)
            cv2.imwrite(str(case/"rectified.png"),warped)
            cv2.imwrite(str(case/"overlay_expected_green_recovered_red.png"),overlay)
            record.update(accepted=report["registration"]["export_allowed"],alignment=report,
                mean_error_mm=float(errors.mean()),max_error_mm=float(errors.max()),
                max_error_px=float(np.linalg.norm(actual-points,axis=1).max()),
                corner_errors_mm=corner_error.tolist(),
                transform_check=evaluate_control_points(transform_points(points,gt),expected_mm,report["source_to_canonical"],layout=layout),
                under_0_5mm=bool(errors.max()<.5 and corner_error.max()<.5))
        except ValueError as exc:
            record.update(accepted=False,reason=str(exc),under_0_5mm=False)
        (case/"metrics.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        results.append({k:v for k,v in record.items() if k not in ("alignment","ground_truth_source_controls","expected_control_mm")})
    accepted=[r for r in results if r["accepted"]]
    summary={"suite":"marker-free supplied-geometry synthetic", "seed":seed,"count":count,
             "accepted":len(accepted),"rejection_rate":1-len(accepted)/count,
             "failure_rate":sum(not r["under_0_5mm"] for r in results)/count,
             "max_accepted_error_mm":max((r["max_error_mm"] for r in accepted),default=None),
             "printer_accuracy":"UNMEASURED; synthetic software test only", "cases":results}
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/physical_alignment"))
    args=parser.parse_args()
    result=run_benchmark(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))


if __name__=="__main__": main()
