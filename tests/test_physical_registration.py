from copy import deepcopy
from dataclasses import replace
import json

import cv2
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, NumberObject, RectangleObject
import pytest
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from printgpt_alignment.document import DocumentPage, load_document
from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_alignment.registration import RegistrationOptions, register_document, transform_points, evaluate_control_points
from printgpt_alignment.printer_calibration import fit_printer, validate_printer, validated_correction, write_target
from printgpt_mvp.pipeline import canonicalize_document, run_offline_document, approve_and_export, _write_toner_outputs

LAYOUT = A4CalibrationLayout(dpi=75)


def document(w=802, h=1000):
    image = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(image, "Original document", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.rectangle(image, (40, 180), (w-40, h-60), (0,0,0), 2)
    return DocumentPage(image, {"source_size_px": [w,h], "kind": "raster"})


def test_802_scan_preview_is_not_registered_and_full_sheet_correction_has_no_padding():
    source = document()
    _, preview = register_document(source, layout=LAYOUT)
    assert not preview["registration"]["export_allowed"]
    assert np.asarray(preview["source_to_canonical"])[1,2] * 297/LAYOUT.raster_height_px == pytest.approx(17.6, abs=.2)
    with pytest.raises(ValueError, match="aspect ratio"):
        register_document(source, layout=LAYOUT, options=RegistrationOptions(mode="full_page", confirmed=True))
    _, corrected = register_document(source, layout=LAYOUT, options=RegistrationOptions(mode="full_page", confirmed=True, allow_nonuniform_resize=True))
    assert corrected["registration"]["export_allowed"]
    mm_points = transform_points([[0,0],[802,1000],[401,500]], corrected["coordinate_system"]["source_to_mm"])
    np.testing.assert_allclose(mm_points, [[0,0],[210,297],[105,148.5]], atol=1e-6)


def test_unknown_geometry_cannot_be_exported(tmp_path):
    path = tmp_path / "scan.png"; cv2.imwrite(str(path), document().image)
    run = run_offline_document(path, np.full(LAYOUT.image_shape,255,np.uint8), tmp_path / "run", layout=LAYOUT)
    with pytest.raises(ValueError, match="preview only"):
        approve_and_export(run, run.suggested_candidate.attempt_id, approved=True)
    assert not (run.output_dir / "print").exists()


def test_dpi_is_evidence_not_automatic_registration(tmp_path):
    path = tmp_path / "scan.png"
    Image.fromarray(document(620,877).image).save(path, dpi=(75,75))
    source = load_document(path)
    assert source.metadata["dpi_xy"] == pytest.approx([75,75], abs=.02)
    _, report = register_document(source, layout=LAYOUT)
    assert not report["registration"]["export_allowed"]
    _, report = register_document(source, layout=LAYOUT, options=RegistrationOptions(mode="metadata", confirmed=True))
    assert report["registration"]["export_allowed"]
    source.metadata["dpi_xy"] = [300,300]
    with pytest.raises(ValueError, match="DPI dimensions disagree"):
        register_document(source, options=RegistrationOptions(mode="metadata", confirmed=True))


def test_exif_rotation_swaps_axes_and_resolution(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.fromarray(document(600,800).image)
    exif = image.getexif(); exif[274] = 6
    image.save(path, exif=exif, dpi=(100,200))
    source = load_document(path)
    assert source.image.shape[:2] == (600,800)
    assert source.metadata["dpi_xy"] == [200,100]


@pytest.mark.parametrize("rotation", [0,90,180,270])
@pytest.mark.parametrize("cropped", [False,True])
def test_pdf_crop_offsets_rotation_and_visible_content(tmp_path, rotation, cropped):
    raw, path = tmp_path / "raw.pdf", tmp_path / "input.pdf"
    pdf = canvas.Canvas(str(raw), pagesize=(210*mm,297*mm))
    for x,y in [(45,60),(160,60),(45,240),(160,240),(105,148)]:
        pdf.circle(x*mm,(297-y)*mm,1*mm,stroke=0,fill=1)
    pdf.showPage(); pdf.save()
    writer = PdfWriter(); page = writer.add_page(PdfReader(raw).pages[0])
    if cropped:
        page.cropbox = RectangleObject([10*mm,15*mm,200*mm,277*mm])
    page.rotate(rotation)
    with path.open("wb") as stream: writer.write(stream)
    source = load_document(path, dpi=150)
    assert source.metadata["page_size_mm"] == pytest.approx([210,297], abs=.001)
    warped, report = register_document(source, layout=LAYOUT, options=RegistrationOptions(mode="metadata", confirmed=True))
    assert report["registration"]["export_allowed"]
    for x,y in [(45,60),(160,60),(45,240),(160,240),(105,148)]:
        px, py = int(round(x/210*LAYOUT.raster_width_px)), int(round(y/297*LAYOUT.raster_height_px))
        assert warped[py-2:py+2,px-2:px+2].min() < 60
    if cropped:
        polygon = np.array(report["observed_polygon_canonical_px"])
        assert polygon[:,1].min() * 297/LAYOUT.raster_height_px == pytest.approx(20, abs=.01)


def test_pdf_userunit_and_non_a4_not_stretched(tmp_path):
    path = tmp_path / "unit.pdf"
    writer = PdfWriter(); page = writer.add_blank_page(105*mm,148.5*mm)
    page[NameObject("/UserUnit")] = NumberObject(2)
    with path.open("wb") as stream: writer.write(stream)
    source = load_document(path)
    assert source.metadata["page_size_mm"] == pytest.approx([210,297], abs=.001)
    _, report = register_document(source, layout=LAYOUT, options=RegistrationOptions(mode="metadata",confirmed=True))
    assert report["registration"]["export_allowed"]
    _, report = register_document(document(600,800), layout=LAYOUT,
        options=RegistrationOptions(mode="full_page",confirmed=True,page_size_mm=(150,200)))
    assert not report["registration"]["export_allowed"]
    np.testing.assert_allclose(transform_points([[600,800]], report["coordinate_system"]["source_to_mm"]), [[150,200]], atol=1e-5)


def test_measured_crop_mapping_and_missing_area_protected(tmp_path):
    path = tmp_path / "crop.png"; cv2.imwrite(str(path), document(450,600).image)
    opts = RegistrationOptions(mode="crop", confirmed=True,dpi_xy=(75,75),origin_mm=(15,25))
    run = run_offline_document(path,np.full(LAYOUT.image_shape,255,np.uint8),tmp_path/"run", layout=LAYOUT, registration=opts)
    np.testing.assert_allclose(transform_points([[0,0],[75,75]],run.alignment["coordinate_system"]["source_to_mm"]), [[15,25],[40.4,50.4]], atol=1e-5)
    assert not run.masks.allowed_region_mask[:74].any()
    assert not run.masks.allowed_region_mask[:,:45].any()
    assert run.alignment["registration"]["export_allowed"]
    with pytest.raises(ValueError, match="origin_mm"):
        register_document(document(), options=replace(opts,origin_mm=None))


@pytest.mark.parametrize("corners", [((0,0),(802,1000),(802,0),(0,1000)), ((-1,0),(802,0),(802,1000),(0,1000)), ((0,0),(1,0),(1,1),(0,1))])
def test_invalid_or_missing_paper_corners_rejected(corners):
    with pytest.raises(ValueError, match="corners"):
        register_document(document(), options=RegistrationOptions(mode="corners",confirmed=True,corners_px=corners))


@pytest.mark.parametrize("seed", range(30))
def test_marker_free_perspective_warp_at_independent_points(seed):
    rng = np.random.default_rng(seed)
    w,h = LAYOUT.raster_width_px, LAYOUT.raster_height_px
    image = np.full((h,w,3),255,np.uint8)
    points = np.array([(int(x*w),int(y*h)) for x in [.15,.5,.85] for y in [.15,.5,.85]], dtype=float)
    for x,y in points.astype(int): cv2.circle(image,(x,y),4,(0,0,0),-1)
    corners = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    quad = corners * rng.uniform(.85,1.1) + [100,100] + rng.uniform(-45,45,(4,2))
    # Quarter-turn source orientation; physical corner order remains explicit.
    quad = np.roll(quad, seed%4, axis=0).astype(np.float32)
    gt = cv2.getPerspectiveTransform(corners,quad)
    source = cv2.warpPerspective(image,gt,(900,1200),borderValue=(100,100,100))
    source = cv2.GaussianBlur(source,(3,3),.5+seed%3*.2)
    shadow = np.linspace(.75,1,source.shape[1])[None,:,None]
    source = np.clip(source*shadow + rng.normal(0,1.2,source.shape),0,255).astype(np.uint8)
    recovered, report = register_document(DocumentPage(source,{}),layout=LAYOUT,
        options=RegistrationOptions(mode="corners",confirmed=True,corners_px=tuple(map(tuple,quad))))
    expected_mm = points * [210/w,297/h]
    measured = []
    for x,y in points.astype(int):
        yy,xx = np.nonzero(cv2.cvtColor(recovered[y-8:y+9,x-8:x+9],cv2.COLOR_BGR2GRAY) < 80)
        assert len(xx)>5
        measured.append([x-8+xx.mean(),y-8+yy.mean()])
    errors = np.linalg.norm((np.array(measured)-points)*[210/w,297/h],axis=1)
    assert errors.max() < .5
    check = evaluate_control_points(transform_points(points,gt),expected_mm,report["source_to_canonical"],layout=LAYOUT)
    assert check["max_error_mm"] < .01


def test_pdf_export_preserves_independent_ink_coordinates(tmp_path):
    mask = np.zeros(LAYOUT.image_shape,np.uint8)
    w,h = LAYOUT.raster_width_px,LAYOUT.raster_height_px
    for x,y in [(30,40),(180,40),(30,257),(180,257),(105,148)]:
        cv2.circle(mask,(round(x*w/210),round(y*h/297)),3,255,-1)
    _, path = _write_toner_outputs(mask,tmp_path,LAYOUT)
    rendered = load_document(path,dpi=300,grayscale=True).image
    for x,y in [(30,40),(180,40),(30,257),(180,257),(105,148)]:
        px,py = round(x*rendered.shape[1]/210),round(y*rendered.shape[0]/297)
        yy,xx = np.nonzero(rendered[py-20:py+21,px-20:px+21] < 80)
        recovered = np.array([px-20+xx.mean(),py-20+yy.mean()])*[210/rendered.shape[1],297/rendered.shape[0]]
        assert np.linalg.norm(recovered-[x,y]) < .5


def calibration_records():
    rows=[]
    F=np.array([[1.002,-.002,1.2],[.003,.998,-.7],[0,0,1.]])
    for j,y in enumerate(np.linspace(25,272,7)):
        for i,x in enumerate(np.linspace(20,190,5)):
            rows.append({"sheet":"calibration-1","role":"fit" if (i+j)%2==0 else "holdout",
                         "expected_mm":[x,y],"measured_mm":(F @ [x,y,1])[:2].tolist()})
    record={"session_id":"fit-session","measurement_source":"physical","settings":{"printer":"Canon imageRUNNER 2425i","tray":"1","driver":"test","side":"simplex","orientation":"portrait","paper":"test stock","scaling":"100%"},"points":rows}
    profile=fit_printer(record)
    verification=[]
    for sheet in range(5):
        for row in rows:
            verification.append(dict(row,sheet=str(sheet), measured_mm=(np.array(row["expected_mm"])+[.01*sheet,-.01*sheet]).tolist()))
    validation={"session_id":"separate-session","measurement_source":"physical","measurement_geometry_verified":True,
                "measurement_uncertainty_mm":.05,"safe_margins_mm":[5]*4,"settings":record["settings"],
                "applied_correction_id":profile["correction_id"],"points":verification}
    return profile,validation,F


def test_printer_fit_and_independent_corrected_repeat_measurements():
    profile,record,F=calibration_records()
    assert profile["status"]=="provisional"
    np.testing.assert_allclose(profile["command_to_physical_mm"],F,atol=1e-10)
    verified=validate_printer(profile,record)
    assert verified["status"]=="measured"
    assert verified["max_repeat_displacement_mm"] == pytest.approx(np.sqrt(2)*.04)
    np.testing.assert_allclose(validated_correction(verified),np.linalg.inv(F),atol=1e-10)


def test_simulated_or_bad_repeat_data_cannot_claim_measured_accuracy():
    profile,record,_=calibration_records()
    record["measurement_source"]="synthetic"
    assert validate_printer(profile,record)["status"]=="not_verified"
    record["measurement_source"]="physical"
    record["points"][0]["measured_mm"][0]+=1
    failed=validate_printer(profile,record)
    assert failed["status"]=="not_verified"
    with pytest.raises(ValueError,match="physical"):
        validated_correction(failed)
    record["points"]=[r for r in record["points"] if r["sheet"]!="4"]
    with pytest.raises(ValueError,match="five"):
        validate_printer(profile,record)


def test_target_is_exact_a4_and_template_requires_real_measurements(tmp_path):
    path=write_target(tmp_path/"target.pdf",tmp_path/"measurements.json")
    page=PdfReader(path).pages[0]
    assert float(page.mediabox.width)/mm == pytest.approx(210,abs=.001)
    data=json.loads((tmp_path/"measurements.json").read_text())
    assert len(data["points"])==35
    with pytest.raises(ValueError,match="finite"):
        fit_printer(data)
    with pytest.raises(FileExistsError,match="preserve"):
        write_target(path,tmp_path/"measurements.json")


def test_source_controls_are_independent_and_enforce_target():
    source=document(620,877)
    mm_points=np.array([(x,y) for x in [30,105,180] for y in [40,148,257]],dtype=float)
    pixels=mm_points*[620/210,877/297]
    opts=RegistrationOptions(mode="full_page",confirmed=True,control_points_px=tuple(map(tuple,pixels)),
                             control_points_mm=tuple(map(tuple,mm_points)),measurement_uncertainty_mm=.05)
    _,report=register_document(source,layout=LAYOUT,options=opts)
    assert report["registration"]["measured_source_accuracy_mm"]==pytest.approx(.05,abs=1e-5)
    assert report["registration"]["export_allowed"]
    _,bad=register_document(source,layout=LAYOUT,options=replace(opts,control_points_mm=tuple(map(tuple,mm_points+[1,0]))))
    assert not bad["registration"]["export_allowed"]


def test_blurred_capture_cannot_be_certified_by_confirmation():
    source=document(620,877)
    blurry=cv2.GaussianBlur(source.image,(31,31),8)
    _,report=register_document(DocumentPage(blurry,source.metadata),layout=LAYOUT,
                              options=RegistrationOptions(mode="full_page",confirmed=True))
    assert not report["registration"]["export_allowed"]
    assert any("sharp" in w for w in report["warnings"])


def test_printer_correction_export_is_inverse_in_mm_and_does_not_mutate_mask(tmp_path):
    mask=np.zeros(LAYOUT.image_shape,np.uint8)
    x,y=105.,148.5
    cv2.circle(mask,(round(x*LAYOUT.raster_width_px/210),round(y*LAYOUT.raster_height_px/297)),4,255,-1)
    original=mask.copy()
    correction=np.array([[1.,0,-1.2],[0,1.,.7],[0,0,1.]])
    png,pdf=_write_toner_outputs(mask,tmp_path,LAYOUT,correction)
    np.testing.assert_array_equal(mask,original)
    for image in [cv2.imread(str(png),0),load_document(pdf,dpi=300,grayscale=True).image]:
        yy,xx=np.nonzero(image<80)
        recovered=np.array([xx.mean()*210/image.shape[1],yy.mean()*297/image.shape[0]])
        assert np.linalg.norm(recovered-[x-1.2,y+.7])<.5


def test_calibrated_profile_export_and_clearance_budget(tmp_path):
    path=tmp_path/"source.png"; cv2.imwrite(str(path),document(620,877).image)
    run=run_offline_document(path,np.full(LAYOUT.image_shape,255,np.uint8),tmp_path/"run",layout=LAYOUT,
                             registration=RegistrationOptions(mode="full_page",confirmed=True))
    profile,record,_=calibration_records()
    verified=validate_printer(profile,record)
    exported=approve_and_export(run,run.suggested_candidate.attempt_id,approved=True,printer_profile=verified)
    manifest=json.loads(exported.manifest_path.read_text())
    assert manifest["printer"]["protection_budget"]=="unverified source accuracy"
    measured_alignment=deepcopy(run.alignment)
    measured_alignment["registration"]["measured_source_accuracy_mm"]=.45
    measured_run=replace(run,alignment=measured_alignment)
    with pytest.raises(ValueError,match="clearance"):
        approve_and_export(measured_run,run.suggested_candidate.attempt_id,approved=True,printer_profile=verified)


def test_non_a4_auto_options_cannot_enable_export():
    from printgpt_alignment.calibration_page import render_calibration_image
    source=DocumentPage(render_calibration_image(LAYOUT),{})
    _,report=register_document(source,layout=LAYOUT,options=RegistrationOptions(page_size_mm=(150,200)))
    assert not report["registration"]["export_allowed"]


def test_low_resolution_full_page_confirmation_is_not_enough():
    source=document(210,297)
    _,report=register_document(source,layout=LAYOUT,options=RegistrationOptions(mode="full_page",confirmed=True))
    assert not report["registration"]["export_allowed"]
    assert report["registration"]["max_sampled_pixel_diagonal_mm"]>1
