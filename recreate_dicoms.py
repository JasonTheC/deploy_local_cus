#!/usr/bin/env python3
"""
Recreate every study's DICOMs locally using the NEW depth-calibration method
(the FOV-axial-height + depth-ruler logic from pacs-transfer-server.py), so the
new pixel spacing can be verified before/without sending anything to PACS.

What it does, per study under US_images/:
  * groups raw frames into series exactly like send_to_pacs (one series per
    .../raw/ leaf folder),
  * computes the per-series median FOV spacing and rebuilds every .dcm into the
    sibling *_dicom folder (overwriting the old ones),
  * prints, per series, the depth used, the FOV axial height, the chosen
    mm/pixel, and the resulting on-image measured depth (Rows * spacing) so a
    15 cm scan should now read ~15 cm.

It does NOT contact PACS and does NOT touch study flags. The DICOM-building and
calibration code is kept byte-for-byte in sync with pacs-transfer-server.py.

Usage:
  python3 recreate_dicoms.py                 # rebuild all studies under US_images/
  python3 recreate_dicoms.py --root US_images
  python3 recreate_dicoms.py --study US_images/unknown/<uuid>
  python3 recreate_dicoms.py --dry-run       # report only, write nothing
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


# --------------------------------------------------------------------------
# Calibration helpers — copied verbatim from pacs-transfer-server.py so this
# script produces identical pixel spacing. Keep in sync if the server changes.
# --------------------------------------------------------------------------
def calibrate_from_depth_ruler(gray, depth_mm):
    """Derive mm-per-pixel from the on-screen depth ruler in the left margin."""
    h, w = gray.shape
    band = gray[:, :max(8, int(0.15 * w))]
    bright_rows = np.where(band.max(axis=1) > 100)[0]
    if len(bright_rows) < 6:
        return None

    ticks = []  # (center_row, dash_length_px)
    start = prev = bright_rows[0]
    for r in list(bright_rows[1:]) + [None]:
        if r is None or r - prev > 3:
            seg = band[start:prev + 1]
            cols = np.where(seg.max(axis=0) > 100)[0]
            if len(cols) >= 2:
                ticks.append(((start + prev) / 2.0, cols[-1] - cols[0] + 1))
            if r is None:
                break
            start = r
        prev = r
    if len(ticks) < 3:
        return None

    centers = np.array([t[0] for t in ticks])
    lengths = np.array([t[1] for t in ticks])
    major = centers[lengths >= 0.75 * lengths.max()]
    if len(major) < 2:
        return None

    px_per_10mm = float(np.median(np.diff(np.sort(major))))
    if px_per_10mm <= 0:
        return None

    mm_per_px = 10.0 / px_per_10mm

    if depth_mm:
        expected_majors = depth_mm / 10.0 + 1
        if not (expected_majors - 2 <= len(major) <= expected_majors + 2):
            return None
    return mm_per_px


def fov_axial_height(gray, bg_thresh=6):
    """Axial (vertical) pixel extent of the imaged field of view."""
    if gray.ndim == 3:
        gray = gray.mean(axis=2).astype(np.uint8)
    mask = (gray > bg_thresh).astype(np.uint8)
    if not mask.any():
        return gray.shape[0]

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lbl == largest).astype(np.uint8)

    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0) & (mask == 0)
    mask[holes] = 1

    cx = w // 2
    rows = np.where(mask[:, max(0, cx - 2):cx + 3].any(axis=1))[0]
    if len(rows) == 0:
        rows = np.where(mask.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1)


# --------------------------------------------------------------------------
# Per-study DICOM rebuild — mirrors the DICOM-build half of send_to_pacs,
# minus the PACS C-STORE and flag writing.
# --------------------------------------------------------------------------
def rebuild_study(study_dir, dry_run=False):
    json_path = os.path.join(study_dir, 'study_info.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        study_info = json.load(f)

    # Collect raw frames grouped one series per raw leaf folder.
    series_images = {}
    for root, dirs, files in os.walk(study_dir):
        if os.path.basename(root) != 'raw':
            continue
        series_key = os.path.relpath(root, study_dir).replace(os.sep, '/')
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                series_images.setdefault(series_key, []).append(os.path.join(root, file))

    if not series_images:
        for root, dirs, files in os.walk(study_dir):
            if 'raw_' not in os.path.basename(root):
                continue
            series_name = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    series_images.setdefault(series_name, []).append(os.path.join(root, file))

    if not series_images:
        return None

    PatientName = study_info.get('patientname', 'Unknown')
    PatientID = study_info.get('patientid') or study_info.get('patientemail') or 'UNKNOWN'
    PatientBirthDate = study_info.get('patientdob', '')
    PatientSex = study_info.get('patientgender', 'O')

    now = datetime.datetime.now()
    default_date = now.strftime('%Y%m%d')
    default_time = now.strftime('%H%M%S')
    if 'processing_timestamp' in study_info:
        try:
            dt = datetime.datetime.fromisoformat(study_info['processing_timestamp'])
            default_date = dt.strftime('%Y%m%d')
            default_time = dt.strftime('%H%M%S')
        except Exception:
            pass

    def clean_dicom_date(d):
        if not d:
            return ''
        s = str(d).replace('-', '').replace('/', '').replace('.', '').strip()
        if not s.isdigit():
            return ''
        return s

    def parse_app_date(d):
        parts = str(d or '').split('_')
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            dd, mm, yyyy = parts
            if len(yyyy) == 4:
                return f"{yyyy}{int(mm):02d}{int(dd):02d}"
        return ''

    def parse_app_time(t):
        parts = str(t or '').split('_')
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            hh, mm, ss = parts
            return f"{int(hh):02d}{int(mm):02d}{int(ss):02d}"
        return ''

    study_date = clean_dicom_date(study_info.get('studydate', '')) \
        or parse_app_date(study_info.get('createddate', '')) \
        or default_date
    study_time = str(study_info.get('studytime', '')).replace(':', '').split('.')[0] \
        or parse_app_time(study_info.get('createdtime', '')) \
        or default_time
    patient_dob = clean_dicom_date(PatientBirthDate)
    sex_map = {'male': 'M', 'female': 'F', 'other': 'O', 'm': 'M', 'f': 'F', 'o': 'O'}
    patient_sex = sex_map.get(str(PatientSex).lower(), 'O')

    study_uuid = study_info.get('studyid') or study_info.get('studyuuid')
    study_instance_uid = study_info.get('studyinstanceuid') \
        or (generate_uid(entropy_srcs=[str(study_uuid)]) if study_uuid else generate_uid())

    organs = study_info.get('organs') or []
    if not organs and study_info.get('organ'):
        organs = [study_info['organ']]
    fallback_study_description = ', '.join(str(o).title() for o in organs)[:64]

    depth_mm = study_info.get('scandepthmm')
    if depth_mm is None:
        depth_alt = study_info.get('scandepthcm', study_info.get('depth'))
        depth_mm = float(depth_alt) * 10.0 if depth_alt is not None else 150.0
    depth_mm = float(depth_mm)

    if fallback_study_description:
        study_description = fallback_study_description
    else:
        seen = []
        for sk in series_images:
            organ0 = sk.split('/')[0]
            if organ0 and organ0 not in seen:
                seen.append(organ0)
        study_description = ', '.join(o.title() for o in seen)[:64] or 'Ultrasound'

    written = 0
    reports = []
    for series_name, img_paths in series_images.items():
        series_instance_uid = generate_uid(entropy_srcs=[str(study_uuid), series_name])
        series_path_parts = series_name.split('/')
        if len(series_path_parts) >= 2:
            organ_raw, orientation_raw = series_path_parts[0], series_path_parts[1]
            side = ''
            organ_name = organ_raw
            for s in ('left', 'right'):
                if organ_raw.startswith(s):
                    side = s
                    organ_name = organ_raw[len(s):]
                    break
            series_description = ' '.join(
                w.title() for w in (side, orientation_raw, organ_name) if w)
        else:
            series_description = series_name.replace('raw_', '').replace('_', ' ').replace('/', ' ').title()

        img_paths = natsorted(img_paths)

        # Median FOV spacing across the whole sweep.
        series_fov_spacing = None
        spacings = []
        fov_heights = []
        for p in img_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is not None:
                fh = fov_axial_height(g)
                if fh > 0:
                    spacings.append(depth_mm / fh)
                    fov_heights.append(fh)
        if spacings:
            series_fov_spacing = float(np.median(spacings))

        ruler_hits = 0
        rows_example = None
        for instance_number, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(img_path)
            if img is None:
                continue

            ds = Dataset()
            ds.PatientName = str(PatientName)
            ds.PatientID = str(PatientID)
            ds.PatientBirthDate = patient_dob
            ds.PatientSex = patient_sex
            ds.StudyDate = study_date
            ds.StudyTime = study_time
            ds.SeriesDate = study_date
            ds.SeriesTime = study_time
            ds.ContentDate = study_date
            ds.ContentTime = study_time
            ds.AccessionNumber = study_info.get('accessionnumber', '')
            ds.ReferringPhysicianName = study_info.get('referringphysicianname', '')

            ds.StudyInstanceUID = study_instance_uid
            ds.StudyDescription = study_description
            ds.SeriesInstanceUID = series_instance_uid
            ds.SeriesDescription = series_description
            ds.SeriesNumber = list(series_images.keys()).index(series_name) + 1
            ds.InstanceNumber = instance_number
            ds.SOPInstanceUID = generate_uid(
                entropy_srcs=[str(study_uuid), series_name, os.path.basename(img_path)])
            ds.Modality = 'US'

            image_2d = np.array(img)
            if len(image_2d.shape) == 3:
                image_2d = np.mean(image_2d, axis=2).astype(float)
            if image_2d.max() == 0:
                image_2d = image_2d + 1.0
            image_2d = ((image_2d / image_2d.max()) * 255).astype(np.uint8)

            pixel_spacing_mm = calibrate_from_depth_ruler(image_2d, depth_mm)
            if pixel_spacing_mm is None:
                pixel_spacing_mm = (series_fov_spacing if series_fov_spacing
                                    else depth_mm / fov_axial_height(image_2d))
            else:
                ruler_hits += 1

            if rows_example is None:
                rows_example = (image_2d.shape[0], pixel_spacing_mm)

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelRepresentation = 0
            ds.HighBit = 7
            ds.BitsStored = 8
            ds.BitsAllocated = 8
            ds.WindowCenter = 128
            ds.WindowWidth = 256
            ds.Columns = image_2d.shape[1]
            ds.Rows = image_2d.shape[0]
            ds.PixelData = image_2d.tobytes()
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.1'

            ds.PixelSpacing = [pixel_spacing_mm, pixel_spacing_mm]
            ds.SequenceOfUltrasoundRegions = [Dataset()]
            ds.SequenceOfUltrasoundRegions[0].RegionSpatialFormat = 1
            ds.SequenceOfUltrasoundRegions[0].RegionDataType = 1
            ds.SequenceOfUltrasoundRegions[0].RegionFlags = 0
            ds.SequenceOfUltrasoundRegions[0].PhysicalUnitsXDirection = 3
            ds.SequenceOfUltrasoundRegions[0].PhysicalUnitsYDirection = 3
            ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX = pixel_spacing_mm
            ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY = pixel_spacing_mm

            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1'
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta = file_meta

            if not dry_run:
                img_path_p = Path(img_path)
                dicom_dir = img_path_p.parent / (img_path_p.parent.name + '_dicom')
                dicom_dir.mkdir(parents=True, exist_ok=True)
                dcm_file_path = dicom_dir / (img_path_p.stem + '.dcm')
                ds.save_as(str(dcm_file_path), enforce_file_format=True)
                print(f"   Wrote {dcm_file_path}")
            written += 1

        med_fov = int(np.median(fov_heights)) if fov_heights else None
        rows = rows_example[0] if rows_example else None
        spacing = series_fov_spacing
        measured_depth_cm = (rows * spacing / 10.0) if (rows and spacing) else None
        reports.append({
            'series': series_description,
            'frames': len(img_paths),
            'depth_mm': depth_mm,
            'median_fov_px': med_fov,
            'rows': rows,
            'spacing_mm_px': spacing,
            'ruler_frames': ruler_hits,
            'measured_depth_cm': measured_depth_cm,
        })

    return {'study_dir': study_dir, 'written': written, 'series': reports}


def main():
    ap = argparse.ArgumentParser(description="Recreate DICOMs with the new depth calibration.")
    ap.add_argument('--root', default='US_images', help='Root of patient/study tree (default: US_images)')
    ap.add_argument('--study', help='Rebuild a single study dir instead of the whole root')
    ap.add_argument('--dry-run', action='store_true', help='Report only; write no .dcm files')
    args = ap.parse_args()

    if args.study:
        study_dirs = [args.study]
    else:
        study_dirs = []
        for patient in sorted(os.listdir(args.root)):
            patient_dir = os.path.join(args.root, patient)
            if not os.path.isdir(patient_dir):
                continue
            for study in sorted(os.listdir(patient_dir)):
                sd = os.path.join(patient_dir, study)
                if os.path.isdir(sd) and os.path.exists(os.path.join(sd, 'study_info.json')):
                    study_dirs.append(sd)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Rebuilding {len(study_dirs)} study(ies)\n")

    total_dcm = 0
    short = []  # series that still measure noticeably off
    for sd in study_dirs:
        res = rebuild_study(sd, dry_run=args.dry_run)
        if not res:
            print(f"·  {sd}: no raw frames / no study_info — skipped")
            continue
        total_dcm += res['written']
        print(f"■  {os.path.relpath(sd, args.root)}  ({res['written']} dcm)")
        for r in res['series']:
            md = r['measured_depth_cm']
            tag = ''
            if md is not None:
                err = md - r['depth_mm'] / 10.0
                tag = f"  measured≈{md:5.1f}cm (Δ{err:+.1f})"
                if abs(err) > 0.5:
                    short.append((os.path.relpath(sd, args.root), r['series'], md, r['depth_mm'] / 10.0))
            ruler = f" ruler={r['ruler_frames']}" if r['ruler_frames'] else ""
            print(f"     {r['series']:<28} n={r['frames']:<4} "
                  f"FOVpx={str(r['median_fov_px']):>4} rows={str(r['rows']):>4} "
                  f"spacing={r['spacing_mm_px']:.4f}mm{ruler}{tag}")

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {total_dcm} DICOM file(s) "
          f"across {len(study_dirs)} study(ies).")
    if short:
        print(f"\n⚠ {len(short)} series measure >0.5 cm from the expected depth "
              f"(expected = scan depth; default 15 cm when no depth in metadata):")
        for sd, series, md, exp in short:
            print(f"   {sd}  {series}: {md:.1f} cm vs {exp:.1f} cm")


if __name__ == '__main__':
    main()
