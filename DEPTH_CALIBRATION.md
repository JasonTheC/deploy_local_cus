# Depth Calibration — New Method

## Goal
Set the DICOM pixel spacing so that an on-image measurement of the scan equals
the true scan depth (e.g. a 15 cm scan measures 15 cm). Must be **agnostic** to:
- vendor (Clarius, 3-in-1, Healson)
- geometry (curvilinear/sector fan **and** linear)

## Inputs
- **Depth value** comes from the app in the upload JSON. Keys are lowercased by
  `sanitise_json`, so read `scandepthmm` first, else `scandepthcm` (×10), else
  fall back to a default. This is the true depth in mm (the app reads it live
  from the probe SDK).
- **The JPEG frames** for each sweep.

## Core idea
`pixel_spacing_mm = depth_mm / live_pixel_height`, applied as square pixels to
both `PixelSpacing` and `SequenceOfUltrasoundRegions[0].PhysicalDeltaX/Y`.

`live_pixel_height` is the **axial (vertical) pixel extent of the field of view
(FOV)** — the lit imaged region, NOT the full image buffer. The depth must map
onto the live pixels only, not the black padding around them.

## How to find the FOV vertical extent (the actual fix)
The current code uses `thr = image_2d.min() + 1` to find live rows. This is the
bug: on an 8-bit image where black ≈ 0, JPEG/compression noise in the black
border clears `min+1`, so the detected region balloons to ~the full image
height. Depth then gets stretched across the black padding, so the live sector
measures short and measuring through the black top gives the full depth.

Replace it with a robust FOV detector:

1. **Threshold with a real floor**, not `min+1`. Use a small intensity floor
   (e.g. `> ~6` on 0–255) so JPEG ringing in the black border is ignored while
   dim deep tissue is kept.
2. **Keep the largest connected component** of the mask. This drops burned-in
   text, the grayscale/depth bar, and stray noise specks that would otherwise
   widen the region.
3. **Fill enclosed holes** in that component. Anechoic structures inside the
   tissue (bladder, cysts, vessels) are black and would otherwise carve a false
   top/bottom end into the FOV. Filling holes removes that risk.
   (Pad the mask by 1 px before the flood-fill so a FOV that touches the frame
   edge still has an outside background to flood from.)
4. **Measure the extent down the CENTRE columns**, not corner-to-corner.
   - For a curvilinear fan, the lateral shoulders of the fan sit above the
     central apex, so a full bounding-box height reads taller than the true
     central scan line — that makes the centred measurement read short
     (this is exactly what produced 13.9 cm for a 15 cm scan).
   - For a linear probe, centre extent == box height, so this stays correct.
   - Take the run from the first to the last lit row within a few central
     columns; that span is `live_pixel_height`.

## Robustness against anechoic "false ends" across a sweep
Compute the spacing per frame, then use the **median over the whole sweep** as
the series spacing. If one frame has an anechoic structure touching the FOV edge
(shortening its detected height), it becomes an outlier and the median rejects
it. Over a full sweep, speckle lights up essentially the entire real FOV, so the
median lands on the true scale.

## Pseudocode
```python
def fov_axial_height(gray, bg_thresh=6):
    # gray: 2D uint8 (0-255)
    mask = gray > bg_thresh
    if not mask.any():
        return gray.shape[0]
    # 1) largest connected component (drops text / grayscale bar / specks)
    mask = largest_connected_component(mask)
    # 2) fill enclosed holes (anechoic bladder/cyst/vessel won't make false ends)
    mask = fill_holes(mask)            # pad by 1px, flood-fill background, invert
    # 3) axial extent down the central columns (agnostic to fan vs linear)
    cx = mask.shape[1] // 2
    rows = where(mask[:, cx-2 : cx+3].any(axis=1))
    if rows is empty:
        rows = where(mask.any(axis=1))
    return rows[-1] - rows[0] + 1

# per series:
spacings = [depth_mm / fov_axial_height(load_gray(p)) for p in frame_paths]
series_spacing_mm = median(spacings)
# then for every frame in the series:
#   PixelSpacing = [series_spacing_mm, series_spacing_mm]
#   PhysicalDeltaX = PhysicalDeltaY = series_spacing_mm
```

`largest_connected_component` / `fill_holes` are available via cv2
(`connectedComponentsWithStats`, `floodFill`) or scipy.ndimage
(`label`, `binary_fill_holes`). cv2 is already a dependency.

## Where this lives
In `pacs-transfer-server.py`, in the per-frame DICOM build inside `send_to_pacs`
(the block that currently computes `pixel_spacing_mm` via the
`thr = image_2d.min() + 1` fallback). The existing `calibrate_from_depth_ruler`
ruler path can be kept as an optional high-precision override when a printed
depth ruler is detected; this FOV method is the agnostic default/fallback that
replaces the broken `min+1` detection.
