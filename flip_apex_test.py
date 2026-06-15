#!/usr/bin/env python3
"""Flip existing reconstructed .nii.gz volumes so the probe-apex axis is on top.

Mirrors what pos_cupy_finalv2.orient_apex_on_top() now does at generation time:
a single flip along the Z (superior-inferior) axis. The fan geometry places the
apex at the LOW end of Z (bottom); flipping Z moves it to the top.

Reads only numpy + gzip (no nibabel), so it runs on the host. The NIfTI header is
preserved byte-for-byte (affine stays identity) — only the voxel data is flipped.

Usage:
    python3 flip_apex_test.py                 # outputs/ -> outputs_oriented/
    python3 flip_apex_test.py --inplace       # overwrite in place (needs write perms)
    python3 flip_apex_test.py --src DIR --dst DIR
"""
import argparse
import glob
import gzip
import os
import struct
import sys

import numpy as np

# NIfTI datatype code -> numpy dtype
DTYPE_MAP = {
    2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
    64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32,
}


def load_nifti_gz(path):
    """Return (header_bytes, voxel_ndarray) from a gzipped NIfTI-1 file."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    dims = struct.unpack_from("<8h", raw, 40)
    nx, ny, nz = dims[1], dims[2], dims[3]
    datatype = struct.unpack_from("<h", raw, 70)[0]
    vox_offset = int(struct.unpack_from("<f", raw, 108)[0])
    if datatype not in DTYPE_MAP:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}")
    dtype = DTYPE_MAP[datatype]
    header = raw[:vox_offset]
    count = nx * ny * nz
    data = np.frombuffer(
        raw, dtype=dtype, offset=vox_offset, count=count
    ).reshape((nx, ny, nz), order="F")
    return header, data


def apex_side(data):
    """Return 'bottom' or 'top' for where the narrow fan apex sits on the Z axis.

    The fan's cross-section grows from the apex (narrow) to the base (wide), so
    populated-voxel count rises from apex to base. Compare the two halves of Z:
    the half with FEWER populated voxels is the apex end.
    """
    c = (data > 0).sum(axis=(0, 1))
    half = len(c) // 2
    low, high = int(c[:half].sum()), int(c[half:].sum())
    return "bottom" if low < high else "top"


def write_nifti_gz(path, header, data):
    with gzip.open(path, "wb") as fh:
        fh.write(header)
        fh.write(np.ascontiguousarray(data).tobytes(order="F"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs", help="source directory")
    ap.add_argument("--dst", default="outputs_oriented", help="destination directory")
    ap.add_argument("--inplace", action="store_true", help="overwrite files in place")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*riv3.nii.gz")))
    if not files:
        print(f"No *riv3.nii.gz files found in {args.src}/")
        return 1

    if not args.inplace:
        os.makedirs(args.dst, exist_ok=True)

    print(f"Found {len(files)} volume(s)\n")
    for f in files:
        name = os.path.basename(f)
        try:
            header, data = load_nifti_gz(f)
            before = apex_side(data)
            out = f if args.inplace else os.path.join(args.dst, name)
            print(f"  {name}")
            if before == "bottom":
                flipped = data[:, :, ::-1]
                write_nifti_gz(out, header, flipped)
                print(f"      apex was at BOTTOM -> flipped to top   wrote {out}")
            else:
                # Already apex-up. Copy through unchanged (or leave in place).
                if not args.inplace:
                    write_nifti_gz(out, header, data)
                print(f"      apex already on TOP -> left as-is")
        except Exception as e:
            print(f"      SKIPPED: {e}")

    dest = args.src if args.inplace else args.dst
    print(f"\nDone. Open the files in {dest}/ — the apex should now be on top.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
