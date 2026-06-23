#!/usr/bin/env python3
"""
PACS Transfer Server
Handles receiving ultrasound images from Android app, processes them,
creates 3D volumes, and sends to PACS server.
"""

import numpy as np
import os
import cv2
import time
import asyncio
import datetime
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom import dcmread
from pynetdicom import AE
import io
import json
import sqlite3
import hashlib
from pathlib import Path
from natsort import natsorted
import nibabel as nib
import torch
import torch.nn.functional as F
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import unpad
import websockets

# Import the CuPy-accelerated 3D reconstruction module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guidance'))

from pos_cupy_finalv2 import process_study as process_study_cupy
CUPY_AVAILABLE = True
print("✓ CuPy-accelerated 3D reconstruction available")

# Studies currently queued or being processed (avoid duplicate enqueues)
STUDIES_IN_FLIGHT = set()


# Server configuration
HOST = "0.0.0.0"
PORT = int(os.environ.get('PACS_TCP_PORT', '8890'))
WEBSOCKET_PORT = int(os.environ.get('PACS_WS_PORT', '7556'))

# Generate RSA key pair for encryption
key = RSA.generate(2048)
public_key = key.publickey()
private_key = key

# Image processing configuration
RESIZE_SIZE = 256
cwd = os.getcwd()
idir = f"{cwd}/US_images"
os.makedirs(idir, exist_ok=True)

# Queue / worker configuration to avoid GPU saturation
# Number of concurrent GPU/3D jobs (set by environment or default to 1)
MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', '1'))
# Maximum queued studies waiting to run
MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', '16'))
# Async queue for study processing
STUDY_QUEUE = None
# Keep worker task references for shutdown
WORKER_TASKS = []

# Color mappings for organs
colour_dict = {
    0: (0, 0, 0),
    1: (255, 0, 0),      # Prostate - Red
    2: (0, 253, 0),      # Bladder - Green
    3: (0, 0, 250),      # Kidney - Blue
}

# Database setup
def init_database():
    """Initialize SQLite database for storing study information."""
    con = sqlite3.connect("CUS.db")
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pt_data TEXT NOT NULL,
            studies TEXT
        )
    """)
    con.commit()
    con.close()

init_database()


def write_to_db(sql_command):
    """Execute SQL command on the database."""
    con = sqlite3.connect("CUS.db")
    cur = con.cursor()
    cur.execute(sql_command)
    con.commit()
    con.close()


def sanitize_dict(d):
    """Sanitize dictionary keys and values."""
    new_dict = {}
    for k, v in d.items():
        new_k = k.lower() if isinstance(k, str) else k
        if isinstance(v, dict):
            new_v = sanitize_dict(v)
        elif isinstance(v, str):
            new_v = v.lower()
            if new_v == "":
                new_v = "unknown"
        else:
            new_v = v
        new_dict[new_k] = new_v
    return new_dict


def sanitise_json(data):
    """Sanitize JSON data."""
    if isinstance(data, dict):
        return sanitize_dict(data)
    return data


def getConfigFromBytes(imagebytes):
    """Extract configuration data from byte stream."""
    config_start = imagebytes.getvalue().find(b'STARTOFCONFIG')
    config_end = imagebytes.getvalue().find(b'ENDOFCONFIG')
    
    if config_start == -1 or config_end == -1:
        return None
    
    config_bytes = imagebytes.getvalue()[config_start + len(b'STARTOFCONFIG'):config_end]
    
    try:
        # Try to decrypt if encrypted
        cipher_rsa = PKCS1_OAEP.new(private_key)
        
        # Extract AES key length (first 4 bytes)
        aes_key_len = int.from_bytes(config_bytes[:4], byteorder='big')
        
        # Extract encrypted AES key
        encrypted_aes_key = config_bytes[4:4 + aes_key_len]
        aes_key = cipher_rsa.decrypt(encrypted_aes_key)
        
        # Extract nonce
        nonce = config_bytes[4 + aes_key_len:4 + aes_key_len + 16]
        
        # Extract encrypted config
        encrypted_config = config_bytes[4 + aes_key_len + 16:]
        
        # Decrypt config
        cipher_aes = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        decrypted_config = cipher_aes.decrypt(encrypted_config)
        
        # Remove padding
        decrypted_config = unpad(decrypted_config, AES.block_size)
        
        config_str = decrypted_config.decode('utf-8')
        r_d = json.loads(config_str)
        return r_d
    except Exception as e:
        print(f"Error decrypting config: {e}")
        # Try plain text
        try:
            config_str = config_bytes.decode('utf-8')
            r_d = json.loads(config_str)
            return r_d
        except Exception as e2:
            print(f"Error parsing plain config: {e2}")
            return None


def zero_trim_ndarray(ndarray):
    """Trim zero-only borders from a 3D numpy array (copied/adapted from pos_mobilenet)."""
    try:
        def get_bounds(a):
            ux = bx = uy = by = uz = bz = 0
            if not a.any():
                return (0, a.shape[0]-1, 0, a.shape[1]-1, 0, a.shape[2]-1)
            for x in range(a.shape[0]):
                if a[x,:,:].any():
                    ux = x
                    break
            for x in range(a.shape[0]-1,-1,-1):
                if a[x,:,:].any():
                    bx = x
                    break
            for x in range(a.shape[1]):
                if a[:,x,:].any():
                    uy = x
                    break
            for x in range(a.shape[1]-1,-1,-1):
                if a[:,x,:].any():
                    by = x
                    break
            for x in range(a.shape[2]):
                if a[:,:,x].any():
                    uz = x
                    break
            for x in range(a.shape[2]-1,-1,-1):
                if a[:,:,x].any():
                    bz = x
                    break
            return (ux,bx,uy,by,uz,bz)

        b = get_bounds(ndarray)
        ndarray = ndarray[b[0]:b[1]+1, b[2]:b[3]+1, b[4]:b[5]+1]
        return ndarray
    except Exception as e:
        print(f"zero_trim_ndarray failed: {e}")
        return ndarray


def place_slice_in_volume(points, image_array, theta_y, center_x, center_y, z_position, volume_shape):
    """Places a 2D image slice into a 3D volume with rotation around the Y-axis.

    Adapted from guidance/pos_mobilenet.place_slice_in_volume for use in the server.
    """
    height, width = image_array.shape
    theta_rad = np.radians(theta_y)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    y_coords, x_coords = np.where(image_array != 0)
    if len(y_coords) == 0:
        return

    orig_x = x_coords - width // 2
    orig_y = y_coords - height // 2

    rot_y = orig_y
    rot_x = (orig_x * cos_theta).astype(int)

    vol_x = (rot_x + center_x).astype(int)
    vol_y = (rot_y + center_y).astype(int)
    vol_z = np.full_like(vol_x, int(z_position))

    in_bounds = (0 <= vol_x) & (vol_x < volume_shape[0]) & \
                (0 <= vol_y) & (vol_y < volume_shape[1]) & \
                (0 <= vol_z) & (vol_z < volume_shape[2])

    vol_x = vol_x[in_bounds]
    vol_y = vol_y[in_bounds]
    vol_z = vol_z[in_bounds]
    pixel_values = image_array[y_coords[in_bounds], x_coords[in_bounds]]

    points[vol_x, vol_y, vol_z] = np.maximum(points[vol_x, vol_y, vol_z], pixel_values)




def get_pacs_nodes(study_info=None):
    """Return the list of PACS destinations to replicate every study to.

    Each node is a dict: {name, ip, port, ae_title}.

    Configuration (all via environment variables, easy to extend):
      Primary node:   PACS_HOST  / PACS_PORT  / PACS_AET
      Fallback nodes:  PACS2_HOST / PACS2_PORT / PACS2_AET
                       PACS3_HOST / PACS3_PORT / PACS3_AET   ... up to PACS9_*

    A node is only added if its *_HOST is set, so adding a second or third
    site is just three extra lines in docker-compose.yml. Ports/AETs default
    to the standard DICOM 104 / "ORTHANC" if omitted.

    Per-study overrides (optional, from the app's metadata):
      study_info['pacs_nodes']  -> explicit list, replaces env config
      study_info['pacs']        -> single dict, merged into the primary node
    """
    def _node(name, ip, port, ae_title):
        return {
            'name': name,
            'ip': ip,
            'port': int(port),
            'ae_title': ae_title,
        }

    # Primary (backward compatible with the previous single-node setup)
    primary = _node(
        os.environ.get('PACS_AET', 'ORTHANC'),
        os.environ.get('PACS_HOST', '127.0.0.1'),
        os.environ.get('PACS_PORT', '4242'),
        os.environ.get('PACS_AET', 'ORTHANC'),
    )
    if study_info and isinstance(study_info.get('pacs'), dict):
        ov = study_info['pacs']
        primary['ip'] = ov.get('ip', primary['ip'])
        primary['port'] = int(ov.get('port', primary['port']))
        primary['ae_title'] = ov.get('ae_title', primary['ae_title'])
        primary['name'] = ov.get('ae_title', primary['name'])

    nodes = [primary]

    # Numbered fallback nodes: PACS2_*, PACS3_*, ... PACS9_*
    for i in range(2, 10):
        host = os.environ.get(f'PACS{i}_HOST')
        if not host:
            continue
        nodes.append(_node(
            os.environ.get(f'PACS{i}_AET', f'PACS{i}'),
            host,
            os.environ.get(f'PACS{i}_PORT', '104'),
            os.environ.get(f'PACS{i}_AET', 'ORTHANC'),
        ))

    # Explicit per-study list overrides the env-based config entirely
    if study_info and isinstance(study_info.get('pacs_nodes'), list):
        nodes = []
        for n in study_info['pacs_nodes']:
            if not n.get('ip'):
                continue
            nodes.append(_node(
                n.get('ae_title', n['ip']),
                n['ip'],
                n.get('port', '104'),
                n.get('ae_title', 'ORTHANC'),
            ))

    return nodes


def store_to_node(node, generated_dcms):
    """C-STORE every generated DICOM to a single PACS node.

    Returns True only if the association was established AND every instance
    was accepted; any connection failure or rejected instance returns False
    so the caller can withhold the pacs_sent.flag and retry later.
    """
    label = f"{node['name']} ({node['ip']}:{node['port']}, AET {node['ae_title']})"
    print(f"→ Sending {len(generated_dcms)} DICOM(s) to PACS {label}")

    ae = AE(ae_title='PYNETDICOM')
    ae.add_requested_context('1.2.840.10008.5.1.4.1.1.6.1')
    try:
        assoc = ae.associate(node['ip'], node['port'], ae_title=node['ae_title'])
    except Exception as e:
        print(f"✗ Could not connect to {label}: {e}")
        return False

    if not assoc.is_established:
        print(f"✗ Association rejected/aborted/never connected for {label}")
        return False

    # 0x0000 = success; 0xB000/B006/B007 = stored with warnings (coercion etc.)
    OK_STATUSES = (0x0000, 0xB000, 0xB006, 0xB007)
    sent_ok = 0
    failures = []
    try:
        for dcm_file in generated_dcms:
            ds_dataset = dcmread(dcm_file)
            status = assoc.send_c_store(ds_dataset)
            code = getattr(status, 'Status', None) if status else None
            if code in OK_STATUSES:
                sent_ok += 1
            else:
                failures.append((dcm_file, code))
                print(f"✗ C-STORE FAILED for {dcm_file} → {node['name']} - status: "
                      f"{f'0x{code:04X}' if code is not None else 'no response'}")
    finally:
        assoc.release()

    print(f"   {node['name']}: {sent_ok}/{len(generated_dcms)} stored OK")
    return not failures


def send_to_pacs(study_dir, study_info):
    """
    Generate DICOMs for a study and replicate them to every configured PACS
    node (see get_pacs_nodes). The study is only marked sent (pacs_sent.flag)
    once *all* nodes have stored every instance, so a fallback node that was
    offline gets retried by the janitor on the next pass.

    Args:
        study_dir: Path to study directory containing images and metadata
    """
    print(f"Sending study to PACS: {study_dir}")


    print(f"Study info: {json.dumps(study_info, indent=2)}")

    # Collect image files grouped into one series per raw/processed leaf folder.
    # Layouts handled:
    #   {study_dir}/{organ}/{orientation}/raw|processed/*.jpg  (current app)
    #   {study_dir}/{orientation}/raw|processed/*.jpg          (older app)
    # The series key is the folder path relative to the study dir, e.g.
    # "rightkidney/transverse/raw", so right and left kidney never merge.
    series_images = {}  # Key: series name, Value: list of image paths

    for root, dirs, files in os.walk(study_dir):
        if os.path.basename(root) not in ('raw', 'processed'):
            continue
        series_key = os.path.relpath(root, study_dir).replace(os.sep, '/')
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                series_images.setdefault(series_key, []).append(os.path.join(root, file))

    # Fallback: legacy raw_* layout
    if not series_images:
        for root, dirs, files in os.walk(study_dir):
            if 'raw_' not in os.path.basename(root):
                continue
            series_name = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    series_images.setdefault(series_name, []).append(os.path.join(root, file))

    if not series_images:
        print("No images found to convert/send")
        return False

    # Get top-level patient/study info fields with fallbacks
    PatientName = study_info.get('patientname', 'Unknown')
    PatientID = study_info.get('patientid') or study_info.get('patientemail') or 'UNKNOWN'
    PatientBirthDate = study_info.get('patientdob', '')
    PatientSex = study_info.get('patientgender', 'O')

    # Format date/time
    now = datetime.datetime.now()
    default_date = now.strftime('%Y%m%d')
    default_time = now.strftime('%H%M%S')
    
    # Try to get timestamp from study info
    if 'processing_timestamp' in study_info:
        try:
            dt = datetime.datetime.fromisoformat(study_info['processing_timestamp'])
            default_date = dt.strftime('%Y%m%d')
            default_time = dt.strftime('%H%M%S')
        except:
            pass

    def clean_dicom_date(d):
        if not d: return ''
        s = str(d).replace('-', '').replace('/', '').replace('.', '').strip()
        # Basic validation: must be digits. 
        # DICOM DA is 8 bytes YYYYMMDD. 
        if not s.isdigit(): return ''
        return s

    def parse_app_date(d):
        """Parse app-format createdDate like '02_06_2026' (DD_MM_YYYY) -> 'YYYYMMDD'."""
        parts = str(d or '').split('_')
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            dd, mm, yyyy = parts
            if len(yyyy) == 4:
                return f"{yyyy}{int(mm):02d}{int(dd):02d}"
        return ''

    def parse_app_time(t):
        """Parse app-format createdTime like '10_33_44' (HH_MM_SS) -> 'HHMMSS'."""
        parts = str(t or '').split('_')
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            hh, mm, ss = parts
            return f"{int(hh):02d}{int(mm):02d}{int(ss):02d}"
        return ''

    # Allow explicit overrides, then fall back to the app's createdDate/createdTime
    study_date = clean_dicom_date(study_info.get('studydate', '')) \
        or parse_app_date(study_info.get('createddate', '')) \
        or default_date

    study_time = str(study_info.get('studytime', '')).replace(':', '').split('.')[0] \
        or parse_app_time(study_info.get('createdtime', '')) \
        or default_time
    
    patient_dob = clean_dicom_date(PatientBirthDate)
    
    # Map sex to DICOM standard
    sex_map = {'male': 'M', 'female': 'F', 'other': 'O', 'm': 'M', 'f': 'F', 'o': 'O'}
    patient_sex = sex_map.get(str(PatientSex).lower(), 'O')

    # Use a single, deterministic StudyInstanceUID per study so re-sends of the
    # same study merge in PACS instead of creating duplicates
    study_uuid = study_info.get('studyid') or study_info.get('studyuuid')
    study_instance_uid = study_info.get('studyinstanceuid') \
        or (generate_uid(entropy_srcs=[str(study_uuid)]) if study_uuid else generate_uid())

    # Fallback study description (used only when a series' own organ can't be
    # determined from its folder path, e.g. legacy raw_* layout)
    organs = study_info.get('organs') or []
    if not organs and study_info.get('organ'):
        organs = [study_info['organ']]
    fallback_study_description = ', '.join(str(o).title() for o in organs)[:64]

    # Get depth from study info (in cm)
    depth_cm = study_info.get('depth', 15)

    # StudyDescription is a single, study-level value shared by every series,
    # so it must describe the whole exam — not one sweep. Use the organs list
    # (falling back to the organs seen across the series folders).
    if fallback_study_description:
        study_description = fallback_study_description
    else:
        seen = []
        for sk in series_images:
            organ0 = sk.split('/')[0]
            if organ0 and organ0 not in seen:
                seen.append(organ0)
        study_description = ', '.join(o.title() for o in seen)[:64] or 'Ultrasound'

    # Process each series separately
    generated_dcms = []
    for series_name, img_paths in series_images.items():
        # Deterministic SeriesInstanceUID per (study, folder) so re-sending the
        # same study merges in PACS instead of duplicating every series
        series_instance_uid = generate_uid(entropy_srcs=[str(study_uuid), series_name])
        # Current layout is "{organ}/{orientation}/{type}" — build a
        # "<Side> <Orientation> <Organ>" label from this series' own folder
        # (e.g. "Right Transverse Kidney"). Side/axis/organ vary *per series*,
        # so this belongs in SeriesDescription — putting it in StudyDescription
        # (a single study-level value) collapsed every series to one label,
        # e.g. all showing "Left Sagital Kidney".
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

        print(f"Processing series: {series_description} ({len(img_paths)} images)")
        
        # Sort images naturally (by timestamp in filename)
        img_paths = natsorted(img_paths)
        
        for instance_number, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            # Build DICOM dataset
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
            ds.SeriesInstanceUID = series_instance_uid  # Same for all images in this series
            ds.SeriesDescription = series_description
            ds.SeriesNumber = list(series_images.keys()).index(series_name) + 1
            ds.InstanceNumber = instance_number
            # Deterministic per image file: a re-send produces the same SOP
            # Instance UID, which PACS treats as the same object (idempotent)
            ds.SOPInstanceUID = generate_uid(
                entropy_srcs=[str(study_uuid), series_name, os.path.basename(img_path)])
            ds.Modality = 'US'

            # Convert to 8-bit grayscale: the US Image Storage SOP class
            # requires 8-bit pixels — 16-bit stores fine in Orthanc but web
            # viewers (Stone/OHIF) render it black
            image_2d = np.array(img)
            if len(image_2d.shape) == 3:
                image_2d = np.mean(image_2d, axis=2).astype(float)
            # Guard against zero-division
            if image_2d.max() == 0:
                image_2d = image_2d + 1.0
            image_2d = ((image_2d / image_2d.max()) * 255).astype(np.uint8)

            # Calculate live pixel region (excluding black borders)
            non_zero_rows = np.any(image_2d > image_2d.min() + 1, axis=1)
            row_indices = np.where(non_zero_rows)[0]
            if len(row_indices) > 0:
                live_pixel_height = row_indices[-1] - row_indices[0] + 1
            else:
                live_pixel_height = image_2d.shape[0]
            # Calculate pixel spacing: depth_cm / live_pixel_height (convert cm to mm)
            pixel_spacing_mm = (depth_cm * 10.0) / live_pixel_height
            
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
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.1'  # Ultrasound
            
            # Add depth and pixel spacing metadata
            ds.PixelSpacing = [pixel_spacing_mm, pixel_spacing_mm]
            ds.SequenceOfUltrasoundRegions = [Dataset()]
            ds.SequenceOfUltrasoundRegions[0].RegionSpatialFormat = 1
            ds.SequenceOfUltrasoundRegions[0].RegionDataType = 1
            ds.SequenceOfUltrasoundRegions[0].RegionFlags = 0
            ds.SequenceOfUltrasoundRegions[0].PhysicalUnitsXDirection = 3
            ds.SequenceOfUltrasoundRegions[0].PhysicalUnitsYDirection = 3
            ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX = pixel_spacing_mm
            ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY = pixel_spacing_mm

            # File meta
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1'
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta = file_meta

            # Save alongside the original image in a *_dicom folder
            img_path_p = Path(img_path)
            dicom_dir = img_path_p.parent / (img_path_p.parent.name + '_dicom')
            dicom_dir.mkdir(parents=True, exist_ok=True)
            dcm_file_path = dicom_dir / (img_path_p.stem + '.dcm')
            ds.save_as(str(dcm_file_path), enforce_file_format=True)
            generated_dcms.append(str(dcm_file_path))

    if not generated_dcms:
        print("No DICOM files generated — nothing to send")
        return False

    # Replicate the same DICOM set to every configured PACS node. Stores are
    # idempotent (deterministic SOP Instance UIDs), so re-sending to a node
    # that already has the study just merges — which is what lets the janitor
    # safely retry a node that was offline on an earlier pass.
    nodes = get_pacs_nodes(study_info)
    print(f"Replicating study to {len(nodes)} PACS node(s): "
          f"{', '.join(n['name'] for n in nodes)}")

    all_nodes_ok = True
    for node in nodes:
        if not store_to_node(node, generated_dcms):
            all_nodes_ok = False

    # Only mark the study done once EVERY node confirmed storage. A partial
    # success leaves the flag unwritten, so the janitor re-enqueues the study
    # later and the missing node(s) get another attempt.
    if not all_nodes_ok:
        print("✗ One or more PACS nodes failed — NOT writing pacs_sent.flag "
              "(janitor will retry the unfinished node(s))")
        return False

    pacs_flag = f"{study_dir}/pacs_sent.flag"
    with open(pacs_flag, 'w') as f:
        f.write(datetime.datetime.now().isoformat())

    print(f"✅ Stored study on all {len(nodes)} PACS node(s); wrote flag {pacs_flag}")
    return True




def _newest_image_mtime(study_dir):
    """Most recent mtime of any image in the study (0 if none)."""
    newest = 0
    for root, dirs, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return newest


def _flag_current(study_dir, flag_name, newest=None):
    """True if <flag_name> exists and is at least as new as the newest image.

    Each processing stage has its own flag (pacs_sent.flag for the C-STORE,
    3d_recon.flag for the 3D reconstruction) so they can be retried
    independently — a study whose DICOMs were sent but whose reconstruction
    failed still gets picked up again by the janitor.
    """
    flag = os.path.join(study_dir, flag_name)
    if not os.path.exists(flag):
        return False
    if newest is None:
        newest = _newest_image_mtime(study_dir)
    try:
        return os.path.getmtime(flag) >= newest
    except OSError:
        return False


async def process_study_async(study_dir):
    """
    Process a complete study: create 3D volume and send to PACS.
    Runs in thread pool to avoid blocking the async event loop.
    """

    # Send to PACS (runs in executor to avoid blocking event loop)
    

    print(f"Processing study: {study_dir}")

    # Load study_info.json to determine procedure type
    json_path = f"{study_dir}/study_info.json"
 
    with open(json_path, 'r') as f:
        study_info = json.load(f)
    
    newest_img = _newest_image_mtime(study_dir)
    loop = asyncio.get_event_loop()

    # PACS C-STORE — skip if the study was already sent since its newest image
    # (its own flag), so a recon-only retry doesn't re-transmit every instance.
    if _flag_current(study_dir, 'pacs_sent.flag', newest_img):
        print(f"PACS send already current — skipping for {study_dir}")
    else:
        pacs_result = await loop.run_in_executor(None, send_to_pacs, study_dir, study_info)

    # Extract patient and study IDs
    patient_id = study_info.get('patientid', 'unknown')
    study_id = study_info.get('studyid', study_info.get('studyuuid', 'unknown'))

    def organ_to_side_target(organ_name):
        organ_name = str(organ_name).lower()
        if 'kidney' in organ_name:
            target = 'kidney'
        elif 'prostate' in organ_name:
            # before the bladder check: "prostatebladder" contains "bladder"
            target = 'prostatebladder'
        elif 'bladder' in organ_name or 'postvoid' in organ_name:
            target = 'bladder'
        else:
            target = 'prostatebladder'
        side = ''
        if 'left' in organ_name:
            side = 'left_'
        elif 'right' in organ_name:
            side = 'right_'
        return side, target

    # Sweep axis folder names the app can produce. The non-transverse axis is
    # "sagital"/"sagittal"/"longitudinal" — the name varies by app version and
    # organ; kidney sweeps in particular arrive as "longitudinal". If it isn't
    # listed here the folder is ignored, no job is built, and the study is
    # silently skipped for 3D reconstruction.
    AXIS_NAMES = ('transverse', 'sagital', 'sagittal', 'longitudinal')

    def axes_with_raw(parent_dir):
        """Maps axis name -> path (relative to study_dir) of its raw folder."""
        found = {}
        for item in sorted(os.listdir(parent_dir)):
            if item.lower() not in AXIS_NAMES:
                continue
            raw_sub = os.path.join(parent_dir, item, 'raw')
            if os.path.isdir(raw_sub):
                found[item.lower()] = os.path.relpath(raw_sub, study_dir)
        return found

    # Build one reconstruction job per organ scanned.
    # Layout A (current app): {study}/{organ}/{axis}/raw/
    # Layout B (older app):   {study}/{axis}/raw/  — organ from study_info
    # Layout C (legacy):      {study}/raw_{side}{target}_{orientation}/
    jobs = []  # dicts: {side, target, transverse_dir, sagital_dir, second_axis}

    def make_job(side, target, axes):
        second_axis = next(
            (a for a in ('sagital', 'sagittal', 'longitudinal') if a in axes), None)
        return {
            'side': side,
            'target': target,
            'transverse_dir': axes.get('transverse'),
            'sagital_dir': axes.get(second_axis) if second_axis else None,
            'second_axis': second_axis or 'sagital',
        }

    top_dirs = [e for e in sorted(os.listdir(study_dir))
                if os.path.isdir(os.path.join(study_dir, e))]

    for entry in top_dirs:
        if entry.lower() in AXIS_NAMES:
            continue
        axes = axes_with_raw(os.path.join(study_dir, entry))
        if axes:
            side, target = organ_to_side_target(entry)
            jobs.append(make_job(side, target, axes))

    if not jobs:
        axes = axes_with_raw(study_dir)
        if axes:
            side, target = organ_to_side_target(study_info.get('organ', 'prostate'))
            jobs.append(make_job(side, target, axes))

    if not jobs:
        # Legacy raw_* folders: derive (side, target) pairs from the dir names
        legacy_pairs = {}
        for entry in top_dirs:
            if not entry.startswith('raw_') or entry.endswith('_dicom'):
                continue
            side, target = organ_to_side_target(entry)
            axes = legacy_pairs.setdefault((side, target), {})
            lower = entry.lower()
            if 'transverse' in lower:
                axes['transverse'] = entry
            elif 'sagit' in lower or 'longitud' in lower:
                axes['sagital'] = entry
        for (side, target), axes in legacy_pairs.items():
            jobs.append(make_job(side, target, axes))

    print(f"Reconstruction jobs: {jobs}")

    # 3D reconstruction is gated by its OWN flag (3d_recon.flag), kept separate
    # from pacs_sent.flag. This is what lets a study whose DICOMs were already
    # sent but whose reconstruction failed (e.g. a transient GPU/host OOM) get
    # retried by the janitor without re-sending to PACS.
    if _flag_current(study_dir, '3d_recon.flag', newest_img):
        print(f"3D reconstruction already current — skipping for {study_dir}")
    else:
        recon_ok = True
        # Process each organ separately (e.g., left kidney, right kidney)
        for job in jobs:
            side, target = job['side'], job['target']
            print(f"\n{'='*60}")
            print(f"Processing: patient={patient_id}, study={study_id}, target={target}, side={side}")
            print(f"{'='*60}")

            r_common = {
                'patientid': patient_id,
                'studyid': study_id,
                'target': target,
                'side': side,
                'studies': {},
                'sagital_dir': job['sagital_dir'],
                'transverse_dir': job['transverse_dir'],
                'second_axis': job['second_axis'],
            }

            print("Passing to CuPy-accelerated 3D reconstruction...")
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, process_study_cupy, r_common)
                print(f"✓ 3D reconstruction completed for {side}{target}")
            except Exception as e:
                recon_ok = False
                print(f"✗ 3D reconstruction failed for {side}{target}: {e}")
                import traceback
                traceback.print_exc()
                # Also persist the traceback into the study folder (mounted on the
                # host) so failures are debuggable without docker access
                try:
                    with open(os.path.join(study_dir, 'processing_error.log'), 'a') as ef:
                        ef.write(f"\n[{datetime.datetime.now().isoformat()}] "
                                 f"reconstruction failed for {side}{target}:\n")
                        ef.write(traceback.format_exc())
                except Exception:
                    pass

        # Flag the study reconstructed only once every organ succeeded (an empty
        # job list counts as done — there is nothing to reconstruct). A partial
        # failure leaves the flag unwritten so the janitor retries later.
        if recon_ok:
            recon_flag = f"{study_dir}/3d_recon.flag"
            with open(recon_flag, 'w') as f:
                f.write(datetime.datetime.now().isoformat())
            print(f"✅ Wrote 3D recon flag {recon_flag}")
        else:
            print("✗ One or more reconstructions failed — NOT writing 3d_recon.flag "
                  "(janitor will retry)")

    print(f"\n✓ All processing completed for {patient_id}/{study_id}")
    return True
   
        
        



async def processImage(image_data, r_d, writer, images_processed, image_list, filename=None):
    """Process a single image from the client."""
    # Extract image bytes
    image_start = image_data.find(b'STARTOFIMAGE')
    image_end = image_data.find(b'ENDOFIMAGE')

    
    image_bytes = image_data[image_start + len(b'STARTOFIMAGE'):image_end]
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Use the directory path from the filename if provided (most reliable)
    if filename and '/' in filename:
        # Filename format: "raw_prostatebladder_transverse/image.jpg"
        sub_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)
    else:
        # Fallback: construct from metadata
        organ = r_d.get('organ', r_d.get('bodypart', 'unknown')).lower()
        if 'kidney' in organ:
            target = 'kidney'
        elif 'prostate' in organ or 'bladder' in organ:
            target = 'prostatebladder'
        else:
            target = 'prostatebladder'  # default
        
        side = ''
        if 'left' in organ:
            side = 'left_'
        elif 'right' in organ:
            side = 'right_'
        
        orientation = r_d.get('orientation', 'transverse')
        sub_dir = f"raw_{side}{target}_{orientation}"
        
        # Try to extract filename timestamp
        if filename:
            base_name = os.path.basename(filename)
        else:
            base_name = None
    
    fpath = f"{idir}/{r_d['patientid']}/{r_d['studyid']}/{sub_dir}"
    os.makedirs(fpath, exist_ok=True)
    
    # Use the base name from the filename if provided
    if filename and base_name:
        # Extract timestamp from pattern (13-digit number)
        parts = base_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').split('_')
        timestamp = None
        for part in parts:
            if part.isdigit() and len(part) == 13:
                timestamp = part
                break
        
        if timestamp:
            fname = f"{fpath}/{timestamp}.jpeg"
        else:
            # Use the original base name
            fname = f"{fpath}/{base_name}"
    else:
        # Generate timestamp
        t = str(int(time.time() * 1000))
        fname = f"{fpath}/{t}.jpeg"
    
    # Save image
    cv2.imwrite(fname, decoded)

    # Track filenames only — frames are re-read from disk when processing
    image_list.append(fname)
    
    # Send acknowledgment
    writer.write(f"Image {images_processed} received".encode())
    await writer.drain()
    
    return decoded


async def process_batch(r_d, image_list, writer):
    """Process a batch of images."""
    print(f"Processing batch of {len(image_list)} images")
    
    # Update study info
    study_dir = f"{idir}/{r_d['patientid']}/{r_d['studyid']}"
    os.makedirs(study_dir, exist_ok=True)
    
    # Save study_info.json
    json_path = f"{study_dir}/study_info.json"
    
    # Load existing or create new study_info
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                study_info = json.load(f)
        except Exception as e:
            print(f"Error reading study_info.json: {e}, creating new")
            study_info = r_d.copy()
    else:
        study_info = r_d.copy()
    
    study_info['num_images'] = len(image_list)
    study_info['processing_timestamp'] = datetime.datetime.now().isoformat()
    
    # Save updated study_info
    with open(json_path, 'w') as f:
        json.dump(study_info, f, indent=4)
    
    # If study is marked as finished, enqueue it FIRST before any drain that might fail
    if r_d.get('finished'):
        print("Study marked as finished, enqueuing for processing...")
        
        # Enqueue the study for processing, respecting queue size
        global STUDY_QUEUE
        if STUDY_QUEUE is None:
            try:
                STUDY_QUEUE = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
            except Exception:
                STUDY_QUEUE = asyncio.Queue()

        enqueue_result = None
        try:
            if study_dir in STUDIES_IN_FLIGHT:
                enqueue_result = "Study already queued for processing"
                print(enqueue_result)
            elif STUDY_QUEUE.full():
                enqueue_result = "Server busy: processing backlog full, try later"
                print(enqueue_result)
            else:
                STUDIES_IN_FLIGHT.add(study_dir)
                await STUDY_QUEUE.put(study_dir)
                enqueue_result = "Study enqueued for processing"
                print(f"✓ {enqueue_result}")
        except Exception as e:
            enqueue_result = f"Failed to enqueue study: {e}"
            print(f"✗ {enqueue_result}")
        
        # Now try to notify client (but don't fail if connection is lost)
        if writer is not None:
            try:
                writer.write(f"Processed {len(image_list)} images\n".encode())
                writer.write("Creating 3D volume and sending to PACS...\n".encode())
                if enqueue_result:
                    writer.write(f"{enqueue_result}\n".encode())
                await writer.drain()
            except Exception as e:
                print(f"Could not send response to client (connection lost): {e}")
    else:
        # Not finished yet, just acknowledge
        if writer is not None:
            try:
                writer.write(f"Processed {len(image_list)} images (not finished yet)".encode())
                await writer.drain()
            except Exception as e:
                print(f"Could not send response to client: {e}")


async def handle_websocket_client(websocket):
    """Handle Android app WebSocket connections.

    Protocol (matches SocketThread.java):
      1. Client sends metadata JSON (text)
      2. For each image: filename text → binary data → server acks with
         {"status":"image_received"} → client sends next
      3. Client sends {"type":"complete"} then closes
    """
    remote = websocket.remote_address
    print(f"[WS] New connection from {remote}")

    r_d = None
    images_processed = 0
    image_list = []
    current_filename = None

    try:
        async for raw_message in websocket:
            # --- Text messages: metadata JSON, filenames, or completion ---
            if isinstance(raw_message, str):
                msg = raw_message.strip()

                # First text message is the metadata JSON
                if r_d is None:
                    try:
                        r_d = json.loads(msg)
                        r_d = sanitise_json(r_d)
                        if 'studyuuid' in r_d and 'studyid' not in r_d:
                            r_d['studyid'] = r_d['studyuuid']
                        if 'patientid' not in r_d:
                            r_d['patientid'] = r_d.get('patientemail', 'unknown')
                        study_dir = f"{idir}/{r_d.get('patientid','unknown')}/{r_d.get('studyid','unknown')}"
                        os.makedirs(study_dir, exist_ok=True)
                        json_path = f"{study_dir}/study_info.json"
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r') as jf:
                                    existing = json.load(jf)
                            except Exception:
                                existing = {}
                            # Each sweep uploads with its own metadata; 'organ' is
                            # last-writer-wins, so keep an accumulating 'organs'
                            # list recording every organ scanned in this study
                            organs = set(existing.get('organs') or [])
                            if existing.get('organ'):
                                organs.add(existing['organ'])
                            if r_d.get('organ'):
                                organs.add(r_d['organ'])
                            existing.update(r_d)
                            existing['organs'] = sorted(organs)
                            with open(json_path, 'w') as jf:
                                json.dump(existing, jf, indent=4)
                        else:
                            r_d.setdefault('patientid', r_d.get('patientid', 'unknown'))
                            r_d.setdefault('studyid', r_d.get('studyid', r_d.get('studyuuid', 'unknown')))
                            r_d.setdefault('finished', False)
                            if r_d.get('organ'):
                                r_d['organs'] = [r_d['organ']]
                            with open(json_path, 'w') as jf:
                                json.dump(r_d, jf, indent=4)
                        print(f"[WS] Received config from {remote}: patient={r_d.get('patientid')} study={r_d.get('studyid')}")
                    except json.JSONDecodeError as e:
                        print(f"[WS] Bad metadata JSON from {remote}: {e}")
                        await websocket.send(json.dumps({"error": "invalid metadata"}))
                        return
                    continue

                # Check for completion signal
                if msg.lower().startswith('{') or msg.lower().startswith('['):
                    try:
                        parsed = json.loads(msg)
                        if parsed.get('type') == 'complete':
                            # Each sweep upload sends 'complete' on its own
                            # connection; only the one flagged sessionComplete
                            # (the last sweep) triggers processing + PACS send.
                            # Older apps don't send the flag — process every time
                            # like before.
                            session_done = bool(r_d.get('sessioncomplete', True))
                            print(f"[WS] Client {remote} marked complete ({images_processed} images, sessionComplete={session_done})")
                            r_d['finished'] = session_done
                            await process_batch(r_d, image_list, None)
                            return
                    except json.JSONDecodeError:
                        pass

                # Otherwise it's a filename
                current_filename = msg
                continue

            # --- Binary messages: raw image data ---
            if isinstance(raw_message, bytes):
                if r_d is None:
                    print(f"[WS] Received image before metadata from {remote}, dropping")
                    continue

                nparr = np.frombuffer(raw_message, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is None:
                    print(f"[WS] Failed to decode image from {remote}")
                    await websocket.send(json.dumps({"status": "image_received"}))
                    continue

                # Filename from Android: "{organ}/{orientation}/{type}/{imagename}"
                # e.g. "rightkidney/sagital/raw/20260611_175122_0000_p0.0_r0.0_y0.0.jpg"
                # Older app versions send "{orientation}/{type}/{imagename}"; the
                # organ then comes from this connection's metadata so right/left
                # sweeps of the same study still land in separate folders.
                img_organ = str(r_d.get('organ', '')).lower()
                img_orientation = r_d.get('orientation', 'transverse')
                img_type = 'raw'  # default
                base_name = current_filename or ''
                if current_filename and '/' in current_filename:
                    parts_path = current_filename.split('/')
                    if len(parts_path) >= 4:
                        img_organ = parts_path[0]
                        img_orientation = parts_path[1]
                        img_type = parts_path[2]  # 'raw' or 'processed'
                        base_name = parts_path[3]
                    elif len(parts_path) == 3:
                        img_orientation = parts_path[0]
                        img_type = parts_path[1]  # 'raw' or 'processed'
                        base_name = parts_path[2]
                    elif len(parts_path) == 2:
                        img_orientation = parts_path[0]
                        base_name = parts_path[1]

                organ_part = f"{img_organ}/" if img_organ else ""
                fpath = f"{idir}/{r_d['patientid']}/{r_d['studyid']}/{organ_part}{img_orientation}/{img_type}"
                os.makedirs(fpath, exist_ok=True)
                fname = f"{fpath}/{base_name}"

                cv2.imwrite(fname, decoded)

                # Track filenames only — holding every decoded frame in memory
                # costs gigabytes per session and the frames are never read
                # again (processing re-reads from disk)
                image_list.append(fname)
                images_processed += 1
                print(f"[WS] Image {images_processed} saved: {fname}")

                # Ack so client sends next image
                await websocket.send(json.dumps({"status": "image_received"}))
                current_filename = None

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[WS] Connection closed by client {remote}: {e}")
    except Exception as e:
        print(f"[WS] Error handling WebSocket from {remote}: {e}")
        import traceback
        traceback.print_exc()


async def handle_client(reader, writer):
    """Handle incoming client connections."""
    client_addr = writer.get_extra_info('peername')
    print(f"New connection from {client_addr}")
    
    connected = True
    
    # Send public key to client
    public_key_pem = public_key.export_key().decode()
    writer.write(public_key_pem.encode())
    writer.write(b"ENDPUBKEY")
    await writer.drain()
    print("Sent public key to client")
    
    while connected:
        imagebytes = io.BytesIO()
        data = b""
        images_processed = 0
        image_list = []
        r_d = None
        current_filename = None
        
        # Read data from client
        while imagebytes.getvalue().find(b"ENDOFFILE") == -1:
            try:
                data = await reader.read(4096)
            except Exception as e:
                print(f"Connection error while reading: {e}")
                connected = False
                break
            if not data:
                connected = False
                break
            imagebytes.write(data)
            
            # Check for config data
            if b'STARTOFCONFIG' in imagebytes.getvalue() and b'ENDOFCONFIG' in imagebytes.getvalue():
                r_d = getConfigFromBytes(imagebytes)
                if r_d:
                    r_d = sanitise_json(r_d)
                    # Map studyuuid to studyid if needed
                    if 'studyuuid' in r_d and 'studyid' not in r_d:
                        r_d['studyid'] = r_d['studyuuid']
                    # Ensure patientid always exists (app sends patientEmail instead)
                    if 'patientid' not in r_d:
                        r_d['patientid'] = r_d.get('patientemail', 'unknown')
                    # Remove config data from imagebytes
                    config_start = imagebytes.getvalue().find(b'STARTOFCONFIG')
                    config_end = imagebytes.getvalue().find(b'ENDOFCONFIG') + len(b'ENDOFCONFIG')
                    remaining_data = imagebytes.getvalue()[:config_start] + imagebytes.getvalue()[config_end:]
                    imagebytes = io.BytesIO()
                    imagebytes.write(remaining_data)
                    print(f"Received config: {r_d}")
                    # Ensure study directory exists and create/update study_info.json
                    try:
                        study_dir = f"{idir}/{r_d.get('patientid','unknown')}/{r_d.get('studyid', r_d.get('studyuuid','unknown'))}"
                        os.makedirs(study_dir, exist_ok=True)
                        json_path = f"{study_dir}/study_info.json"
                        # If file exists, merge/overwrite with incoming config; otherwise create it
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r') as jf:
                                    existing = json.load(jf)
                            except Exception:
                                existing = {}
                            # Merge: update existing with keys from r_d, keeping a
                            # running list of every organ scanned in this study
                            organs = set(existing.get('organs') or [])
                            if existing.get('organ'):
                                organs.add(existing['organ'])
                            if r_d.get('organ'):
                                organs.add(r_d['organ'])
                            existing.update(r_d)
                            existing['organs'] = sorted(organs)
                            with open(json_path, 'w') as jf:
                                json.dump(existing, jf, indent=4)
                        else:
                            # Ensure some defaults
                            r_d.setdefault('patientid', r_d.get('patientid', 'unknown'))
                            r_d.setdefault('studyid', r_d.get('studyid', r_d.get('studyuuid', 'unknown')))
                            r_d.setdefault('finished', False)
                            with open(json_path, 'w') as jf:
                                json.dump(r_d, jf, indent=4)
                    except Exception as e:
                        print(f"Error writing study_info.json: {e}")
                else:
                    writer.write(b'Config error: could not parse config')
                    await writer.drain()
                    connected = False
                    break
                if r_d is None:
                    writer.write(b'Config error: config not set')
                    await writer.drain()
                    return None
            
            # Check for filename data
            elif b'STARTOFFILENAME' in imagebytes.getvalue() and b'ENDOFFILENAME' in imagebytes.getvalue():
                start = imagebytes.getvalue().find(b'STARTOFFILENAME')
                end = imagebytes.getvalue().find(b'ENDOFFILENAME')
                name_bytes = imagebytes.getvalue()[start + len(b'STARTOFFILENAME'):end]
                current_filename = name_bytes.decode('utf-8')
                #print(f"Receiving image: {current_filename}")
                
                # Remove from buffer
                remaining = imagebytes.getvalue()[:start] + imagebytes.getvalue()[end + len(b'ENDOFFILENAME'):]
                imagebytes = io.BytesIO()
                imagebytes.write(remaining)

            # Check for image data
            elif b'STARTOFIMAGE' in imagebytes.getvalue() and b'ENDOFIMAGE' in imagebytes.getvalue():
                # Extract image data
                image_start = imagebytes.getvalue().find(b'STARTOFIMAGE')
                image_end = imagebytes.getvalue().find(b'ENDOFIMAGE') + len(b'ENDOFIMAGE')
                image_data = imagebytes.getvalue()[image_start:image_end]
                remaining_data = imagebytes.getvalue()[:image_start] + imagebytes.getvalue()[image_end:]
                imagebytes = io.BytesIO()
                imagebytes.write(remaining_data)
                
                # Process image
                print(r_d)
                decoded = await processImage(image_data, r_d, writer, images_processed, image_list, current_filename)
                images_processed += 1
                current_filename = None # Reset for next image
        
        if not connected:
            break
        
        if r_d is None:
            writer.write("No config received".encode())
            await writer.drain()
            continue
        
        # Process collected images
        if image_list:
            writer.write("Batch received, processing...".encode())
            await writer.drain()
            await process_batch(r_d, image_list, writer)
    
    try:
        writer.close()
        await writer.wait_closed()
    except ConnectionResetError:
        pass
    print(f"Connection closed: {client_addr}")


async def run_server():
    """Start the PACS transfer server."""
    # Initialize the study queue and start worker tasks
    global STUDY_QUEUE, WORKER_TASKS
    if STUDY_QUEUE is None:
        try:
            STUDY_QUEUE = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        except Exception:
            STUDY_QUEUE = asyncio.Queue()

    async def _worker(worker_id: int):
        print(f"Worker {worker_id} started")
        while True:
            study_dir = await STUDY_QUEUE.get()
            try:
                print(f"Worker {worker_id} processing: {study_dir}")
                try:
                    await process_study_async(study_dir)
                except Exception as e:
                    print(f"Worker {worker_id} error processing {study_dir}: {e}")
            finally:
                STUDIES_IN_FLIGHT.discard(study_dir)
                try:
                    STUDY_QUEUE.task_done()
                except Exception:
                    pass

    # Janitor: uploads can die mid-session (network drop), in which case the
    # final sweep's sessionComplete never arrives and the study would sit
    # unprocessed forever. Periodically pick up any study whose images have
    # settled (no new files for a while) but which hasn't been sent to PACS
    # since its newest image. Processing is idempotent (deterministic DICOM
    # UIDs), so a later re-upload + reprocess just merges.
    JANITOR_INTERVAL = 60          # seconds between scans
    JANITOR_SETTLE_SECS = 180      # study must be quiet this long
    JANITOR_RETRY_SECS = 900       # min gap between attempts per study
    janitor_last_try = {}

    async def _janitor():
        print("Janitor started (recovers studies with interrupted uploads)")
        while True:
            await asyncio.sleep(JANITOR_INTERVAL)
            try:
                now = time.time()
                for patient in os.listdir(idir):
                    patient_dir = os.path.join(idir, patient)
                    if not os.path.isdir(patient_dir):
                        continue
                    for study in os.listdir(patient_dir):
                        study_dir = os.path.join(patient_dir, study)
                        if not os.path.isdir(study_dir) or study_dir in STUDIES_IN_FLIGHT:
                            continue
                        if not os.path.exists(os.path.join(study_dir, 'study_info.json')):
                            continue
                        if now - janitor_last_try.get(study_dir, 0) < JANITOR_RETRY_SECS:
                            continue
                        newest = _newest_image_mtime(study_dir)
                        if newest == 0 or now - newest < JANITOR_SETTLE_SECS:
                            continue  # no images yet / still uploading
                        # Re-enqueue if EITHER the PACS send or the 3D
                        # reconstruction is missing/stale relative to the newest
                        # image. Tracked by separate flags so a failed
                        # reconstruction is retried even when the DICOMs were
                        # already sent (and vice-versa).
                        pacs_done = _flag_current(study_dir, 'pacs_sent.flag', newest)
                        recon_done = _flag_current(study_dir, '3d_recon.flag', newest)
                        if pacs_done and recon_done:
                            continue  # already fully handled
                        if STUDY_QUEUE.full():
                            continue
                        janitor_last_try[study_dir] = now
                        STUDIES_IN_FLIGHT.add(study_dir)
                        await STUDY_QUEUE.put(study_dir)
                        print(f"🧹 Janitor enqueued unfinished study: {study_dir}")
            except Exception as e:
                print(f"Janitor error: {e}")

    # Launch workers + janitor
    for i in range(max(1, MAX_CONCURRENT_JOBS)):
        t = asyncio.create_task(_worker(i))
        WORKER_TASKS.append(t)
    WORKER_TASKS.append(asyncio.create_task(_janitor()))

    # Start TCP server (legacy protocol)
    tcp_server = await asyncio.start_server(handle_client, HOST, PORT)
    tcp_addr = tcp_server.sockets[0].getsockname()

    # Start WebSocket server (Android app protocol)
    ws_server = await websockets.serve(handle_websocket_client, HOST, WEBSOCKET_PORT)

    print(f"PACS Transfer Server running")
    print(f"   TCP  listening on {HOST}:{PORT} (legacy)")
    print(f"   WS   listening on {HOST}:{WEBSOCKET_PORT} (Android)")
    print(f"   Images will be saved to: {idir}")
    print(f"   Press Ctrl+C to stop")

    async with tcp_server, ws_server:
        await asyncio.gather(tcp_server.serve_forever(), ws_server.wait_closed())


if __name__ == "__main__":
    print("=" * 60)
    print("PACS Transfer Server")
    print("=" * 60)

    asyncio.run(run_server())
   
