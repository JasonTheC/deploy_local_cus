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

# Import the CuPy-accelerated 3D reconstruction module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guidance'))

from pos_cupy_finalv2 import process_study as process_study_cupy
CUPY_AVAILABLE = True
print("✓ CuPy-accelerated 3D reconstruction available")

# Global dict to accumulate images in memory
study_images = {}


# Server configuration
HOST = "0.0.0.0"
PORT = 8890  # Different port from guidanceWS3.py

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




def send_to_pacs(study_dir, study_info):
    """
    Send study to PACS server.
    This is a placeholder - implement actual PACS C-STORE here.
    
    Args:
        study_dir: Path to study directory containing images and metadata
    """
    print(f"Sending study to PACS: {study_dir}")


    print(f"Study info: {json.dumps(study_info, indent=2)}")

    # Build PACS config defaults (can be overridden by study_info['pacs'])
    # Inside Docker the Orthanc service is reachable as "orthanc"; override
    # via PACS_HOST / PACS_PORT / PACS_AET environment variables.
    pacs_cfg = {
        'ae_title': os.environ.get('PACS_AET', 'ORTHANC'),
        'port': int(os.environ.get('PACS_PORT', '4242')),
        'ip': os.environ.get('PACS_HOST', '127.0.0.1')
    }
    if isinstance(study_info.get('pacs'), dict):
        pacs_cfg.update(study_info.get('pacs'))

    # Collect image files grouped by series (raw_* directories)
    series_images = {}  # Key: directory name, Value: list of image paths
    for root, dirs, files in os.walk(study_dir):
        # Only process files in raw_* directories
        if 'raw_' not in os.path.basename(root):
            continue
        series_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                if series_name not in series_images:
                    series_images[series_name] = []
                series_images[series_name].append(os.path.join(root, file))

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

    # Study description from the scanned organ (e.g. "Prostate")
    study_description = str(study_info.get('organ', '')).title()
    
    # Get depth from study info (in cm)
    depth_cm = study_info.get('depth', 15)

    # Process each series separately
    generated_dcms = []
    for series_name, img_paths in series_images.items():
        # Generate one SeriesInstanceUID per series (folder)
        series_instance_uid = generate_uid()
        series_description = series_name.replace('raw_', '').replace('_', ' ').title()
        
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
            ds.SOPInstanceUID = generate_uid()
            ds.Modality = 'US'

            # convert to grayscale and 16-bit similar to guidance.dicom_util
            image_2d = np.array(img)
            if len(image_2d.shape) == 3:
                image_2d = np.mean(image_2d, axis=2).astype(float)
            # Guard against zero-division
            if image_2d.max() == 0:
                image_2d = image_2d + 1.0
            image_2d = ((image_2d / image_2d.max()) * 65535).astype(np.uint16)

            # Calculate live pixel region (excluding black borders)
            non_zero_rows = np.any(image_2d > image_2d.min() + 100, axis=1)
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
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
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

    # Send all DICOMs in a single association
    print(f"Attempting to send {len(generated_dcms)} DICOM files to PACS {pacs_cfg['ip']}:{pacs_cfg['port']}")
    ae = AE(ae_title='PYNETDICOM')
    ae.add_requested_context('1.2.840.10008.5.1.4.1.1.6.1')
    assoc = ae.associate(pacs_cfg['ip'], pacs_cfg['port'], ae_title=pacs_cfg['ae_title'])

    if not assoc.is_established:
        print('Association rejected, aborted or never connected')
        return False

    try:
        for dcm_file in generated_dcms:
            ds_dataset = dcmread(dcm_file)
            status = assoc.send_c_store(ds_dataset)
            print(f"Sent {dcm_file} - Status: {status}")
    finally:
        assoc.release()

    # Mark finished with flag
    pacs_flag = f"{study_dir}/pacs_sent.flag"
    with open(pacs_flag, 'w') as f:
        f.write(datetime.datetime.now().isoformat())

    print(f"✅ Sent study to PACS and wrote flag {pacs_flag}")
    return True




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
    
    loop = asyncio.get_event_loop()
    pacs_result = await loop.run_in_executor(None, send_to_pacs, study_dir, study_info)
   
    # Extract patient and study IDs
    patient_id = study_info.get('patientid', 'unknown')
    study_id = study_info.get('studyid', study_info.get('studyuuid', 'unknown'))
    
    # Scan the study directory to find what raw_ directories actually exist
    study_path = f"US_images/{patient_id}/{study_id}"
    raw_dirs = []
    if os.path.isdir(study_path):
        for item in os.listdir(study_path):
            if item.startswith('raw_') and os.path.isdir(os.path.join(study_path, item)):
                raw_dirs.append(item)
    
    # Extract unique target+side combinations from directory names
    # e.g., raw_left_kidney_sagital -> (left_, kidney)
    studies_to_process = set()
    for dirname in raw_dirs:
        # Remove raw_ prefix and orientation suffix
        parts = dirname.replace('raw_', '').split('_')
        # Filter out orientation keywords
        parts = [p for p in parts if p not in ['sagital', 'transverse']]
        
        # Reconstruct target and side
        if not parts:
            continue
        
        side = ''
        target = ''
        if parts[0] in ['left', 'right']:
            side = parts[0] + '_'
            target = '_'.join(parts[1:]) if len(parts) > 1 else ''
        else:
            target = '_'.join(parts)
        
        if target:
            studies_to_process.add((side, target))
    
    print(f"Found {len(studies_to_process)} target(s) to process: {studies_to_process}")
    
    # Process each target separately (e.g., left kidney, right kidney)
    for side, target in studies_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: patient={patient_id}, study={study_id}, target={target}, side={side}")
        print(f"{'='*60}")
        
        r_common = {
            'patientid': patient_id,
            'studyid': study_id,
            'target': target,
            'side': side,
            'studies': {}
        }
        
        print("Passing to CuPy-accelerated 3D reconstruction...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, process_study_cupy, r_common)
            print(f"✓ 3D reconstruction completed for {side}{target}")
        except Exception as e:
            print(f"✗ 3D reconstruction failed for {side}{target}: {e}")
            import traceback
            traceback.print_exc()
    
    if study_dir in study_images:
        del study_images[study_dir]
    
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
    
    # Accumulate images in memory for direct processing
    study_dir = f"{idir}/{r_d['patientid']}/{r_d['studyid']}"
    if study_dir not in study_images:
        study_images[study_dir] = {'sag': {'images': [], 'files': []}, 'trans': {'images': [], 'files': []}}
    
    orientation = 'trans' if 'transverse' in sub_dir else 'sag'
    study_images[study_dir][orientation]['images'].append(decoded)
    study_images[study_dir][orientation]['files'].append(fname)
    
    # Collect images for batch processing
    image_list.append((decoded, fname))
    
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
            if STUDY_QUEUE.full():
                enqueue_result = "Server busy: processing backlog full, try later"
                print(enqueue_result)
            else:
                await STUDY_QUEUE.put(study_dir)
                enqueue_result = "Study enqueued for processing"
                print(f"✓ {enqueue_result}")
        except Exception as e:
            enqueue_result = f"Failed to enqueue study: {e}"
            print(f"✗ {enqueue_result}")
        
        # Now try to notify client (but don't fail if connection is lost)
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
        try:
            writer.write(f"Processed {len(image_list)} images (not finished yet)".encode())
            await writer.drain()
        except Exception as e:
            print(f"Could not send response to client: {e}")


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
                            # Merge: update existing with keys from r_d
                            existing.update(r_d)
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
    
    writer.close()
    await writer.wait_closed()
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
                try:
                    STUDY_QUEUE.task_done()
                except Exception:
                    pass

    # Launch workers
    for i in range(max(1, MAX_CONCURRENT_JOBS)):
        t = asyncio.create_task(_worker(i))
        WORKER_TASKS.append(t)

    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"PACS Transfer Server running on {addr}")
    print(f"   Listening on {HOST}:{PORT}")
    print(f"   Images will be saved to: {idir}")
    print(f"   Press Ctrl+C to stop")
    
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    print("=" * 60)
    print("PACS Transfer Server")
    print("=" * 60)

    asyncio.run(run_server())
   
