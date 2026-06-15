import cupy as cp
import numpy as np
import os, cv2, time, json, torch
from natsort import natsorted
import nibabel as nib
from PIL import Image
from scipy.ndimage import gaussian_filter

torch.set_num_threads(30)
anatomy_to_colour = {"prostate": 1, "bladder": 2, "kidney": 3}
procedure_list = {"prostatebladder": ["prostate", "bladder"],
                    "prostate": ["prostate", "bladder"], 
                    "bladder": ["prostate", "bladder"],
                    "right_kidney":["kidney"],
                    "left_kidney":["kidney"],
                    "kidney":["kidney"]}

idir = 'US_images'
odir = 'outputs'
mdir = idir
os.makedirs(odir, exist_ok=True)

# Load trained models
print("Loading trained mobilenet segmentation models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

modelDict = {}
# Use absolute path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(script_dir, "mobilenet_models")

if os.path.exists(modelPath):
    for x in os.listdir(modelPath):
        if x.endswith('.pth'):
            model = torch.load(f"{modelPath}/{x}", weights_only=False)
            modelDict[x.split("_")[0]] = model.to(device).eval()
    print(f"Loaded {len(modelDict)} models")
else:
    print(f"Warning: Model directory not found at {modelPath}")
    print("AI segmentation will not be available")

def prepare_image_batch(images, target_size=256):
    """Prepare batch of images for inference"""
    prepared = []
    for img in images:
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        prepared.append(img)
    return np.stack(prepared)

def run_inference_batch(model, images, device, target_size=256, batch_size=32):
    """Run inference in chunks — a full sweep (~300 frames) in one batch
    overflows an 8GB GPU's activation memory."""
    mask_chunks = []
    conf_chunks = []
    for start in range(0, len(images), batch_size):
        prepared = prepare_image_batch(images[start:start + batch_size], target_size)
        input_tensor = torch.from_numpy(prepared).unsqueeze(1).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)
            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1).cpu().numpy()
                mask_chunks.append(np.argmax(probs, axis=1).astype(np.uint8))
                conf_chunks.append(np.max(probs, axis=1))
            else:
                probs = torch.sigmoid(output).squeeze(1).cpu().numpy()
                mask_chunks.append((probs > 0.5).astype(np.uint8))
                conf_chunks.append(probs)

        del input_tensor, output

    # Free GPU memory
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

    return np.concatenate(mask_chunks), np.concatenate(conf_chunks)

def get_bounds(a):
    """Fast bounds calculation using CuPy"""
    # Convert to CuPy if numpy
    if isinstance(a, np.ndarray):
        a_gpu = cp.asarray(a)
    else:
        a_gpu = a
    
    if not a_gpu.any():
        return (0, a_gpu.shape[0]-1, 0, a_gpu.shape[1]-1, 0, a_gpu.shape[2]-1)
    
    x_any = a_gpu.any(axis=(1,2))
    y_any = a_gpu.any(axis=(0,2))
    z_any = a_gpu.any(axis=(0,1))
    
    ux = int(cp.argmax(x_any).get())
    bx = int(len(x_any) - cp.argmax(x_any[::-1]).get() - 1)
    uy = int(cp.argmax(y_any).get())
    by = int(len(y_any) - cp.argmax(y_any[::-1]).get() - 1)
    uz = int(cp.argmax(z_any).get())
    bz = int(len(z_any) - cp.argmax(z_any[::-1]).get() - 1)
    
    return (ux, bx, uy, by, uz, bz)

def zero_trim_ndarray(ndarray):
    """Trim zeros from array"""
    b = get_bounds(ndarray)
    return ndarray[b[0]:b[1]+1, b[2]:b[3]+1, b[4]:b[5]+1]

def process_images(r_d, img_array=None, flist=None):
    """Process images and run inference"""
    print("\nProcessing images for inference...")
    # Use actual_dir if provided, otherwise try to find it
    base_study = f'{mdir}/{r_d["patientid"]}/{r_d["studyid"]}'
    orient = r_d.get("orientation", "transverse")

    if 'actual_dir' in r_d and isinstance(r_d['actual_dir'], str) and r_d['actual_dir']:
        fpath = f'{base_study}/{r_d["actual_dir"]}/'
    else:
        # The second axis goes by several names depending on app version/organ
        if orient in ('sagital', 'sagittal', 'longitudinal'):
            orient_aliases = [orient] + [a for a in ('sagital', 'sagittal', 'longitudinal') if a != orient]
        else:
            orient_aliases = [orient]

        # Try new layout first: {orientation}/raw/
        fpath = None
        for o in orient_aliases:
            new_layout = f'{base_study}/{o}/raw/'
            if os.path.isdir(new_layout):
                fpath = new_layout
                break
        if fpath is None:
            # Fallback: legacy raw_ prefix patterns
            patterns = []
            for o in orient_aliases:
                patterns += [
                    f'raw_{r_d["side"]}{r_d["target"]}_{o}',
                    f'{r_d["side"]}{r_d["target"]}_{o}',
                    f'raw_{r_d["target"]}_{o}',
                    f'{r_d["target"]}_{o}'
                ]
            for pattern in patterns:
                test_path = f'{base_study}/{pattern}/'
                if os.path.isdir(test_path):
                    fpath = test_path
                    break
            if fpath is None:
                fpath = f'{base_study}/raw_{r_d["side"]}{r_d["target"]}_{orient}/'
    print(r_d)
    print(fpath)
    target_list = procedure_list[r_d["target"]]
    if r_d["target"] not in procedure_list:
        print(f"Unknown target {r_d['target']}, skipping")
        return [], {}, np.array([]), []
    targets_int = [anatomy_to_colour[t] for t in target_list]
    
    if img_array is None:
        files = natsorted(os.listdir(fpath))
        valid_imgs = []
        valid_files = []
        
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(f"{fpath}/{f}"):
                img = cv2.imread(f"{fpath}/{f}")
                if img is not None:
                    valid_imgs.append(img)
                    valid_files.append(f)
        
        if not valid_imgs:
            raise RuntimeError(f"No valid images for {r_d}")

        # The probe stream can change resolution mid-sweep (depth/view change),
        # but np.stack and the volume builder need uniform frames — resize
        # outliers to the sweep's dominant size
        from collections import Counter
        shape_counts = Counter(im.shape[:2] for im in valid_imgs)
        target_hw = shape_counts.most_common(1)[0][0]
        if len(shape_counts) > 1:
            print(f"Mixed frame sizes {dict(shape_counts)} — resizing all to {target_hw}")
            valid_imgs = [im if im.shape[:2] == target_hw
                          else cv2.resize(im, (target_hw[1], target_hw[0]))
                          for im in valid_imgs]

        img_array = np.stack(valid_imgs)
    else:
        if isinstance(img_array, np.ndarray) and img_array.ndim == 4:
            valid_imgs = [img_array[i] for i in range(img_array.shape[0])]
        else:
            valid_imgs = img_array  # assume list
        valid_files = flist if flist is not None else [f"image_{i}.png" for i in range(len(valid_imgs))]
        img_array = np.stack(valid_imgs) if not isinstance(img_array, np.ndarray) or img_array.ndim != 3 else img_array
    
    # Select model
    key = r_d["target"].lower()
    if 'prostate' in key or 'bladder' in key:
        model_key = 'bladder'
    elif 'kidney' in key:
        model_key = 'kidney'
    else:
        model_key = 'bladder'
    print(model_key)
    model = modelDict.get(model_key, next(iter(modelDict.values())))
    
    # Run inference
    start_time = time.time()
    print(f'Running inference on {len(valid_imgs)} images...')
    pred_masks, confidence_maps = run_inference_batch(model, valid_imgs, device)
    print(f'Inference done in {time.time() - start_time:.2f} seconds.')
    
    # Process predictions
    contiguous_segments = []
    max_bbox = {"height": 0, "width": 0, "target": targets_int[0] if targets_int else 1}
    current_sublist = []
    
    for idx in range(len(valid_imgs)):
        pred_mask = pred_masks[idx]
        confidence_map = confidence_maps[idx]
        h, w = valid_imgs[idx].shape[:2]
        ni = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(confidence_map, (w, h), interpolation=cv2.INTER_LINEAR)
        ni[confidence_map < 0.1] = 0
        u = np.unique(ni)
        
        if any(t in u for t in targets_int):
            current_sublist.append(valid_files[idx])
        else:
            if current_sublist:
                contiguous_segments.append(current_sublist)
                current_sublist = []
        
        for target in targets_int:
            if target in u:
                coords = np.column_stack(np.where(ni == target))
                if coords.size:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    height, width = y_max - y_min, x_max - x_min
                    if height > max_bbox["height"] or width > max_bbox["width"]:
                        max_bbox = {"height": height, "width": width, "img": valid_files[idx], 
                                    "target": target, "y_min": y_min, "x_min": x_min, 
                                    "y_max": y_max, "x_max": x_max}
                break
    
    if current_sublist:
        contiguous_segments.append(current_sublist)
    
    return contiguous_segments, max_bbox, img_array, valid_files

def place_slice_in_volume(points, image_array, theta, center_x, center_y, center_z, volume_shape, orientation='transverse'):
    """Place 2D slice into 3D volume with fan geometry - CuPy accelerated version"""
    height, width = image_array.shape
    
    # Convert image to CuPy
    if isinstance(image_array, np.ndarray):
        image_array_gpu = cp.asarray(image_array)
    else:
        image_array_gpu = image_array
    
    # Get non-zero pixels upfront
    mask = image_array_gpu != 0
    if not mask.any():
        return
    
    if abs(theta) < 0.01:
        # No rotation case
        y_coords, x_coords = cp.where(mask)
        pixel_values = image_array_gpu[mask]

        # Any non-transverse sweep (sagital/sagittal/longitudinal) uses the
        # sagittal placement geometry
        if orientation != 'transverse':
            vol_x = cp.full_like(x_coords, center_x)
            vol_y = y_coords
            vol_z = x_coords
        else:
            vol_x = x_coords
            vol_y = y_coords
            vol_z = cp.full_like(x_coords, center_z)
    else:
        # Fan geometry
        x_indices, y_indices = cp.indices([height, width])
        
        theta_rad = cp.deg2rad(theta)
        half_complement = (180 - theta) / 2
        
        # CuPy doesn't have errstate, so we just compute and clean up NaN/Inf after
        A = x_indices / cp.sin(cp.deg2rad(half_complement))
        A = A * cp.sin(theta_rad)
        z1 = A * cp.sin(cp.deg2rad(half_complement))
        x1 = z1 / cp.tan(theta_rad)
        
        x1 = cp.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
        z1 = cp.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_coords = y_indices[mask]
        x_coords_3d = x1[mask]
        z_coords_3d = z1[mask]
        pixel_values = image_array_gpu[mask]

        if orientation != 'transverse':
            vol_x = (z_coords_3d + center_x).astype(cp.int32)
            vol_y = y_coords.astype(cp.int32)
            vol_z = (x_coords_3d + center_z).astype(cp.int32)
        else:
            vol_x = (x_coords_3d + center_x).astype(cp.int32)
            vol_y = y_coords.astype(cp.int32)
            vol_z = (z_coords_3d + center_z).astype(cp.int32)
    
    # Bounds check
    in_bounds = (0 <= vol_x) & (vol_x < volume_shape[0]) & \
                (0 <= vol_y) & (vol_y < volume_shape[1]) & \
                (0 <= vol_z) & (vol_z < volume_shape[2])
    
    vol_x = vol_x[in_bounds]
    vol_y = vol_y[in_bounds]
    vol_z = vol_z[in_bounds]
    pixel_values = pixel_values[in_bounds]
    
    # Convert to CPU for indexing (CuPy advanced indexing can be slow)
    vol_x_cpu = cp.asnumpy(vol_x)
    vol_y_cpu = cp.asnumpy(vol_y)
    vol_z_cpu = cp.asnumpy(vol_z)
    pixel_values_cpu = cp.asnumpy(pixel_values)
    
    # Update volume
    points[vol_x_cpu, vol_y_cpu, vol_z_cpu] = np.maximum(points[vol_x_cpu, vol_y_cpu, vol_z_cpu], pixel_values_cpu)

def orient_apex_on_top(volume_np):
    """Ensure the probe-apex axis sits at the top of the saved volume.

    The rebuild fans slices around the apex; depending on sweep direction the apex
    can land at either end of Z. The fan cross-section grows from apex (narrow) to
    base (wide), so the Z-half with fewer populated voxels is the apex end. Flip Z
    only when that apex end is at the bottom.
    """
    counts = (volume_np > 0).sum(axis=(0, 1))
    half = len(counts) // 2
    if counts[:half].sum() < counts[half:].sum():  # apex (narrow end) at bottom
        return np.ascontiguousarray(volume_np[:, :, ::-1])
    return volume_np

def get_voxel(patient_data, r_d, img_array, flist):
    """Create 3D volume from 2D slices - CuPy accelerated"""
    indices = patient_data["studies"][r_d["studyid"]]["indices"]
    
    # Volume size
    multi = 2
    offset = 20
    s = (int(800 * multi + offset), int(800 * multi + offset), int(800 * multi + offset))
    
    print(f"Creating volume of size {s}")
    # Keep volume on CPU for now (memory consideration)
    points = np.zeros(s, dtype=np.uint8)
    
    center_x, center_y, center_z = s[0] // 2, s[1] // 2, s[2] // 2
    orientation = r_d.get('orientation', 'transverse')
    
    # Angle setup
    target_angle = 120.0
    num_slices = len(flist)
    angle_interval = target_angle / num_slices if num_slices > 1 else 1.0
    
    print(f"{orientation}: {num_slices} slices, angle_interval={angle_interval:.2f}°")
    
    # Reduced interpolation for speed
    interpolation_factor = 3
    
    # Calculate angles
    all_angles = []
    for i in range(len(flist)):
        start_angle = i * angle_interval
        all_angles.append((i, start_angle, flist[i], False))
        
        if i + 1 < len(flist):
            next_angle = (i + 1) * angle_interval
            for interp_step in range(1, interpolation_factor + 1):
                alpha = interp_step / (interpolation_factor + 1)
                interp_angle = start_angle + alpha * (next_angle - start_angle)
                all_angles.append((i, interp_angle, flist[i], True))
    
    # Process slices
    print(f"Processing {len(all_angles)} angle steps...")
    for idx, (i, theta, fname, is_interpolated) in enumerate(all_angles):
        if idx % 100 == 0:
            print(f"  {idx}/{len(all_angles)}")
        
        if i >= len(img_array):
            continue
        
        img = img_array[i]
        if img.ndim == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        if img_gray is None:
            continue
        
        place_slice_in_volume(points, img_gray, theta, center_x, center_y, center_z, s, orientation)
    
    # Trim
    print("Trimming volume...")
    points = zero_trim_ndarray(points)
    
    # Quick gap fill - process in chunks to avoid GPU memory overflow
    mask = points > 0
    if not mask.all():
        print("Quick gap fill with CuPy (chunked processing)...")
        max_iter = 5
        
        # Size chunks from actual free VRAM: each z-slice needs roughly
        # 12 bytes/voxel across points/sums/counts + astype temporaries.
        # (A fixed 200 was tuned for a 16GB card and OOMs on 8GB.)
        try:
            free_b, _total_b = cp.cuda.Device().mem_info
            bytes_per_slice = points.shape[0] * points.shape[1] * 12
            chunk_size = max(16, min(200, int(free_b * 0.5 / bytes_per_slice)))
        except Exception:
            chunk_size = 64
        chunk_size = min(chunk_size, points.shape[2])
        print(f"Gap fill chunk size: {chunk_size} slices")
        num_chunks = (points.shape[2] + chunk_size - 1) // chunk_size
        
        for it in range(max_iter):
            any_changes = False
            
            for chunk_idx in range(num_chunks):
                z_start = chunk_idx * chunk_size
                z_end = min(z_start + chunk_size, points.shape[2])
                
                # Add overlap for neighbor operations
                z_fetch_start = max(0, z_start - 1)
                z_fetch_end = min(points.shape[2], z_end + 1)
                
                # Move chunk to GPU
                chunk = points[:, :, z_fetch_start:z_fetch_end]
                points_gpu = cp.asarray(chunk)
                
                # Create working arrays for this chunk
                sums = cp.zeros_like(points_gpu, dtype=cp.float32)
                counts = cp.zeros_like(points_gpu, dtype=cp.uint8)
                
                # 6-neighbor average
                sums[:-1, :, :] += points_gpu[1:, :, :].astype(cp.float32)
                counts[:-1, :, :] += (points_gpu[1:, :, :] > 0).astype(cp.uint8)
                sums[1:, :, :] += points_gpu[:-1, :, :].astype(cp.float32)
                counts[1:, :, :] += (points_gpu[:-1, :, :] > 0).astype(cp.uint8)
                sums[:, :-1, :] += points_gpu[:, 1:, :].astype(cp.float32)
                counts[:, :-1, :] += (points_gpu[:, 1:, :] > 0).astype(cp.uint8)
                sums[:, 1:, :] += points_gpu[:, :-1, :].astype(cp.float32)
                counts[:, 1:, :] += (points_gpu[:, :-1, :] > 0).astype(cp.uint8)
                sums[:, :, :-1] += points_gpu[:, :, 1:].astype(cp.float32)
                counts[:, :, :-1] += (points_gpu[:, :, 1:] > 0).astype(cp.uint8)
                sums[:, :, 1:] += points_gpu[:, :, :-1].astype(cp.float32)
                counts[:, :, 1:] += (points_gpu[:, :, :-1] > 0).astype(cp.uint8)
                
                fill_locs = (points_gpu == 0) & (counts > 0)
                
                if fill_locs.any():
                    any_changes = True
                    mask_nonzero = counts > 0
                    sums[mask_nonzero] /= counts[mask_nonzero].astype(cp.float32)
                    points_gpu[fill_locs] = cp.rint(sums[fill_locs]).astype(cp.uint8)
                
                # Write back to CPU (excluding overlap regions)
                local_start = z_start - z_fetch_start
                local_end = local_start + (z_end - z_start)
                points[:, :, z_start:z_end] = cp.asnumpy(points_gpu[:, :, local_start:local_end])
                
                # Free GPU memory for this chunk
                del points_gpu, sums, counts, fill_locs
                cp.get_default_memory_pool().free_all_blocks()
            
            if not any_changes:
                break
            
            print(f"  Iteration {it + 1}/{max_iter}")
        
        print(f"Gap fill done in {it + 1} iteration(s)")
    
    # Apply gentle blur to remove pixelated look without creating artifacts
    print("Applying smoothing to remove pixelation...")
    # Use scipy (CPU) to avoid CuPy compilation issues
    mask = points > 0
    blurred = gaussian_filter(points.astype(np.float32), sigma=0.8)
    points = np.where(mask, blurred, 0).astype(np.uint8)

    # The fan apex (probe axis) comes out at the bottom — flip so it sits on top.
    print("Orienting probe-apex axis to top...")
    points = orient_apex_on_top(points)

    # Save
    output_path = f"{odir}/{r_d['patientid']}_{r_d['studyid']}_{r_d['side']}{r_d['target']}_{orientation}_riv3.nii.gz"
    print(f"Saving to {output_path}")
    nii = nib.Nifti1Image(points, affine=np.eye(4))
    nib.save(nii, output_path)
    
    # Aggressively free GPU memory after volume generation
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    print("GPU memory cleared")

def process_study(r_common):
    """Process a single study - transverse and sagital"""
    patient_id = r_common["patientid"]
    study_id = r_common["studyid"]
    target = r_common["target"]
    side = r_common.get('side', '')
    
    print(f"\n{'='*60}")
    print(f"Processing {patient_id}/{study_id}/{side}{target}")
    print(f"{'='*60}")
    
    # Second axis: "sagital" for pelvic organs, "longitudinal" for kidneys.
    # The label drives the output filename; geometry treats anything
    # non-transverse as a sagittal-style sweep.
    second_axis = r_common.get('second_axis') or 'sagital'
    print(f"\n--- {second_axis.upper()} ---")
    r_s = r_common.copy()
    r_s["orientation"] = second_axis
    if 'sagital_dir' in r_common:
        r_s['actual_dir'] = r_common['sagital_dir']
    print('starting process_images for sagital...')
    # A sweep can be missing entirely when an upload died mid-session —
    # reconstruct whatever orientations actually arrived instead of aborting
    try:
        contiguous_segments_sag, max_bbox_sag, img_array_sag, files_sag = process_images(r_s)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"⚠ {second_axis} sweep unavailable ({e}) — continuing without it")
        contiguous_segments_sag, max_bbox_sag, img_array_sag, files_sag = [], {}, None, []

    default_x = img_array_sag.shape[0] // 2 if img_array_sag is not None else 256
    default_y = img_array_sag.shape[1] // 2 if img_array_sag is not None else 256
    indices_sag = {max_bbox_sag.get("target", 1): {
        "y1": max_bbox_sag.get("y_min", default_y),
        "x1": max_bbox_sag.get("x_min", default_x),
        "y2": max_bbox_sag.get("y_max", default_y),
        "x2": max_bbox_sag.get("x_max", default_x)
    }}

    # Transverse
    print("\n--- TRANSVERSE ---")
    r_t = r_common.copy()
    r_t["orientation"] = "transverse"
    if 'transverse_dir' in r_common:
        r_t['actual_dir'] = r_common['transverse_dir']
    try:
        contiguous_segments_trans, max_bbox_trans, img_array_trans, files_trans = process_images(r_t)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"⚠ transverse sweep unavailable ({e}) — continuing without it")
        contiguous_segments_trans, max_bbox_trans, img_array_trans, files_trans = [], {}, None, []

    if contiguous_segments_trans:
        longest = max(contiguous_segments_trans, key=len)
        transverse_stats = {"min": longest[0], "max": longest[-1]}
    elif len(files_trans):
        transverse_stats = {"min": files_trans[0], "max": files_trans[-1]}
    else:
        transverse_stats = {}
    
    # Setup indices
    if "studies" not in r_common:
        r_common["studies"] = {}
    if study_id not in r_common["studies"]:
        r_common["studies"][study_id] = {}
    r_common["studies"][study_id]["indices"] = indices_sag
    r_common["studies"][study_id]["transverse_stats"] = transverse_stats
    
    # Generate volumes for whichever sweeps are present
    if img_array_trans is not None and len(files_trans):
        print("\n--- GENERATING TRANSVERSE VOLUME ---")
        get_voxel(r_common, r_t, img_array_trans, files_trans)
    else:
        print("\n(no transverse sweep — skipping transverse volume)")

    if img_array_sag is not None and len(files_sag):
        print(f"\n--- GENERATING {second_axis.upper()} VOLUME ---")
        get_voxel(r_common, r_s, img_array_sag, files_sag)
    else:
        print(f"\n(no {second_axis} sweep — skipping {second_axis} volume)")
    
    # Final cleanup after entire study
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Completed {patient_id}/{study_id}/{side}{target}")
    print(f"{'='*60}\n")

def process_all_patients(base_dir=mdir):
    """Process all patients"""
    patients = natsorted(os.listdir(base_dir))
    print(f"Found {len(patients)} patients to process.")
    
    for patient_id in patients:
        patient_path = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
            
        studies = natsorted(os.listdir(patient_path))
        for study_id in studies:
            study_path = os.path.join(patient_path, study_id)
            if not os.path.isdir(study_path):
                continue
                
            print(f"\nChecking patient={patient_id} study={study_id}")
            
            # Find actual directories in study path
            all_dirs = [d for d in os.listdir(study_path)
                       if os.path.isdir(os.path.join(study_path, d))]

            # --- New layout: {orientation}/raw/ or {orientation}/processed/ ---
            new_orientations = {}
            for d in all_dirs:
                raw_sub = os.path.join(study_path, d, 'raw')
                proc_sub = os.path.join(study_path, d, 'processed')
                if os.path.isdir(raw_sub) or os.path.isdir(proc_sub):
                    low = d.lower()
                    if 'sagital' in low:
                        new_orientations.setdefault('sagital', True)
                    elif 'transverse' in low:
                        new_orientations.setdefault('transverse', True)

            # --- Legacy layout: raw_{side}{target}_{orientation} ---
            raw_dirs = [d for d in all_dirs if 'processed' not in d and
                       (d.startswith('raw_') or any(x in d for x in ['kidney', 'prostate', 'bladder']))]
            studies_found = {}
            for d in raw_dirs:
                clean = d.replace('raw_', '') if d.startswith('raw_') else d
                if '_sagital' in clean:
                    orientation = 'sagital'
                    prefix = clean.split('_sagital')[0]
                elif '_transverse' in clean:
                    orientation = 'transverse'
                    prefix = clean.split('_transverse')[0]
                else:
                    continue
                if 'kidney' not in prefix and 'prostate' not in prefix and 'bladder' not in prefix:
                    continue
                if prefix not in studies_found:
                    studies_found[prefix] = {'sagital': [], 'transverse': []}
                studies_found[prefix][orientation].append(d)
            
            # Process each unique target/side combination
            for part_dir, orientations in studies_found.items():
                # Determine target and side
                if 'kidney' in part_dir:
                    target = 'kidney'
                elif 'prostate' in part_dir or 'bladder' in part_dir:
                    target = 'prostatebladder'
                else:
                    continue
                
                if 'left' in part_dir:
                    side = 'left_'
                elif 'right' in part_dir:
                    side = 'right_'
                else:
                    side = ''
                
                # Check if already processed
                if os.path.exists(f"{odir}/{patient_id}_{study_id}_{side}{target}_sagital_riv3.nii.gz"):
                    #print(f"Already exists: {patient_id}_{study_id}_{side}{target}_sagital, skipping.")
                    continue
                
                # Check we have both orientations
                if not orientations['sagital']:
                    print(f"Missing sagital directory for {part_dir}, skipping.")
                    continue
                if not orientations['transverse']:
                    print(f"Missing transverse directory for {part_dir}, skipping.")
                    continue
                
                # Use the directory without number suffix if available, otherwise the highest numbered one
                def select_dir(dir_list):
                    # Prefer unnumbered
                    for d in dir_list:
                        if not any(d.endswith(f'_{i}') for i in range(10)):
                            return d
                    # Otherwise return the last one (highest number)
                    return sorted(dir_list)[-1]
                
                sagital_dir = select_dir(orientations['sagital'])
                transverse_dir = select_dir(orientations['transverse'])
                
                print(f"Using sagital: {sagital_dir}, transverse: {transverse_dir}")
                
                # Check study_info.json
                if not os.path.isfile(study_path + "/study_info.json"):
                    print(f"No study_info.json for {study_id}/{target}, skipping.")
                    continue
                
                r_common = json.loads(open(study_path + "/study_info.json").read())
                r_common["patientid"] = patient_id
                r_common["studyid"] = study_id
                r_common["target"] = target
                r_common['side'] = side
                r_common['sagital_dir'] = sagital_dir
                r_common['transverse_dir'] = transverse_dir
                
                # Process this study
                try:
                    process_study(r_common)
                except Exception as e:
                    print(f"ERROR processing {patient_id}/{study_id}/{side}{target}: {e}")
                    import traceback
                    traceback.print_exc()

            # --- New layout: {orientation}/raw/ — derive target from metadata ---
            if not studies_found and new_orientations:
                info_path = os.path.join(study_path, "study_info.json")
                if not os.path.isfile(info_path):
                    continue
                info = json.loads(open(info_path).read())
                organ = info.get('organ', 'prostate').lower()
                if 'kidney' in organ:
                    target = 'kidney'
                elif 'bladder' in organ or 'postvoid' in organ:
                    target = 'bladder'
                else:
                    target = 'prostatebladder'
                side = ''
                if 'left' in organ:
                    side = 'left_'
                elif 'right' in organ:
                    side = 'right_'

                out_sag = f"{odir}/{patient_id}_{study_id}_{side}{target}_sagital_riv3.nii.gz"
                if os.path.exists(out_sag):
                    continue

                r_common = {
                    'patientid': patient_id,
                    'studyid': study_id,
                    'target': target,
                    'side': side,
                    'studies': {},
                }
                # Pass actual orientation dir names so process_images finds them
                for d in all_dirs:
                    low = d.lower()
                    if 'sagital' in low and os.path.isdir(os.path.join(study_path, d, 'raw')):
                        r_common['sagital_dir'] = f'{d}/raw'
                    elif 'transverse' in low and os.path.isdir(os.path.join(study_path, d, 'raw')):
                        r_common['transverse_dir'] = f'{d}/raw'

                try:
                    process_study(r_common)
                except Exception as e:
                    print(f"ERROR processing {patient_id}/{study_id}/{side}{target}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    process_all_patients()
