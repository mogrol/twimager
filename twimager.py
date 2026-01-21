import os
import sys
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle
import vptree
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from imagehash import phash
import json
import av
import time

from rife.rife_align import rife_align_image

EXTENSIONS_IMAGE = (".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff")
EXTENSIONS_VIDEO = (".webm", ".mkv", ".mp4", ".avi", ".mpg", ".mpeg", ".avs", ".avsi")

def ask(question, default=False):
    while True:
        if default:
            return input(f"{question} [Y/n]: ").lower() not in ("n", "no")
        else:
            return input(f"{question} [y/N]: ").lower() in ("y", "yes")

def format_time(seconds):
    """Converts seconds into a human-readable string."""
    periods = [
        ('h', 3600),
        ('m', 60),
        ('s', 1)
    ]

    result = []
    for suffix, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            result.append(f"{int(period_value)}{suffix}")

    return " ".join(result) if result else "0s"

def color_transfer(target=None, reference=None):
    if target is None or reference is None:
        return None

    mean_in = np.mean(target, axis=(0, 1), keepdims=True)
    mean_ref = np.mean(reference, axis=(0, 1), keepdims=True)
    std_in = np.std(target, axis=(0, 1), keepdims=True)
    std_ref = np.std(reference, axis=(0, 1), keepdims=True)
    #img_arr_out = (target - mean_in) / std_in * std_ref + mean_ref
    #img_arr_out[img_arr_out < 0] = 0
    #img_arr_out[img_arr_out > 255] = 255
    return np.clip(
        (target - mean_in) / std_in * std_ref + mean_ref,
        0,
        255
    ).astype(np.uint8)

def get_comb_score(frame):
    img = frame.to_ndarray(format="gray")
    top, mid, bot = img[:-2, :], img[1:-1, :], img[2:, :]

    # Spatial difference check for combing zig-zags
    comb = np.abs(mid.astype(np.int32) - (top.astype(np.int32) + bot.astype(np.int32)) / 2)


    return np.mean(comb > 22) # Slightly higher threshold for telecine precision

def analyze_video_type(path, frames_to_check=50):
    container = av.open(path)
    stream = container.streams.video[0]

    # Check container-level field order
    order = getattr(stream.codec_context, "field_order", "unknown")

    results = []

    # Seek to 20% to avoid logos/static
    container.seek(int(stream.duration * 0.2) if stream.duration else 0, stream=stream)

    for i, frame in enumerate(container.decode(stream)):
        if i >= frames_to_check: break
        results.append(get_comb_score(frame) > 0.01) # Boolean: is this frame combed?

    container.close()

    # Pattern Analysis
    combed_count = sum(results)
    combed_ratio = combed_count / len(results)

    # Look for the 3:2 pattern (Telecine signature)
    # A simplified check: Telecine usually has ~40% combed frames (2 out of 5)
    if 0.3 <= combed_ratio <= 0.5:
        return "Telecined", "fieldmatch,decimate"

    elif combed_ratio > 0.7:
        # Standard interlaced (most frames show combing in motion)
        filter_str = "bwdif" if order == "unknown" else f"bwdif=parity={1 if 'tt' in order else 0}"

        return "Interlaced", filter_str
    else:
        return "Progressive", None

class Hash:
    BLUR_FILTER = ImageFilter.GaussianBlur(3)

    @staticmethod
    def create(image, hash_size=8, blur=False):
        try:
            if isinstance(image, np.ndarray):
                # Convert once and reuse via the local variable 'img'
                img = Image.fromarray(image)
                if blur:
                    img = img.filter(Hash.BLUR_FILTER)

                return phash(img, hash_size)

            elif isinstance(image, Image.Image):
                if blur:
                    return phash(image.filter(Hash.BLUR_FILTER), hash_size)

                return phash(image, hash_size)

            return None

        except Exception:
            return None

def hamming_distance(a, b):
    return a - b

def get_image(path, file_or_index, type):
    if type == "images":
        return load_image(os.path.join(path, file_or_index))

    if type == "video":
        return load_frame(path, file_or_index)

def load_image(file):
    try:
        return cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        #return cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[..., :3]
    except Exception:
        return None

load_frame_lock = threading.Lock()
def load_frame(path, index):
    container = None

    if path == opt_hr:
        container = hr_container
        stream = hr_container.streams.video[0]
        keyframe_map = hr_keyframe_map
    elif path == opt_lr:
        container = lr_container
        stream = lr_container.streams.video[0]
        keyframe_map = lr_keyframe_map

    if container is None:
        return None

    # Use lock for safety as this method can be access from multiple threads at the same time.
    with load_frame_lock:
        kf_index, kf_pts = max([f for f in keyframe_map if f[0] <= index], key=lambda x: x[0])

        container.seek(kf_pts, stream=stream)

        # Manually decode frames one-by-one until we reach the correct frame
        for frame in container.decode(stream):
            # Don't rely on frame.index as it might not exist. Instead use our own generated indices
            if kf_index == index:
                # Convert to ndarray and return
                return frame.to_ndarray(format="bgr24")

            if kf_index > index:
                # If we passed it, something went wrong with the stream indexing
                break

            kf_index += 1

    return None

def crop_image(image, scale=1.0):
    height, width = image.shape[:2]

    if opt_crop_border:
        #crop_pixels = opt_crop_border * scale * 2

        #if crop_pixels < width and crop_pixels < height:
        #    return image[int(opt_crop_border * scale):-int(opt_crop_border * scale), int(opt_crop_border * scale):-int(opt_crop_border * scale)]
        return image[int(opt_crop_border * scale):-int(opt_crop_border * scale), int(opt_crop_border * scale):-int(opt_crop_border * scale)]
        #else:
        #    print(f"Image dimensions of {file} is smaller than crop border. Image skipped...")
        #    return None

    elif opt_crop_size:
        #if width < opt_crop_size[0] * scale or height < opt_crop_size[1] * scale:
        #    print(f"Image dimensions of {file} is smaller than crop size. Image skipped...")
        #    return None

        x = int(((width - (opt_crop_size[1] * scale)) * 0.5))
        y = int(((height - (opt_crop_size[0] * scale)) * 0.5))

        return image[y:y + int(opt_crop_size[1] * scale), x:x + int(opt_crop_size[0] * scale)]

    elif opt_crop_coordinates:
        top = int(opt_crop_coordinates[0] * scale)
        left = int(opt_crop_coordinates[1] * scale)
        right = int(top + (opt_crop_coordinates[2] * scale))
        bottom = int(left + (opt_crop_coordinates[3] * scale))

        return image[top:bottom, left:right]

    return image

def process_file(file, scale, cut=False, blur=False):
    image = load_image(file)
    if image is None:
        return

    key = os.path.basename(file)

    return process_image(image, key, scale, cut, blur)

def process_image(image, key, scale, cut=False, blur=False):
    height, width = image.shape[:2]

    if opt_crop_border:
        crop_pixels = opt_crop_border * scale * 2

        if crop_pixels < width and crop_pixels < height:
            return None

    elif opt_crop_size:
        if width < opt_crop_size[0] * scale or height < opt_crop_size[1] * scale:
            return None

    elif opt_crop_coordinates:
        top = int(opt_crop_coordinates[0] * scale)
        left = int(opt_crop_coordinates[1] * scale)
        right = int(top + (opt_crop_coordinates[2] * scale))
        bottom = int(left + (opt_crop_coordinates[3] * scale))

        if width < right or height < bottom:
            return None

    if cut:
        ratio = width / height

        # if the aspect ratio is off by 25%, cut out a part of it and use to create an additional hash
        # this helps with finding HR images which are taller or wider than the LR image.
        if ratio < 0.75: #0.75:
            offset = int((height - width) * 0.5)
            hash = Hash.create(image[offset:offset+width, 0:width], HASH_SIZE, blur)
            if hash:
                return (hash, key)
                #target_list[hash] = key
        elif ratio > 1.35: #1.25:
            offset = int((width - height) * 0.5)
            hash = Hash.create(image[0:height, offset:offset+height], HASH_SIZE, blur)
            if hash:
                return (hash, key)
                #target_list[hash] = key

    hash = Hash.create(image, HASH_SIZE, blur)

    if hash:
        return (hash, key)
        #target_list[hash] = key

    return None

def find_image_match(hr_hash):
    report = None
    results = lr_tree.get_all_in_range(hr_hash, DISTANCE_THRESHOLD_MATCH)

    if not results:
        return 0, report

    hr_file = hr_items[hr_hash]
    hr_image = get_image(opt_hr, hr_items[hr_hash], opt_hr_type)
    if hr_image is None:
        return 0, report

    previous_distance = None
    aligned_hr_image = None
    aligned_lr_image = None

    for result in results:
        lr_hash = result[1]
        lr_file = lr_items[lr_hash]
        lr_image = get_image(opt_lr, lr_file, opt_lr_type)
        if lr_image is None:
            continue

        hr_image_aligned = align_image(hr_image, lr_image, opt_scale, align_method=opt_align_method, match_method=opt_match_method)

        if hr_image_aligned is None:
            continue

        if opt_align_rife:
            if opt_scale > 1.0:
                hr_image_aligned = rife_align_image(
                    hr_image_aligned,
                    cv2.resize(lr_image, (int(lr_image.shape[1] * opt_scale), int(lr_image.shape[0] * opt_scale)), interpolation=cv2.INTER_AREA)
                )

            else:
                hr_image_aligned = rife_align_image(
                    hr_image_aligned,
                    lr_image
                )

        if opt_crop:
            lr_image = crop_image(lr_image)
            hr_image_aligned = crop_image(hr_image_aligned, scale=opt_scale)

        aligned_distance = hamming_distance(
            Hash.create(lr_image, HASH_SIZE_ALIGNED),
            Hash.create(hr_image_aligned, HASH_SIZE_ALIGNED)
        )

        if aligned_distance > DISTANCE_THRESHOLD_ALIGNED:
            continue

        if previous_distance and aligned_distance >= previous_distance:
            continue

        previous_distance = aligned_distance

        aligned_hr_image = hr_image_aligned
        aligned_hr_file = hr_file

        aligned_lr_image = lr_image
        aligned_lr_file = lr_file

    if aligned_hr_image is None or aligned_lr_image is None:
        return 0, report

    if opt_dry:
        return 1, report

    hr_image = color_transfer(aligned_hr_image, aligned_lr_image) if opt_color_transfer and not opt_dry else aligned_hr_image
    hr_file = aligned_hr_file

    lr_image = aligned_lr_image
    lr_file = aligned_lr_file

    if opt_debug:
        pass
        """
        height, width = hr_image.shape[:2]
        preview = np.zeros((int(height), int(width * 3) + 3, 3), dtype=np.uint8)
        preview[0:height, 0:width] = hr_image
        preview[0:height, width + 1:int(width * 2) + 1] = cv2.resize(lr_image, (width, height))
        preview[0:height, int(width * 2) + 2:int(width * 3) + 2] = cv2.addWeighted(cv2.resize(hr_image, (width, height)), 0.5, cv2.resize(lr_image, (width, height)), 0.5, 0.0)

        string = "HR"
        cv2.putText(preview, string, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        string = "LR"
        cv2.putText(preview, string, (width + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (width + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        string = "Blend"
        cv2.putText(preview, string, (int(width * 2) + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (int(width * 2)+ 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Twimager Debug Window", preview)

        del preview

        key = cv2.waitKeyEx()
        if key == 27: # esc
            raise KeyboardInterrupt
        elif key == 10 or key == 3014656: # n or del
            cv2.destroyAllWindows()
            return 0, report

        cv2.destroyAllWindows()
        """

    if opt_filename == "hr":
        if opt_hr_type == "video":
            lr_dest_name = hr_dest_name = f"{os.path.basename(opt_hr)}_{hr_file}"
        else:
            lr_dest_name = hr_dest_name = os.path.splitext(hr_file)[0]
    elif opt_filename == "lr":
        if opt_lr_type == "video":
            lr_dest_name = hr_dest_name = f"{os.path.basename(opt_lr)}_{lr_file}"
        else:
            lr_dest_name = hr_dest_name = os.path.splitext(lr_file)[0]
    else:
        lr_dest_name = os.path.splitext(lr_file)[0]
        hr_dest_name = os.path.splitext(hr_file)[0]

    if opt_prefix and opt_suffix:
        hr_dest_path = os.path.join(opt_dest_hr, f"{opt_prefix_string}_{hr_dest_name}_{opt_suffix_string}.png")
        lr_dest_path = os.path.join(opt_dest_lr, f"{opt_prefix_string}_{lr_dest_name}_{opt_suffix_string}.png")
    elif opt_prefix:
        hr_dest_path = os.path.join(opt_dest_hr, f"{opt_prefix_string}_{hr_dest_name}.png")
        lr_dest_path = os.path.join(opt_dest_lr, f"{opt_prefix_string}_{lr_dest_name}.png")
    elif opt_suffix:
        hr_dest_path = os.path.join(opt_dest_hr, f"{hr_dest_name}_{opt_suffix_string}.png")
        lr_dest_path = os.path.join(opt_dest_lr, f"{lr_dest_name}_{opt_suffix_string}.png")
    else:
        hr_dest_path = os.path.join(opt_dest_hr, f"{hr_dest_name}.png")
        lr_dest_path = os.path.join(opt_dest_lr, f"{lr_dest_name}.png")

    if opt_output in ("hr", "all"):
        cv2.imencode(".png", hr_image)[1].tofile(hr_dest_path)

    if opt_output in ("lr", "all"):
        cv2.imencode(".png", lr_image)[1].tofile(lr_dest_path)

    if opt_report:
        if opt_output == "all":
            report = (hr_dest_path, lr_dest_path)
        elif opt_output == "hr":
            report = (hr_dest_path, os.path.join(opt_lr, lr_file))
        elif opt_output == "lr":
            report = (os.path.join(opt_hr, hr_file), lr_dest_path)

    return 1, report

def shift_image(src, dst, scale=1.0, align_method="transform", match_method="bf"):
    if scale > 1.0:
        dst_gray = np.array(Image.fromarray(dst).convert("L").resize((int(dst.shape[1] * scale), int(dst.shape[0] * scale)), resample=Image.Resampling.BICUBIC))
    else:
        dst_gray = np.array(Image.fromarray(src).convert("L"))

    src_gray = np.array(Image.fromarray(dst).convert("L").resize((dst_gray.shape[1], dst_gray.shape[0]), resample=Image.Resampling.BICUBIC))

    # Optionally apply a Hanning window to reduce edge effects
    hanning_window = cv2.createHanningWindow(dst_gray.shape[::-1], cv2.CV_64F)
    dst_windowed = np.multiply(dst_gray.astype(np.float64), hanning_window)
    src_windowed = np.multiply(src_gray.astype(np.float64), hanning_window)

    # Compute Fourier Transforms of both images
    dst_dft = cv2.dft(np.float32(dst_windowed), flags=cv2.DFT_COMPLEX_OUTPUT)
    src_dft = cv2.dft(np.float32(src_windowed), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Calculate phase correlation
    shift, _ = cv2.phaseCorrelate(dst_dft, src_dft)

    # Use the shift to align the second image
    rows, cols = dst.shape[:2]
    aligned_src = cv2.warpAffine(src, np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]]), (cols, rows))

    return aligned_src

def align_image(src, dst, scale=1.0, align_method="transform", match_method="bf"):
    src_height, src_width = src.shape[:2]
    dst_height, dst_width = dst.shape[:2]

    MIN_SCORE = 20
    GOOD_MATCH_PERCENT = 0.7
    RESIZE = 512 if 512 < min(dst_width, dst_height) else min(dst_width, dst_height)

    def get_matches(descriptor1, descriptor2, method="bf"):
        if method == "knn":
            return get_matches_knn(descriptor1, descriptor2)
        else:
            return get_matches_bf(descriptor1, descriptor2)

    def get_matches_knn(descriptor1, descriptor2):
        matcher = cv2.BFMatcher() # faster matching but less accurate

        matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
        results1 = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < GOOD_MATCH_PERCENT * n.distance:
                    results1.append(m)

        matches = matcher.knnMatch(descriptor2, descriptor1, k=2)
        results2 = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < GOOD_MATCH_PERCENT * n.distance:
                    results2.append(m)

        results = []
        for match1 in results1:
            match1_query_index = match1.queryIdx
            match1_train_index = match1.trainIdx

            for match2 in results2:
                match2_query_index = match2.queryIdx
                match2_train_index = match2.trainIdx

                if (match1_query_index == match2_train_index) and (match1_train_index == match2_query_index):
                    results.append(match1)

        return results

    def get_matches_bf(descriptor1, descriptor2):
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

    def calculate_score(matches, keypoint1, keypoint2):
        return 100 * (matches / min(keypoint1, keypoint2))

    # Resize the intermediate images. If scale is 1.0 increase the size slightly to to avoid aliasing when transforming
    # or warping the image
    align_scale = scale if scale > 1.0 else scale + 0.5
    align_width = int(dst_width * align_scale)
    align_height = int(dst_height * align_scale)

    # Feature detection on large images takes an unnecessarily long time with no real benefit when it comes to the
    # amount of points matched. So we'll scale both src and dst to RESIZE as this not only speeds up the process but
    # also improves accuracy
    src_ratio = src_width / src_height
    if src_ratio < 1:
        src_width_scaled = int(RESIZE * src_ratio)
        src_height_scaled = RESIZE
    else:
        src_width_scaled = RESIZE
        src_height_scaled = int(RESIZE / src_ratio)

    dst_ratio = dst_width / dst_height
    if dst_ratio < 1:
        dst_width_scaled = int(RESIZE * dst_ratio)
        dst_height_scaled = RESIZE
    else:
        dst_width_scaled = RESIZE
        dst_height_scaled = int(RESIZE / dst_ratio)

    # Convert to grayscale and resize using PIL. As to why PIL is used instead of OpenCV, read: https://zuru.tech/blog/the-dangers-behind-image-resizing
    src_gray = np.array(Image.fromarray(cv2.convertScaleAbs(src, 0, 1.5)).convert("L", dither=Image.Dither.NONE).resize((src_width_scaled, src_height_scaled), resample=Image.Resampling.LANCZOS))
    #src_gray = np.array(Image.fromarray(src).convert("L", dither=Image.Dither.NONE).resize((src_width_scaled, src_height_scaled), resample=Image.Resampling.LANCZOS))
    #cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (src_width_scaled, src_height_scaled), interpolation=cv2.INTER_AREA)
    dst_gray = np.array(Image.fromarray(cv2.convertScaleAbs(dst, 0, 1.5)).convert("L", dither=Image.Dither.NONE).resize((dst_width_scaled, dst_height_scaled), resample=Image.Resampling.LANCZOS))
    #dst_gray = np.array(Image.fromarray(dst).convert("L", dither=Image.Dither.NONE).resize((dst_width_scaled, dst_height_scaled), resample=Image.Resampling.LANCZOS))
    #cv2.resize(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (dst_width_scaled, dst_height_scaled), interpolation=cv2.INTER_AREA)

    #cv2.imshow("src_gray", src_gray)
    #cv2.imshow("dst_gray", dst_gray)
    #cv2.waitKey()

    sift = cv2.SIFT_create(nfeatures=256) # 512
    src_keypoints, src_descriptors = sift.detectAndCompute(src_gray, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dst_gray, None)

    if len(src_keypoints) == 0 or len(dst_keypoints) == 0 or len(src_descriptors) == 0 or len(dst_descriptors) == 0:
        return None

    matches = get_matches(src_descriptors, dst_descriptors, method=match_method)
    score = calculate_score(len(matches), len(src_keypoints), len(dst_keypoints))

    # Skip the alignment step if feature point matching score is less than MIN_SCORE
    if score < MIN_SCORE:
        return None

    # Scale the matched points for src and dst to match the desired aligned image size
    src_x_scale = (src_width / src_width_scaled)
    src_y_scale = (src_height / src_height_scaled)
    src_keypoints_scaled = []
    for keypoint in src_keypoints:
        x = keypoint.pt[0] * src_x_scale
        y = keypoint.pt[1] * src_y_scale
        size = keypoint.size * max(src_x_scale, src_y_scale)
        src_keypoints_scaled.append(cv2.KeyPoint(x, y, size))

    src_keypoints = src_keypoints_scaled

    dst_x_scale = dst_width / dst_width_scaled
    dst_y_scale = dst_height / dst_height_scaled
    dst_keypoints_scaled = []
    for keypoint in dst_keypoints:
        x = (keypoint.pt[0] * dst_x_scale) * align_scale
        y = (keypoint.pt[1] * dst_y_scale) * align_scale
        size = keypoint.size * max(dst_x_scale, dst_y_scale) * align_scale
        dst_keypoints_scaled.append(cv2.KeyPoint(x, y, size))

    dst_keypoints = dst_keypoints_scaled

    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if align_method == "transform":
        #matrix = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)[0]
        matrix = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)[0]
    elif align_method == "warp":
        #matrix = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)[0]
        matrix = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)[0]

    if matrix is None:
        return None

    if align_method == "transform":
        if scale > 1.0:
            return cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA) # Use INTER_AREA interpolation as it's the most accurate of the ones available in OpenCV (as of the time when writing this)
        else:
            # if scale is 1.0, warp and resize down to the desired size
            return np.array(
                Image.fromarray(
                    cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA)
                ).resize(
                    (int(dst_width * scale), int(dst_height * scale)),
                    resample=Image.Resampling.BOX #LANCZOS
                )
            )
            #return cv2.resize(cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)
    elif align_method == "warp":
        if scale > 1.0:
            return cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA) # Use INTER_AREA interpolation as it's the most accurate of the ones available in OpenCV (as of the time when writing this)
        else:
            # if scale is 1.0, warp and resize down to the desired size
            return np.array(Image.fromarray(cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), resample=Image.Resampling.BOX))
            #return cv2.resize(cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)

    return None

if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        prog="twimager.py",
        description="A script to find matching images and align them.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--hr",
        help="Path to folder containing HR (target) images",
        required=True,
        type=str
    )
    parser.add_argument(
        "--lr",
        help="Path to folder containing LR (reference) images which to align the HR image to",
        required=True,
        type=str
    )
    parser.add_argument(
        "--dest",
         help="Path to folder where images should be written.",
         required=False,
         type=str,
         default=None
    )
    parser.add_argument(
        "--output",
        help="What to write to the desination folder (default: all)\n" +
             "- lr - Only LR images\n" +
             "- hr - Only the HR images\n" +
             "- all - Both LR and HR (the folders LR and HR will be created in the destination folder if needed)\n\n",
        required=False,
        type=str,
        default="all",
        choices=["lr", "hr", "all"]
    )
    parser.add_argument(
        "--hash_size",
        help="Hash size to use when comparing aligned images (default: 32).\n\n" +
             "The default should work on most images. If there are a lot of misses due to slight detail differences, raising it to 48 or 64\n" +
             "could help. As long as the HR and/or LR images aren't too degraded.\n\n",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "--threshold",
        help="Aligned image match threshold (25-100), if the perceptual distance between the LR and aligned HR image is below the threshold it won't be saved (default: 90)\n\n" +
             "The default value should work for most scenarios, as long as the LR and/or HR images aren't overly noisy, lowering it might help.\n" +
             "Please note that setting the threshold too low might cause some aligned images to falsly be deemed successful.\n\n",
        required=False,
        type=int,
        default=90,
        metavar="[25-100]"
    )
    parser.add_argument(
        "--scale",
        help="Aligned image scale. The size of aligned HR images will be a multiplier of SCALE and the matched LR image size (default: 2.0)",
        required=False,
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--crop",
        help="Crop the LR image before aligning.\n" +
            "- Use \"--crop 10\" and 10 pixels will be removed around the image.\n" +
            "- Use \"--crop 640 480\" and the image will be cropped to 640 width and 480 height from the center.\n" +
            "- Use \"--crop 10 20 400\" 300 and the image will be cropped to 400 width and 300 height starting 10 pixels from the top of the image and 20 pixels from the left.\n\n",
        required=False,
        type=int,
        default=False,
        nargs="+"
    )
    parser.add_argument(
        "--align_method",
        help="What method to use when aligning images, transform or warp (default: transform)\n" +
             " - transform - Use affine transformation to align image\n" +
             " - warp - Use perspective transform to align image\n\n",
        required=False,
        type=str,
        default="transform",
        choices=["transform", "warp"]
    )
    parser.add_argument(
        "--align_rife",
        help="Additional alignment pass using rife",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--match_method",
        help="What method to use when finding matching points between images, knn or bf (default: bf)\n" +
             " - bf - Brute force matching with cross checking\n" +
             " - knn - Use affine transformation to align image (slightly faster but less accurate)\n\n",
        required=False,
        type=str,
        default="bf",
        choices=["bf", "knn"]
    )
    parser.add_argument(
        "--color_transfer",
        help="Transfer colors from the LR image to the aligned HR image",
        required=False,
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--filename",
        help="What filename to use when saving the matched/aligned images (default: hr)\n" +
             "- hr - Filename will be the same as the input HR\n" +
             "- lr - Filename will be the same as the input LR\n" +
             "- keep - Filename will be the same as the input filename\n\n",
        type=str,
        required=False,
        default="hr",
        choices=["hr", "lr", "keep"]
    )
    parser.add_argument(
        "--prefix",
        help="Prefix to append to the filename when writing images.",
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        "--suffix",
        help="Suffix to append to the filename when writing images.",
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        "--limit",
        help="Limit the amount of aligned images, please note that this isn't exact due to threading",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "--save",
        help="Save generated hashes to file. Filename will be \"twimager_{basename of the LR or HR folder}.pickle\"",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--report",
        help="Save a report in the destination folder, the report can be used by twimager-review.",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--debug",
        help="Bring up a window showing the results when image alignment is successful. Press \"esc\" to abort the alignment process and \"n\" or \"del\" to skip the image. Pressing any other key will accept. Threading will be disabled when this option is used",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--video_skip",
        help="If LR or HR input is a video, use this to skip this many frames from the start and end of the video.",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--dry",
        help="Dry run. No images will be written. Only find matching images and attempt to align them.",
        action="store_true",
        required=False,
        default=False
    )

    args = parser.parse_args()

    #
    # deprecated/changed
    #
    if hasattr(args, "method"):
        parser.error("The --method option has been renamed to --align-method, please change your command line input.")
        sys.exit(0)

    #
    # evaluate args and store options
    #
    opt_lr = os.path.abspath(args.lr)
    opt_hr = os.path.abspath(args.hr)

    opt_output = args.output

    if args.threshold < 25 or args.threshold > 100:
        parser.error("Threshold value can't be below 25 or above 100, please adjust it.")

    opt_threshold = args.threshold

    opt_dest = os.path.abspath(args.dest) if args.dest else None

    opt_crop = False
    opt_crop_border = False
    opt_crop_size = False
    opt_crop_coordinates = False
    if args.crop:
        opt_crop = True
        if len(args.crop) == 1:
            opt_crop_border = args.crop
            opt_crop_type = "border"
        elif len(args.crop) == 2:
            opt_crop_size = args.crop
            opt_crop_type = "center"
        elif len(args.crop) == 4:
            opt_crop_coordinates = args.crop
            opt_crop_type = "coordinates"
        else:
            parser.error(
                "Invalid crop args, --crop only accepts one, two or four arguments.\n" +
                "Usage:\n" +
                "- Use \"--crop 10\" and 10 pixels will be removed around the image.\n" +
                "- Use \"--crop 640 480\" and the image will be cropped to 640 width and 480 height from the center.\n" +
                "- Use \"--crop 10 20 400 300\" and the image will be cropped to 400 width and 300 height starting 10 pixels from the  of the image and 20 pixels from the left.\n",
            )
    else:
        opt_crop_type = None

    opt_align_method = args.align_method
    opt_align_rife = args.align_rife
    opt_match_method = args.match_method

    opt_scale = 1.0
    if args.scale:
        opt_scale = args.scale

    opt_color_transfer = args.color_transfer
    opt_save = args.save
    opt_report = args.report
    opt_debug = args.debug
    opt_limit = args.limit

    opt_filename = args.filename
    opt_prefix = False
    if args.prefix:
        opt_prefix = True
        opt_prefix_string = args.prefix

    opt_suffix = False
    if args.suffix:
        opt_suffix = True
        opt_suffix_string = args.suffix

    opt_dry = args.dry
    if opt_dry and opt_report:
        print("Running --dry together with --report isn't supported, report will be disabled...")
        opt_report = False

    if not opt_dry and not opt_dest:
        parser.error("--dest is required unless --dry is used, exiting...")
        sys.exit()

    opt_video_skip = args.video_skip

    opt_dest_lr = os.path.join(opt_dest, "LR") if opt_output == "all" else opt_dest
    opt_dest_hr = os.path.join(opt_dest, "HR") if opt_output == "all" else opt_dest

    SPACING = 25

    HASH_SIZE = 8
    HASH_SIZE_ALIGNED = args.hash_size

    # A hamming distance of 10 is usually considered to be the same image when doing perceptual image matching with a hash size
    # of 8. But we wish to be slightly more generous since we wish to be able to also match heavily degraded images.
    DISTANCE_THRESHOLD_MATCH = 11
    DISTANCE_THRESHOLD_ALIGNED = int((HASH_SIZE_ALIGNED ** 2) * (1 - (opt_threshold / 100)) + 0.5) if opt_threshold > 0 and opt_threshold < 100 else None # 360

    print("Path to HR images/video".ljust(SPACING), ":", args.hr)
    print("Path to LR images/video".ljust(SPACING), ":", args.lr)
    print("Destination path".ljust(SPACING), ":", args.dest)
    print("Output".ljust(SPACING), ":", {"hr": "hr (HR Images)", "lr": "lr (LR images)", "all": "all (LR and HR images)"}[args.output])
    print("Threshold".ljust(SPACING), ":", args.threshold)
    print("Hash size".ljust(SPACING), ":", args.hash_size)
    print("Scale".ljust(SPACING), ":", args.scale)
    print("Crop".ljust(SPACING), ":", args.crop, opt_crop_type)
    print("Align Method".ljust(SPACING), ":", args.align_method)
    print("Rife Align".ljust(SPACING), ":", args.align_rife)
    print("Match Method".ljust(SPACING), ":", args.match_method)
    print("Color transfer".ljust(SPACING), ":", args.color_transfer)
    print("Filename".ljust(SPACING), ":", {"hr": "Use filename from HR image/video", "lr": "Use filename from LR image/video", "keep": "Keep the filenames from th HR and LR images"}[args.filename])
    print("File name prefix".ljust(SPACING), ":", args.prefix)
    print("File name suffix".ljust(SPACING), ":", args.suffix)
    print("Limit".ljust(SPACING), ":", args.limit)
    print("Save hash tables".ljust(SPACING), ":", args.save)
    print("Debug mode".ljust(SPACING), ":", args.debug, {True: "(threading will be disabled)", False: ""}[args.debug])
    print("Dry run".ljust(SPACING), ":", args.dry)

    print("-----")

    if not ask("Continue?", True):
        sys.exit()

    #
    # check provided paths
    #
    if not os.path.exists(opt_hr):
        print("HR folder doesn't exist, exiting...")
        sys.exit()

    if os.path.isdir(opt_hr):
        opt_hr_type = "images"
    elif os.path.isfile(opt_hr) and opt_hr.endswith(EXTENSIONS_VIDEO):
        opt_hr_type = "video"
    else:
        print("HR must be a folder or video, exiting...")
        sys.exit()

    if os.path.isdir(opt_lr):
        opt_lr_type = "images"
    elif os.path.isfile(opt_lr) and opt_lr.endswith(EXTENSIONS_VIDEO):
        opt_lr_type = "video"
    else:
        print("LR must be a folder or video, exiting... ")
        sys.exit()

    if not opt_dry:
        if not os.path.exists(opt_dest):
            try:
                os.makedirs(opt_dest, exist_ok=True)
            except Exception as error:
                print("Destination folder doesn't exist and couldn't be created, exiting...")
                print(error)
                sys.exit()

        if os.path.exists(opt_dest):
            if opt_output == "all":
                if os.path.exists(opt_dest_hr) and os.listdir(opt_dest_hr) or os.path.exists(opt_dest_lr) and os.listdir(opt_dest_lr):
                    if ask("Destination isn't empty, delete existing files before proceeding?", False):
                        shutil.rmtree(opt_dest_hr)
                        shutil.rmtree(opt_dest_lr)
            else:
                if os.path.exists(opt_dest) and os.listdir(opt_dest):
                    if ask("Destination isn't empty, delete existing files before proceeding?", False):
                        shutil.rmtree(opt_dest_hr)

            if opt_output == "all":
                os.makedirs(opt_dest_hr, exist_ok=True)
                os.makedirs(opt_dest_lr, exist_ok=True)
            else:
                os.makedirs(opt_dest, exist_ok=True)

    hr_hash_file = os.path.join(os.getcwd(), f"twimager_{os.path.basename(opt_hr)}.pickle")
    lr_hash_file = os.path.join(os.getcwd(), f"twimager_{os.path.basename(opt_lr)}.pickle")

    #print(hr_hash_file)
    #print(lr_hash_file)

    """
    hr_files = [
        file
        for file in os.listdir(opt_hr)
        if file.lower().endswith(EXTENSIONS_IMAGE)
    ]

    lr_files = [
        file
        for file in os.listdir(opt_lr)
        if file.lower().endswith(EXTENSIONS_IMAGE)
    ]
    """

    hr_items = None
    lr_items = None

    if os.path.isfile(hr_hash_file) and os.path.exists(lr_hash_file):
        if ask("HR and LR hash files detected, do you wish to load them?", True):
            try:
                with open(hr_hash_file, "rb") as f:
                    hr_items = pickle.load(f)

                    #for hr_file in list(hr_items.values()):
                    #    if hr_file not in hr_files:
                    #        raise ValueError
            except ValueError:
                hr_items = None
                print(f"Invalid HR dictionary detected, recreation neccesary.")

            except Exception:
                print("HR dictionary file couldn't be loaded, recreation neccesary.")

            else:
                print("HR dictionary file loaded.")

                if opt_hr_type == "video":
                    hr_container = av.open(opt_hr, "r")

            try:
                with open(lr_hash_file, "rb") as f:
                    lr_items = pickle.load(f)

                    #for lr_file in list(lr_items.values()):
                    #    if lr_file not in lr_files:
                    #        raise ValueError
            except ValueError:
                lr_items = None
                print(f"Invalid LR dictionary detected, rereation neccesary.")

            except Exception:
                print("LR dictionary file couldn't be loaded, recreation neccesary.")
                sys.exit(0)


            else:
                print("LR dictionary file loaded.")

                if opt_lr_type == "video":
                    lr_container = av.open(opt_lr, "r")


    elif os.path.isfile(hr_hash_file) and ask("HR hash file detected, do you wish to load it?"):
        try:
            with open(hr_hash_file, "rb") as f:
                hr_items = pickle.load(f)

                #for hr_file in list(hr_items.values()):
                #    if hr_file not in hr_files:
                #        raise ValueError

        except ValueError:
            hr_items = None
            print(f"Invalid HR dictionary detected, recreation neccesary.")


        except Exception:
            print("HR dictionary file couldn't be loaded")

        else:
            print("HR dictionary file loaded.")

            if opt_hr_type == "video":
                hr_container = av.open(opt_hr, "r")

    elif os.path.isfile(lr_hash_file) and ask("LR hash file detected, do you wish to load it?"):
        try:
            with open(lr_hash_file, "rb") as f:
                lr_items = pickle.load(f)

                #for lr_file in list(lr_items.values()):
                #    if lr_file not in lr_files:
                #        raise ValueError

        except ValueError:
            lr_items = None
            print(f"Invalid LR dictionary detected, recreation neccesary.")


        except Exception:
            print("LR dictionary file couldn't be loaded")

        else:
            print("LR dictionary file loaded.")

            if opt_lr_type == "video":
                lr_container = av.open(opt_lr, "r")

    #
    # hr/source images
    #
    if hr_items is None:
        hr_items = {}

        if opt_hr_type == "images":
            hr_files = [
                file
                for file in os.listdir(opt_hr)
                if file.lower().endswith(EXTENSIONS_IMAGE)
            ]

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_file, os.path.join(opt_hr, file), scale=opt_scale, cut=True, blur=False) for file in hr_files}
                kwargs = {
                    "desc": "Generating hashes for HR images",
                    "total": len(futures),
                    "unit": "it",
                    "unit_scale": False,
                    "leave": True
                }
                try:
                    for f in tqdm(as_completed(futures), **kwargs):
                        result = f.result()
                        if result:
                            hash, key = result
                            hr_items[hash] = key

                except KeyboardInterrupt:
                    print("User interrupt, exiting...")
                    executor.shutdown(wait=False)
                    for future in futures:
                        future.cancel()
                    sys.exit()

            del hr_files

        elif opt_hr_type == "video":
            opt_hr_video_type, opt_hr_video_filter = analyze_video_type(opt_hr)

            hr_keyframe_map = []

            hr_container = av.open(opt_hr, "r")

            hr_stream = hr_container.streams.video[0]
            hr_stream.thread_type = "AUTO"

            """
            if opt_hr_video_filter is not None:
                print(f"{opt_hr_video_type} detected, enabling deinterlacing...")
                graph = av.filter.Graph()
                link = graph.add_buffer(template=hr_stream)

                filt = graph.add(opt_hr_video_filter)
                link.link(filt)
                link = filt

                sink = graph.add("buffersink")
                link.link(sink)
                graph.configure()
            """

            frame_index = 0
            frame_count = hr_stream.frames

            # If frame_count is 0 we've encountered a "broken" video container, count manually.
            if frame_count == 0:
                print(f"HR video metadata doesn't contain a frame count, analyzing packets...")

                for packet in hr_container.demux(video=0):
                    if packet.dts is not None:
                        frame_count += 1
                    #hr_container.close()

                hr_container.seek(0, stream=hr_stream)

            if opt_video_skip:
                frame_start = opt_video_skip
                frame_end = frame_count - (opt_video_skip * 2)

                frame_count = frame_end
            else:
                frame_start = None
                frame_end = None

            with tqdm(total=frame_count, desc="Generating hashes for HR video frames") as pbar:
                for frame in hr_container.decode(video=0):
                    pbar.update(1)

                    if frame.key_frame:
                        hr_keyframe_map.append((frame_index, frame.pts))

                    if frame_start and frame_index < frame_start:
                        frame_index += 1
                        continue

                    if frame_end and frame_index > frame_end:
                        break

                    # Convert to ndarray in BGR 24 format
                    hash = Hash.create(frame.to_ndarray(format="bgr24"), HASH_SIZE) # or rgb24

                    # Filter out frames which are near duplicates, we're more harsh with HR than LR since HR will generally be "cleaner" and thus provide less false duplicates
                    if not hash or any(hamming_distance(hash, previous_hash) <= 16 for previous_hash in hr_items.keys()):
                        frame_index += 1
                        continue

                    hr_items[hash] = frame_index

                    frame_index += 1

            hr_container.seek(0, stream=hr_stream)
            # Don't close this here. We'll repurpose it for the image matching.
            #hr_container.close()

        if opt_save:
            try:
                with open(hr_hash_file, "wb") as f:
                    pickle.dump(hr_items, f)

            except Exception as error:
                print("HR hashes couldn't be saved:", error)

            else:
                print("HR hashes saved to", hr_hash_file)

    #
    # lr/target images
    #
    if lr_items is None:
        lr_items = {}

        if opt_lr_type == "images":
            lr_files = [
                file
                for file in os.listdir(opt_lr)
                if file.lower().endswith(EXTENSIONS_IMAGE)
            ]

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_file, os.path.join(opt_lr, file), scale=1, cut=False, blur=False) for file in lr_files}
                kwargs = {
                    "desc": "Generating hashes for LR images",
                    "total": len(futures),
                    "unit": "it",
                    "unit_scale": False,
                    "leave": True
                }
                try:
                    for f in tqdm(as_completed(futures), **kwargs):
                        result = f.result()
                        if result:
                            hash, key = result
                            lr_items[hash] = key
                except KeyboardInterrupt:
                    print("User interrupt, exiting...")
                    executor.shutdown(wait=False)
                    for future in futures:
                        future.cancel()
                    sys.exit()

            """
            if opt_save:
                try:
                    with open(lr_hash_file, "wb") as f:
                        pickle.dump(lr_items, f)

                except Exception as error:
                    print("LR hashes couldn't be saved:", error)

                else:
                    print("LR hashes saved to", lr_hash_file)

            for lr_file in list(lr_items.values()):
                if lr_file not in lr_files:
                    print(f"Missmatch detected between available files and the ones in the LR dictionary. Please recreate it by running twimager with the --save option before attempting to load it again.")
                    sys.exit(0)
            """

            del lr_files

        elif opt_lr_type == "video":
            opt_lr_video_type, opt_lr_video_filter = analyze_video_type(opt_lr)

            lr_keyframe_map = []

            lr_container = av.open(opt_lr, "r")

            lr_stream = lr_container.streams.video[0]
            lr_stream.thread_type = "AUTO"

            """
            if opt_lr_video_filter is not None:
                print(f"{opt_lr_video_type} detected, enabling deinterlacing...")
                graph = av.filter.Graph()
                link = graph.add_buffer(template=lr_stream)

                filt = graph.add(opt_lr_video_filter)
                link.link(filt)
                link = filt

                sink = graph.add("buffersink")
                link.link(sink)
                graph.configure()
            """

            # Iterate and convert
            frame_index = 0
            frame_count = lr_stream.frames
            if frame_count == 0:
                print(f"LR video metadata doesn't contain a frame count, analyzing packets...")

                for packet in lr_container.demux(video=0):
                    if packet.dts is not None:
                        frame_count += 1
                        #lr_container.close()

                lr_container.seek(0, stream=lr_stream)

            if opt_video_skip:
                frame_start = opt_video_skip
                frame_end = frame_count - (opt_video_skip * 2)

                frame_count = frame_end
            else:
                frame_start = None
                frame_end = None

            with tqdm(total=frame_count, desc="Generating hashes for LR video frames") as pbar:
                for frame in lr_container.decode(video=0):
                    pbar.update(1)

                    if frame.key_frame:
                        lr_keyframe_map.append((frame_index, frame.pts))

                    if frame_start and frame_index < frame_start:
                        frame_index += 1
                        continue

                    if frame_end and frame_index > frame_end:
                        break

                    # Convert to ndarray in BGR 24 format
                    hash = Hash.create(frame.to_ndarray(format="bgr24"), HASH_SIZE) # or rgb24

                    # Filter out frames which are near duplicates, we're less harsh with LR than HR as LR usually have more artifacts which can skew the results
                    if not hash or any(hamming_distance(hash, previous_hash) <= 6 for previous_hash in lr_items.keys()):
                        frame_index += 1
                        continue

                    lr_items[hash] = frame_index

                    frame_index += 1

            lr_container.seek(0, stream=lr_stream)
            # Don't close this here. We'll repurpose it for the image matching.
            #lr_container.close()

        if opt_save:
            try:
                with open(lr_hash_file, "wb") as f:
                    pickle.dump(lr_items, f)

            except Exception as error:
                print("LR hashes couldn't be saved:", error)

            else:
                print("LR hashes saved to", lr_hash_file)

    lr_tree = vptree.VPTree(list(lr_items.keys()), hamming_distance)
    print("Vantage-point tree created.")

    info = "Aligned image estimation"

    if opt_lr_type == "video":
        print("LR frames:", len(lr_items))
    else:
        print("LR images:", len(lr_items))

    if opt_hr_type == "video":
        print("HR frames:", len(hr_items))
    else:
        print("HR images:", len(hr_items))

    count = 0

    if opt_report:
        report = {
            "options": {
                "hr": opt_hr,
                "lr": opt_lr,
                "dest": opt_dest,
                "output": opt_output,
                "scale": opt_scale,
                "crop": opt_crop,
                "align_method": opt_align_method,
                "match_method": opt_match_method,
                "color_transfer": opt_color_transfer,
                "filename": opt_filename,
                "prefix": opt_prefix,
                "suffix": opt_suffix,
                "limit": opt_limit,
                "save": opt_save,
                "debug": opt_debug,
                "dry": opt_dry
            },
            "images": []
        }

    if not opt_debug:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(find_image_match, hr_item) for hr_item in hr_items}
            kwargs = {
                "total": len(futures),
                "unit": "it",
                "unit_scale": False,
                "leave": True
            }
            progress = tqdm(as_completed(futures), **kwargs)
            progress.set_description(f"{info}: {count} -")

            try:
                for f in progress:
                    try:
                        result_count, result_report = f.result()
                        count += result_count

                        if result_report:
                            report["images"].append(result_report)

                        progress.set_description(f"{info}: {count} -")

                        if opt_limit and count >= opt_limit:
                            progress.close()

                            print("Limit reached, exiting...")

                            executor.shutdown(wait=False)
                            for future in futures:
                                future.cancel()
                            sys.exit()

                    except Exception as error:
                        print("An error occurred:", error)
                        executor.shutdown(wait=False)
                        for future in futures:
                            future.cancel()
                        sys.exit()

            except KeyboardInterrupt:
                print("User interrupt, exiting...")
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                sys.exit()
    else:
        try:
            progress = tqdm(list(hr_items.keys()))
            progress.set_description(f"{info}: {count} -")

            for hr_item in progress:
                result_count, result_report = find_image_match(hr_item)
                count += result_count

                if result_report:
                    report["images"].append(result_report)

                progress.set_description(f"{info}: {count} -")

                if opt_limit and count >= opt_limit:
                    progress.close()

                    print("Limit reached, exiting...")

                    sys.exit()

        except KeyboardInterrupt:
            print("User interrupt, exiting...")
            sys.exit()

if opt_report:
    with open(os.path.join(opt_dest, f"twimager_report.json"), "w", encoding="utf8") as f:
        json.dump(report, f, indent=4)

#if not opt_dry:
print(f"Aligned images count: {len(os.listdir(opt_dest_lr))}")
print(f"Time taken: {format_time(time.perf_counter() - start_time)}")
