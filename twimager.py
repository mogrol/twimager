import os
import sys
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import vptree
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from imagehash import phash
import json

EXTENSIONS_IMAGE = (".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff")

def ask(question, default=False):
    while True:
        if default:
            return input(f"{question} [Y/n]: ").lower() not in ("n", "no")
        else:
            return input(f"{question} [y/N]: ").lower() in ("y", "yes")

def color_transfer(target=None, reference=None):
    if target is None or reference is None:
        return None

    mean_in = np.mean(target, axis=(0, 1), keepdims=True)
    mean_ref = np.mean(reference, axis=(0, 1), keepdims=True)
    std_in = np.std(target, axis=(0, 1), keepdims=True)
    std_ref = np.std(reference, axis=(0, 1), keepdims=True)
    img_arr_out = (target - mean_in) / std_in * std_ref + mean_ref
    img_arr_out[img_arr_out < 0] = 0
    img_arr_out[img_arr_out > 255] = 255

    return img_arr_out.astype("uint8")

def create_hash(image, hash_size=8, blur=False):
    try: 
        image = Image.fromarray(image).filter(ImageFilter.GaussianBlur(3)) if blur else Image.fromarray(image)
        return phash(image, hash_size)
    except Exception:
        return None

def hamming_distance(a, b):
    return a - b

def load_image(file):
    try:
        return cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[..., :3]
    except Exception:
        return None

def crop_image(image, file=None, scale=1.0):
    height, width = image.shape[:2]

    if opt_crop_border:
        crop_pixels = opt_crop_border * scale * 2

        if crop_pixels < width and crop_pixels < height:
            return image[int(opt_crop_border * scale):-int(opt_crop_border * scale), int(opt_crop_border * scale):-int(opt_crop_border * scale)]
        else:
            print(f"Image dimensions of {file} is smaller than crop border. Image skipped...")
            return None

    elif opt_crop_size:
        x = int(((width - (opt_crop_size[1] * scale)) * 0.5))
        y = int(((height - (opt_crop_size[0] * scale)) * 0.5))

        return image[y:y + int(opt_crop_size[1] * scale), x:x + int(opt_crop_size[0] * scale)]

    elif opt_crop_coordinates:
        return image[int(opt_crop_coordinates[0] * scale):int(opt_crop_coordinates[3] * scale), int(opt_crop_coordinates[1] * scale):int(opt_crop_coordinates[2] * scale)]

    return image

def process_file(file, target_list, cut=False, blur=False):
    image = load_image(file)
    if image is None:
        return

    if cut:
        height, width = image.shape[:2]

        ratio = width / height

        # if the aspect ratio is off by 25%, cut out a part of it and use to create an additional hash
        # this helps with finding HR images which are taller or wider than the LR image.
        if ratio < 0.75: #0.75:
            offset = int((height - width) * 0.5)
            hash = create_hash(image[offset:offset+width, 0:width], HASH_SIZE, blur)
            if hash:
                target_list[hash] = os.path.basename(file)
        elif ratio > 1.35: #1.25:
            offset = int((width - height) * 0.5)
            hash = create_hash(image[0:height, offset:offset+height], HASH_SIZE, blur)
            if hash:
                target_list[hash] = os.path.basename(file)

    hash = create_hash(image, HASH_SIZE)
    if hash:
        target_list[hash] = os.path.basename(file)

def find_image_match(hr_hash):
    report = None
    results = lr_tree.get_all_in_range(hr_hash, DISTANCE_THRESHOLD_MATCH)

    if not results:
        return 0, report

    hr_file = hr_items[hr_hash]
    hr_image = load_image(os.path.join(opt_hr, hr_items[hr_hash]))
    if hr_image is None:
        return 0, report

    previous_distance = None
    aligned_hr_image = None
    aligned_lr_image = None

    for result in results:
        lr_hash = result[1]
        lr_file = lr_items[lr_hash]
        lr_image = load_image(os.path.join(opt_lr, lr_file))
        if lr_image is None:
            continue

        hr_image_aligned = align_image(hr_image, lr_image, opt_scale, align_method=opt_align_method, match_method=opt_match_method)
        if hr_image_aligned is None:
            continue

        if opt_crop:
            lr_image = crop_image(lr_image, file=lr_file)
            hr_image_aligned = crop_image(hr_image_aligned, file=hr_file, scale=opt_scale)

        aligned_distance = hamming_distance(
            create_hash(lr_image, HASH_SIZE_ALIGNED,),
            create_hash(hr_image_aligned, HASH_SIZE_ALIGNED)
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
        lr_dest_name = hr_dest_name = os.path.splitext(hr_file)[0]
    elif opt_filename == "lr":
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

def align_image(src, dst, scale=1.0, align_method="transform", match_method="bf"):
    MIN_SCORE = 20
    GOOD_MATCH_PERCENT = 0.7
    RESIZE = 768

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
    
    src_height, src_width = src.shape[:2]
    dst_height, dst_width = dst.shape[:2]

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
    src_gray = np.array(Image.fromarray(src).convert("L", dither=Image.Dither.NONE).resize((src_width_scaled, src_height_scaled), resample=Image.Resampling.LANCZOS))
    #cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (src_width_scaled, src_height_scaled), interpolation=cv2.INTER_AREA)
    dst_gray = np.array(Image.fromarray(dst).convert("L", dither=Image.Dither.NONE).resize((dst_width_scaled, dst_height_scaled), resample=Image.Resampling.LANCZOS))
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
            return np.array(Image.fromarray(cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), resample=Image.Resampling.LANCZOS))
            #return cv2.resize(cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)
    elif align_method == "warp":
        if scale > 1.0:
            return cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA) # Use INTER_AREA interpolation as it's the most accurate of the ones available in OpenCV (as of the time when writing this)
        else:
            # if scale is 1.0, warp and resize down to the desired size
            return np.array(Image.fromarray(cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), resample=Image.Resampling.LANCZOS))
            #return cv2.resize(cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)

    return None

if __name__ == "__main__":
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
    opt_dest_lr = os.path.join(opt_dest, "LR") if opt_output == "all" else opt_dest
    opt_dest_hr = os.path.join(opt_dest, "HR") if opt_output == "all" else opt_dest

    opt_crop = False
    opt_crop_border = False
    opt_crop_size = False
    opt_crop_coordinates = False
    if args.crop:
        opt_crop = True
        if len(args.crop) == 1:
            opt_crop_border = args.crop
        elif len(args.crop) == 2:
            opt_crop_size = args.crop
        elif len(args.crop) == 4:
            opt_crop_coordinates = args.crop
        else:
            parser.error(
                "Invalid crop args, --crop only accepts one, two or four arguments.\n" +
                "Usage:\n" +
                "- Use \"--crop 10\" and 10 pixels will be removed around the image.\n" +
                "- Use \"--crop 640 480\" and the image will be cropped to 640 width and 480 height from the center.\n" +
                "- Use \"--crop 10 20 400\" 300 and the image will be cropped to 400 width and 300 height starting 10 pixels from the left of the image and 20 pixels from the top.\n",
            )

    opt_align_method = args.align_method
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

    SPACING = 20

    HASH_SIZE = 8
    HASH_SIZE_ALIGNED = args.hash_size

    # A hamming distance of 10 is usually considered to be the same image when doing perceptual image matching with a hash size
    # of 8. But we wish to be slightly more generous since we wish to be able to also match heavily degraded images.
    DISTANCE_THRESHOLD_MATCH = 11
    DISTANCE_THRESHOLD_ALIGNED = int((HASH_SIZE_ALIGNED ** 2) * (1 - (opt_threshold / 100)) + 0.5) if opt_threshold > 0 and opt_threshold < 100 else None # 360

    print("Path to HR images".ljust(SPACING), ":", args.hr)
    print("Path to LR images".ljust(SPACING), ":", args.lr)
    print("Destination path".ljust(SPACING), ":", args.dest)
    print("Output".ljust(SPACING), ":", {"hr": "hr (HR Images)", "lr": "lr (LR images)", "all": "all (LR and HR images)"}[args.output])
    print("Threshold".ljust(SPACING), ":", args.threshold)
    print("Hash size".ljust(SPACING), ":", args.hash_size)
    print("Scale".ljust(SPACING), ":", args.scale)
    print("Crop".ljust(SPACING), ":", args.crop)
    print("Align Method".ljust(SPACING), ":", args.align_method)
    print("Match Method".ljust(SPACING), ":", args.match_method)
    print("Color transfer".ljust(SPACING), ":", args.color_transfer)
    print("Filename".ljust(SPACING), ":", {"hr": "Use filename from HR image", "lr": "Use filename from LR image", "keep": "Keep the filenames from th HR and LR images"}[args.filename])
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

    if not os.path.exists(opt_lr):
        print("LR folder doesn't exist, exiting...")
        sys.exit()

    #if opt_dest and not os.path.exists(opt_dest) and not os.makedirs(opt_dest, exist_ok=True):
    #    print("Destination folder doesn't exist and couldn't be created, exiting...")
    #    sys.exit()
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

    hr_items = None
    lr_items = None

    if os.path.exists(hr_hash_file) and os.path.exists(lr_hash_file):
        if ask("HR and LR hash files detected, do you wish to load them?", True):
            try:
                with open(hr_hash_file, "rb") as f:
                    hr_items = pickle.load(f)

                    for hr_file in list(hr_items.values()):
                        if hr_file not in hr_files:
                            raise ValueError
            except ValueError:
                hr_items = None
                print(f"Invalid HR dictionary detected, recreation neccesary.")
            except Exception:
                print("HR dictionary file couldn't be loaded, recreation neccesary.")
            else:
                print("HR dictionary file loaded.")

            try:
                with open(lr_hash_file, "rb") as f:
                    lr_items = pickle.load(f)

                    for lr_file in list(lr_items.values()):
                        if lr_file not in lr_files:
                            raise ValueError
            except ValueError:
                lr_items = None
                print(f"Invalid LR dictionary detected, rereation neccesary.")
            except Exception:
                print("LR dictionary file couldn't be loaded, recreation neccesary.")
                sys.exit(0)
            else:
                print("LR dictionary file loaded.")
    elif os.path.exists(hr_hash_file) and ask("HR hash file detected, do you wish to load it?"):
        try:
            with open(hr_hash_file, "rb") as f:
                hr_items = pickle.load(f)

                for hr_file in list(hr_items.values()):
                    if hr_file not in hr_files:
                        raise ValueError
        except ValueError:
            hr_items = None
            print(f"Invalid HR dictionary detected, recreation neccesary.")
        except Exception:
            print("HR dictionary file couldn't be loaded")
        else:
            print("HR dictionary file loaded.")
    elif os.path.exists(lr_hash_file) and ask("LR hash file detected, do you wish to load it?"):
        try:
            with open(lr_hash_file, "rb") as f:
                lr_items = pickle.load(f)

                for lr_file in list(lr_items.values()):
                    if lr_file not in lr_files:
                        raise ValueError
        except ValueError:
            lr_items = None
            print(f"Invalid LR dictionary detected, recreation neccesary.")
        except Exception:
            print("LR dictionary file couldn't be loaded")
        else:
            print("LR dictionary file loaded.")

    #
    # hr/source images
    #
    if hr_items is None:
        hr_items = {}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_file, os.path.join(opt_hr, file), hr_items, True) for file in hr_files}
            kwargs = {
                "desc": "Generating hashes for HR images",
                "total": len(futures),
                "unit": "it",
                "unit_scale": False,
                "leave": True
            }
            try:
                for f in tqdm(as_completed(futures), **kwargs):
                    f.result()
            except KeyboardInterrupt:
                print("User interrupt, exiting...")
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                sys.exit()

        if opt_save:
            try:
                with open(hr_hash_file, "wb") as f:
                    pickle.dump(hr_items, f)
            except Exception as error:
                print("HR hashes couldn't be saved:", error)
            else:
                print("HR hashes saved to", hr_hash_file)

    del hr_files

    #
    # lr/target images
    #
    if lr_items is None:
        lr_items = {}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_file, os.path.join(opt_lr, file), lr_items, False, blur=False) for file in lr_files}
            kwargs = {
                "desc": "Generating hashes for LR images",
                "total": len(futures),
                "unit": "it",
                "unit_scale": False,
                "leave": True
            }
            try:
                for f in tqdm(as_completed(futures), **kwargs):
                    f.result()
            except KeyboardInterrupt:
                print("User interrupt, exiting...")
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                sys.exit()


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

    del lr_files

    lr_tree = vptree.VPTree(list(lr_items.keys()), hamming_distance)
    print("Vantage-point tree created.")

    info = "Aligned image estimation"

    print("LR Images:", len(lr_items))
    print("HR Images:", len(hr_items))

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

if not opt_dry:
    print(f"Done, aligned image count: {len(os.listdir(opt_dest_lr))}")
