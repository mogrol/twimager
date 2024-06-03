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
from PIL import Image
#import scipy.fftpack
from imagehash import phash, dhash, average_hash

# A hash size of 8 is in most cases enough to identify image matches # and
# to check if alignment it successful. It'll only potentially fail when there
# are two images which are already very similar.
HASH_SIZE = 8
HASH_SIZE_ALIGNED = 32

# Use hamming distance 14 to catch images which are event remotely similar
# and distance 10 when doing the comparison between the paired and aligned image
DISTANCE_THRESHOLD_MATCH = 15
DISTANCE_THRESHOLD_ALIGNED = 60

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

def create_hash(image, hash_size=4, hash_type="phash"):
    try: 
        if hash_type == "phash":
            return phash(Image.fromarray(image), hash_size)
        elif hash_type == "dhash":
            return dhash(Image.fromarray(image), hash_size)
        elif hash_type == "average_hash":
            return average_hash(Image.fromarray(image), hash_size)
        else:
            return None
    except Exception:
        return None

def hamming_distance(a, b):
    return a - b

def load_image(file):
    try:
        return cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[:,:,:3]
    except Exception:
        return None

def crop_image(image, scale=1.0):
    if opt_crop_border:
        return image[int(opt_crop_border * scale):-int(opt_crop_border * scale), int(opt_crop_border * scale):-int(opt_crop_border * scale)]

    elif opt_crop_size:
        x = int(((image.shape[1] - (opt_crop_size[1] * scale)) * 0.5))
        y = int(((image.shape[0] - (opt_crop_size[0] * scale)) * 0.5))

        return image[y:y + int(opt_crop_size[1] * scale), x:x + int(opt_crop_size[0] * scale)]

    elif opt_crop_coordinates:
        return image[int(opt_crop_coordinates[0] * scale):int(opt_crop_coordinates[3] * scale), int(opt_crop_coordinates[1] * scale):int(opt_crop_coordinates[2] * scale)]

    return image

def process_file(file, target_list, cut=False):
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
            hash = create_hash(image[offset:offset+width, 0:width], HASH_SIZE)
            if hash:
                target_list[hash] = os.path.basename(file)
        elif ratio > 1.35: #1.25:
            offset = int((width - height) * 0.5)
            hash = create_hash(image[0:height, offset:offset+height], HASH_SIZE)
            if hash:
                target_list[hash] = os.path.basename(file)

    hash = create_hash(image, HASH_SIZE)
    if hash:
        target_list[hash] = os.path.basename(file)

def find_image_match(hr_hash):
    results = lr_tree.get_all_in_range(hr_hash, DISTANCE_THRESHOLD_MATCH)

    if not results:
        return 0

    result = min(results, key=lambda x: x[0])

    hr_file = hr_items[hr_hash]
    hr_image = load_image(os.path.join(opt_hr, hr_file))
    if hr_image is None:
        return 0

    match_distance = result[0]

    lr_hash = result[1]
    lr_file = lr_items[lr_hash]
    lr_image = load_image(os.path.join(opt_lr, lr_file))
    if lr_image is None:
        return 0

    #if opt_crop:
    #    lr_image = crop_image(lr_image)

    hr_image = align_image(hr_image, lr_image, opt_scale, opt_method)
    if hr_image is None:
        return 0

    aligned_distance = hamming_distance(
        create_hash(lr_image, HASH_SIZE_ALIGNED, "phash"),
        create_hash(hr_image, HASH_SIZE_ALIGNED, "phash")
    )

    if opt_crop:
        lr_image = crop_image(lr_image)
        hr_image = crop_image(hr_image, opt_scale)

    height, width = lr_image.shape[:2]

    if aligned_distance > DISTANCE_THRESHOLD_ALIGNED:
        return 0

    if opt_debug:
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
            return 0

        cv2.destroyAllWindows()

    if opt_color_transfer:
        hr_image = color_transfer(hr_image, lr_image)

    if opt_filename == "hr":
        lr_dest_name = hr_dest_name = os.path.splitext(hr_file)[0]
    elif opt_filename == "lr":
        lr_dest_name = hr_dest_name = os.path.splitext(lr_file)[0]
    else:
        lr_dest_name = os.path.splitext(lr_file)[0]
        hr_dest_name = os.path.splitext(hr_file)[0]

    if not opt_dry:
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

    return 1

def align_image(src, dst, scale=1.0, method="transform"):
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    sift = cv2.SIFT_create(500)

    # The match point limit is low since sometimes four is all that's needed align an image. if the
    # alignment doesn't work the image should be it should be filtered out when doing the perceptual
    # match between the aligned images
    MIN_MATCH_POINTS = 8
    GOOD_MATCH_PERCENT = 0.7

    src_height, src_width = src.shape[:2]
    src_min_size = min(src_width, src_height)

    dst_height, dst_width = dst.shape[:2]
    dst_min_size = min(dst_width, dst_height)

    # Using full sized images while doing the matching will take an unnecessarily long time with no real benefit
    # when it comes to the amount of points matched. So if it's shorter side of the image is more than 512 pixels
    # we'll shrink it down before doing the matching.
    RESIZE = 512 #min([480, src_width, src_height, dst_width, dst_height]) #480 #512

    SRC_RESIZE = min(RESIZE, src_min_size) #RESIZE if min(src_width, src_height) > RESIZE else min(src_width, src_height)
    DST_RESIZE = min(RESIZE, dst_min_size) #RESIZE if min(dst_width, dst_height) > RESIZE else min(dst_width, dst_height)

    # Resize the intermediate images. If scale is 1.0 increase the size slightly to to avoid aliasing when transforming
    # or warping the image
    align_scale = scale if scale > 1.0 else scale + 0.5
    align_width = round(dst_width * align_scale)
    align_height = round(dst_height * align_scale)

    src_ratio = SRC_RESIZE / src_min_size
    src_width_scaled = round(src_width * src_ratio)
    src_height_scaled = round(src_height * src_ratio)

    dst_ratio = DST_RESIZE / dst_min_size
    dst_width_scaled = round(dst_width * dst_ratio)
    dst_height_scaled = round(dst_height * dst_ratio)

    # Resizing using cv2.INTER_AREA for interpolation results in slightly better feature detection compared to cv2.INTER_LANCZOS4
    src_gray = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (src_width_scaled, src_height_scaled), interpolation=cv2.INTER_AREA)
    dst_gray = cv2.resize(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (dst_width_scaled, dst_height_scaled), interpolation=cv2.INTER_AREA)

    src_keypoints, src_descriptors = sift.detectAndCompute(src_gray, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dst_gray, None)

    if len(src_keypoints) == 0 or len(dst_keypoints) == 0 or len(src_descriptors) == 0 or len(dst_descriptors) == 0:
        return None

    matches = matcher.knnMatch(src_descriptors, dst_descriptors, k=2)

    good_matches = []
    for match in matches:
        try:
            m, n = match

            if m.distance < GOOD_MATCH_PERCENT * n.distance:
                good_matches.append(m)
        except Exception:
            pass

    if len(good_matches) < MIN_MATCH_POINTS:
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

    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if method == "transform":
        matrix = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)[0]
    elif method == "warp":
        matrix = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)[0]

    if matrix is None:
        return None

    if method == "transform":
        if scale > 1.0:
            return cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA)
        else:
            # if scale is 1.0, warp and resize down to the desired size
            return cv2.resize(cv2.warpAffine(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)
    elif method == "warp":
        if scale > 1.0:
            return cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA) # flags=cv2.INTER_AREA
        else:
            # if scale is 1.0, warp and resize down to the desired size
            return cv2.resize(cv2.warpPerspective(src, matrix, (align_width, align_height), flags=cv2.INTER_AREA), (int(dst_width * scale), int(dst_height * scale)), interpolation=cv2.INTER_AREA)

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
        "--scale",
        help="Aligned image scale. The size of aligned HR images will be a multiplier of SCALE and the matched LR image size (default: 2.0)",
        required=False,
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--crop",
        help="Crop the images after aligning. The crop is based on the LR size image and scaled according to SCALE when cropping the HR image\n" +
            "- Use \"--crop 10\" and 10 pixels will be removed around the image.\n" +
            "- Use \"--crop 640 480\" and the image will be cropped to 640 width and 480 height from the center.\n" +
            "- Use \"--crop 10 20 400\" 300 and the image will be cropped to 400 width and 300 height starting 10 pixels from the left of the image and 20 pixels from the top.\n\n",
        required=False,
        type=int,
        default=False,
        nargs="+"
    )
    parser.add_argument(
        "--method",
        help="What method to use when aligning images, transform or warp (default: transform)\n" +
             " - transform - Use affine transformation to align image\n" +
             " - warp - Use perspective transform to align image\n\n",
        required=False,
        type=str,
        default="transform",
        choices=["transform", "warp"]
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
        "--debug",
        help="Bring up a window showing the results when image alignment is successful. Press \"esc\" to abort the alignment process and \"n\" or \"del\" to skip the image. Pressing any other key will accept. Threading will be disabled when this option is used",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--dry",
        help="Dry run - Match or align images but dont write any output.",
        action="store_true",
        required=False,
        default=False
    )

    args = parser.parse_args()

    #
    # evaluate args and store options
    #
    opt_lr = os.path.abspath(args.lr)
    opt_hr = os.path.abspath(args.hr)

    opt_output = args.output

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

    opt_method = args.method

    opt_scale = 1.0
    if args.scale:
        opt_scale = args.scale

    opt_color_transfer = args.color_transfer
    opt_save = args.save
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

    if not opt_dry and not opt_dest:
        parser.error("--dest is required if --dry is missing")

    SPACING = 35

    print("Path to HR images".ljust(SPACING), args.hr)
    print("Path to LR images".ljust(SPACING), args.lr)
    print("Destination path".ljust(SPACING), args.dest)
    print("Output".ljust(SPACING), {"hr": "hr (HR Images)", "lr": "lr (LR images)", "all": "all (LR and HR images)"}[args.output])
    print("Scale".ljust(SPACING), args.scale)
    print("Crop".ljust(SPACING), args.crop)
    print("Method".ljust(SPACING), args.method)
    print("Color transfer".ljust(SPACING), args.color_transfer)
    print("Filename".ljust(SPACING), {"hr": "Use filename from HR image", "lr": "Use filename from LR image", "keep": "Keep the filenames from th HR and LR images"}[args.filename])
    print("File name prefix".ljust(SPACING), args.prefix)
    print("File name suffix".ljust(SPACING), args.suffix)
    print("Limit".ljust(SPACING), args.limit)
    print("Save hash tables".ljust(SPACING), args.save)
    print("Debug mode".ljust(SPACING), args.debug, {True: "(threading will be disabled)", False: ""}[args.debug])
    print("Dry run".ljust(SPACING), args.dry)

    print("--------------------")

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

    if opt_dest and not os.path.exists(opt_dest):
        print("Destination folder doesn't exist, exiting...")
        sys.exit()
    elif not opt_dry:
        if os.path.exists(opt_dest):
            if os.listdir(opt_dest):
                if ask("Destination folder isn't empty, delete existing files and folders before proceeding?", False):
                    shutil.rmtree(opt_dest)

                os.makedirs(opt_dest, exist_ok=True)

            if opt_output == "all":
                os.makedirs(opt_dest_hr, exist_ok=True)
                os.makedirs(opt_dest_lr, exist_ok=True)

    hr_hash_file = os.path.join(os.getcwd(), f"twimager_{HASH_SIZE}_{os.path.basename(opt_hr)}.pickle")
    lr_hash_file = os.path.join(os.getcwd(), f"twimager_{HASH_SIZE}_{os.path.basename(opt_lr)}.pickle")

    hr_items = {}
    lr_items = {}

    if os.path.exists(hr_hash_file) and os.path.exists(lr_hash_file):
        if ask("HR and LR hash files detected, do you wish to load them?"):
            try:
                with open(hr_hash_file, "rb") as f:
                    hr_items = pickle.load(f)
            except Exception:
                print("HR hash file couldn't be loaded")
            else:
                print("HR hash file loaded...")

            try:
                with open(lr_hash_file, "rb") as f:
                    lr_items = pickle.load(f)
            except Exception:
                print("LR hash file couldn't be loaded")
            else:
                print("LR hash file loaded...")
    elif os.path.exists(hr_hash_file) and ask("HR hash file detected, do you wish to load it?"):
        try:
            with open(hr_hash_file, "rb") as f:
                hr_items = pickle.load(f)
        except Exception:
            print("LR hash file couldn't be loaded")
        else:
            print("LR hash file loaded...")
    elif os.path.exists(lr_hash_file) and ask("LR hash file detected, do you wish to load it?"):
        try:
            with open(lr_hash_file, "rb") as f:
                lr_items = pickle.load(f)
        except Exception:
            print("LR hash file couldn't be loaded")
        else:
            print("LR hash file loaded...")

    if not hr_items:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_file, os.path.join(opt_hr, file), hr_items, True) for file in os.listdir(opt_hr)}
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

    #
    # lr/target images
    #
    if not lr_items:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_file, os.path.join(opt_lr, file), lr_items, False) for file in os.listdir(opt_lr)}
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

    print("Creating vantage-point tree...")
    lr_tree = vptree.VPTree(list(lr_items.keys()), hamming_distance)

    info = "Aligned image estimation"

    print("Matching and aligning images...", len(hr_items))

    count = 0

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
                        result = f.result()
                        count += result

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
                result = find_image_match(hr_item)
                count += result

                progress.set_description(f"{info}: {count} -")

                if opt_limit and count >= opt_limit:
                    progress.close()

                    print("Limit reached, exiting...")

                    sys.exit()

        except KeyboardInterrupt:
            print("User interrupt, exiting...")
            sys.exit()

if not opt_dry:
    print(f"Done, aligned image count: {len(os.listdir(opt_dest_lr))}")
