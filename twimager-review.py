import os
import sys
import argparse
import numpy as np
import cv2
import json

def decrease(index, max, min=0):
    index -= 1

    if index < min:
        index = max - 1
    elif index >= max:
        index = min

    return index

def increase(index, max, min=0):
    index += 1

    if index < min:
        index = max - 1
    elif index >= max:
        index = min

    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="twimager-review.py",
        description="A script to review the results of Twimager.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--report",
        help="Report file to read",
        required=True,
        type=str
    )
    parser.add_argument(
        "--scale",
        help="Preview scale (based on the HR image size)",
        required=False,
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--interpolation",
        help="Algorithm to use when images are scaled (default: lanczos)",
        required=False,
        type=str,
        default="lanczos",
        choices=["cubic", "lanczos", "area"]
    )

    args = parser.parse_args()

    if not os.path.isfile(args.report):
        print(f"Report file doesn't exist, exiting...")
        sys.exit()

    try:
        with open(args.report, "r", encoding="utf8") as f:
            report = json.load(f)
    except Exception as error:
        print(f"Error reading report file - {error}")
        sys.exit()

    INTERPOLATION_MAP = {
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    SPACING = 20

    print()
    print("Options Used")
    print("-----")
    for key in report["options"]:
        print(key.ljust(SPACING), ":", report["options"][key])

    index = 0
    direction = "down"
    window = "Twimager Review"

    cv2.namedWindow(window)

    while True:
        hr, lr = report["images"][index]

        if not os.path.isfile(lr) or not os.path.isfile(hr):
            if direction == "down":
                index = increase(index, max=len(report["images"]))
            else:
                index = decrease(index, max=len(report["images"]))
            continue

        if not os.path.isfile(hr):
            print(f"Image {hr} doesn't exist, exiting...")
            sys.exit()

        if not os.path.isfile(lr):
            print(f"Image {lr} doesn't exist, exiting...")
            sys.exit()

        hr_image = cv2.imdecode(np.fromfile(hr, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[..., :3]
        lr_image = cv2.imdecode(np.fromfile(lr, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[..., :3]

        height = int(hr_image.shape[0] * args.scale)
        width = int(hr_image.shape[1] * args.scale)

        hr_image = cv2.resize(hr_image, (width, height), interpolation=INTERPOLATION_MAP[args.interpolation])
        lr_image = cv2.resize(lr_image, (width, height), interpolation=INTERPOLATION_MAP[args.interpolation])

        preview = np.zeros((int(height), int(width * 3) + 3, 3), dtype=np.uint8)
        preview[0:height, 0:width] = hr_image
        preview[0:height, width + 1:int(width * 2) + 1] = lr_image
        preview[0:height, int(width * 2) + 2:int(width * 3) + 2] = cv2.addWeighted(hr_image, 0.5, lr_image, 0.5, 0.0)

        string = "HR"
        cv2.putText(preview, string, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        string = "LR"
        cv2.putText(preview, string, (width + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (width + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        string = "Blend"
        cv2.putText(preview, string, (int(width * 2) + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (int(width * 2)+ 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.525, (255, 255, 255), 1, cv2.LINE_AA)

        string = f"{index+1}/{len(report['images'])}"
        cv2.putText(preview, string, (2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.625, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, string, (2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.625, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window, preview)

        key = cv2.waitKeyEx()

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            sys.exit()
        elif key == 27: #esc
            cv2.destroyAllWindows()
            sys.exit()
        elif key == 2162688 or key == 2490368 or key == 2424832: #pgup / upkey / leftkey
            direction = "up"
            index = decrease(index, max=len(report["images"]))
        elif key == 2228224 or key == 2621440 or key == 2555904: #pgdwon / downkey / rightkey
            direction = "down"
            index = increase(index, max=len(report["images"]))
        elif key == 3014656: # del
            os.remove(hr)
            os.remove(lr)

            del report["images"][index]

            with open(args.report, "w", encoding="utf8") as f:
                json.dump(report, f, indent=4)

        if index < 0:
            index = len(report["images"]) - 1
        elif index >= len(report["images"]):
            index = 0
