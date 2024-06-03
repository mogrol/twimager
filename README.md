# Twimager

A python script to match find matching images and attempt to align them.

This tool doesn't rely on matching filenames, instead image matching is done by first creating a [perceptual hash](https://en.wikipedia.org/wiki/Perceptual_hashing) for all images in the specified folders. The hashes are then used to find images which are within a specified [hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) of each other.

Each potential match is scanned for matching features using [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform), if enough matching features are detected, an attempt to align the images are made. After alignment, a perceptual hash of the aligned image is created and if the distance between the paired and aligned image passes another threshold the alignment is considered successful and the image(s) are saved.

[Examples](https://slow.pics/s/DI1F262n) of images aligned using Twimager. The images are also available in the [examples folder](https://github.com/mogrol/twimager/examples/).

> [!NOTE]
> Twimager works best on images which aren't distorted. For those cases you can try to use Twimager as an intermediate step to find matching images. And then [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) (available in [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)) for alignment.

### Installation
Requires Python 3.10 or newer.
```
git clone https://github.com/mogrol/twimager
cd twimager
pip install -r requirements.txt
```

### Usage

```
python twimager.py --lr <path to lr (reference) images folder> --hr <path to hr (target) images folder> --dest <path to destination folder>
```

### Options

| Argument | Description |
|:-|:-|
| &#x2011;&#x2011;hr | Path to folder containing HR (target) images |
| &#x2011;&#x2011;lr | Path to folder containing LR (reference) images which to align the HR image to |
| &#x2011;&#x2011;dest | Path to folder where images should be written. |
| &#x2011;&#x2011;output | What to write to the desination folder (default: `all`)<ul><li>`lr` - Only LR images</li><li>`hr` - Only the HR images</li><li>`all` - Both LR and HR (the folders LR and HR will be created in the destination folder if needed)</li></ul>
| &#x2011;&#x2011;scale | Aligned image scale. The size of aligned HR images will be a multiplier of SCALE and the matched LR image size (default: `2.0`) |
| &#x2011;&#x2011;crop | Crop the images after aligning. The crop is based on the LR size image and scaled according to SCALE when cropping the HR image<ul><li>`--crop 10` will remove 10 pixels around the image.</li><li>`--crop 640 480` will crop the image to 640 width and 480 height from the center.</li><li>`--crop 10 20 400 300` will crop the image to 400 width and 300 height starting 10 pixels from the left and 20 pixels from the top.</li></ul>|
| &#x2011;&#x2011;method | What method to use when aligning images (default: `transform`).<ul><li>`transform` - Use affine transformation to align image</li><li>`warp` - Use perspective transform to align image</li></ul> |
| &#x2011;&#x2011;color_transfer | Transfer colors from the LR image to the aligned HR image |
| &#x2011;&#x2011;filename | What filename to use when writing the matched or aligned images<ul><li>`hr` Filename will be the same as the input HR</li><li>`lr` - Filename will be the same as the input LR</li><li>`keep` - Filename will be the same as the input filename</li></ul> |
| &#x2011;&#x2011;prefix | Prefix to append to the filename when writing images. |
| &#x2011;&#x2011;suffix | Suffix  to append to the filename when writing images. |
| &#x2011;&#x2011;limit| Limit the amount of aligned images , please note that this isn't exact due to threading |
| &#x2011;&#x2011;debug | Bring up a window showing the results when image alignment is successful. Press `esc` to abort the alignment process and `n` or `del` to skip the image. Pressing any other key will accept. Threading will be disabled when this option is used |
| &#x2011;&#x2011;dry | Dry run.  - No images will be written. Only find matching images and attempt to align them. |

### Acknowledgements

In no particular order.

Thanks to Dr. Neal Krawetz for [perceptual hashing](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html), Johannes Buchner for [ImageHash](https://github.com/JohannesBuchner/imagehash), Rickard Sjögren for [VP-Tree](https://github.com/RickardSjogren/vptree), Adrian Rosebrock at [pyimagesearch](https://pyimagesearch.com/) for the articles on image matching and alignment (registration) and the [OpenCV Team](https://opencv.org/)


### License
[MIT](https://github.com/mogrol/twimager/LICENSE)