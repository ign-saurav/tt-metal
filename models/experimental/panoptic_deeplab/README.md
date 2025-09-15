# Panoptic Deeplab

## Platforms:
    Wormhole (n150)

## Introduction
Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, etc) to pixels belonging to thing classes.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
```
pytest models/experimental/panoptic_deeplab/tests/test_panoptic_deeplab.py
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- end-2-end perf is `_` FPS
  ```
  pytest
  ```

### Demo
```
python models/experimental/panoptic_deeplab/demo/panoptic_deeplab_demo.py --input <input image path> --output <output image to be stored path>
```
**Note:** Output images will be saved in the `panoptic_deeplab_predictions/` folder.

#### Single Device (BS=1):
##### Custom Images:
- Use the following command to run demo :
  ```
  pytest models/experimental/panoptic_deeplab/tests/test_panoptic_deeplab.py
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/experimental/panoptic_deeplab/resources/` and run:
  ```
  pytest models/experimental/panoptic_deeplab/tests/test_panoptic_deeplab.py
  ```

## Details
- The entry point to the `Panoptic Deeplab` is located at:`models/experimental/panoptic_deeplab/tt/panoptic_deeplab.py`..
- Supported Input Resolution - `(512, 1024)` - (Height, Width).
