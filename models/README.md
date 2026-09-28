# Model weights

Not vendored. `visage.models` downloads them on first use into this directory:

- `face_detection_yunet_2023mar.onnx` (about 230 KB) — YuNet
- `face_recognition_sface_2021dec.onnx` (about 37 MB) — SFace

Source: the [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Haar cascades ship inside `opencv-python-headless` and need no download.

Pin **4.10.0.84**. OpenCV 5.0's Python wheel no longer exposes `cv2.CascadeClassifier`, so a newer wheel silently removes the Viola–Jones path.
