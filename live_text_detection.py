
"""
live_text_detection_ready.py

Robust live text detection demo using EasyOCR + OpenCV.

- Sets a safe workaround for OpenMP duplicate runtime on Windows.
- Provides helpful error messages if dependencies are missing.
- Runs OCR in a background thread to keep UI responsive.
- CLI options for camera/input/save/gpu/confidence/resize.
- Use the Python interpreter where you've installed dependencies.
"""

import os
# --- Workaround for OpenMP duplicate runtime on Windows ---
# This is a practical workaround often required on Windows when libraries
# include different OpenMP runtimes (e.g., numpy, intel-mkl, torch).
# It's generally fine for demos; if you prefer a stricter fix, recreate env via conda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Optional

# Defensive imports with user-friendly messages
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is not available in this Python environment.")
    print("Install it in the same interpreter you are using:")
    print("  python -m pip install opencv-python")
    print("If you're using conda, try:")
    print("  conda install -c conda-forge opencv")
    raise

try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is not available. Install it:")
    print("  python -m pip install numpy")
    raise

try:
    import easyocr
except Exception as e:
    print("ERROR: EasyOCR is not available. Install it:")
    print("  python -m pip install easyocr")
    raise

# Optional: don't import torch explicitly here; EasyOCR may import torch internally
# but if you want GPU control, EasyOCR's reader uses 'gpu' flag during creation.

# ---------- Config ----------
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_results(frame: np.ndarray, results: List[Tuple[List[List[float]], str, float]], conf_thresh: float):
    """Draw bounding boxes and text for OCR results on the frame."""
    for bbox, text, prob in results:
        try:
            if prob < conf_thresh:
                continue
            pts = np.array(bbox).astype(int)
            if pts.shape[0] >= 4:
                # draw polygon
                cv2.polylines(frame, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                tl = pts[0]
            else:
                # fallback: bounding rectangle
                x_coords = pts[:, 0] if pts.size else [0]
                y_coords = pts[:, 1] if pts.size else [0]
                x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
                x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                tl = (x_min, y_min)
            display_text = f"{text} ({prob:.2f})"
            (w, h), _ = cv2.getTextSize(display_text, FONT, 0.6, 1)
            cv2.rectangle(frame, (tl[0], tl[1] - h - 6), (tl[0] + w, tl[1] + 4), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (tl[0], tl[1] - 2), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            # don't let drawing errors crash the loop
            continue

def ocr_worker(reader: easyocr.Reader, frame: np.ndarray):
    """Run EasyOCR on the provided frame (callable for background thread)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return reader.readtext(rgb)

def parse_args():
    p = argparse.ArgumentParser(description="Live Text Detection (robust demo)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--input", type=str, default=None, help="Path to input video file (optional)")
    p.add_argument("--skip", type=int, default=8, help="Process OCR every N frames (default 8)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold to display (0-1)")
    p.add_argument("--save", type=str, default=None, help="Optional path to save annotated output (MP4)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for EasyOCR if available")
    p.add_argument("--width", type=int, default=None, help="Resize display width for performance (preserves aspect ratio)")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting Live Text Detection")

    # Try to create EasyOCR reader
    try:
        reader = easyocr.Reader(["en"], gpu=bool(args.gpu))
    except Exception as e:
        logging.warning("EasyOCR reader initialization failed with gpu=%s. Trying cpu fallback. Error: %s", args.gpu, e)
        reader = easyocr.Reader(["en"], gpu=False)

    # Open video source
    if args.input:
        cap = cv2.VideoCapture(args.input)
        logging.info("Opening input video: %s", args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
        logging.info("Opening camera index: %s", args.camera)

    if not cap.isOpened():
        logging.error("Could not open video source. Check camera index or input file path.")
        sys.exit(1)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        logging.info("Saving annotated output to %s (w=%d h=%d fps=%.2f)", args.save, w, h, fps)

    frame_count = 0
    latest_results = []
    executor = ThreadPoolExecutor(max_workers=1)
    pending: Optional[Future] = None

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("Stream ended or camera disconnected.")
                break

            # optional resize
            if args.width:
                h0, w0 = frame.shape[:2]
                new_w = args.width
                new_h = int(h0 * (new_w / w0))
                frame = cv2.resize(frame, (new_w, new_h))

            frame_count += 1

            # Submit OCR job every N frames if none pending
            if frame_count % args.skip == 0 and (pending is None or pending.done()):
                try:
                    frame_for_ocr = frame.copy()
                    pending = executor.submit(ocr_worker, reader, frame_for_ocr)
                except Exception as e:
                    logging.exception("Failed to submit OCR job: %s", e)
                    pending = None

            # Retrieve finished OCR job
            if pending is not None and pending.done():
                try:
                    latest_results = pending.result()
                except Exception as e:
                    logging.exception("OCR worker failed: %s", e)
                    latest_results = []
                pending = None

            # Draw results and FPS
            draw_results(frame, latest_results, args.conf)
            now = time.time()
            if prev_time != 0:
                instant_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
                fps = 0.9 * fps + 0.1 * instant_fps
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.8, (0, 255, 255), 2)

            cv2.imshow("Live OCR - press q to quit", frame)
            if writer:
                # ensure same width/height
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Quit requested by user.")
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user (KeyboardInterrupt).")
    finally:
        logging.info("Cleaning up...")
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
        logging.info("Exited cleanly.")

if __name__ == "__main__":
    main()
