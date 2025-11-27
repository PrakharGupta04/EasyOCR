"""
live_text_detection_enhanced.py

Enhanced live OCR demo:
 - Threaded EasyOCR + OpenCV (from original base). See original: :contentReference[oaicite:1]{index=1}
 - Features:
    * Text-to-speech (pyttsx3) non-blocking
    * Auto-translate (googletrans) overlay (English -> Hindi)
    * Copy detected text to clipboard (press 'c') via pyperclip
    * Save detected text to timestamped log file
    * Mode filtering: all / numbers / words via --mode
 - Improved OCR accuracy:
    * Preprocessing (CLAHE, denoise, sharpening, upscaling)
    * Deskew attempt
    * Dual-pass OCR (original + preprocessed) and merge results
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Optional
import math
import re
import threading

# Defensive imports
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is not available. Install: python -m pip install opencv-python")
    raise

try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is not available. Install: python -m pip install numpy")
    raise

try:
    import easyocr
except Exception as e:
    print("ERROR: EasyOCR is not available. Install: python -m pip install easyocr")
    raise

# Optional features: TTS, translator, clipboard
try:
    import pyttsx3
    _HAS_TTS = True
except Exception:
    _HAS_TTS = False

try:
    from googletrans import Translator as GTTranslator
    _HAS_GT = True
except Exception:
    _HAS_GT = False

try:
    import pyperclip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# ---------- Config ----------
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------- Preprocessing + OCR improvements -----------------

def _deskew_image(gray):
    """
    Conservative deskew: estimate dominant line angle using Hough on edges.
    Rotate only when angle magnitude >= 1 deg.
    """
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi/180.0, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray, 0.0
        angles = []
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            if angle < -45:
                angle += 90
            if angle > 45:
                angle -= 90
            angles.append(angle)
        if len(angles) == 0:
            return gray, 0.0
        median_angle = float(np.median(angles))
        if abs(median_angle) < 1.0:
            return gray, 0.0
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, median_angle
    except Exception:
        return gray, 0.0

def _preprocess_for_ocr(frame):
    """
    Preprocess frame to improve OCR:
     - convert to grayscale
     - upscale (if small)
     - bilateral filter (denoise preserving edges)
     - CLAHE (contrast)
     - unsharp mask (sharpen)
     - small morphology to remove speckle
     - deskew attempt
    Returns: deskewed_bgr (BGR) and deskewed_gray
    """
    try:
        bgr = frame.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Upscale small frames (limit upscaling)
        h, w = gray.shape
        max_dim = max(w, h)
        if max_dim < 1000:
            scale = min(2.0, 1000.0 / max_dim)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Denoise (preserve edges)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Unsharp mask (slight sharpen)
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
        sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # Morphology to clean small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(sharpen, cv2.MORPH_OPEN, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

        deskewed, angle = _deskew_image(morph)
        deskewed_bgr = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        return deskewed_bgr, deskewed
    except Exception:
        # fallback: original
        return frame.copy(), cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

def ocr_worker(reader: easyocr.Reader, frame: np.ndarray):
    """
    Enhanced OCR worker:
     - Run EasyOCR on original RGB and preprocessed RGB
     - Merge results by text (case-insensitive), keep highest-confidence bbox
     - Return list of (bbox, text, prob)
    """
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prep_bgr, prep_gray = _preprocess_for_ocr(frame)
        prep_rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)

        results_all = []

        # Pass 1: original
        try:
            r1 = reader.readtext(rgb, detail=1, paragraph=False)
            if r1:
                results_all.extend(r1)
        except Exception:
            pass

        # Pass 2: preprocessed
        try:
            r2 = reader.readtext(prep_rgb, detail=1, paragraph=False)
            if r2:
                results_all.extend(r2)
        except Exception:
            pass

        # Merge by text (keep max prob)
        merged = {}
        for bbox, text, prob in results_all:
            key = text.strip()
            if key == "":
                continue
            key_l = key.lower()
            if key_l in merged:
                if prob > merged[key_l][2]:
                    merged[key_l] = (bbox, text, prob)
            else:
                merged[key_l] = (bbox, text, prob)

        return list(merged.values())

    except Exception:
        # fallback single-pass safe read
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return reader.readtext(rgb)
        except Exception:
            return []

# ----------------- Drawing & helpers -----------------

def draw_results(frame: np.ndarray, results: List[Tuple[List[List[float]], str, float]], conf_thresh: float):
    """Draw bounding boxes and text for OCR results on the frame."""
    for bbox, text, prob in results:
        try:
            if prob < conf_thresh:
                continue
            pts = np.array(bbox).astype(int)
            if pts.shape[0] >= 4:
                cv2.polylines(frame, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                tl = pts[0]
            else:
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
            continue

def is_numberish(s: str):
    s = s.strip()
    s2 = s.replace(",", "").replace(" ", "")
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+", s2))

# ----------------- Main CLI & loop -----------------

def parse_args():
    p = argparse.ArgumentParser(description="Live Text Detection (enhanced + improved OCR)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--input", type=str, default=None, help="Path to input video file (optional)")
    p.add_argument("--skip", type=int, default=8, help="Process OCR every N frames (default 8)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold to display (0-1)")
    p.add_argument("--save", type=str, default=None, help="Optional path to save annotated output (MP4)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for EasyOCR if available")
    p.add_argument("--width", type=int, default=None, help="Resize display width for performance (preserves aspect ratio)")
    p.add_argument("--mode", type=str, default="all", choices=["all", "numbers", "words"],
                   help="Filter mode: all (default), numbers (show only numeric results), words (hide pure numbers)")
    p.add_argument("--no-tts", action="store_true", help="Disable TTS even if pyttsx3 is installed")
    p.add_argument("--no-translate", action="store_true", help="Disable translation overlay")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting Live Text Detection (enhanced)")

    # EasyOCR reader
    try:
        reader = easyocr.Reader(["en"], gpu=bool(args.gpu))
    except Exception as e:
        logging.warning("EasyOCR reader init with gpu=%s failed; falling back to cpu. Error: %s", args.gpu, e)
        reader = easyocr.Reader(["en"], gpu=False)

    # TTS
    engine = None
    tts_executor = None
    if _HAS_TTS and (not args.no_tts):
        try:
            engine = pyttsx3.init()
            tts_executor = ThreadPoolExecutor(max_workers=1)
            logging.info("Text-to-Speech enabled (pyttsx3).")
        except Exception as e:
            logging.warning("pyttsx3 init failed: %s. TTS disabled.", e)
            engine = None
            tts_executor = None
    else:
        if not _HAS_TTS:
            logging.info("pyttsx3 not installed — TTS unavailable.")
        else:
            logging.info("TTS disabled via --no-tts.")

    # Translator
    translator = None
    if _HAS_GT and (not args.no_translate):
        try:
            translator = GTTranslator()
            logging.info("Translator (googletrans) enabled.")
        except Exception as e:
            logging.warning("Translator init failed: %s. Translation disabled.", e)
            translator = None
    else:
        if not _HAS_GT:
            logging.info("googletrans not installed — translation unavailable.")
        else:
            logging.info("Translation disabled via --no-translate.")

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

    # Prepare log file
    log_path = f"detected_text_log_{time.strftime('%Y%m%d')}.txt"
    try:
        log_file = open(log_path, "a", encoding="utf-8")
    except Exception as e:
        logging.warning("Could not open log file for writing: %s. Continuing without logging.", e)
        log_file = None

    frame_count = 0
    latest_results = []
    executor = ThreadPoolExecutor(max_workers=1)
    pending: Optional[Future] = None

    last_spoken = ""
    last_logged_text = ""
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

            # Filtering by mode (all / numbers / words)
            filtered = []
            for bbox, text, prob in latest_results:
                if prob < args.conf:
                    continue
                if args.mode == "numbers":
                    if not is_numberish(text):
                        continue
                elif args.mode == "words":
                    if is_numberish(text):
                        continue
                filtered.append((bbox, text, prob))
            latest_results = filtered

            # draw detected boxes and text
            draw_results(frame, latest_results, args.conf)

            # Prepare consolidated text
            full_text = " ".join([t[1] for t in latest_results]).strip()

            # Translation overlay (English -> Hindi) if translator available
            translated_text = ""
            if translator and full_text:
                try:
                    trans = translator.translate(full_text, dest="hi", src="en")
                    translated_text = getattr(trans, "text", str(trans))
                except Exception as e:
                    logging.debug("Translation failed: %s", e)
                    translated_text = ""

            if translated_text:
                x = 10
                y = frame.shape[0] - 10
                max_chars = 60
                lines = [translated_text[i:i+max_chars] for i in range(0, len(translated_text), max_chars)]
                for i, line in enumerate(reversed(lines)):
                    cv2.putText(frame, line, (x, y - (i*22)), FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Non-blocking TTS: speak full_text only if changed
            if engine and full_text and full_text != last_spoken:
                def _speak(text):
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        logging.debug("TTS error: %s", e)
                try:
                    tts_executor.submit(_speak, full_text)
                    last_spoken = full_text
                except Exception as e:
                    logging.debug("Failed to submit TTS job: %s", e)

            # Logging to file (append with timestamp) if new
            if log_file and full_text and full_text != last_logged_text:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{timestamp}] {full_text}\n")
                    log_file.flush()
                    last_logged_text = full_text
                except Exception as e:
                    logging.debug("Failed to write to log file: %s", e)

            # Draw FPS
            now = time.time()
            if prev_time != 0:
                instant_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
                fps = 0.9 * fps + 0.1 * instant_fps
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.8, (0, 255, 255), 2)

            # Show overlay of consolidated detected text (top-left)
            if full_text:
                max_chars = 50
                lines = [full_text[i:i+max_chars] for i in range(0, len(full_text), max_chars)]
                for i, line in enumerate(lines[:3]):  # limit to 3 lines
                    cv2.putText(frame, line, (10, 50 + i*22), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Live OCR - press q to quit (c=copy to clipboard)", frame)
            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Quit requested by user.")
                break
            # Copy to clipboard on 'c' key
            if key == ord("c"):
                if not _HAS_CLIP:
                    print("pyperclip not installed — can't copy to clipboard. Install: pip install pyperclip")
                else:
                    try:
                        to_copy = full_text
                        pyperclip.copy(to_copy)
                        print("Copied to clipboard:", to_copy)
                    except Exception as e:
                        logging.warning("Failed to copy to clipboard: %s", e)

    except KeyboardInterrupt:
        logging.info("Interrupted by user (KeyboardInterrupt).")
    finally:
        logging.info("Cleaning up...")
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
        if tts_executor:
            try:
                tts_executor.shutdown(wait=False)
            except Exception:
                pass
        logging.info("Exited cleanly.")

if __name__ == "__main__":
    main()
