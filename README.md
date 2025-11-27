**EasyOCR – Enhanced Live Text Detection**

Real-time Text Detection system built using EasyOCR, OpenCV, and Python, upgraded with advanced text preprocessing, text-to-speech (TTS), translation, clipboard copy, filtering, and automatic logging.

This tool reads any text shown to the camera — printed documents, mobile screens, handwritten text, boards, labels — and enhances accuracy with preprocessing such as CLAHE, denoising, sharpening, morphology, and deskewing.

🚀 Features
🔍 Enhanced OCR Accuracy

Dual-pass OCR (original + preprocessed frame)

CLAHE contrast enhancement

Noise removal

Sharpening

Morphological cleanup

Auto deskew

Result merging (highest confidence kept)

🗣️ Text-to-Speech (TTS)

Reads detected text aloud automatically using pyttsx3.

🌏 Automatic Translation (English ➝ Hindi)

Detected text is translated and displayed at the bottom of the screen.
(Uses Google Translate API via googletrans)

📋 Copy to Clipboard

Press c → instantly copies all detected text to clipboard.

📝 Automatic Logging

All detected text is saved to:

detected_text_log_YYYYMMDD.txt

🎯 Filtering Options

Choose what type of text to detect:

--mode all
--mode numbers
--mode words

📦 Requirements

Make sure Python 3.9 is installed.

Install dependencies:

pip install -r requirements.txt


If you installed everything manually:

pip install pyttsx3 googletrans==4.0.0-rc1 pyperclip


Core packages:

easyocr==1.7.1

opencv-python==4.9.0.80

numpy==1.26.4

torch==2.2.1

torchvision==0.17.1

Feature packages:

pyttsx3

googletrans==4.0.0-rc1

pyperclip

▶️ How to Run
Webcam
python live_text_detection_enhanced.py --camera 0

Improve performance
python live_text_detection_enhanced.py --skip 10

Disable TTS or Translation
python live_text_detection_enhanced.py --no-tts --no-translate

Detect only numbers
python live_text_detection_enhanced.py --mode numbers

Save output video
python live_text_detection_enhanced.py --save output.mp4

📂 Project Structure
EasyOCR/
│
├── live_text_detection_enhanced.py
├── live_text_detection.py
├── requirements.txt
└── detected_text_log_YYYYMMDD.txt (auto-created)

🙌 Credits

EasyOCR by JaidedAI

OpenCV

PyTorch

Google Translate API

pyttsx3 TTS Engine

Author: Prakhar Gupta 
