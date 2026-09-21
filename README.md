# Object Area AI

AI-powered object segmentation and area measurement web application.

## Features

- Camera capture
- Image upload
- Positive and negative object marking
- SAM 2 object segmentation
- Pixel-area calculation
- Optional real-world area calculation in cm² using a known reference length
- Download segmented result

## Project structure

```text
area_ai/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .gitignore
└── README.md
```

## Run locally

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## How it works

1. Upload an image or capture one with the camera.
2. Add positive points inside the object.
3. Optionally add negative points to exclude unwanted regions.
4. Run segmentation.
5. The app displays the segmented object and pixel area.
6. To estimate cm², select two points on a known-size reference and enter its length in centimeters.

## Real-world area limitation

A normal photograph does not contain enough information to determine physical area by itself. Calibration requires a known reference length in the image, and accuracy depends on perspective, camera angle, and the reference and object being on approximately the same plane.

## AI model

The backend uses `facebook/sam2-hiera-tiny` through Hugging Face Transformers. The model is downloaded automatically on the first segmentation request.

## Hardware note

SAM 2 is substantially heavier than a normal HTML/CSS/JavaScript application. A GPU-equipped computer is recommended for faster inference. Low-RAM systems may need a cloud/Colab environment for the AI backend.

## GitHub deployment note

GitHub stores the source code, but GitHub Pages cannot run the Python/SAM 2 backend. For a public online deployment, host the Python backend on a Python-capable service and configure the frontend to call that backend.
