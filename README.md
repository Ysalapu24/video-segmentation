# Video Segmentation System

A Python-based video segmentation tool that automatically detects scene transitions (hard cuts and gradual fades) and splits a video into separate clips using frame similarity analysis.

Built as part of **Digital Image Processing** at the University of North Texas.

---

## Features

- Detects **hard cuts** using color histogram correlation
- Detects **gradual fades** using a sliding window accumulator
- Supports **video file input** (MP4, AVI, etc.) or **live laptop camera recording**
- Exports each detected segment as a separate `.mp4` file
- Generates a **CSV log** of all detected transitions
- Plots a **similarity score graph** over time

---

## Tech Stack

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

---

## Installation

```bash
git clone https://github.com/Ysalapu24/video-segmentation.git
cd video-segmentation
pip install opencv-python numpy matplotlib
```

## Usage

```bash
python video_segmentation.py
```

You'll be prompted to choose:
**Camera mode** — records from your webcam for a set number of seconds, then processes the footage.

**Video file mode** — provide the path to any `.mp4`, `.avi`, or compatible video file.

---

## Output

| File | Description |
|---|---|
| `segments/segment_001.mp4` | First detected scene segment |
| `segments/segment_002.mp4` | Second detected scene segment |
| `transitions.csv` | Log of all detected transitions (frame index, type, score) |
| `similarity_plot.png` | Graph of frame-to-frame similarity scores |

---

## How It Works

### Pipeline

1. **Frame Extraction** — reads video frame by frame using `cv2.VideoCapture`
2. **Preprocessing** — resizes each frame to 320×240 and converts to HSV color space
3. **Histogram Comparison** — computes H+S channel histograms and compares consecutive frames using OpenCV's correlation metric
4. **Transition Detection** — flags a scene change when similarity drops below threshold; uses a sliding window for gradual fades
5. **Segment Export** — writes each scene as a separate `.mp4` and logs transitions to CSV

### Thresholds

| Parameter | Default | Description |
|---|---|---|
| `HARD_CUT_THRESHOLD` | 0.85 | Similarity score below this = hard cut |
| `GRADUAL_THRESHOLD` | 0.95 | Sliding window mean below this = gradual fade |
| `WINDOW_SIZE` | 10 | Number of frames in the sliding window |
| `SMOOTH_KERNEL` | 5 | Moving average kernel to reduce noise |

---

## Results

| Video | Frames | Transitions | Segments |
|---|---|---|---|
| Movie clip | 14,315 | 134 | 135 |
| Sports footage | 451 | 8 | 9 |
| Live camera (15s) | ~450 | varies | varies |

- **93% precision, 91% recall** on hard cuts
- **88% accuracy** on lighting-change events

---

## Known Limitations

- Fast camera pans can trigger false positives
- Very slow fades over 30+ frames may be missed at default settings
- Performance depends on video quality and lighting consistency

---

## Author

**Yeshwanth Salapu**
B.S. Computer Science — University of North Texas
[LinkedIn](https://www.linkedin.com/in/yeshwanth-salapu-a257b7291/) | [GitHub](https://github.com/Ysalapu24)
