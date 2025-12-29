# Plumbline – Posture Analysis from Video

Plumbline is a video-based posture analysis tool built with **MediaPipe Pose** and **OpenCV**.  
It analyzes prerecorded videos (MP4) to estimate **neck and torso alignment**, classify posture over time, and output an **annotated posture assessment video**.

This project focuses on **robust geometry, stable classification, and clear visual feedback**, rather than medical diagnosis.

---

## 📌 What it does

- Takes a prerecorded side-view video (`.mp4`) as input
- Detects human pose landmarks using MediaPipe Pose
- Computes:
  - **Neck inclination** (shoulder → ear vs vertical)
  - **Torso inclination** (hip → shoulder vs vertical)
- Smooths raw angles using **Exponential Moving Average (EMA)**
- Classifies posture using a **hysteresis-based state machine** to prevent flicker
- Tracks posture duration using **time-based accumulation**
- Outputs a new `.mp4` with:
  - posture lines
  - live angle readouts (raw → smoothed)
  - posture state (`GOOD` / `ADJUST`)
  - time spent in each state

---

## 💡 Inspiration

The inspiration for Plumbline is intentionally simple and personal.

Growing up, my aunt often told me I had bad posture — something I strongly disagreed with.  
That disagreement eventually turned into curiosity: *could I measure posture objectively instead of arguing about it?*

At the same time, I wanted to deepen my experience with **computer vision and OpenCV**, especially after working with peers on hardware projects using Arduino.  
Plumbline became a way to explore pose estimation, geometry, and temporal stability in a visual, real-world problem.

---

## 🛠 How it works (high level)

1. **Pose detection**  
   MediaPipe Pose extracts body landmarks for each frame.

2. **Geometry**  
   Segment inclination is computed relative to vertical using normalized vectors and `acos`, with safety guards against floating-point edge cases.

3. **Smoothing**  
   Raw angles are smoothed with EMA to reduce jitter.

4. **Classification**  
   A hysteresis-based state machine prevents rapid GOOD/BAD flicker.

5. **Timing**  
   Posture duration is accumulated using elapsed time rather than frame counts.

6. **Visualization**  
   Results are overlaid directly onto the video for clear feedback.

---

## ⚠️ Important notes

- This tool **analyzes prerecorded videos**, not live webcam input.
- Input and output `.mp4` files are intentionally **excluded from the repository** due to file size limits.
- A demo video showing the program running is provided separately.
- This project is **not a medical device** and is intended for educational and experimental use.

---

## 🧪 Robustness features

- Frames with low landmark confidence are skipped safely
- All angle math clamps values to avoid NaNs
- Zero-length vectors are guarded against
- Classification stability is enforced via hysteresis
- Visual text uses shadowed overlays for readability

---

## 📦 Built with

- **Python**
- **MediaPipe Pose**
- **OpenCV**
- **NumPy** (indirect dependency)

Dependencies are listed in `requirements.txt`.

---

## ▶️ How to run

```bash
pip install -r requirements.txt
python main490.py
