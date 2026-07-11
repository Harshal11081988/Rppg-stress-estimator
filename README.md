# 💓 Webcam Stress & Recovery Estimator (rPPG)

Estimates heart rate and heart-rate variability (HRV) from a short face
video — no wearable hardware required — using **remote
photoplethysmography (rPPG)**: detecting your pulse from subtle,
invisible-to-the-eye color changes in facial skin caused by blood flow.

> ⚠️ **Experimental wellness tool, not a medical device.** Accuracy
> depends on lighting, camera quality, and how still you stay during
> recording. Not for diagnosis or treatment decisions.

## How it works

1. **Face detection** (OpenCV Haar cascade, bundled — no external
   download needed) locates your face in each video frame, with a
   center-crop fallback if detection briefly fails.
2. A **forehead region of interest** is isolated (strong pulse signal,
   minimal motion artifacts from eyes/mouth).
3. The **mean green-channel value** of that region is tracked across
   every frame — this is the raw rPPG signal.
4. The signal is **detrended and bandpass-filtered** (42–200 BPM range)
   to isolate the pulse component from lighting drift and motion.
5. **Peak detection** on the filtered signal finds individual
   heartbeats, from which heart rate and HRV metrics (SDNN, RMSSD) are
   computed.
6. A simple heuristic combines elevated heart rate + reduced HRV into
   a directional **stress index** (not a clinical score).

No external dataset or network access is required at any point — this
runs entirely on the uploaded video.

## Project structure

```
rppg-stress-estimator/
├── app.py               # Streamlit app (deploy this)
├── rppg.py              # Core signal-processing pipeline
├── test_pipeline.py      # Self-test: verifies HR recovery on synthetic data
├── requirements.txt
└── README.md
```

## Setup (local)

```bash
git clone <your-repo-url>
cd rppg-stress-estimator
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Optional: verify the pipeline works correctly on your machine
python test_pipeline.py

# Run the app
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub as-is — no data files or training step
   needed (unlike EEG-based projects, this has zero external dataset
   dependency).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect
   your GitHub repo, and point it at `app.py`.
3. Done. Users upload a short video directly in the browser.

## Recording tips (for good readings)

- 20–30 seconds, facing the camera, well-lit (avoid backlighting)
- Stay relatively still; avoid talking during the clip
- Plain background helps but isn't required

## Tech stack

- **OpenCV (headless)** — face detection via Haar cascades
- **SciPy** — bandpass filtering, peak detection
- **NumPy** — signal array processing
- **Streamlit** — UI and video upload
- **Plotly** — waveform and interval visualizations

## Limitations & honest caveats

- Accuracy is meaningfully lower than a real pulse oximeter or ECG,
  especially under poor lighting, camera compression, or head motion
- The "stress index" is a simplified heuristic for demonstration, not
  a validated clinical HRV-stress model
- Works best with good lighting and minimal motion — this is a known
  limitation of rPPG generally, not just this implementation

## Possible extensions

- Live webcam mode via `streamlit-webrtc` instead of video upload
- Multi-region signal fusion (forehead + cheeks) for robustness
- POS or CHROM rPPG algorithms (more advanced than green-channel-only)
- Trend tracking across multiple sessions (requires persistent storage)
