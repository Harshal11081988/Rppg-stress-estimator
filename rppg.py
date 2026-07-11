"""
rppg.py
========
Core signal-processing pipeline for remote photoplethysmography (rPPG):
estimating heart rate and heart-rate variability (HRV) from subtle
color changes in facial skin captured on video.

How it works (the short version):
Every heartbeat pushes a small pulse of blood into facial capillaries,
causing a tiny, invisible-to-the-eye change in how much green light
skin reflects. By averaging the green channel over a skin region
across many video frames, that pulse becomes a periodic signal we can
bandpass-filter and measure -- no wearable hardware required.

This is an experimental wellness signal, not a medical device. It is
sensitive to lighting, motion, and video quality.
"""

import numpy as np
import cv2
from scipy.signal import butter, filtfilt, detrend, find_peaks

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Plausible human resting/active heart rate range, used both for the
# bandpass filter and for sanity-checking detected peaks.
MIN_HR_BPM = 42
MAX_HR_BPM = 200


def _load_face_cascade():
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("Failed to load bundled Haar cascade for face detection.")
    return cascade


def detect_face_bbox(gray_frame, cascade, last_bbox=None):
    """
    Detect a face bounding box (x, y, w, h) in a grayscale frame.
    Falls back to the last known bbox, then to a centered rectangle,
    so the pipeline stays robust to occasional missed detections
    (common with head movement / lighting changes in real video).
    """
    faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        # Pick the largest detected face (most likely the main subject)
        return max(faces, key=lambda f: f[2] * f[3])

    if last_bbox is not None:
        return last_bbox

    h, w = gray_frame.shape[:2]
    side = int(min(h, w) * 0.4)
    return (w // 2 - side // 2, h // 2 - side // 2, side, side)


def forehead_roi(bbox):
    """
    Given a face bounding box, return a sub-rectangle over the forehead --
    a region with strong, motion-tolerant pulsatile signal and few
    features (eyes/mouth) that introduce motion artifacts.
    """
    x, y, w, h = bbox
    roi_x = x + int(w * 0.25)
    roi_y = y + int(h * 0.08)
    roi_w = int(w * 0.5)
    roi_h = int(h * 0.18)
    return (roi_x, roi_y, roi_w, roi_h)


def extract_raw_signal(video_path, max_seconds=40, progress_callback=None):
    """
    Read a video file and extract the mean green-channel value of the
    forehead ROI for every frame.

    Returns:
        raw_signal: 1D numpy array, one value per frame
        fps: frames per second of the source video
        face_detected_ratio: fraction of frames where a face was actually
            detected (vs. fallback ROI used) -- a data-quality indicator
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file. Try a different format (mp4/mov/avi).")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0  # sane fallback if the container doesn't report fps correctly

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * fps)
    n_frames_to_read = min(total_frames, max_frames) if total_frames > 0 else max_frames

    cascade = _load_face_cascade()
    raw_signal = []
    last_bbox = None
    detected_count = 0
    frame_idx = 0

    while frame_idx < n_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) > 0:
            bbox = max(faces, key=lambda f: f[2] * f[3])
            last_bbox = bbox
            detected_count += 1
        else:
            bbox = detect_face_bbox(gray, cascade, last_bbox)

        rx, ry, rw, rh = forehead_roi(bbox)
        rx, ry = max(rx, 0), max(ry, 0)
        roi = frame[ry:ry + rh, rx:rx + rw]

        if roi.size == 0:
            # ROI fell off-frame; reuse the previous value to keep timing intact
            raw_signal.append(raw_signal[-1] if raw_signal else 0.0)
        else:
            green_channel = roi[:, :, 1]  # BGR order in OpenCV -> index 1 is green
            raw_signal.append(float(np.mean(green_channel)))

        frame_idx += 1
        if progress_callback and frame_idx % 5 == 0:
            progress_callback(min(frame_idx / n_frames_to_read, 1.0))

    cap.release()

    if len(raw_signal) < fps * 5:
        raise ValueError(
            f"Video too short or unreadable ({len(raw_signal)} frames read). "
            "Use at least a 10-15 second clip."
        )

    face_detected_ratio = detected_count / len(raw_signal)
    if progress_callback:
        progress_callback(1.0)

    return np.array(raw_signal), fps, face_detected_ratio


def bandpass_filter(signal_1d, fps, low_bpm=MIN_HR_BPM, high_bpm=MAX_HR_BPM):
    """
    Detrend and bandpass-filter the raw signal to the plausible heart
    rate frequency range, isolating the pulse component from lighting
    drift and slow motion artifacts.
    """
    detrended = detrend(signal_1d)
    nyquist = fps / 2.0
    low_hz = (low_bpm / 60.0) / nyquist
    high_hz = (high_bpm / 60.0) / nyquist
    high_hz = min(high_hz, 0.99)  # stay safely under Nyquist
    b, a = butter(3, [low_hz, high_hz], btype="band")
    filtered = filtfilt(b, a, detrended)
    return filtered


def compute_hr_hrv(filtered_signal, fps):
    """
    Detect pulse peaks in the filtered signal and compute heart rate
    and time-domain HRV metrics (SDNN, RMSSD) from inter-beat intervals.

    Returns a dict with hr_bpm, sdnn_ms, rmssd_ms, ibi_ms (array),
    and n_beats. Returns None values if too few peaks were found to
    compute a reliable estimate.
    """
    min_distance = int(fps * 60.0 / MAX_HR_BPM)
    peaks, _ = find_peaks(filtered_signal, distance=max(min_distance, 1))

    if len(peaks) < 4:
        return {
            "hr_bpm": None, "sdnn_ms": None, "rmssd_ms": None,
            "ibi_ms": np.array([]), "n_beats": len(peaks),
        }

    peak_times_s = peaks / fps
    ibi_ms = np.diff(peak_times_s) * 1000.0

    # Filter out physiologically implausible intervals (likely false peaks)
    valid = (ibi_ms > (60000.0 / MAX_HR_BPM)) & (ibi_ms < (60000.0 / MIN_HR_BPM))
    ibi_ms_clean = ibi_ms[valid]

    if len(ibi_ms_clean) < 3:
        return {
            "hr_bpm": None, "sdnn_ms": None, "rmssd_ms": None,
            "ibi_ms": ibi_ms, "n_beats": len(peaks),
        }

    hr_bpm = 60000.0 / np.mean(ibi_ms_clean)
    sdnn_ms = float(np.std(ibi_ms_clean, ddof=1))
    rmssd_ms = float(np.sqrt(np.mean(np.diff(ibi_ms_clean) ** 2)))

    return {
        "hr_bpm": float(hr_bpm), "sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms,
        "ibi_ms": ibi_ms_clean, "n_beats": len(peaks),
    }


def compute_stress_index(hr_bpm, rmssd_ms):
    """
    Heuristic 0-100 "stress index" combining elevated heart rate with
    reduced HRV (lower RMSSD = less parasympathetic/"rest and digest"
    activity, generally associated with higher sympathetic/stress load).

    This is a simplified educational heuristic, not a validated
    clinical stress score.
    """
    if hr_bpm is None or rmssd_ms is None:
        return None, "Insufficient signal"

    # Normalize HR: 60bpm -> 0, 100bpm -> 1 (clipped)
    hr_component = np.clip((hr_bpm - 60) / 40, 0, 1)
    # Normalize RMSSD: 60ms (high HRV, relaxed) -> 0, 10ms (low HRV, stressed) -> 1
    rmssd_component = np.clip((60 - rmssd_ms) / 50, 0, 1)

    stress_index = float(np.clip(100 * (0.4 * hr_component + 0.6 * rmssd_component), 0, 100))

    if stress_index < 30:
        label = "Relaxed"
    elif stress_index < 60:
        label = "Moderate"
    else:
        label = "Elevated"

    return stress_index, label
