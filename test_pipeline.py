"""
test_pipeline.py
==================
Generates a synthetic test video with a KNOWN embedded heart rate
(via a pulsating green channel in a center region) and runs it through
the full rppg.py pipeline to verify HR/HRV recovery works correctly
before this ships. Not part of the deployed app -- a build-time sanity
check only.
"""

import numpy as np
import cv2
import tempfile
import os
from rppg import extract_raw_signal, bandpass_filter, compute_hr_hrv, compute_stress_index


def generate_synthetic_video(path, true_bpm=72, duration_s=25, fps=30, width=320, height=240):
    """
    Create a video with a skin-tone-colored rectangle whose green
    channel pulses at a known BPM, simulating a real pulse signal.
    No actual face is drawn (Haar cascade won't detect one), which
    exercises the fallback center-ROI path used for real videos with
    occasional missed detections.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    n_frames = int(duration_s * fps)
    t = np.arange(n_frames) / fps
    true_hz = true_bpm / 60.0
    # Small-amplitude pulse on top of a skin-tone base color, plus mild noise
    pulse = 8 * np.sin(2 * np.pi * true_hz * t) + np.random.normal(0, 1.5, n_frames)

    for i in range(n_frames):
        frame = np.full((height, width, 3), (120, 150, 180), dtype=np.uint8)  # BGR skin-ish tone
        green_val = np.clip(150 + pulse[i], 0, 255)
        frame[:, :, 1] = green_val
        # add slight per-pixel noise so it's not a flat color field
        noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return true_bpm


def main():
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, "synthetic_test.mp4")

    print("Generating synthetic test video with known heart rate...")
    true_bpm = generate_synthetic_video(video_path, true_bpm=72, duration_s=25)
    print(f"  Embedded true HR: {true_bpm} BPM")

    print("Running extraction pipeline...")
    raw_signal, fps, face_ratio = extract_raw_signal(video_path, max_seconds=30)
    print(f"  Frames processed: {len(raw_signal)}, fps: {fps}, face-detected ratio: {face_ratio:.2f}")

    filtered = bandpass_filter(raw_signal, fps)
    result = compute_hr_hrv(filtered, fps)
    print(f"  Detected HR: {result['hr_bpm']:.1f} BPM (true: {true_bpm})")
    print(f"  SDNN: {result['sdnn_ms']:.1f} ms, RMSSD: {result['rmssd_ms']:.1f} ms, beats: {result['n_beats']}")

    stress_idx, stress_label = compute_stress_index(result["hr_bpm"], result["rmssd_ms"])
    print(f"  Stress index: {stress_idx:.1f} ({stress_label})")

    error = abs(result["hr_bpm"] - true_bpm)
    print(f"\nHR estimation error: {error:.2f} BPM")
    assert error < 5.0, f"HR estimate off by more than 5 BPM (got {error:.2f})"
    assert result["sdnn_ms"] is not None
    assert stress_idx is not None

    print("\nALL PIPELINE CHECKS PASSED")

    os.remove(video_path)


if __name__ == "__main__":
    main()
