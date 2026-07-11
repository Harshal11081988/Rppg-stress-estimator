"""
Webcam-based Stress & Recovery Estimator (rPPG)
--------------------------------------------------
Upload a short face video and this app estimates your heart rate and
heart-rate variability (HRV) purely from subtle skin color changes --
no wearable hardware required. HRV is used as a proxy for autonomic
nervous system balance (stress vs. recovery).

⚠️ This is an experimental wellness tool, not a medical device. Results
are sensitive to lighting, motion, and video quality, and should not be
used for diagnosis or treatment decisions.

Run locally with: streamlit run app.py
"""

import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from rppg import extract_raw_signal, bandpass_filter, compute_hr_hrv, compute_stress_index

st.set_page_config(page_title="rPPG Stress Estimator", page_icon="💓", layout="wide")


def main():
    st.title("💓 Webcam Stress & Recovery Estimator")
    st.caption(
        "Estimates heart rate and heart-rate variability (HRV) from a short face video "
        "using remote photoplethysmography (rPPG) — detecting your pulse from tiny color "
        "changes in facial skin, no wearable required."
    )

    st.warning(
        "⚠️ **Experimental wellness tool, not a medical device.** Accuracy depends heavily "
        "on lighting, camera quality, and staying still. Don't use this for diagnosis or "
        "treatment decisions."
    )

    with st.expander("📋 Tips for a good reading"):
        st.markdown(
            "- Record **20-30 seconds** of steady, well-lit video facing the camera\n"
            "- Use natural or bright indoor light — avoid backlighting\n"
            "- Keep your head relatively still; avoid talking during the clip\n"
            "- Plain background helps but isn't required\n"
            "- Supported formats: MP4, MOV, AVI"
        )

    uploaded_file = st.file_uploader("Upload a face video", type=["mp4", "mov", "avi", "webm", "m4v"])

    if uploaded_file is None:
        st.info("Upload a video above to get started.")
        return

    # Save to a temp file since OpenCV needs a filesystem path
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    try:
        st.video(uploaded_file)

        progress_bar = st.progress(0.0, text="Extracting pulse signal from video...")

        def update_progress(frac):
            progress_bar.progress(frac, text=f"Extracting pulse signal... {int(frac * 100)}%")

        raw_signal, fps, face_ratio = extract_raw_signal(video_path, progress_callback=update_progress)
        progress_bar.empty()

        if face_ratio < 0.3:
            st.warning(
                f"⚠️ Face was only detected in {face_ratio * 100:.0f}% of frames — results may "
                "be less reliable. Try better lighting or facing the camera more directly."
            )

        filtered_signal = bandpass_filter(raw_signal, fps)
        result = compute_hr_hrv(filtered_signal, fps)

        if result["hr_bpm"] is None:
            st.error(
                "Couldn't detect a reliable pulse signal from this video. Try a longer, "
                "steadier clip with better lighting."
            )
            return

        stress_idx, stress_label = compute_stress_index(result["hr_bpm"], result["sdnn_ms"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Heart Rate", f"{result['hr_bpm']:.0f} BPM")
        col2.metric("SDNN (HRV)", f"{result['sdnn_ms']:.0f} ms")
        col3.metric("RMSSD (HRV)", f"{result['rmssd_ms']:.0f} ms")
        col4.metric("Stress Index", f"{stress_idx:.0f}/100", stress_label)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Extracted Pulse Waveform")
            time_axis = np.arange(len(filtered_signal)) / fps
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_axis, y=filtered_signal, mode="lines", name="Filtered pulse signal"))
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Amplitude (filtered)",
                height=350, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Beat-to-Beat Intervals")
            if len(result["ibi_ms"]) > 0:
                fig_ibi = go.Figure()
                fig_ibi.add_trace(go.Scatter(
                    y=result["ibi_ms"], mode="lines+markers", name="Inter-beat interval (ms)"
                ))
                fig_ibi.update_layout(
                    xaxis_title="Beat number", yaxis_title="Interval (ms)",
                    height=350, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_ibi, use_container_width=True)
            else:
                st.info("Not enough beats detected to plot intervals.")

        st.markdown("---")
        with st.expander("How to read these numbers"):
            st.markdown(
                "- **Heart Rate (BPM):** beats per minute, averaged over the clip\n"
                "- **SDNN:** standard deviation of beat-to-beat intervals — overall HRV magnitude\n"
                "- **RMSSD:** a measure more specific to short-term, parasympathetic "
                "(\"rest and digest\") activity. Higher RMSSD generally reflects better "
                "recovery/relaxation; lower RMSSD often accompanies stress or fatigue\n"
                "- **Stress Index:** a simplified heuristic combining elevated heart rate with "
                "reduced HRV — not a validated clinical score, just a directional signal"
            )

    finally:
        os.unlink(video_path)


if __name__ == "__main__":
    main()
