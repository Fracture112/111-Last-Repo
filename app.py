import streamlit as st
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import matplotlib.pyplot as plt
import re
from skimage.feature import canny
from scipy.ndimage import center_of_mass

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def detect_crack_origin(gray):
    edges = canny(gray, sigma=2)
    y_coords, x_coords = np.nonzero(edges)
    if len(x_coords) == 0:
        return None
    cy, cx = center_of_mass(edges)
    return int(cx), int(cy)

def detect_beach_marks(gray):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    gradient_magnitude = np.mean(np.abs(sobel))
    return gradient_magnitude > 2.2  # LOWERED THRESHOLD TO MATCH VISUAL

def detect_chevron_marks(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(lap)
    return variance > 400  # RAISED TO AVOID FALSE POSITIVES

def analyze_fracture_features(image):
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    h, w = gray.shape
    symmetry = abs(np.mean(edges[:h//3]) - np.mean(edges[2*h//3:]))

    origin = detect_crack_origin(gray)
    beach = detect_beach_marks(gray)
    chevron = detect_chevron_marks(gray)

    failure_mode = "Fatigue" if beach else "Overload"
    stress_type = "Reversed Bending" if symmetry < 15 else "Bending"
    origin_count = "1" if origin else "0"

    table = f"""
| Feature | Analysis |
|--------|----------|
| Failure Mode | {failure_mode} |
| Type of Stress | {stress_type} |
| Beach Marks | {"Present" if beach else "Not visible"} |
| Chevron Marks | {"Present" if chevron else "Absent"} |
| Origin Count | {origin_count} |
| Additional Notes | Crack origin at {origin if origin else "unknown"}, symmetry = {symmetry:.1f} |
"""

    summary_info = {
        "failure_mode": failure_mode,
        "stress_type": stress_type,
        "beach": beach,
        "chevron": chevron,
        "origin": origin,
        "symmetry": symmetry
    }

    return edges, summary_info, table

def generate_summary(summary_info):
    description = f"""
This fracture shows a {summary_info["failure_mode"]} failure. 
Visual inspection detected {"clear beach marks" if summary_info["beach"] else "no visible beach marks"} 
and {"chevron marks" if summary_info["chevron"] else "no chevron marks"}.
Crack origin was {'detected at ' + str(summary_info["origin"]) if summary_info["origin"] else 'not clearly detected'}.
The fracture appears {"symmetric, suggesting reversed bending" if summary_info["symmetry"] < 15 else "asymmetric, indicating simple bending or overload"}.
"""

    prompt = f"""
Write a concise technical summary based on this fracture description:

"""{description}"""

Focus on metallurgy and failure mechanics.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional metallurgical failure analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# --- STREAMLIT UI ---
st.set_page_config(page_title="ANDALAN FRACTOGRAPHY SOLVER", layout="centered")
st.title("🧠 ANDALAN FRACTOGRAPHY SOLVER – Chat-Synced Final")

uploaded_file = st.file_uploader("Upload a fracture image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    with st.spinner("Analyzing image..."):
        edges, summary_info, table = analyze_fracture_features(image)

    st.subheader("Edge Detection")
    st.image(edges, clamp=True, use_column_width=True)

    st.subheader("Fracture Table")
    st.markdown(table)

    with st.spinner("Generating summary..."):
        summary = generate_summary(summary_info)

    st.markdown("### Summary")
    st.markdown(summary)
else:
    st.info("👆 Upload a fracture image to begin.")