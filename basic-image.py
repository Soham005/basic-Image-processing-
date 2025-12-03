import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Interactive Image Processing App 🚀")
st.write("Upload an image and perform various operations!")

# -------------------------
# IMAGE UPLOAD SECTION
# -------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    st.image(img_np, caption="Original Image", use_column_width=True)

    # -------------------------
    # IMAGE PROPERTIES
    # -------------------------
    st.subheader("Image Properties 📐")
    st.write(f"**Shape:** {img_np.shape}")
    st.write(f"**Height:** {img_np.shape[0]}")
    st.write(f"**Width:** {img_np.shape[1]}")
    st.write(f"**Channels:** {img_np.shape[2] if len(img_np.shape)==3 else 1}")

    # -----------------------------------
    # BLACK & WHITE IMAGE
    # -----------------------------------
    st.subheader("Convert to Black & White 🖤")
    if st.button("Convert"):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Black & White Image", use_column_width=True)

    # -----------------------------------
    # ROTATE IMAGE
    # -----------------------------------
    st.subheader("Rotate Image ↩️")
    angle = st.selectbox("Choose rotation", [90, 180, 270])

    if st.button("Rotate"):
        rotated = cv2.rotate(img_np, 
                             cv2.ROTATE_90_CLOCKWISE if angle == 90 else
                             cv2.ROTATE_180 if angle == 180 else
                             cv2.ROTATE_90_COUNTERCLOCKWISE)
        st.image(rotated, caption=f"Rotated {angle}°", use_column_width=True)

    # -----------------------------------
    # MIRROR / FLIP IMAGE
    # -----------------------------------
    st.subheader("Flip / Mirror Image 🪞")
    flip_type = st.selectbox("Flip Type", ["Horizontal", "Vertical"])

    if st.button("Flip"):
        flipped = cv2.flip(img_np, 1 if flip_type=="Horizontal" else 0)
        st.image(flipped, caption=f"{flip_type} Flip", use_column_width=True)

    # -----------------------------------
    # OBJECT DETECTION USING CONTOURS (No DL)
    # -----------------------------------
    st.subheader("Detect Objects Without Deep Learning 🔍")

    if st.button("Detect Objects"):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img_np.copy()
        cv2.drawContours(output, contours, -1, (0,255,0), 2)

        st.image(output, caption=f"Objects Detected: {len(contours)}", use_column_width=True)

    # -----------------------------------
    # IMAGE CROP SECTION
    # -----------------------------------
    st.subheader("Crop the Image ✂️")

    x1 = st.number_input("x1", min_value=0, max_value=img_np.shape[1])
    y1 = st.number_input("y1", min_value=0, max_value=img_np.shape[0])
    x2 = st.number_input("x2", min_value=0, max_value=img_np.shape[1])
    y2 = st.number_input("y2", min_value=0, max_value=img_np.shape[0])

    if st.button("Crop Image"):
        if x2 > x1 and y2 > y1:
            cropped = img_np[int(y1):int(y2), int(x1):int(x2)]
            st.image(cropped, caption="Cropped Image", use_column_width=True)
        else:
            st.error("Invalid crop coordinates!")
