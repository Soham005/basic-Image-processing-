import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Interactive Image Processing", layout="wide")

# ---------- Utility functions ----------

def load_image_bgr(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def show_image_bgr(img_bgr, caption=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, use_column_width=True)

def to_grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def get_properties(img_bgr):
    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1
    return {
        "Width": w,
        "Height": h,
        "Channels": channels,
        "Shape": img_bgr.shape,
        "Data Type": str(img_bgr.dtype),
        "Total Pixels": img_bgr.size
    }

def rotate_image(img_bgr, angle):
    mapping = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img_bgr, mapping.get(angle, None))

def mirror_image(img_bgr):
    return cv2.flip(img_bgr, 1)

def make_grid(img_bgr, rows=4, cols=4):
    h, w = img_bgr.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    grid_img = img_bgr.copy()

    for r in range(1, rows):
        y = r * cell_h
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)

    for c in range(1, cols):
        x = c * cell_w
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)

    return grid_img

def detect_objects(img_bgr, min_area=500):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_img = img_bgr.copy()
    count = 0

    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(obj_img, (x, y), (x + w), (y + h), (0, 255, 0), 2)
            count += 1

    return obj_img, count


# ---------- UI Header ----------
st.title("🧠📷 Interactive Image Processing App")
st.markdown("Perform all basic image processing operations interactively.")


# ---------- File Upload ----------
uploaded_file = st.file_uploader("📤 Upload JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])

if uploaded_file:

    img_bgr = load_image_bgr(uploaded_file)

    # Tabs layout
    tabs = st.tabs([
        "📸 Show Image",
        "⚫ Grayscale",
        "📐 Properties",
        "🔄 Rotate",
        "🪞 Mirror",
        "🔳 Grid",
        "🕵️ Object Detection",
        "✨ All-in-One"
    ])

    # 1 - Original Image
    with tabs[0]:
        st.header("Original Image")
        show_image_bgr(img_bgr)

    # 2 - Grayscale
    with tabs[1]:
        st.header("Black & White Conversion")
        gray = to_grayscale(img_bgr)
        st.image(gray, caption="Grayscale", clamp=True, use_column_width=True)

    # 3 - Properties
    with tabs[2]:
        st.header("Image Properties")
        props = get_properties(img_bgr)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

    # 4 - Rotate
    with tabs[3]:
        st.header("Rotate Image")
        angle = st.radio("Select rotation angle:", [90, 180, 270], horizontal=True)
        rotated = rotate_image(img_bgr, angle)
        show_image_bgr(rotated, f"Rotated {angle}°")

    # 5 - Mirror
    with tabs[4]:
        st.header("Mirror Image (Horizontal Flip)")
        mirrored = mirror_image(img_bgr)
        show_image_bgr(mirrored, "Mirrored Image")

    # 6 - Grid
    with tabs[5]:
        st.header("4×4 Grid on Image")
        grid = make_grid(img_bgr, 4, 4)
        show_image_bgr(grid, "Image with Grid")

    # 7 - Object Detection
    with tabs[6]:
        st.header("Object Detection (No Deep Learning)")
        min_area = st.slider("Minimum Object Area", 100, 5000, 500)
        obj_img, count = detect_objects(img_bgr, min_area)
        st.write(f"**Detected objects:** {count}")
        show_image_bgr(obj_img, "Objects Detected")

    # 8 - All Operations
    with tabs[7]:
        st.header("All Operations Combined")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            show_image_bgr(img_bgr)

            st.subheader("Grayscale")
            st.image(to_grayscale(img_bgr), clamp=True)

            st.subheader("Grid (4×4)")
            show_image_bgr(make_grid(img_bgr), "Grid")

        with col2:
            st.subheader("Rotation (90/180/270)")
            show_image_bgr(rotate_image(img_bgr, 90), "90°")
            show_image_bgr(rotate_image(img_bgr, 180), "180°")
            show_image_bgr(rotate_image(img_bgr, 270), "270°")

            st.subheader("Mirror Image")
            show_image_bgr(mirror_image(img_bgr))

        st.subheader("Object Detection")
        obj_img_all, count_all = detect_objects(img_bgr)
        st.write(f"**Objects detected:** {count_all}")
        show_image_bgr(obj_img_all)

else:
    st.info("Please upload an image to begin.")
