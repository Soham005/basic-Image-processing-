# 🖼️ Basic Image Processing Web App  
A simple and interactive **Streamlit-based** web application that performs fundamental image processing tasks using **OpenCV** and **Pillow**.

This app allows you to:
- Upload an image  
- Convert to grayscale  
- Rotate (90°, 180°, 270°)  
- Flip horizontally/vertically  
- View image properties  
- Detect basic objects without deep learning  
- Download processed images  

---

## 🚀 Live Demo  
👉 *Add your Streamlit Cloud link here*  
Example:  
https://your-app-name.streamlit.app/

---

## 📌 Features  
### ✅ Upload & Preview  
Supports JPG, PNG, JPEG formats.

### 🎨 Image Processing Options  
- Grayscale conversion  
- Automatic object detection (contour-based)  
- Rotate image in multiple directions  
- Mirror flip (horizontal/vertical)  
- Crop tool (optional if you add it)

### 🧠 Object Detection  
Uses **edge detection + contour analysis**  
No ML/DL required.  

---

## 🛠️ Tech Stack  
- **Streamlit** – Web interface  
- **OpenCV (Headless)** – Image processing  
- **NumPy** – Arrays & pixels  
- **Pillow (PIL)** – Image handling  

---

## 📂 Project Structure  
📁 your-repo/ /n
│── app.py /n
│── requirements.txt /n
│── README.md /n
│── LICENSE /n
└── sample_images/ /n


---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
