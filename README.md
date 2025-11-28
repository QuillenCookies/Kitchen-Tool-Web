# 📷 Real-Time Object Detection with Wiki Lookup

A real-time computer vision application built with **Streamlit**, **OpenCV**, and **MobileNet SSD**. This project detects objects via webcam stream and provides an interactive dashboard to view detailed information about identified classes (Wiki).

## 🚀 Features

- **Real-Time Detection:** Low-latency video streaming and processing using `streamlit-webrtc`.
- **Object Classification:** Identifies 8 classes of objects (Chopsticks, Fork, Knife (Butter), Plate, etc.) using the MobileNet SSD model.
- **Interactive Dashboard:** - Adjust confidence threshold dynamically.
- **Smart Linking:** Click on detected object labels in the results table to navigate to a dedicated "Wiki" sub-page for detailed information.
- **Data Visualization:** Real-time bounding boxes and confidence scores overlay.

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit** (Frontend & UI)
- **OpenCV** (Image Processing & DNN Module)
- **MobileNet SSD** (Pre-trained Caffe Model)
- **Pandas & NumPy** (Data handling)

## ⚙️ Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the repository

```bash
git clone [https://github.com/QuillenCookies/Kitchen-Tool-Web]
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to mange dependencies.

**For Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**For masOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure you have `requirements.txt` ready (or install manually).

```bash
pip install -r requirements.txt
```

If you need to update Streamlit to the latest version:

```bash
pip install --upgrade streamlit
```

### 4. Run the App

```bash
streamlit run Menu.py
```

## 📂 Project Structure

```TEST
├── Detection_models
├── models/                  # Caffe model files (MobileNetSSD)
├── objects
├── pages_modules/
│   └── about_us.py          # About us page
    └── home.py              # Page for object detection
    └── wikipedia.py         # Wiki page for object
├── sample_utils
    └── download.py          # Sample resource for Detection
├── streamlit
    └── config.toml          # Config page used for design
├── Menu.py                  # Main application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🤝 Creadits & Inspiration

- The Github takes inspiration from this [Streamlit Discussion](https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669).
