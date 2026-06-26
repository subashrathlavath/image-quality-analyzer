# 🖼️ Image Quality Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-green)

This project is an **AI-powered Image Quality Analyzer** that classifies uploaded images as **Good** or **Bad** and detects issues like **blur, dullness, or low contrast**. It is built using a **ResNet34 CNN model** trained on a custom dataset and deployed with **Streamlit** for an interactive web app experience.

---

## 🚀 Features

* Upload any image and instantly get quality classification.
* Detects issues like blur, darkness, or dullness.
* Powered by **PyTorch (ResNet34)** for image classification.
* Simple and interactive UI built with **Streamlit**.
* Deployed on **Streamlit Community Cloud**.
* Real-time feedback with confidence scores.

---

## 🎥 Demo

Watch the demo video below:

[![Watch Demo](https://img.youtube.com/vi/I3FEKGYdayQ/0.jpg)](https://youtu.be/I3FEKGYdayQ?si=vZ-VfPIEQJFU_Lxg)

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/subashrathlavath/image-quality-analyzer.git
cd image-quality-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app locally:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── model.pth              # Trained ResNet34 model weights
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── demo.gif               # Demo animation
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **PyTorch** | Deep learning framework |
| **Torchvision** | Pre-trained models & image utilities |
| **OpenCV** | Image processing |
| **Pillow** | Image handling |
| **Streamlit** | Web app framework & deployment |

---

## 📊 Model Details

- **Architecture:** ResNet34 (Pre-trained on ImageNet, fine-tuned on custom dataset)
- **Input Size:** 224×224 pixels
- **Output:** Binary classification (Good/Bad) with confidence score
- **Supported Formats:** JPG, PNG, BMP, TIFF

---

## 💡 Usage

1. Open the Streamlit app
2. Upload an image using the file uploader
3. Click **Analyze** to process the image
4. View the classification result and detected issues
5. Download the analysis report (optional)

### Example Output
```
Classification: Good ✓
Confidence: 94.2%
Issues Detected: None
```

---

## 📈 Future Work

- Add support for detecting noise, compression artifacts, and color imbalance
- Improve model accuracy using EfficientNet / Vision Transformers
- Add confidence visualization and detailed analysis charts
- Deploy on Docker + AWS/GCP for production scalability
- Add batch processing for multiple images
- Support for video frame analysis

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please reach out via GitHub Issues.

---

**Made with ❤️ by [subashrathlavath](https://github.com/subashrathlavath)**
