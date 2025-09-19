# 🖼️ Image Quality Analyzer

This project is an **AI-powered Image Quality Analyzer** that classifies uploaded images as **Good** or **Bad** and detects issues like **blur, dullness, or low contrast**.  
It is built using a **ResNet34 CNN model** trained on a custom dataset and deployed with **Streamlit** for an interactive web app experience.

---

## 🚀 Features

* Upload any image and instantly get quality classification.
* Detects issues like blur, darkness, or dullness.
* Powered by **PyTorch (ResNet34)** for image classification.
* Simple and interactive UI built with **Streamlit**.
* Deployed on **Streamlit Community Cloud**.

---

## 🎥 Demo

 Watch the demo video below:
<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## ⚙️ Installation

1. Clone the repo:

````
git clone https://github.com/your-username/image-quality-analyzer.git
cd image-quality-analyzer 
````
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the app locally:
```
streamlit run app.py
```
4. Project Structure
```
.
├── app.py               # Streamlit app script
├── model.pth            # Trained ResNet34 model
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── demo.mp4             # Demo video (optional, if you want to keep locally)
```
🛠️ Tech Stack
```
- Python
- PyTorch
- Torchvision
- OpenCV
- Pillow
- Streamlit for deployment
```
## Future Work

- Add support for detecting noise, compression artifacts, and color imbalance.

- Improve model accuracy using EfficientNet / Vision Transformers.

- Add confidence visualization for predictions.

- Deploy on Docker + AWS/GCP for production scalability.
