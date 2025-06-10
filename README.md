# 🍇 Grape Leaf Disease Detection Using CNN and Streamlit

This project uses a Convolutional Neural Network (CNN) to detect and classify grape leaf diseases from images. It includes both a Jupyter Notebook for model training and a Streamlit web app for user-friendly image prediction.

## 📌 Project Description

This system classifies grape leaves into four categories:

- **Black Rot**
- **Esca (Black Measles)**
- **Healthy**
- **Leaf Blight (Isariopsis Leaf Spot)**

The pipeline includes image preprocessing, CNN-based training, and a Streamlit web interface for disease detection.

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/grape-leaf-disease-detection.git
   cd grape-leaf-disease-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional if using provided `.h5` model):
   - Run the Jupyter Notebook to preprocess and train the model.

4. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Features

- CNN-based classification of grape leaf diseases
- Image input through a simple web UI using Streamlit
- Model accuracy/loss visualization
- Confusion matrix and classification report
- Easy-to-use interface for farmers/agronomists

---

## 📁 Folder Structure

```
grape-leaf-disease-detection/
├── app.py                      # Streamlit web app
├── leaf_disease_coloured.h5   # Trained model file
├── notebook.ipynb             # Model training and evaluation notebook
├── x_train_coloured.pickle    # Training features
├── y_train_coloured.pickle    # Training labels
├── x_test_coloured.pickle     # Testing features
├── y_test_coloured.pickle     # Testing labels
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 🔐 Authentication Flow

This project does not include user authentication. It is a single-user inference tool. You can deploy it as a public or local web tool.

---

## 📊 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy, Matplotlib, Seaborn

---

## ✅ Results

- Achieved accurate predictions on grape leaf disease dataset.
- Visualized performance metrics and predictions.
- Real-time image predictions via a lightweight interface.

---

## 📄 License

This project is for educational and research purposes. You may modify and reuse it with attribution.

---

## 💡 Suggestions

- Use a larger dataset for better generalization.
- Add data augmentation to improve robustness.
- Deploy the model on cloud (e.g., AWS, Heroku).
