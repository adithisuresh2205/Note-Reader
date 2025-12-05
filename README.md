# ✍️ Handwritten English Characters & Digits Recognition

A deep-learning model that accurately reads **handwritten digits (0–9), uppercase (A–Z), and lowercase letters (a–z)** using an EfficientNet-based neural network.

---

## 🚀 Features

* Recognizes **62 classes** (0–9, A–Z, a–z)
* Uses **EfficientNetB0** with mixed-precision for fast training
* Handles augmented datasets from Kaggle
* Reads and predicts characters from custom images
* GPU-optimized, lightweight, and easy to deploy

---

## 📌 Project Structure

* **Training Pipeline:** Dataset loading → normalization → model training
* **Model:** EfficientNetB0 + Dense layers
* **Inference Script:** Upload an image → get predicted character and confidence

---

## 🧠 Model Architecture

* EfficientNetB0 (pretrained on ImageNet)
* Custom classification head
* 62-class softmax output

---

## 🗂 Dataset

Dataset used: **Handwritten English Characters & Digits (Kaggle)**

* Includes augmented images
* Combined train/test folders
* Supports RGB images resized to 128×128

---

## 🧪 Training

```bash
# Train the model
model.fit(train_ds, epochs=50, validation_data=val_ds)
```

### Output Metrics

* Training Accuracy
* Validation Accuracy
* Test Accuracy

---

## 🔍 Prediction Example

```python
predict_character("a.jpg")
# Output: Predicted Character: a (Confidence: 98.52%)
```

---

## 📦 Installation

```bash
pip install opendatasets kaggle tensorflow keras matplotlib seaborn scikit-learn
```

---

## 💾 Saving & Loading Model

```python
model.save("optimized_model.keras")
model = keras.models.load_model("optimized_model.keras")
```

---

## 🖼 Sample Inference Code

```python
pred = predict_character("/content/a.jpg")
print("Predicted:", pred)



