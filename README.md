# 🩺 DermaV1 – Skin Disease Classification Model



**DermaV1** is a machine learning project developed to **detect and classify common skin diseases** from images. Trained using a curated dataset of skin conditions, the model takes an image input and predicts the type of skin disease, helping to demonstrate how AI can support early medical diagnosis and accessibility.

---

## 🚀 Features

- 📸 **Image-based skin disease recognition**
- 🧠 Built with **supervised ML techniques**
- 🧪 Includes full pipeline: **preprocessing → training → evaluation**
- 🖼️ **Web scraping** used to collect training data
- 📊 Exploratory data analysis included
- 📂 Modular code structure for easy scaling

---

## 🧱 Project Structure

| File | Description |
|------|-------------|
| `data_preprocessing.py` | Prepares and augments image data for training |
| `data_analysis.py` | Performs EDA (Exploratory Data Analysis) on dataset |
| `model_building.py` | Builds the machine learning model architecture |
| `model_training.py` | Trains the model on the dataset |
| `model_evaluation.py` | Evaluates accuracy, precision, and other metrics |
| `web scrapper.py` | Collects images from the web for training data |
| `main.py` | Runs the complete inference pipeline |
| `imagefodify.py` / `imgflipper.py` | Data augmentation scripts |
| `LICENSE` | MIT License |

---

## 🧠 Model Information

- Type: **Image Classification Model**
- Framework: `scikit-learn`, `OpenCV`, and `matplotlib`
- Language: **Python**
- Input: JPG/PNG images of skin conditions
- Output: Predicted class of skin disease

---

## 📷 Example Use

```bash
python main.py --image path/to/test_image.jpg
