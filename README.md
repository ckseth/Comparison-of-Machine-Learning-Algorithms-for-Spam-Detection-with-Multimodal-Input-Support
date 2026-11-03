# Comparison of Machine Learning Algorithms for Spam Detection with Multimodal Input Support

[![Conference](https://img.shields.io/badge/Conference-IEEE%20APCIT%202024-blue.svg)](https://ieeexplore.ieee.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)

Official repository for the research paper:  
**"Comparison of Machine Learning Algorithms for Spam Detection with Multimodal Input Support"**  
Presented at the *2024 Asia Pacific Conference on Innovation in Technology (APCIT)*, Mysuru, India (IEEE).

---

## 👥 Authors & Affiliations

- **Anshika Chamoli** — Department of Computer Science and Engineering, Graphic Era Hill University, Dehradun, India
- **Rahul Chauhan** — Department of Computer Science and Engineering, Graphic Era Hill University, Dehradun, India
- **Niharika Bisht** — Department of Computer Science and Engineering, Graphic Era Hill University, Dehradun, India
- **Swati Devliyal** — Department of Computer Science and Engineering, Graphic Era Deemed to be University, Dehradun, India
- **Ravi Ranjan Kumar** — Chitkara University Institute of Engineering and Technology, Chitkara University, Punjab, India

---

## 📌 Abstract

The exponential proliferation of messaging, mobile communication, and digital exchange has triggered an unprecedented surge in unsolicited, fraudulent, and malicious spam communications. In addition to consuming network bandwidth and cloud storage, malicious spam exposes users to credential phishing, malware distribution, and financial theft. 

This research presents a rigorous empirical benchmark comparing machine learning algorithms for spam classification on the **UCI SMS Spam Collection dataset**. The feature extraction framework leverages Term Frequency–Inverse Document Frequency (TF-IDF) with unigram and bigram token representations. Seven supervised machine learning algorithms are implemented and systematically compared:
1. **Random Forest (RF)**
2. **Support Vector Machine (Linear SVM)**
3. **Multinomial Naive Bayes (MNB)**
4. **Decision Tree (DT)**
5. **Gradient Boosting Classifier (GBC)**
6. **Logistic Regression (LR)**
7. **K-Nearest Neighbors (KNN)**

Furthermore, the framework expands traditional text spam filters into a **Multimodal Input Support Architecture**, capable of processing:
- **Raw Text / SMS messages**
- **PDF Documents** (automated text extraction and inspection)
- **Image Inputs via Optical Character Recognition (OCR)** (extracting embedded spam banners and phishing posters)

---

## 📐 Architecture & Workflow

### Multimodal Spam Detection Workflow
The multimodal pipeline takes diverse input formats (Text, PDF, Images), extracts the underlying linguistic tokens via specialized parsers and OCR engines, runs TF-IDF vectorization, and passes the vector to the trained ML model for classification and heuristic indicator summarization.

![Multimodal Spam Detection Workflow](figures/multimodal_spam_detection_workflow.png)

### Proposed Methodology Flowchart
The step-by-step methodology outlining data collection, duplicate purging, tokenization, stopword removal, n-gram TF-IDF vectorization, model training, k-fold hyperparameter validation, and metric evaluation.

![Proposed Methodology Flowchart](figures/proposed_methodology_flowchart.png)

---

## 📊 Experimental Results

All models were evaluated on an 80:20 stratified split of the UCI SMS Spam Collection dataset using Accuracy, Precision, Recall, and F1-Score.

### Benchmark Comparison Table

| Rank | Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **98.07%** | **100.00%** | 84.73% | **91.74%** |
| 🥈 | **Support Vector Machine (SVM)** | 97.87% | 97.39% | 85.50% | 91.06% |
| 🥉 | **Multinomial Naive Bayes** | 97.49% | 100.00% | 80.15% | 88.98% |
| 4 | **Decision Tree** | 97.00% | 89.68% | **86.26%** | 87.94% |
| 5 | **Gradient Boosting** | 96.23% | 100.00% | 70.23% | 82.51% |
| 6 | **Logistic Regression** | 95.94% | 97.85% | 69.47% | 81.25% |
| 7 | **K-Nearest Neighbors (KNN)** | 92.94% | 100.00% | 44.27% | 61.38% |

### Performance Comparison Graph
![Model Comparison](figures/model_comparison.png)

### Confusion Matrix (Top Performing Model: Random Forest)
![Confusion Matrix](figures/confusion_matrix.png)

---

## 🗂 Repository Structure

```plaintext
├── paper/                                   # Research paper documents
│   ├── Comparison_of_Machine_Learning_Algorithms_for_Spam_Detection_with_Multimodal_Input_Support.docx
│   ├── Comparison_of_Machine_Learning_Algorithms_for_Spam_Detection_with_Multimodal_Input_Support.pdf
│   └── Intelligent_Email_Spam_Detection_A_Machine_Learning-Based_Approach.pdf
├── src/                                     # Source code implementation
│   ├── sms_spam_detection.py                # 7 ML models training and evaluation pipeline
│   ├── multimodal_detector.py               # Multimodal pipeline (Text, PDF, Image OCR)
│   └── SMS_Colab.py                         # Google Colab compatible standalone script
├── dataset/                                 # Datasets used in the research
│   ├── SMSSpamCollection                    # Cleaned UCI SMS Spam Collection file
│   ├── sms+spam+collection.zip              # Original UCI archive
│   ├── spambase.zip                         # UCI Spambase dataset
│   └── README.md                            # Dataset description
├── figures/                                 # High-resolution diagrams & result charts
│   ├── multimodal_spam_detection_workflow.png
│   ├── proposed_methodology_flowchart.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── sample_multimodal_image.jpeg
├── results/                                 # Generated evaluation metrics
│   ├── model_comparison_results.csv
│   └── results_summary.md
├── Research_Paper_and_Project_Assets.zip     # Complete bundled archive of all project documents
├── SMS.PY                                   # Main root execution script
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # MIT License
└── README.md                                # Project documentation
```

---

## 📦 Complete Project Zip Download

For convenience, all documents, code, figures, datasets, and paper files are packaged into a single zip archive:
- **[Download Complete Project Archive (ZIP)](Research_Paper_and_Project_Assets.zip)**

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Clone this repository:
```bash
git clone https://github.com/ckseth/Comparison-of-Machine-Learning-Algorithms-for-Spam-Detection-with-Multimodal-Input-Support.git
cd Comparison-of-Machine-Learning-Algorithms-for-Spam-Detection-with-Multimodal-Input-Support
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional for Image OCR)*: If you want to run OCR on image files, install the Tesseract OCR engine on your system:
- **Windows**: Install via `winget install UB-Mannheim.TesseractOCR` or download installer.
- **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

---

## 💻 Usage

### 1. Train and Benchmark the 7 ML Algorithms
Run the benchmark script:
```bash
python SMS.PY
```
Or use the modular script with arguments:
```bash
python src/sms_spam_detection.py --dataset dataset/SMSSpamCollection --output-dir .
```

### 2. Run Multimodal Spam Detection
Test classification across different modalities:

**Text Modality:**
```bash
python src/multimodal_detector.py --modality text --input "Congratulations! You have won a $1,000 Walmart Gift Card. Click here to claim."
```

**PDF Document Modality:**
```bash
python src/multimodal_detector.py --modality pdf --input "path/to/document.pdf"
```

**Image OCR Modality:**
```bash
python src/multimodal_detector.py --modality image --input "figures/sample_multimodal_image.jpeg"
```

---

## 📖 Citation

If you find this research paper or codebase useful in your academic work, please cite:

```bibtex
@inproceedings{chamoli2024comparison,
  title={Comparison of Machine Learning Algorithms for Spam Detection with Multimodal Input Support},
  author={Chamoli, Anshika and Chauhan, Rahul and Bisht, Niharika and Devliyal, Swati and Kumar, Ravi Ranjan},
  booktitle={2024 Asia Pacific Conference on Innovation in Technology (APCIT)},
  year={2024},
  publisher={IEEE},
  address={Mysuru, India}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
