# Edge Computing Flame & Cigarette Detection System for Industrial Safety

An edge-computing prototype combining computer vision and machine learning to support safety monitoring in high-risk industrial environments, such as fuel stations and petrochemical facilities. The system employs a prioritized alert hierarchy to detect open flames and smoking violations in real time.

---

## Key Objectives & Overview

In environments with high explosion hazards, early hazard detection is critical. This system focuses on:

- **Hierarchical Hazard Detection:** Prioritizes flame detection before triggering secondary classification layers to reduce unnecessary processing.
- **Low-Latency Edge Processing:** Designed to process live video streams locally on compatible edge hardware.
- **False Alarm Mitigation:** Uses multi-stage filtering to reduce transient detections and improve alert stability.

---

## Engineering Approach & Methodology

### 1. Prioritized Flame Detection — OpenCV HSV Color Masking

- Transforms real-time video frames into the **HSV (Hue, Saturation, Value)** color space.
- Applies color thresholding and masking to identify potential flame-like regions.
- **Priority Pipeline:** Potential fire hazards receive the highest alert priority, reducing unnecessary inference overhead when a flame condition is detected.

### 2. Smoking Violation Classification — RBF Kernel SVM

- Applies machine learning inference to relevant image regions when no higher-priority flame alert is active.
- **Image Preprocessing:** Resizes input images to a standardized resolution of \(64 \times 64\) pixels.
- **Model Architecture:** Uses a Support Vector Classifier (`SVC`) with a **Radial Basis Function (RBF) kernel**.
- **Class Balancing:** Uses `class_weight='balanced'` to help mitigate the effects of imbalanced training data.

### 3. Multi-Layer False Alarm Filtering

The system incorporates multiple filtering mechanisms intended to improve detection stability under varying lighting and environmental conditions:

- **Standard Deviation-Based Scene Filtering:** Filters low-variation image regions that may provide insufficient visual information.
- **Confidence Thresholding:** Applies a decision threshold before issuing classification-based alerts, where supported by the model configuration.
- **Temporal Buffer Windows:** Evaluates consecutive frames to reduce alerts caused by transient detections.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV (`cv2`) |
| Machine Learning | scikit-learn (`SVC`, SVM) |
| Model Persistence | Joblib |
| Scientific Computing | NumPy |

---

## Detection Demonstrations

| Flame Detection (`ATES TEHLIKE`) | Cigarette Violation (`SIGARA TEHLIKESI`) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/696b8bb6-0948-46c4-8004-3381f080b5ef" width="420" alt="ates" /> | <img src="https://github.com/user-attachments/assets/961864e6-1bf0-476f-bf0c-b59c5ad0de03" width="420" alt="sigara" /> |

The demonstrations are intended to illustrate prioritized threat classification and temporal alert states, including `GENEL TEHLIKE`.

To display images hosted in the repository, replace the placeholders with your actual image paths. For example:

```markdown
| Flame Detection | Cigarette Violation |
|:---:|:---:|
| ![Flame Detection](images/ates_tehlike.png) | ![Cigarette Detection](images/sigara_tehlikesi.png) |
```

---

## Model & Setup

### 1. Download the Pretrained Model

Due to file size constraints, the trained SVM model (`tehlike_modeli.pkl`) may be hosted externally.

- **[Download Trained Weights (`tehlike_modeli.pkl`)](https://drive.google.com/file/d/1hdwBzAkhW3D_2FkCnN_V6OQUWUEPUE95/view?usp=sharing)**

Download the model and place `tehlike_modeli.pkl` in the repository's root directory before running inference.

### 2. Installation

Install the required Python packages:

```bash
pip install opencv-python scikit-learn joblib numpy
```

### 3. Execution

Run real-time detection using a webcam or a supported video stream:

```bash
python kamera.py
```

Optionally, retrain the SVM classifier using the training script:

```bash
python egitim.py
```

**Note:** Ensure that the script filenames, model paths, and dependencies match the actual repository structure before execution.

---

## Limitations & Safety Disclaimer

This repository presents a computer vision and machine learning prototype for industrial safety monitoring. Detection performance may vary with lighting, camera quality, environmental conditions, and training data.

- HSV-based flame detection relies on color characteristics and may confuse flames with visually similar light sources.
- SVM classification performance depends on the quality, diversity, and representativeness of the training dataset.
- Detection results should be evaluated using appropriate test data and documented performance metrics.
- This prototype is not a replacement for certified fire detection systems, industrial safety equipment, or established emergency response procedures.

---

## License & Proprietary Rights

**Copyright © 2026 Remziye Azra Sifil . All Rights Reserved.**

This project, including its source code, trained model weights, algorithmic design, and research methodology, is claimed as proprietary work and is intended to be protected under applicable intellectual property laws.

- **No Unauthorized Distribution:** Unauthorized copying, modification, redistribution, relicensing, or commercial deployment of this work is prohibited, subject to applicable law and any rights or permissions that may apply.
- **Academic & Research Inquiries:** Anyone wishing to use, reference, reproduce, or build upon this work beyond legally permitted uses should contact the repository owner to discuss permission requirements.
- **Contact:** Contact the repository owner through GitHub or submit an inquiry using the repository's issue tracker.

**Usage Notice:** Unless a separate license or written permission is provided, no license is granted to reuse or redistribute this project's proprietary materials. This notice does not override applicable legal exceptions or third-party rights.

