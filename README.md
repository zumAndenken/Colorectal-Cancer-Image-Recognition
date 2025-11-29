# Large Language Models for Image Classification

**🔬 An Open-Source Project for Evaluating LLMs in Multi-Domain Image Classification**

---

## 📖 Project Overview

This repository contains evaluation scripts and tools for assessing the performance of vision-enabled LLMs in image classification tasks. The project supports four distinct datasets covering different application domains:

1. **AID** - Aerial scene classification (remote sensing)
2. **CRC100K** - Colorectal cancer histopathology tissue classification
3. **GIPD** - Satellite image land use classification
4. **MHIST** - Colon polyp histopathology classification

Each dataset has its own dedicated classification script that leverages local LLM services (via Ollama) to perform image classification with optional few-shot learning capabilities using K-Nearest Neighbors (KNN) for dynamic example selection.

---

## 🔑 Key Components

### Classification Scripts
- **`AID.py`**: AID dataset classification script for aerial scene classification
- **`CRC.py`**: CRC100K dataset classification script for colorectal cancer histopathology tissue classification
- **`GIPD.py`**: GIPD dataset classification script for satellite image land use classification
- **`MHIST.py`**: MHIST dataset classification script for colon polyp histopathology classification

### Core Modules
- **`KNN.py`**: K-Nearest Neighbors module for intelligent few-shot example selection using ResNet50 feature extraction and cosine similarity

### Datasets
- **`Datasets/AID/`**: Aerial scene images organized by class folders
- **`Datasets/CRC100K/`**: Colorectal cancer histopathology images organized by tissue type
- **`Datasets/GIPD/`**: Satellite land use images organized by land use type
- **`Datasets/MHIST/`**: Colon polyp histopathology images organized by polyp type

### Features
- **Zero-Shot & Few-Shot Learning**: Support for both zero-shot classification and few-shot learning with dynamic KNN-based example selection
- **Multi-Domain Support**: Evaluation frameworks for remote sensing, histopathology, and satellite imagery
- **Comprehensive Evaluation**: Automatic generation of confusion matrices, accuracy metrics, and detailed Excel reports
- **Concurrent Processing**: Multi-threaded support for efficient batch processing

---

## 📁 Project Structure

```
.
├── AID.py              # AID dataset classification script
├── CRC.py              # CRC100K dataset classification script
├── GIPD.py             # GIPD dataset classification script
├── MHIST.py            # MHIST dataset classification script
├── KNN.py              # K-Nearest Neighbors module for example selection
├── Datasets/           # Dataset directories
│   ├── AID/            # Aerial scene images
│   ├── CRC100K/        # Colorectal cancer histopathology images
│   ├── GIPD/           # Satellite land use images
│   └── MHIST/          # Colon polyp histopathology images
└── README.md           # Project description
```
## 📜 License

The project is open-sourced under the Apache 2.0 License. For details, please see the LICENSE file.
---

## 📬 Contact

If you have any questions or collaboration intentions, please contact us:

- 📧 Email: sunlianshan@sust.edu.cn(Lianshan Sun);1241917171@qq.com (Diandong Liu)
---
