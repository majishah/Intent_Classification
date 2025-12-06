# Resource-Aware Conditional Cascaded (RACC-SLM) Framework for Zero-Shot Intent Detection in Real-Time Voice Streams

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)](https://link.springer.com/journal/607)

> **Official Repository for the manuscript submitted to Springer - Computing.**

This repository implements a **Resource-Aware Conditional Cascaded Framework** for real-time Spoken Language Understanding (SLU) on edge devices. Unlike cloud-centric approaches, this system leverages **Small Language Models (SLMs)** and a hierarchical pruning strategy to achieve sub-50ms inference latency without requiring task-specific training data.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Novelty](#-key-features--novelty)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Detailed Documentation](#-detailed-documentation)
- [Citation](#-citation)

---

## 🔭 Project Overview

Real-time intent detection in smart environments often faces a trade-off between accuracy (Cloud LLMs) and privacy/latency (Edge devices). This project bridges that gap by orchestrating a pipeline that combines:
1.  **Acoustic Gating:** Lightweight VAD to filter noise and silence.
2.  **Efficient Transcription:** Distilled Speech-to-Text models (`faster-distil-whisper`).
3.  **Zero-Shot Inference:** Entailment-based SLMs (`nli-MiniLM2`) using a hierarchical search.

The system is designed to run on constrained hardware (e.g., IoT Hubs, Raspberry Pi) while maintaining robust performance in multi-user environments.

---

## 🚀 Key Features & Novelty

### 1. Resource-Aware Conditional Cascade
Instead of a linear pipeline, the system employs an **Early-Exit Strategy**. Heavy computation (ASR/NLU) is only triggered when acoustic gating conditions (VAD + SNR check) are met, significantly reducing energy consumption.

### 2. Edge-Native Zero-Shot Classification
We utilize **`nli-MiniLM2-L6-H768`**, a 66M parameter Small Language Model (SLM). This allows for:
*   **Privacy:** No audio or text leaves the device.
*   **Adaptability:** New intents can be added simply by updating the configuration, without model retraining.

### 3. Hierarchical Intent Pruning
To optimize inference speed, we replace standard $O(N)$ linear search with a **Coarse-to-Fine** approach:
*   **Level 1:** Broad Domain Classification (e.g., *Home Automation* vs. *Media*).
*   **Level 2/3:** Specific Intent Sub-graph activation.
*   *Result:* ~80% reduction in classification FLOPs per inference.

---

## 🏗 System Architecture

The workflow follows the **Conditional Cascaded Framework** described in the manuscript:

1.  **Input Stream:** Real-time audio capture via `PyAudio`.
2.  **Stage 1: Acoustic Gating:** 
    *   **VAD:** `Silero VAD` isolates speech segments.
    *   **Preprocessing:** Gain control and noise suppression ensure signal clarity.
3.  **Stage 2: Transcription:**
    *   **Engine:** `Faster-Whisper` (Distilled Medium/Small).
    *   **Metric:** Computes average log-probability for confidence estimation.
4.  **Stage 3: Hierarchical Inference:**
    *   **Encoder:** `Cross-Encoder` utilizing `nli-MiniLM2`.
    *   **Logic:** Hierarchical pruning calculates semantic entailment between the transcript and candidate labels.

---

## 📂 Repository Structure

This repository is organized to facilitate reproducibility and code review:

```text
├── models/                  # Directory for storing downloaded model weights
├── utils.py                 # Helper functions for logging and formatted output
├── config.py                # Configuration: thresholds, model paths, intent hierarchy
├── speech_recognizer.py     # Core ASR module (VAD + Faster-Whisper + Buffering)
├── intent_classifier.py     # NLU module (Zero-Shot Hierarchical Logic)
├── main.py                  # Orchestrator: Initializes system and runs the loop
├── requirements.txt         # Python dependencies
│
├── docs/
│   ├── Average_Log_Probabilty_Calculation.md  # Math behind ASR confidence
│   ├── Hierarchical_Intent_Classification.md  # Explanation of the pruning logic
│   └── WorkFlow_Explanation.md                # End-to-end system data flow
│
└── README.md                # Project documentation
```

---

## 📚 Detailed Documentation

To support cross-validation of our methodology, we provide three focused documentation files:

*   **[WorkFlow_Explanation.md](docs/WorkFlow_Explanation.md)**: A comprehensive guide to the system's architecture, data flow, and module interactions.
*   **[Hierarchical_Intent_Classification.md](docs/Hierarchical_Intent_Classification.md)**: Details the Coarse-to-Fine reasoning process and how the Zero-Shot NLI model scales to new intents.
*   **[Average_Log_Probabilty_Calculation.md](docs/Average_Log_Probabilty_Calculation.md)**: Explains the mathematical basis for the confidence metrics used in the gating mechanism.

---

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   PortAudio (for microphone access)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/edge-native-intent-detection.git
cd edge-native-intent-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
*Key libraries: `torch`, `faster-whisper`, `transformers`, `pyaudio`, `numpy`, `silero-vad`.*

### Step 3: Model Setup
The system will automatically download the required models (`faster-distil-whisper` and `nli-MiniLM2`) on the first run. Ensure you have an active internet connection for the initial setup.

---

## 🖥 Usage

To start the real-time intent detection system:

```bash
python main.py
```

### Operational Loop:
1.  **Calibration:** The system measures ambient noise levels to set VAD thresholds.
2.  **Listening:** Waits for speech input (Passive Mode).
3.  **Processing:** Upon speech detection, it transcribes and classifies intent (Active Mode).
4.  **Output:** Results are printed to the console with timestamps and confidence scores.

**Configuration:**
Modify `config.py` to adjust:
*   `VAD_THRESHOLD`: Sensitivity of speech detection.
*   `INTENT_HIERARCHY`: Add or remove intents to test zero-shot adaptability.

---

## 📄 Citation

If you use this code or methodology in your research, please cite the associated manuscript:

```bibtex
@article{YourName2025EdgeNative,
  title={Beyond Large Models: Effective Zero-Shot Intent Detection with SLMs for Real-Time Edge Applications},
  author={Your Name and Co-Authors},
  journal={Springer Computing},
  year={2025},
  note={Under Review}
}
```

---

**Contact:** [Your Name] | [Your Email]

