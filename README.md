# Resource-Aware Conditional Cascaded (RACC-SLM) Framework
## Zero-Shot Intent Detection via Semantic Mapping on Edge Architectures

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

### Overview

This repository hosts the implementation and experimental validation of the **Resource-Aware Conditional Cascaded (RACC-SLM)** framework. This research addresses the fundamental trade-off between semantic understanding and computational efficiency in real-time Spoken Language Understanding (SLU).

By utilizing **Semantic Mapping protocols**—transforming abstract schema labels into descriptive natural language definitions—this system enables robust Zero-Shot classification using Small Language Models (SLMs) such as `nli-MiniLM2-L6-H768`. The architecture is optimized for edge-native deployment, achieving sub-second latency without dependency on cloud-based Large Language Models (LLMs).

---

### Key Technical Contributions

1.  **Semantic Mapping Protocol:**
    Implements a methodology that translates raw dataset tags (e.g., `iot_hue_lighton`) into natural language definitions (e.g., *"Turn on the lights"*). This aligns user utterances with the pre-trained Natural Language Inference (NLI) manifold, improving Zero-Shot F1-scores by **~38%** over baselines.

2.  **Edge-Cloud Latency Optimization:**
    Demonstrates that local SLM inference (236ms) significantly outperforms cloud-based API calls (~1.2s latency due to network RTT and SSL overhead) while maintaining high semantic robustness (80% Top-5 Accuracy on CLINC150).

3.  **Hierarchical & Flat Classification Strategies:**
    Includes implementations for both Hierarchical (Coarse-to-Fine) and Optimized Flat classification architectures to handle large-scale ontologies (up to 150 classes).

4.  **Microservice Architecture:**
    Deploys the intent engine as an isolated FastAPI microservice, decoupling the NLU logic from audio capture subsystems for modular scalability.

---

### Repository Structure

The directory structure is organized into datasets, preprocessing utilities, core source code, and experimental validation scripts.

```text
INTENT_CLASSIFICATION/
├── datasets/                        # Benchmark datasets (Raw and Processed)
│   ├── clinic_data.json             # CLINC150 (OOS) Dataset (150 classes)
│   ├── snips_train.csv              # SNIPS Dataset (7 classes)
│   ├── slurp_data.jsonl             # Original SLURP Dataset
│   ├── slurp_final_clean.jsonl      # Curated/Cleaned SLURP subset
│   └── ... (Intermediate data files)
│
├── models/                          # Local storage for Transformer weights
│
├── pre_proc_tools/                  # Data Engineering & Inspection Tools
│   ├── analyze_slurp.py             # Schema extraction for SLURP
│   ├── cleanup_slurp.py             # Noise reduction and outlier removal
│   ├── clinic_diagnosis.py          # Confusion matrix analysis for CLINC150
│   ├── inspect_clinic.py            # Structure analysis for OOS data
│   └── ... (Dataset generation scripts)
│
├── src/
│   └── core.py                      # Core engine logic and class definitions
│
├── test/                            # Experimental Validation Suites
│   ├── clinic_experiment.py         # Full 150-class Zero-Shot evaluation
│   ├── edge_tradeoff_experiment.py  # CPU/GPU Latency benchmarking
│   ├── hierarchical_experiment.py   # Multi-stage inference tests
│   ├── natural_language_exp.py      # Semantic Map validation
│   ├── snips_experiment.py          # Control group validation
│   └── ... (Ablation studies)
│
├── intent_classifier.py             # Main inference entry point
├── listener.py                      # Audio capture and VAD orchestration
└── requirements.txt                 # Project dependencies
```

---

### System Architecture

The framework operates on a conditional execution pipeline designed to minimize resource usage on constrained hardware:

1.  **Input Processing:** Audio is captured and transcribed via a distilled ASR model.
2.  **Semantic Projection:** The transcribed text is paired with a dynamic list of candidate definitions based on the active domain context.
3.  **Cross-Encoder Inference:** The `nli-MiniLM2-L6` model computes entailment scores between the user query and the semantic definitions.
4.  **Decision Logic:**
    *   **Exact Match (Top-1):** High-confidence immediate execution.
    *   **Intent Retrieval (Top-3):** Fallback for disambiguation or clarification loops.

---

### Performance Benchmarks

#### 1. Zero-Shot Accuracy (CLINC150 Dataset)
*Evaluation on 150 distinct intents with no training data.*

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Exact Match Accuracy (Top-1)** | **57.63%** | Precise identification of the single correct label. |
| **Broad Recognition (Top-5)** | **79.80%** | Successful retrieval of the correct semantic neighborhood. |
| **Weighted Precision** | **0.6985** | High reliability in confident predictions. |

#### 2. Latency Profile (Edge vs. Cloud)
*Comparative analysis on single-thread CPU hardware vs. standard API benchmarks.*

| Architecture | Model | Inference Latency | Network Overhead | Total Time |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed (Local)** | MiniLM-L6 | **236 ms** | **0 ms** | **~236 ms** |
| **Cloud Baseline** | GPT-4o-mini | ~400 ms | ~800 ms | ~1,200 ms |

*Result:* The proposed local architecture offers a **5x speedup** in total interaction time compared to cloud-dependent solutions.

---

### Installation and Setup

**Prerequisites:**
*   Python 3.10 or higher
*   PyTorch (CUDA recommended for lower latency, CPU supported)

**1. Clone the Repository:**
```bash
git clone <repository_url>
cd INTENT_CLASSIFICATION
```

**2. Initialize Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

### Usage

#### Running the Intent Recognition Microservice
To start the FastAPI service that handles semantic classification:

```bash
python intent_classifier.py
```
*The service will expose a REST endpoint at `http://localhost:8005/understand_intent`.*

#### Reproducing Experiments
To replicate the benchmarks presented in the manuscript, run the scripts located in the `test/` directory:

*   **For Latency Benchmarking:**
    ```bash
    python test/edge_tradeoff_experiment.py
    ```
*   **For CLINC150 Zero-Shot Evaluation:**
    ```bash
    python test/clinic_experiment_quick.py
    ```

---

### Contact

For inquiries regarding the architecture or experimental data, please refer to the corresponding author details in the associated manuscript.
