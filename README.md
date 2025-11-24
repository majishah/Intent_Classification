**Introduction**

This project implements a streamlined real-time Speech Intent Recognition system capable of detecting speech, transcribing audio using Faster-Whisper, and interpreting user intent through a structured hierarchical zero-shot classification model.

To maintain transparency and reproducibility we includes three dedicated documentation files, each providing focused explanations of major components:

1. Average_Log_Probabilty_Calculation.md – Describes how transcription confidence is computed using average log-probabilities.
2. Hierarchical_Intent_Classification.md – Details the three-level intent classification design and its reasoning process.
3. WorkFlow_Explanation.md – Gives a complete overview of the full system workflow, architecture, and module interactions.

These files are included in the repository under the same names, allowing readers to quickly explore the logic, mathematics, and architecture behind this work.

**What We Can Find in This Repository**

This repository is purposefully structured to give reviewers a complete picture of both the system’s design and the reasoning behind each major component. Inside, readers can find:

✔️ A Complete System Overview
Covers the full pipeline from audio capture → VAD → transcription → intent detection.

✔️ Modular Architecture Breakdown
Clearly separated modules for audio processing, transcription, utilities, and intent reasoning.

✔️ Detailed Technical Methods
Including VAD detection, gain control, noise suppression, Whisper decoding, and zero-shot classification logic.

✔️ Step-by-Step Workflow Description
A transparent execution flow showing how audio is detected, buffered, processed, and converted into high-level intent.

✔️ Model Design Insights
Why Faster-Whisper was selected, why a hierarchical intent structure was used, and how zero-shot NLI benefits scalability.

✔️ Rationale Behind Design Choices
Focus on reliability, low-latency decisions, interpretability, modularity, and suitability for real-time applications.

✔️ Package & Environment Requirements
Complete package installation instructions and model directory expectations for full reproducibility.


**Contribution & Novelty Summary**

This system introduces a cohesive and transparent approach to speech understanding by integrating multiple complex tasks into a unified, efficient pipeline. Its novelty lies in:

Hierarchical Zero-Shot Intent Classification
    Instead of a flat list of intents, the system breaks down understanding into progressively refined levels, improving accuracy, interpretability, and scalability — all without requiring any model training.

Confidence-Aware Speech Transcription
  Using average log-probability values from Faster-Whisper, it provides interpretable confidence metrics that guide downstream decision-making.

Fully Modular Architecture
  Each subsystem — VAD, transcription, intent classification — is isolated, replaceable, and clearly documented using the three supporting files in the repository.

Real-Time, Lightweight Operation
  The system performs all tasks in real time on CPU-level hardware while retaining high robustness and clarity of processing.

Overall, this work demonstrates a novel, expandable, and reliable design for speech-driven intent recognition, suitable for assistants, robotics, IoT-based edge systems, and other resource-constrained environments where efficient real-time processing is essential.
