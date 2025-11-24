**Introduction**
This project implements a streamlined real-time Speech Intent Recognition system capable of detecting speech, transcribing audio using Faster-Whisper, and interpreting user intent through a structured hierarchical zero-shot classification model.

To maintain transparency and reproducibility we includes three dedicated documentation files, each providing focused explanations of major components:

1. Average_Log_Probabilty_Calculation.md – Describes how transcription confidence is computed using average log-probabilities.
2. Hierarchical_Intent_Classification.md – Details the three-level intent classification design and its reasoning process.
3. WorkFlow_Explanation.md – Gives a complete overview of the full system workflow, architecture, and module interactions.

These files are included in the repository under the same names, allowing readers to quickly explore the logic, mathematics, and architecture behind this work.




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
