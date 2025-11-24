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


**System Capabilities**

✔️ Real-Time Speech Detection

The system continuously listens to the microphone and uses Silero VAD to identify when real speech begins and ends.
It enhances audio quality by applying:

Gain control to amplify low-volume speech

Noise suppression to remove background disturbances

Dynamic buffering to isolate clean speech segments

This ensures only meaningful audio is processed, which drastically improves transcription accuracy.

✔️ High-Efficiency Transcription

The project uses Faster-Whisper, a lightweight and CPU-friendly implementation of Whisper, to convert speech into text with high accuracy.

The transcription pipeline also provides average log-probability, a crucial confidence metric:

Values are always negative because log(probability) of numbers between 0 and 1 produces negative values.

Values closer to zero indicate higher confidence.

Summing log-probabilities prevents numerical underflow and gives stable confidence estimation.

A complete explanation is available in: Average_Log_Probabilty_Calculation.md

✔️ Hierarchical Intent Classification

After transcription, the system interprets the meaning behind the spoken text using a three-level intent classification hierarchy:

Level 1 – Broad Category

Conversation-Oriented, Task-Oriented, Entertainment

Level 2 – Subcategory

Refines the meaning based on the Level 1 decision.

Level 3 – Fine-Grained Interpretation

Pinpoints the exact intent (e.g., Greeting vs. Small-Talk, Music vs. Movies).

**Why Hierarchy?**

Increases accuracy by narrowing options step-by-step

Prevents misclassification across unrelated categories

Makes the system more interpretable

Allows effortless expansion as new intents are added

Intent classification relies on a zero-shot NLI model, meaning no training data is required.
The model semantically compares text to labels and chooses the closest match.

A deeper explanation is available in: Hierarchical_Intent_Classification.md



**Architecture Overview**

Your system is split into five purposeful modules: 
WorkFlow_Explanation

**1. config.py**

Holds all configuration:
    Model paths
    Audio settings
    VAD thresholds
    Hierarchical intent labels
    Logging formats

This file defines the "personality" of the entire system and determines how the pipeline behaves in real time.

**2. utils.py**

Provides helper functions:
    Logging setup
    Structured printing of intent results
    Clean display of predicted labels, scores, and ranking

This makes debugging and understanding predictions significantly easier.

**3. speech_recognizer.py**

The most complex module, responsible for:
    Initializing microphone stream (PyAudio)
    Applying gain + noise suppression
    Running VAD to isolate speech
    Buffering and segmenting audio
    Converting audio segments into WAV
    Triggering a beep sound during detection
    Transcribing segments through Faster-Whisper

This module is built with threading, locks, and asynchronous utilities to ensure real-time responsiveness.

**4. intent_classifier.py**

Runs the zero-shot transformer pipeline to classify user intent.
Handles all three hierarchical levels:
    Level 1 → Level 2 → Level 3
    Broad → Specific → Most Specific

Each level narrows down the next set of possible labels, boosting accuracy and interpretability.

**5. main.py**

The orchestrator.
It:
    Boots system
    Runs calibration
    Starts listener
    Continuously processes buffered audio
    Transcribes text
    Performs hierarchical intent classification
    Logs and prints all results
    Handles commands (pause / resume / stop)

This is the “glue” that binds the entire pipeline into a functional real-time application.



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
