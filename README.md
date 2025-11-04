# Enhanced Face Tracer

This repository contains an implementation and initial exploration of the **FaceTracer** system, based on the paper "FaceTracer: Unveiling Source Identity from Swapped Face Images and Videos for Fraud Prevention", for my final-year project under Prof. Zhongqun Zhang.

The goal is to reproduce the core functionality of source identity tracing and explore potential improvements or applications, particularly in the context of fraud prevention.

## Features (Implemented/Explored)
*   **Source Identity Extraction:** Implementation of the core FaceTracer pipeline for identifying the source identity in face-swapped images/videos.
*   **Model Support:** Tested with ViT backbone and AAMSoftmax components (using provided HiRes model).
*   **Identity Consistency Analysis:** Initial analysis of consistency across frames (e.g., similarity scores > 0.85 observed on test data).
*   **Cross-frame Matching:** Verification of source identity consistency across frames of the same video.
*   **Processing Pipeline:** Adapted the original codebase for execution (initially tested on Google Colab).

## Project Structure (Based on Original + Additions)
*   `main.py` - Main execution script (adapted from original or created for testing).
*   `models/` - Model architectures (from original FaceTracer).
*   `utils/` - Utility functions (from original FaceTracer).
*   `scripts/` - Helper scripts (from original FaceTracer or added for testing).
*   `checkpoints/` - Model weights (downloaded separately as per original instructions).
*   `input_videos/` - Input data (placeholder or test data).
*   `output_videos/` - Output results (placeholder or results from test runs).
*   `notebooks/` - (Optional) Jupyter notebooks used for initial testing/analysis if applicable.

## Status
*   **Successfully Reproduced:** The core FaceTracer pipeline has been executed, and source identity extraction has been performed on test data.
*   **Initial Findings:** Observed high identity consistency for the same source across frames. The system effectively isolates source identity from the target face in swapped inputs.
*   **Next Steps:** Explore potential improvements identified (e.g., multi-model ensembles, temporal consistency for videos, real-time optimization) as discussed with the supervisor.

## Quick Start
1.  **Clone this repository:**
    ```bash
    git clone https://github.com/Coachemmy/Enhanced-Face-Tracer.git # Or your public link
    cd Enhanced-Face-Tracer
    ```
2.  **Set up the environment:** Follow the requirements specified in the original FaceTracer repository or adapt based on your successful Colab setup. (e.g., `pip install -r requirements.txt` or install PyTorch, numpy, etc.)
3.  **Download Model Weights:** Obtain the required model files (e.g., HiRes) from the original FaceTracer repository's instructions.
4.  **Prepare Input Data:** Place your face-swapped video/image data in the `input_videos/` folder (or adapt the path in the script).
5.  **Run the Pipeline:** Execute the main script or notebook used for processing (e.g., `python main.py` or run the Colab notebook logic).

## Reference
*   **Original Paper:** [FaceTracer: Unveiling Source Identity from Swapped Face Images and Videos for Fraud Prevention](https://arxiv.org/abs/2401.08252)
*   **Original Repository:** [https://github.com/zzy224/FaceTracer](https://github.com/zzy224/FaceTracer)
