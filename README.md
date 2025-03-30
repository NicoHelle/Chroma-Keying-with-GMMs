
# Real-Time Chroma Keying using Gaussian Mixture Models (GMM/BGMM)

This project implements an advanced real-time Chroma Keying application leveraging probabilistic models, specifically **Gaussian Mixture Models (GMM)** and **Bayesian Gaussian Mixture Models (BGMM)**. It's designed to provide robust segmentation even under challenging lighting conditions where classical Chroma Key methods typically fail.

---

## Overview

Classical Chroma Key methods, widely used in film and broadcasting, often struggle under uneven lighting, shadows, noise, and varying illumination. This project addresses these limitations by employing probabilistic segmentation models to achieve more accurate background removal.

A graphical user interface (**GUI**) enables users to manually select background samples to precisely train the GMM or BGMM, thereby enhancing the segmentation quality significantly compared to classical methods.

---

## Features

- **Real-Time Processing:**  
  Optimized for real-time video processing using GPU acceleration (CUDA).
  
- **Probabilistic Models:**
  - **Gaussian Mixture Models (GMM)**
  - **Bayesian Gaussian Mixture Models (BGMM)** for improved segmentation.

- **Preprocessing and Postprocessing:**
  - Adjustable brightness and contrast.
  - Various noise filtering techniques (CUDA Bilateral Filter, Non-Local Means).
  - Morphological operations (erosion, dilation) and alpha blending for smoother results.

- **Interactive GUI:**  
  Built with PyQt5 to allow intuitive user interaction for sample area selection and parameter adjustment (brightness, contrast, filtering parameters, threshold).

- **Visualization:**  
  Includes functionality to visualize GMM/BGMM model boundaries and sampled data points in a 3D YUV color space.

---

## Requirements

- **Python 3.12.4**
- **OpenCV 4.10.0** (with CUDA 12.6.2 support)
- **scikit-learn 1.4.2**
- **PyQt5 5.15.10**
- **CuPy 13.4.0**
- **Matplotlib 3.9.2**

*(Recommended Hardware: NVIDIA GPU, e.g., RTX 3080 or similar.)*

---

## Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/NicoHelle/Chroma-Keying-with-GMMs.git
cd Chroma-Keying-with-GMMs
```

2. **Install dependencies:**  
```bash
pip install -r requirements.txt
```

(Alternatively, install manually as listed above.)

3. **Run the application:**  
```bash
python main.py
```

---

## Application Workflow

- **Sample Selection:**  
  Select background areas in the GUI to train the GMM/BGMM model.

- **Parameter Tuning:**  
  Adjust preprocessing and model parameters via sliders and checkboxes in real-time.

- **Visualization:**  
  Visualize segmentation quality and probabilistic classification in real-time.

---

## File Structure

- `main.py` - Main pipeline and entry point of the application.
- `ChromaKeying.py` - Core implementation of the GMM and BGMM segmentation logic.
- `gui.py` - Graphical user interface built with PyQt5.
- `preprocess.py` - Image preprocessing methods.
- `postprocess.py` - Image postprocessing methods.
- `plotandvisualize.py` - Visualization functions for model insights.

---

## Results & Performance

Experimental results demonstrate significantly improved segmentation precision and robustness compared to traditional methods. Performance gains were achieved through GPU acceleration.

*(Detailed evaluation is provided in the [full thesis document](Masterarbeit_NicoHelle_4022929-komprimiert.pdf).)*

---

## References

For technical details, see the comprehensive research in:
- **[Chroma Keying mithilfe von Gaussian Mixture Models (Master Thesis, Nico Helle)](Masterarbeit_NicoHelle_4022929-komprimiert.pdf)**

---

## License

This project is available under the MIT License.

---

## Author

- **Nico Helle**  
  Wilhelm-Schickard-Institut für Informatik, Universität Tübingen  
  *(Supervisor: Prof. Dr. Andreas Schilling & Prof. Dr.-Ing. Hendrik P. A. Lensch)*

---

© 2025 Nico Helle. All rights reserved.
