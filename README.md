# XAI Robustness Benchmark 🛡️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

**Does your model keep the same "reasoning" even when it makes the correct prediction?**

The **XAI Robustness Benchmark** is a toolkit designed to evaluate the stability of Explainable AI (XAI) methods—such as Grad-CAM and Integrated Gradients—under meaningful distribution shifts (noise, blur, corruption). It helps researchers and engineers answer critical questions:
- *Is the model right for the right reasons?*
- *Does the explanation hold up when the image is slightly corrupted?*
- *Which explanation method is more trustworthy for this specific model?*

## ✨ Features

- **Interactive Dashboard**: A powerful Streamlit app to visualize explanation stability in real-time.
- **Enhanced Visualizations**: Interactive Plotly charts with detailed tooltips and high-contrast stability tracking.
- **Live Lab Playground**: A 4-column comparison view to analyze Original vs. Corrupted images and their heatmaps side-by-side.
- **Multi-Explainer Support**: Built-in support for **Grad-CAM** and **Integrated Gradients**.
- **Distribution Shift Library**: Easy-to-use corruptions including Gaussian Noise, Motion Blur, and Brightness shifts.
- **Quantitative Metrics**: Measures stability using SSIM (Structural Similarity) and Cosine Similarity.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/your-username/xai-robustness-benchmark.git
cd xai-robustness-benchmark

# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Dashboard

Calculated metrics and visualized results can be explored interactively.

```bash
streamlit run src/app.py
```
> Navigate to `http://localhost:8501` in your browser.

### 3. Run a Benchmark Experiment (CLI)

To run a full sweep of experiments (e.g., assessing robustness across 5 severity levels of corruption):

```bash
# Basic run with default config
python src/main.py --config configs/default_benchmark.yaml

# Override parameters
python src/main.py --shift gaussian_noise --severity 3 --output_dir outputs/experiment_v1
```

---

## 📂 Repository Structure

```
xai-robustness-benchmark/
├── configs/             # YAML configuration files for experiments
├── data/                # Dataset storage (CIFAR-10, etc.)
├── notebooks/           # Jupyter notebooks for prototyping
├── outputs/             # Results, logs, and saved checkpoints
├── src/                 # Source code
│   ├── datasets/        # Data loading and transforms
│   ├── explainers/      # XAI implementations (GradCAM, IG)
│   ├── models/          # Model architectures (ResNet, etc.)
│   ├── metrics/         # Stability metrics (SSIM, Cosine)
│   ├── shifts/          # Image corruption logic
│   ├── app.py           # Streamlit dashboard entry point
│   ├── main.py          # CLI benchmark entry point
│   └── utils.py         # Helpers
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 📊 Methodology

### 1. Model Training
We train a standard ResNet-18 on CIFAR-10. The goal isn't just high accuracy, but to study the *features* learned by this standard model.

### 2. Attribution Stability
For a given image $x$ and its corrupted version $x'$, we generate explanations $E(x)$ and $E(x')$. We then compute the **Stability Score**:
$$ \text{Stability} = \text{SSIM}(E(x), E(x')) $$
High stability means the model looks at the same regions despite the corruption.

### 3. Shift Library
We currently support:
- **Noise**: Gaussian, Shot, Impulse
- **Blur**: Gaussian, Motion, Glass
- **Weather**: Fog, Brightness
- **Digital**: Pixelation, JPEG Compression

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a Pull Request.

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
