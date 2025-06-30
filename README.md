# 🧠 Unified Diffusion Toolkit: WSDD + Cycle‑GAN‑Diffusion

[![License](https://img.shields.io/badge/license-MIT-blue)]()

A unified PyTorch-based repository combining two advanced generative frameworks:

- 🔁 **Cycle‑GAN‑Diffusion**: High-quality unpaired image translation using Cycle-GAN with diffusion-based refinement.
- 💨 **WSDD – Weight‑Shared Distilled Diffusion**: A compact, fast diffusion model using progressive knowledge distillation and weight sharing.

---

## 📌 Table of Contents

- [Project 1: WSDD](#project-1-wsdd)  
  - [Overview](#overview)  
  - [Installation](#installation)  
  - [Training](#training)  
  - [Sampling](#sampling)  
  - [Benchmark & Results](#benchmark--results)
- [Project 2: Cycle‑GAN‑Diffusion](#project-2-cycle-gan-diffusion)  
  - [Overview](#overview-1)  
  - [Installation](#installation-1)  
  - [Training](#training-1)  
  - [Diffusion Translation](#diffusion-translation)  
  - [Evaluation & Metrics](#evaluation--metrics)
- [Citation](#citation)  
- [License](#license)  

---

## 🎯 Project 1: WSDD – Weight‑Shared Distilled Diffusion

### Overview

WSDD compresses standard diffusion models by:

- Reusing weights across denoising steps.
- Applying knowledge distillation to learn fewer but more effective steps.
- Maintaining visual quality with up to **8× faster sampling**.

### Installation

```bash
cd wsdd
pip install -r requirements.txt
