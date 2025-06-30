# 🔄 Cycle‑GAN‑Diffusion

[![License](https://img.shields.io/badge/license-MIT-blue)]()

A **PyTorch** & **Diffusers** based framework that combines the power of Cycle-GANs with diffusion models for high-quality image-to-image translation without paired data.

---

## 📌 Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training Cycle‑GAN](#training-cycle-gan)  
  - [Diffusion‑Driven Translation](#diffusion-driven-translation)  
- [Configuration](#configuration)  
- [Examples](#examples)  
- [Evaluation & Metrics](#evaluation--metrics)  
- [Citation](#citation)  
- [License](#license)  

---

## 📈 Overview

This repository marries **Cycle-GAN** unsupervised domain translation with **diffusion-based refinement**, enabling:

1. **Unpaired translation** (e.g., photo ↔ painting).
2. **Detail enhancement** using diffusion to sharpen GAN outputs.
3. **Modular training pipeline** for GAN and diffusion components.

---

## ✨ Key Features

- 🔁 Cycle-GAN backbone for unpaired domain mapping  
- 🎯 Diffusion model for visual refinement  
- 🧩 Hugging Face [`diffusers`](https://github.com/huggingface/diffusers)-compatible  
- 🛠️ Options for end-to-end training or pre-trained Cycle-GAN + post-hoc diffusion  
- ⚙️ CUDA/FP16 support  

---

## 🛠️ Installation

```bash
git clone https://github.com/naimurborno/cycle-gan-diffsuion.git
cd cycle-gan-diffsuion

# Install dependencies
pip install -r requirements.txt
