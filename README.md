# 📌 Official Implementation of "LightSBB-M: Bridging Schrödinger and Bass for Generative Modeling"

This repository contains the official implementation of the paper ["Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation"](https://arxiv.org/abs/2503.02943).

Authors: Alexandre Alouadi, Pierre Henry-Labordère, Grégoire Loeper, Othmane Mazhar, Huyên Pham, Nizar Touzi

Contact: alexandre.alouadi@bnpparibas.com; huyen.pham@polytechnique.edu

If you notice any errors or have suggestions for improvement, please feel free to reach out to us.

## Abstract
The Schrödinger Bridge and Bass (SBB) formulation, which jointly controls drift and volatility, is an established extension of the classical Schrödinger Bridge (SB).  Building on this framework, we introduce **LightSBB‑M**, an algorithm that computes the optimal SBB transport plan in only a few iterations. The method exploits a dual representation of the SBB objective to obtain analytic expressions for the optimal drift and volatility, and it incorporates a tunable parameter β > 0 that interpolates between pure drift (the Schrödinger  Bridge) and pure volatility (Bass martingale transport). We show that LightSBB‑M achieves the lowest 2‑Wasserstein distance on synthetic datasets against state‑of‑the‑art Schrödinger  Bridge and diffusion baselines with up to 32% improvement. We also illustrate the generative capability of the framework on an unpaired image‑to‑image translation task (*adult* → *child* faces in FFHQ). These findings demonstrate that **LightSBB‑M** provides a scalable, high‑fidelity SBB solver that outperforms existing SB and diffusion baselines across both synthetic and real‑world generative tasks.

## 📂 Project Structure
```
/LightSBB-M
│── data                    # Will be created when runing the notebook
│── light_sbb               # Model's architecture, training and running files
│── alae_experiments.ipynb  # Example usage with image generation
```

## Setup
To clone the repository, run:
```bash
git clone https://github.com/alexouadi/LightSBB-M.git
cd LightSBB-M
```

To install the necessary dependencies for this project, make sure you use python 3.11 and run:
```bash
pip install -r requirements.txt
```

## Usage
You can use the implemented functions directly by importing them into your script, run the Jupyter notebook to see practical examples:
```bash
jupyter alae_experiments.ipynb
```
