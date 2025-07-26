# MLOps Assignment 3: End-to-End MLOps Pipeline

---

##  Objective

This project demonstrates an end-to-end MLOps pipeline using a simple machine
learning model. It includes:
- Model training using Scikit-learn and PyTorch
- Containerization using Docker
- CI/CD automation with GitHub Actions
- Manual quantization of model parameters

---

##  Branch Structure

| Branch Name | Purpose |
|------------------|-------------------------------------------------------------------------|
| `main` | Initial project setup with README and .gitignore |
| `dev` | Model development with training script using Scikit-learn |
| `docker_ci` | Docker integration and CI/CD workflow using GitHub Actions |
| `quantization` | Manual quantization of model weights and performance comparison
|

---

##  Dataset &amp; Model

- **Dataset:** California Housing dataset from `sklearn.datasets`
- **Models:**
- **Linear Regression** using Scikit-learn
- **Single-layer PyTorch model** initialized with weights from the Scikit-learn model
- **Quantized model:** manually quantized parameters to `uint8` format

---

##  Pipeline Overview

###  `dev` Branch
- Trains a `LinearRegression` model (`train.py`)
- Saves the model using `joblib`

###  `docker_ci` Branch
- Adds `Dockerfile` and `predict.py` script for model testing
- Configures `.github/workflows/ci.yml` for CI/CD with:
- Training
- Docker build and test
- Image push to DockerHub

###  `quantization` Branch

- Converts the trained model into a quantized PyTorch model (`quantize.py`)
- Performs manual quantization and inference with dequantized weights
- Saves both unquantized and quantized parameter files using `joblib`

---

##  CI/CD Details

Workflow defined in `.github/workflows/ci.yml` runs on every push:
1. **Train Model:** Run `train.py` to generate `.joblib` file
2. **Build Docker Image:** Based on the provided `Dockerfile`
3. **Test Container:** Run `predict.py` inside the container
4. **Push Image:** To DockerHub on success

---

##  Quantization Results

| Metric | Original Sklearn Model | Quantized Model |
|------------------|------------------------|--------------------------|
| R² Score | *0.5758* | *0.5715* |
| Model Size (KB) | *0.35 KB* | *0.33 KB* |

---

##  Repository Links

- **GitHub:** [View on Github](https://github.com/karthik-kumar-g/mlops-housing-pipeline.git)
- **DockerHub:** [View on Dockerhub](https://hub.docker.com/layers/karthikkumarg/mlops-housing-image/latest/images/sha256:8c48d2f5c304704caa5370ab78068f0befac0a65c501275eba66df95a07c6035?uuid=b63c6072-2d7d-43f3-a743-60077f16367a%0A)

---

##  Documentation

Complete documentation, with screenshots and step-by-step explanations of:
- Git branching
- Model training and saving
- Dockerfile and container verification
- GitHub Actions workflow
- Manual quantization and comparison metrics

is available in the PDF report.

---

##  Final Notes

- All tasks are implemented in **separate branches** as per instructions
- No branches are merged back into `main`
- All scripts and configurations are tested for reproducibility

---
