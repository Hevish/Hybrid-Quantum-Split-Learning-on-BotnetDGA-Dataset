# HQSL: A Hybrid Quantum Neural Network for Split Learning

This repository demonstrates how **Hybrid Quantum Neural Networks** can be integrated with **Split Learning (SL)** — a framework we refer to as **HQSL**.

HQSL enables multiple **resource-constrained clients** to collaboratively train a machine learning model by splitting the model between the clients and a powerful **Hybrid Quantum Server**. This approach is particularly suitable for privacy-preserving and distributed training scenarios like botnet domain detection.

> 📂 This is a code repository with working examples. Full datasets or additional experimental details can be provided upon reasonable request.

---

## 📁 Repository Structure

```
.
├── folder1/ to folder5/        # Experimental data splits as per k-fold x-val
├── results/                    # Output logs
├── centralized_classical.py    # Centralized classical baseline
├── centralized_hybrid.py       # Centralized HQSL model (before splitting)
├── divideDataset.py            # Dataset preprocessing and splitting
├── hybrid_split_v3.zip         # Legacy or backup hybrid models/configs
├── split_classical.py          # Basic classical split learning
├── split_hybrid.py             # Basic HQSL (2-client setup)
├── split_N_classical.py        # N-client classical split learning
├── split_N_hybrid.py           # N-client HQSL implementation
└── README.md                   # You're here!
```

---

## Getting Started

1. **Install dependencies**

2. **Prepare the dataset**
   Make sure the dataset is accessible, then run:
   ```bash
   python divideDataset.py
   ```

3. **Run a model**
   Choose from the centralized or split versions:
   ```bash
   python centralized_hybrid.py        # HQSL model without split
   python split_hybrid.py              # HQSL with 2 clients
   python split_N_hybrid.py            # HQSL with N clients
   ```

---

## Models Explained

| Script                    | Description                                        |
|--------------------------|----------------------------------------------------|
| `centralized_hybrid.py`  | HQSL model before splitting; all components together |
| `split_hybrid.py`        | HQSL across two devices: client + quantum server   |
| `split_N_hybrid.py`      | HQSL scaled to N clients                           |
| `centralized_classical.py` | Centralized classical baseline                   |
| `split_classical.py`     | Classical split learning (


---


## Results

Results, plots, or model outputs will be saved in the `results/' folder.

---

## Motivation

Botnet-based Domain Generation Algorithms (DGAs) are difficult to detect with traditional methods. HQSL offers a scalable, distributed, and quantum-enhanced approach to improve generalization and privacy in learning models for DGA detection.

---
## Notes

This repo assumes familiarity with quantum machine learning, split learning, and botnet DGA detection.

If you are running quantum components, make sure to configure a backend such as IBM Qiskit or Pennylane.
