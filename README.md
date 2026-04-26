# bookCodePython: Numerical Simulation of Optical Wave Propagation

This repository provides the official Python implementation of the simulation framework from the textbook **"Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"**.

It transitions the core wave optics algorithms—including Fresnel and Fraunhofer propagation, phase screen generation, and atmospheric turbulence modeling—into an accessible Python environment using NumPy and Matplotlib.

---

## 📚 Reference & Citation

If you use this code in your research or professional work, please cite the original textbook:

> Jason D. Schmidt, _Numerical Simulation of Optical Wave Propagation with Examples in MATLAB_, SPIE, Bellingham, WA (2010).

- [SPIE Digital Library](https://www.spiedigitallibrary.org/ebooks/PM/Numerical-Simulation-of-Optical-Wave-Propagation-with-Examples-in-MATLAB/eISBN-9780819483270/10.1117/3.866274)
- [Amazon Reference](https://www.amazon.com/Numerical-Simulation-Propagation-Examples-Monograph/dp/0819483265/)


---

## 📝 A Note on the Python Implementation

**Direct Correspondence:** The primary goal of this repository is to maintain a one-to-one correspondence with the MATLAB examples provided in the textbook.

To assist readers in connecting the book material with the Python implementation, the code is intentionally written to mirror the MATLAB structure, including folder structure, variable names, and comments. As a result, it is not optimized for maximum execution speed, nor does it strictly follow "Pythonic" idioms (such as heavy use of list comprehensions or object-oriented abstractions) that might obscure the underlying physics for those following along with the book.

**Enhancement:** Unlike the original code provided on the disc with the book, this Python version includes integrated plotting routines to visualize key outputs immediately.

---

## 📂 Project Structure

```text
bookCodePython/
├── examples/           # Textbook example scripts organized by chapter
├── requirements.txt    # Optional: Python dependencies
├── LICENSE
├── OpticalWavePropSim.py
├── README.md
└── requirements.txt
```

---

## 🚀 Installation & Quick Start

### 1. Requirements

Ensure you have the core scientific stack installed:

```bash
pip install numpy matplotlib scipy
```

### 2. Setup

Clone the repository and add the root directory to your PYTHONPATH, or simply run the examples from the root:

```bash
git clone https://github.com/jdschmidt-opticalscientist/bookCodePython.git
cd bookCodePython
```

### 3. Run an Example

```bash
python examples/Chapter9/pt_source_atmos_setup.py
```

---

##⚖️ License

The Python source code in this repository is licensed under the BSD 3-Clause License. See the LICENSE file for full details.

Note: This license applies specifically to this Python translation. The original MATLAB code and textbook content remain under the copyright of SPIE.

## Development Quality

![Python CI](https://github.com/jdschmidt-opticalscientist/bookCodePython/actions/workflows/main.yml/badge.svg?branch=main)

We use GitHub Actions to ensure code quality through automated linting and unit testing.
