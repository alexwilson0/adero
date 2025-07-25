![Logo](images/logo.png)
# ADERO: Aircraft Design Exploration with Regression & Optimisation
## A Machine Learning Based Design Tool to Aid Early Aircraft Design & Optimisation


### Description

This repository contains a multi-stage machine learning-based tool which provides an accelerated approach to early aircraft wing design. ADERO cuts a lengthy design process down to as little as a minute. The user inputs their contraints like cruise altitude, aircraft velocity, and more; then the system draws upon several surrogate models, trained on organic data, to produce an aerodynamically and structurally viable wing. This is packaged inside a robust GUI that enables inspeciton of the 3D wing geometry, its internal structure, and the generated data values. An additional functionality in the GUI enables the user to use the individual surrogate models to predict aerodynamic/structural performance given a set of geometric inputs.

![Main GUI](images/gui.png)

### Features

- Predicts wing geometry and internal structure from input mission profiles
- Alternatively predict performance based on geometric or structural inputs calling surrogate models directly
- Combines multiple optimisation modules (aerofoil, wing-box, planform, structural)
- Fully interactive GUI for streamlined user experience
- Large ML models managed with DVC and stored in Amazon S3

### Tech Stack

- Python >=3.8
- Keras / TensorFlow
- Scikit-learn, NumPy, Plotly, pymoo, PyQt 
- DVC for model versioning
- Amazon S3 for model storage

### How to Clone and Run

ADERO uses [DVC](https://dvc.org) and Amazon S3 to manage large ML models. Follow these steps to download and run the app:

**1. Clone the Repository**
```bash  
git clone https://github.com/alexwilson0/adero.git  
cd adero
```
**2. Set Up the Environment**
```bash  
conda env create -f environment.yml  
conda activate adero  
```
**3. Install DVC with S3 Support**
```bash  
pip install "dvc[s3]"  
```
**4. Pull the ML Models from S3**
```bash  
dvc pull  
```
**5. Launch the Application**
```bash  
python adero/gui.py  
```
### Know the Team
Charlie Williams - Team Manager, Aerodynamics (cpw1g21@soton.ac.uk)  
Alfred Vincent - Aerodynamics (ajv1g21@soton.ac.uk)  
Alexander Wilson - ML & Software (atcw1g21@soton.ac.uk)  
Darius Wurzbacher - ML & Software (dw3g21@soton.ac.uk)  
Mingxuan Soong - Structural (ms10n20@soton.ac.uk)  
David Collyer - Structural (dc3g20@soton.ac.uk)

Ondrej Hovorka - Supervisor  
Andrea Da Ronch - Co-Supervisor  

This project was originally conceived and developed as a group design project for The University of Southampton.
