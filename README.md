# Distance-Encoding Biomorphic-Informational Neural Networks (DEBI-NN)
This repository contains information about the DEBI-NN concept, implemented by researchers of the Medical University of Vienna (MUV), Vienna, Austria. The development is conducted at the Center for Medical Physics and Biomedical Engineering (main developer: Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at), under the umbrella of projects, conducted by the [Applied Quantum Computing (AQC) group](https://mpbmt.meduniwien.ac.at/en/research/quantum-computing/).

Note that the DEBI-NN describes a fully-connected neural network scheme. While we understand that - especially in the shadow of modern deep learning approaches - this model scheme may not be practically relevant for real-life applications, we consider that the DEBI-NN concept is worthy to investigate in scientific experimental settings. The first logical step of this investigation is to build fully-connected networks for the basic understanding of this model scheme's abilities and properties.

## DEBI-NN in Action
For a video example of how a DEBI-NN is trained over an open-source tabular data, see:

[YouTube: Distance-Encoding Biomorphic-Informational (DEBI) Neural Network training process example](https://youtu.be/S4Dj5qc7Rno)

## Intended Use
We explicitly state that the DEBI-NN binary does not describe any product and it is not intended to be used in any real-life, especially not in clinical settings. This repository contains description and examples of a research tool for experimenting with biomorphic and informational neural network schemes. Therefore, the contents of this repository shall not be used either partly or fully for rendering clinical decisions or to be included in any product. We waive any liabilities for any damage caused by the executables.


## License
Software, apps, algorithms, written by MUV employees are protected by copyright. MUV has the exclusive right-of-use with all rights reserved. 

In case you wish to use our solution in a commercial environment please contact the Technology Transfer Office (TTO) of MUV: https://www.meduniwien.ac.at/technologietransfer

Contact the correspondig author (Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at) in case you are interested in research collaborations regarding the utilization of DEBI-NNs.

## Usability
Given, that the purpose with this work is to conduct research investigations and to communicate the findings of such research, repeatability of our findings is important for us. Note, however, that our current DEBI-NN implementaton is CPU-only. Hence, the execution time of building DEBI-NNs is currently considerably slower than open-source (e.g., scikit-learn or TensorFlow) conventional NN algorithms, that are operating with GPUs. Our C++ implementation relies on parallel computing on CPU (OMP), and hence, does not require the presence of any GPUs to utilize it.


## Access
See the tabular and MNIST datasets and their DEBI-NN settings we utilized for our study under "Examples".

See the Windows and Linux executables in "Binaries".


### Use Case 1: Building and cross-validating DEBI-NN models
See example executions under "Examples/Executions" for two tabular and two MNIST datasets and their respective settings. The folder containing the "Examples" subfolder is the project folder to build and cross-validate DEBI-NN models. All subfolders in the project folder (e.g., "Executions") and all subfolders under those subfolders will be handled as DEBI-NN cross-validation tasks that are executed sequentially. This is to support batch processing of the same datasets with trying e.g., different settings. The result of each execution will be stored in an automatically-created "Log" subfolder within the folder of the given execution. Two files will be stored: one for logging the cross-validation confusion matrix as well as loss analytics and one for logging the train-validate performance of models across iterations until early stopping is initiatied of till the training reached the maximum iteration count.

**Note**:

We use Mersenne Twister C++ random generators in our implementation by default with a fixed seed (123). Nevertheless, various code parts may still end-up with non-deterministic outcomes. Specificially, differences in job management of operating systems (OS), floating point value representations, external libraries as well as CPU parallelization (Open MP) may result in non-determinsitic results. Therefore, re-executing the same cross-validaiton might lead to fluctuations in the cross-validation results. In case of tabular data we have 100 Monte Carlo folds, hence, across 100 folds this effect is minimial. Nevertheless, the MNIST imaging datasets have only one pre-defined train-validate-test splits, which may result in higher predictive performance fluctuations.

### Use Case 2: Visualizing built DEBI-NN models in a 3D viewer
See an example of DEBI-NN models built over one fold of a tabular data under "Examples/Models". You can use the project folder containing "Models" subfolder and "-v" as additional argument to run the GUI application which allows you to observe and interact with DEBI-NN models in a 3D viewer. These models demonstrate one example training iteration of a training process (for demonstration purposes, early stopping was switched off when creating and saving these models to result in 1000 model instances over 1000 training iterations).

**Note**:
- Loading up 1000 pre-built model schemes may take a few seconds and during the loading process the GUI might be in a non-responsive stage. Patience is your friend here, until we implement a progress bar.
- In case you save a view or batch save multiple views in the 3D viewer application, a "Screenshots" folder will be created under your project folder. In this folder your screenshots will be stored in PNG format.
- We created 1000 model variants and placed them under "Examples/Models" for your convenience. Note that currently, there is no way to store models being built during a cross-validation process. We will update our code soon to allow this option.


### How to run DEBI-NN on Linux
To run DEBI-NN with Use Case 1, launch the test application:
```
./TestApplication/TestApplication /path/to/project
```
where `/path/to/project` is the project path containing the subfolder "Executions" (see Access for use cases).


To run DEBI-NN with Use Case 2, launch the test application:

```
./TestApplication/TestApplication /path/to/project -v
```
where `/path/to/project` is the project path containing the subfolder "Models" (see Access for use cases) and -v indicates Visualization mode.

Requirements

- cmake 3.16.3
- GNU Make 4.2.1
- gcc 9.4.0
- Qt 6.3.0 (hard requirement)
- For the installation of Qt 6.3.0, refer to the official installation guide: doc.qt.io

### How to run DEBI-NN on Windows

To run DEBI-NN with Use Case 1, launch the test application:
```
TestApplication/TestApplication.exe /path/to/project
```
where `/path/to/project` is the project path containing the subfolder "Executions" (see Access for use cases).


To run DEBI-NN with Use Case 2, launch the test application:

```
TestApplication/TestApplication.exe /path/to/project -v
```
where `/path/to/project` is the project path containing the subfolder "Models" (see Access for use cases) and -v indicates Visualization mode.


## Citation
Please cite this article as: L. Papp, D. Haberl, B. Ecsedi et al., DEBI-NN: Distance-encoding
biomorphic-informational neural networks for minimizing the number of trainable parameters.
Neural Networks (2023), doi: https://doi.org/10.1016/j.neunet.2023.08.026.

For the published paper see https://www.sciencedirect.com/science/article/pii/S089360802300446X.
