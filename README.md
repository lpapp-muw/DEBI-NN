# Distance-Encoding Biomorphic-Informational Neural Networks (DEBI-NN)
This repository contains information about the DEBI-NN concept, implemented by researchers of the Medical University of Vienna (MUV), Vienna, Austria. The development is conducted at the Center for Medical Physics and Biomedical Engineering (main developer: Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at), under the umbrella of projects, conducted by the [Applied Quantum Computing (AQC) group](https://mpbmt.meduniwien.ac.at/en/research/quantum-computing/).

Note that the DEBI-NN describes a fully-connected neural network scheme. While we understand that - especially in the shadow of modern deep learning approaches - this model scheme may not be practically relevant for some real-life applications, we consider that the novelty of the DEBI-NN concept requires a thorough investigation, which shall be conducted in fully-connected network configurations first.

## DEBI-NN in Action
For a video example of how a DEBI-NN is trained over an open-source tabular data, see:

[YouTube: Distance-Encoding Biomorphic-Informational (DEBI) Neural Network training process example](https://youtu.be/S4Dj5qc7Rno)

## Intended Use
We explicitly state that the DEBI-NN source code does not describe any product and it is not intended to be used in any real-life, especially not in clinical settings. This repository contains description, code and examples of a research tool for experimenting with biomorphic and informational neural network schemes. Therefore, it shall not be used either partly or fully for rendering clinical decisions or to be included in any product. We waive any liabilities for any damage caused by the code.

Please note that our paper describing DEBI-NNs as well as its predictive performance comparison with conventional NNs is currently in review. We strongly believe in the values of scientific integrity. Therefore, as long as our paper is in review and not accepted for publication, we do not recommend to rely on our code.

With that being said, we strongly recommend to investigate the contents of this repository only if you are the reviewer of our paper. Note that we will update this repository with up-to-date information in case our paper undergoes a proper peer review and its publication stage changes.

## License
Software, apps, algorithms, written by MUV employees are protected by copyright. MUV has the exclusive right-of-use. All rights reserved.
Contact the Technology Transfer Office (TTO) of MUV in case you wish to use our solution in a commercial environment:
https://www.meduniwien.ac.at/web/en/research/technologies-and-services/technology-transfer-office/


## Usability
Given, that the purpose with this work is to conduct research investigations and to communicate the findings of such research, readability of our code is important for us. Therefore, to support the understanding of our work, our implementation is currently CPU-only. Hence, the execution time of building DEBI-NNs is currently considerably slower than open-source (e.g., scikit-learn or TensorFlow) conventional NN algorithms, that are operating with GPUs. Our C++ implementation relies on parallel computing on CPU (OMP), and hence, does not require the presence of any GPUs to utilize it.


## Access
See the tabular and MNIST datasets and their DEBI-NN settings we utilized for our study under "Examples".

See the Doxygen documentation of the DEBI implementation under "Documentation". Here, open index.html to get access to the documentation.

See the C++ implementation under "Source".

### Use Case 1: Building and cross-validating DEBI-NN models
See example executions under "Examples/Executions" for two tabular and two MNIST datasets and their respective settings. You can use the "Examples" folder as project folder to build and cross-validate DEBI-NN models over executions found in the above mentioned folder. All subfolders under "Executions" will be handled as DEBI-NN cross-validation tasks that are executed sequentially. The result of each execution will be stored in an automatically-created "Log" subfolder within the folder of the given execution. Two files will be stored: one for logging the cross-validation confusion matrix as well as loss analytics and one for logging the train-validate performance of models across iterations until early stopping is initiatied of till the training reached the maximum iteration count.

**Note**:

We use Mersenne Twister C++ random generators in our implementation by default with a fixed seed (123). Nevertheless, various code parts may still end-up with non-deterministic outcomes. Specificially, differences in job management of operating systems (OS), floating point value representations, external libraries as well as CPU parallelization (Open MP) may result in non-determinsitic results. Therefore, re-executing the same cross-validaiton might lead to fluctuations in the cross-validation results. In case of tabular data we have 100 Monte Carlo folds, hence, across 100 folds this effect is minimial. Nevertheless, the MNIST imaging datasets have only one pre-defined train-validate-test splits, which may result in higher predictive performance fluctuations.

### Use Case 2: Visualizing built DEBI-NN models in a 3D viewer
See an example of DEBI-NN models built over one fold of a tabular data under "Examples/Models". You can use the "Examples" folder as project folder and uncomment [visualizeModels( projectFolder ); in main.cpp](https://github.com/lpapp-muw/DEBI-NN/blob/502e3fc03fce911c51c7b2db9ed6e4b52e172e9f/Source/DEBI-Qt6.3/TestApplication/main.cpp#L108) to run the GUI application which allows you to observe and interact with DEBI-NN models in a 3D viewer. These models demonstrate one example training iteration of a training process (for demonstration purposes, early stopping was switched off when creating and saving these models to result in 1000 model instances over 1000 training iterations).

**Note**:
- Loading up 1000 pre-built model schemes may take a few seconds and during the loading process the GUI might be in a non-responsive stage. Patience is your friend here, until we implement a progress bar.
- In case you save a view or batch save multiple views in the 3D viewer application, a "Screenshots" folder will be created under your project folder. In this folder your screenshots will be stored in PNG format.
- We created 1000 model variants and placed them under "Examples/Models" for your convenience. Note however, that Use Case 1 by default will not store models across training iterations, because this creates an overhead in the execution time. If you are interested to store models of a training process, go to [GeneticAlgorithmOptimizer.cpp](https://github.com/lpapp-muw/DEBI-NN/blob/ed4c7decf596d3ce3de2e64201d0b28b5e67ed45/Source/DEBI-Qt6.3/Evaluation/GeneticAlgorithmOptimizer.cpp) and change the values of the past two class member variables [mIsAlphaToSavePerIteration(false)](https://github.com/lpapp-muw/DEBI-NN/blob/ed4c7decf596d3ce3de2e64201d0b28b5e67ed45/Source/DEBI-Qt6.3/Evaluation/GeneticAlgorithmOptimizer.cpp#L57) and [mModelSaveTargetFolder( "" )](https://github.com/lpapp-muw/DEBI-NN/blob/ed4c7decf596d3ce3de2e64201d0b28b5e67ed45/Source/DEBI-Qt6.3/Evaluation/GeneticAlgorithmOptimizer.cpp#L58) accordingly in the constructor (lines 57-58). We acknowledge that this approach is currently non-convenient, but up until now we focused on executing a scientifically-sound project where certain convenience features got to the back of our TODO list.

## Install DEBI-NN on Linux
The installation of DEBI-NN was tested on Ubuntu 20.04.4 LTS (64-bit) with the requirements below. Except for *Qt*, there are no hard requirements.

### Requirements
- cmake 3.16.3
- GNU Make 4.2.1
- gcc 9.4.0
- Qt 6.3.0 (**hard requirement**)

For the installation of *Qt 6.3.0*, refer to the official installation guide: [doc.qt.io](https://doc.qt.io/qt-6/linux.html)

### How to build DEBI-NN on Linux
To run DEBI-NN on Linux, download or clone this repo and make a build directory:
```
git clone https://github.com/lpapp-muw/DEBI-NN.git
cd DEBI-NN/Source/DEBI-Qt6.3
mkdir build
```
To configure the build, change into the build directory and run CMake:
```
cd build
cmake ../ -DCMAKE_PREFIX_PATH=/path/to/Qt/6.3.0/gcc_64/
```
where `../` points to the the DEBI-NN source directory and `/path/to/Qt/6.3.0/gcc_64/` to the path of the Qt library.

After the configuration has run, CMake will have generated Unix Makefiles for building DEBI-NN. To run the build, execute make in the build directory:
```
make -jN
```
where `N` specifies the number of parallel jobs you want to use for building.

To run DEBI-NN, launch the test application:
```
./TestApplication/TestApplication /path/to/project
```
where `/path/to/project` is the project path containing the subfolder "Executions" and "Models" (see Access for use cases).


## Citation
For more information about DEBI-NN, please read the following paper (**manuscript in submission**):
```
[TITLE AUTHORS DOI]
```
