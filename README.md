# Distance-Encoding Biomorphic-Informational Neural Networks (DEBI-NN)
This repository contains information about the DEBI-NN concept, implemented by researchers of the Medical University of Vienna (MedUni Wien), Vienna, Austria. The development is conducted at the Center for Medical Physics and Biomedical Engineering (main developer: Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at), under the umbrella of projects, conducted by the [Applied Quantum Computing (AQC) group](https://mpbmt.meduniwien.ac.at/en/research/quantum-computing/).

Note that the DEBI-NN describes a fully-connected neural network scheme. Implementing convolutional and vision transformer DEBI-NNs is currently in progress. For now, DEBI-NNs are ideal candidates to build prediction models in small, imbalanced (medical imaging or clinical) datasets.

## DEBI-NN in Action
For a video example of how a DEBI-NN is trained over an open-source tabular data, see:

[YouTube: Distance-Encoding Biomorphic-Informational (DEBI) Neural Network training process example](https://youtu.be/S4Dj5qc7Rno)

## License

This repository has Creative Commons Attribution-Noncommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0) license.

In case you wish to use our solution in a commercial environment please contact the Technology Transfer Office (TTO) of MedUni Wien: https://www.meduniwien.ac.at/technologietransfer

Contact the corresponding author (Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at) in case you are interested in research collaborations regarding the utilization of DEBI-NNs and if you wish to access the source code of the project. Currently, no source code is shared here due to the fact that LLM data grabbers reportedly do not respect copyright law.

## Intended Use
We explicitly state that the DEBI-NN binary does not describe any product and it is not intended to be used in any real-life, especially not in clinical settings. This repository contains description and examples of a research tool for experimenting with novel biomorphic neural network schemes. Therefore, the contents of this repository shall not be used either partly or fully for rendering clinical decisions or to be included in any product. We waive any liabilities for any damage caused by the executables.

## Handbook

We wrote an extensive handbook titled "Mastering Distance-Encoding Biomorphic Neural Networks – The DEBI-NN Handbook" which provides detailed explanations about the general concept and behaviour of DEBI-NNs, and it details all parameters modifiable for training DEBI-NN models on tabular data.

To access the Handbook see DOI: 10.5281/zenodo.15827443 or https://doi.org/10.5281/zenodo.15827443.


## Usability
Given, that the purpose with this work is to conduct research investigations and to communicate the findings of such research, repeatability of our findings is important for us. Note, however, that our current DEBI-NN implementation is CPU-only. Hence, the execution time of building DEBI-NNs is currently considerably slower than open-source (e.g., scikit-learn or TensorFlow) conventional NN algorithms, that are operating with GPUs. Our C++ implementation relies on parallel computing on CPU (OMP), and hence, does not require the presence of any GPUs to utilize it.


## Access

See the best models we identified in our latest study in "BestModels". Here, the saved models are located that can be loaded by the DEBI-NN application and can be visualized in a viewer. For details see the Handbook.

See the DEBI-NN eecutables in "Binaries".

See the consolidated results of the DEBI-NN execution in "Results". These results are derived by relying on the data stored in "Executions".


## How to run DEBI-NN

Read the Handbook for all details regarding the usage of the DEBI-NN approach.

**Note**:

We use Mersenne Twister C++ random generators in our implementation with fixed seeds. Nevertheless, differences in job management of operating systems (OS), floating point value representations, external libraries as well as CPU parallelization (Open MP) may result in non-deterministic results. What we guarantee is that on the same computer and configuration, the same repeated execution yields the same result.


## Citation

### Handbook

Papp, L. (2025). Mastering Distance-Encoding Biomorphic Neural Networks – The DEBI-NN Handbook (1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15827443

### PAPER I

Our first paper introducing the concept of DEBI-NNs is published in Elsevier Neural Networks.

Please cite this article as: L. Papp, D. Haberl, B. Ecsedi et al., DEBI-NN: Distance-encoding
biomorphic-informational neural networks for minimizing the number of trainable parameters.
Neural Networks (2023), doi: https://doi.org/10.1016/j.neunet.2023.08.026.

For the published paper see https://www.sciencedirect.com/science/article/pii/S089360802300446X

### PAPER II

Our second paper focusing on the effect of regularization s on DEBI-NN predictive performance is currently in review.

