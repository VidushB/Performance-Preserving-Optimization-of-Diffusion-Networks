# Performance-Preserving-Optimization-of-Diffusion-Networks

Final project for HPML Spring 23 at NYU
The project we decided to work with is about the optimization of diffusion networks. As we will see in the latter sections of the report, the main idea of the project revolves around the possible ways to optimize the training of diffusion networks, by making use of PyTorch Profiling.

Preparing the Dataset and reproducable:

We download the CIFAR-10 dataset from Kaggle as a zip. The zip file gets converted to a folderset using our code in cifar10loader.py. 

To run code, you will need to install accelerate, ema_pytorch and einops along with torch. Also make sure that the directory in your training functions matches the location of CIFAR-10 training set within your system. Similarly, ensure that the bash files are given access to the directory containing your training code.


This repo contains all the necessary files needed to run the code, and also the reports.
To run the training of the diffusion model, you can just clone this repository in your machine (GPU available machine)
and run **nvbb training.ipynb**. The file includes all the necessary imports in the first few lines. There are also different python attempt files and their outputs, each with respective specifics incorporated. 
Note that you can also run **evaluating.ipynb** to evaluate and see the performance of the model, but this process isn't strongly related to the optimization of training process for diffusion models. 


**RESULTS**
Inferences we made:
1. The convolutional backpropagation is the main bottleneck.
2. Having AMP vs no AMP sped up the CPU runtime for 1 GPU, it also slightly
improved losses.
3. 2 GPUs gave a slight speedup compared to 1 GPU
4. On 2 GPUs, AMP didn’t improve runtimes
5. The convolutional backpropagation is the main bottleneck.
6. In two GPUs, data is parallelized, so the model spends less time on backprop.

Stats:
1-1 GPU with AMP : CPU 28.3 CUDA 25.5    Total runtime - 502
2-1 GPU w/out AMP : CPU 42.4 CUDA 41.09   Total runtime - 428
3-2 GPUs with AMP : CPU 39.7 CUDA 25.4    Total runtime - 572
3-2 GPUs w/out AMP : CPU 38.2 CUDA 25.3  Total runtime - 559
