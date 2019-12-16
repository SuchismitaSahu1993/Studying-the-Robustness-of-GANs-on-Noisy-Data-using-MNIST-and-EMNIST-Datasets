# Studying-the-Robustness-of-GANs-on-Noisy-Data-using-MNIST-and-EMNIST-Datasets

### Suchismita Sahu
Department of Electrical and Computer Engineering<br>
Viterbi School of Engineering<br>
University of Southern California<br>
suchisms@usc.edu

### Supervised By - Dr.Anand A. Joshi
Research Assistant Professor of Electrical and Computer Engineering<br>
Department of Electrical and Computer Engineering<br>
Viterbi School of Engineering<br>
University of Southern California<br>
ajoshi@usc.edu

## 1. Introduction
Here we study the problem of learning generators from noisy data on MNIST handwritten digit dataset that gives insights about the robustness of Generative Adversarial Networks towards Noisy Data. The dataset is corrupted for experiments in two scenarios; Corruption by Gaussian noise and corruption by EMNIST dataset. We show experimentally that Generative Adversarial Networks are robust to noise in the data upto a certain threshold and various artifacts are introduced in the output generated images on increasing the noise above the threshold.
![MNIST_GAN](https://miro.medium.com/max/1416/1*6zMZBE6xtgGUVqkaLTBaJQ.png)

## 1.1. Generative Adversarial Network
Generative Adversarial Networks are a class of generative models that can generate new content. GANs have been really successful in many current applications including generating synthetic datases for research, neural style transfer, image de-noising etc. GANs aim to learn the underlying distribution of the training data. The Generator and Discriminator are trained in such a way that the generator leans to create data such that the discriminator isn't able to distinguish it as fake anymore. The optimization function that a GAN solves is:
![Objective_Function](https://miro.medium.com/max/2344/1*FbQLpEVQKsMSK-c7_5KcWw.png)
Where, D and G are the discriminator and generator respectively optimized over function classes of our choise and N is the distribution of the latent random noise vector.

### Types of GANs:
## 2. Architecture Implemented

## 3. Literature Review/ Related Work


## 4. Experiements
## 4.1. Data
The dataset used here is the MNIST handwritten digits recognition dataset which has about 60,000 images in training and . The dataset can be found at 

## 4.2. Code and Baselione

## 4.2. Dataset Created for Experiments

## 4.3. Experiement Done

## 5. Results and Discussion

## 6. Reproducing Results
## 6.1. Usage
To run the ipython notebook, upload the notebook on Google Colab. The code downloads both MNIST and EMNIST datasets from online. Change the Input to the Create_Dataset Function to reproduce the result with varying Noise percentages to the GAN.


## 6. References
1. https://papers.nips.cc/paper/8229-robustness-of-conditional-gans-to-noisy-labels.pdf 
2. https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29
3. 
