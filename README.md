# Additive-Manufacturing-Acoustics-Semisupervised-Learning
This repo hosts the codes that were used in journal work "Semi-supervised monitoring of laser powder bed fusion process based on acoustic emissions".
# Journal link
https://doi.org/10.1080/17452759.2021.1966166

![Additive manufacturing setup](https://user-images.githubusercontent.com/39007209/199726535-4fb1a1f6-6299-4072-aa44-173a8c94ac4d.jpg)

# Overview

One of my first works in monitoring the metal-based laser powder bed fusion process (LPBF) using machine learning and acoustic emissions (AE) published in the Journal of Virtual and Physical Prototyping (Taylor & Francis Group). The article proposes a semi-supervised strategy by which the defect-free regime can be differentiated from the anomalies by familiarising generative models only with the distribution of acoustic signatures corresponding to the defect-free regime. Two generative models based on Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) were used to monitor process state as the laser interacts with Inconel powders.

![Semi supervised](https://user-images.githubusercontent.com/39007209/199726399-758d663c-f80c-40f0-a782-6711378fcc26.jpg)

# Generative Models

Autoencoders and their variants find their applications predominant in image denoising (Gondara 2016), dimensionality reduction (Mahmud, Huang, and Fu 2020), feature extraction (Nishizaki 2017), image generation (Pandiyan et al. 2019; Cai, Gao, and Ji 2019), machine translation (Pagnoni, Liu, and Li 2018), and anomaly detection (Sakurada and Yairi 2014; Hahn and Mechefske 2021; Pandiyan et al. 2021). An autoencoder architecture generally consists of a pair of networks, namely an encoder and a decoder whose purpose is to learn the identity function for the data distribution they had been trained on. The encoder-decoder combination learns the data representation efficiently in a dense manner and reconstructs the original input. The encoder network maps the original data x∈ X, to z belonging to low dimensional latent space z using a function Ф. Subsequently, the decoder network recreates x'∈ X similar to the original data from z by a function Ψ, as depicted in Equations (1) and (2):


![CodeCogsEqn(2)](https://user-images.githubusercontent.com/39007209/199745977-b344e46a-b3f8-4e98-bc79-7aa713cdcbd6.gif)


During training, the model learns to retain the minimal information to encode the original data X so that it can be regenerated as the output on the other side by back-propagating the reconstruction loss as presented in Equation (3), which is the difference between the input and output. Once the autoencoder has been trained, we both have an encoder and a decoder to reconstruct the input. However, still, there are chances of overfitting as the latent space is not regularised. Variational autoencoder (VAE) is one of the types of autoencoders where latent space distribution is regularised during the training. The VAE provides a probabilistic manner for describing an observation in latent space. Thus, instead of building an encoder that outputs a single value to describe each latent state attribute, we will make our encoder define a probability distribution for each latent feature. In other words, the encoder does not directly map to the latent space as depicted in Figure 1(a), instead it generates two quantities, mean (μ) and variance describing the distribution .

Unlike vanilla autoencoders (An and Cho 2015), the loss function of the VAE network consists of two terms. The first term maximizes the reconstruction likelihood similar to Equation (3). The second term, also known as the Kullback–Leibler (KL) divergence, encourages the learned distribution q(z|x) to be identical to the true prior distribution p(z), for each dimension j of the latent space as depicted in Equation (4). The KL divergence score ensures that the distribution learned q is similar to the true existing distribution p.

![VAE](https://user-images.githubusercontent.com/39007209/199726688-9c932de7-d771-40b0-a744-ab96f2d30e36.jpg)

A Generative Adversarial Network (GAN) is based on the idea that two adversarial networks, a generative network G and a discriminative network D that are set against one another during model training. The goal of the generative network's is to create new distribution samples that are different but still reminiscent enough from the training data. The goal of the discriminator network is to differentiate the synthetic distribution created by the generator network from the original training set.Based on the set objective, the two networks iteratively improve during training such that the generator network is capable of creating synthetic data resembling the actual distribution.The training of the two networks requires a loss function, which primarily depends on the second network. The update of the weights does not occur simultaneously in the networks. The loss function for the vanilla GANs are of the form shown in Equation (5), where D is the discriminator, G is the generator, p_z (z) is the input noise distribution, p_data (x) is the original data distribution, and p_g (x) is the generated distribution. The objective of the architecture is to maximize the discriminator (D) and minimize the generator (G). V is the sum of the Expected log-likelihoods for real and generated data. The loss function aims to move p_g (x) towards p_data (x) for an optimal D.

![CodeCogsEqn(4)](https://user-images.githubusercontent.com/39007209/199747757-c1b1dfbe-f1ae-4798-a0cc-227f275738db.gif)

GANomalies are recent variants of the GAN network architectures where — based on known input — the network would generate a manifold representation of the input. However, when unusual input is encoded, its reconstruction can be poor, which can be used for anomaly detection. GANomaly (Akcay, Atapour-Abarghouei, and Breckon 2018), AnoGan (Schlegl et al. 2019), and Efficient-GAN-Anomaly (Zenati et al. 2018) are the adversarial networks based on GAN architecture for identifying anomalies and outliers.



 
![GAN](https://user-images.githubusercontent.com/39007209/199726830-a512716c-8c8a-49c1-82aa-829d23cc106c.jpg)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Acoustics-Semisupervised-Learning
cd Additive-Manufacturing-Acoustics-Semisupervised-Learning
python VAE/VAE_Main.py
python GANOmaly/GAN_Main.py
```
#Data
https://polybox.ethz.ch/index.php/s/Yv7c7BS6KLZAKHD

# Results

![Model prediction](https://user-images.githubusercontent.com/39007209/199726999-526a5fd0-ef51-49e4-8499-3510d7b3463e.jpg)


# Citation
```
@article{pandiyan2021semi,
  title={Semi-supervised Monitoring of Laser powder bed fusion process based on acoustic emissions},
  author={Pandiyan, Vigneashwara and Drissi-Daoudi, Rita and Shevchik, Sergey and Masinelli, Giulio and Le-Quang, Tri and Log{\'e}, Roland and Wasmer, Kilian},
  journal={Virtual and Physical Prototyping},
  volume={16},
  number={4},
  pages={481--497},
  year={2021},
  publisher={Taylor \& Francis}
}
```
