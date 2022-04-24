---
description: Ko & Lim / Unsupervised Detection of Adversarial Examples with Model Explanations / KDD-2021
---



# Unsupervised Detection of Adversarial Examples with Model Explanations

> Ko, G., & Lim, G. (2021). Unsupervised Detection of Adversarial Examples with Model Explanations. arXiv preprint arXiv:2107.10480.

## 1. Problem Definition  

In the last few years, adversarial attacks are one of the main issues in security threats. It alters the behavior of a deep neural network by utilizing data samples which have been subtly modified. Adversarial perturbations, even simple ones, can affect deep neural networks. In this case, the model may produce incorrect results and cause damage to the security system. The following example of adversarial attack on a panda image will give you an idea of what adversarial examples look like. A small perturbation is applied to the original image so that the attacker is successfully misclassifying it as a gibbon with high confidence.

<figure>
<img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/0.png"><figcaption align = "center"><i>Figure 1: An adversarial perturbation can manipulate a classifier to misclassify a panda as a gibbon.</i>
</figcaption>
</figure>



## 2. Motivation  

In order to identify which input data may have undergone adversarial perturbation, the new technique uses deep learning explainability methods. The idea came from observing that adding small noise to inputs affected their explanations greatly. As a consequence, the perturbed image will produce abnormal results when it is run through an explainability algorithm.  

In many existing detection-based defenses, adversarial attacks are detected with supervised methods or by modifying current networks, which often require a great deal of computational power and can sometimes lead to a loss of accuracy on normal examples. Several previous works used pre-generated adversarial examples, which resulted in subpar performance against unknown attacks. Additionally, they result in a high computational cost due to the large dimension in model explanations. While other existing methods require less computation power, their transformations lack generalization so it may only work for the specified dataset.  

In contrast to many previous attempts, the proposed method uses an unsupervised method to detect the attack. It does not rely on pre-generated adversarial samples, making it a simple yet effective method for detecting adversarial examples.

## 3. Method  

In this method, a saliency map is used as an explanation map to detect adversarial examples. For image inputs, each pixel is scored based on its contribution to the final output of the deep learning model and shown on a heatmap.  

<figure>
<img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/00.png"><figcaption align = "center"><i>Figure 2: Examples of saliency map based on importance or contribution of each pixel.</i>
</figcaption>
</figure>


There are three steps in this method:

1. Generating input explanations  

   By using explainability techniques, inspector networks create saliency maps based on the data examples used to train the original model (target classifier). With Φ<sup>𝑐</sup> as a set of input explanations of output label 𝑐, we get:

   <figure>
   <img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/eq1.png">
   </figure>

2. Training reconstructor networks  

   By using the saliency maps, the inspector trains reconstruction networks (autoencoder) which are capable of recreating each class' explanation. An explanation map is then produced for a given image input. For example, in a handwritten digit case, it will need ten reconstructor networks. When an input image is classified by the target classifier as a “1”, the image is then entered to the class “1” reconstructor network and a saliency map is produced.  

   The training process is done by optimizing:

   <figure>
   <img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/eq2.png">
   </figure>


   where LΦ(𝜃; ·) is a reconstruction loss for parameterized network 𝑔(𝜃; ·) on Φ.

3. Separating adversarial examples.  

   The networks are trained on unperturbed examples. Hence, when presented with an adversarial example (abnormal explanation), the reconstruction network will produce poor results, making it possible for the inspector to detect adversarially perturbed images. If the reconstruction error (𝜙′) of a given input (𝑥′) is higher than given threshold 𝑡′<sub>𝑐</sub> then the input is expected to be an adversarial example.

## 4. Experiment

### Experiment setup

The method is evaluated on the MNIST dataset. A simple CNN network is used as target classifier, saliency maps are generated using input gradients method, and all reconstructor networks consist of one single hidden layer autoencoder. Adversarial examples are generated using FGSM, PGD, and MIM methods and performance evaluation is measured using Area Under the ROC Curve (AUC).

### Result

**Effect of input perturbations on explanations**  

<figure>
<img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/1.png"><figcaption align = "center"><i>Figure 3: Input, gradient, and reconstruction of an example MNIST image and adversarial examples crafted using the image. For each attack, adversarial example with 𝜖 = 0.1 is created.</i>
</figcaption>
</figure>


Adversarial perturbation on input proved to lead to an obvious alteration in their explanation. The above figure shows that reconstructions of adversarial explanations have more noise than those of non-adversarial explanations. 

**Adversarial detection performance**  

<figure>
<img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/2.png"><figcaption align = "center"><i>Figure 4: Area under the Receiver Operating Characteristic (ROC) curve obtained according to the attack’s severity (parameterized by 𝜖), for (a) FGSM, (b) PGD, and (c) MIM attacks. For each class label, our proposed detector’s performance is recorded using adversarial examples created using given (attack, epsilon) pair. Grey areas show the min-max range of AUC, and black lines show average value of AUC across different class labels.</i>
</figcaption>
</figure>


Overall, the method has difficulty on detecting adversarial examples with low noise level (𝜖 < 0.1). However, in the standard setting for MNIST dataset (𝜖 = 0.1), the experimental result shows that this method has relatively high performance with average AUC of 0.9583 for FGSM, 0.9942 for PGD, 0.9944 for MIM.

**Quantitative comparison to previous approaches**  

<figure>
<img width="140" src=".gitbook/2022-spring-assets/NabilahMuallifah_1/3.png"><figcaption align = "center"><i>Table 1: Comparison on adversarial detection accuracy of the proposed (Ko & Lim) and existing approaches. The best and the second best results are highlighted in boldface and underlined texts, espectively. All benchmarks are done on MNIST dataset.</i>
</figcaption>
</figure>


The above table shows that the proposed method has better or on-par accuracy compared with previously existing works. 

## 5. Conclusion

As a means of securing deep learning models, the paper proposed model explanations that are critical in repairing vulnerability in deep neural networks. A new method is suggested to identify which input data may have undergone adversarial perturbation based on model explainability. Small adversarial perturbation will greatly affect model explanation and produce abnormal results. 

According to the results of the experiment utilizing the MNIST dataset, adversarial explanation maps are present in all adversarial attack approaches. This proves that the method is attack-agnostic and therefore does not require pre-generated adversarial samples and generalized to unseen attacks. The unsupervised detection approach was also found to be capable of detecting various adversarial examples with performance comparable to or better than existing methods. Moreover, the unsupervised defense method using model explanations is efficient to detect adversarial attacks as it only requires a single training for reconstructor networks.

Despite all the advantages described previously, using the MNIST dataset to evaluate the method is considered rather straightforward. The dataset may fail to replicate the complexities of real-world adversarial attacks, making its applicability to more complex cases questionable. In the future, further evaluation is needed to check the performance of the proposed method on a more complex and realistic dataset of adversarial attacks.

## Author Information

Gihyuk Ko

* Carnegie Mellon University
* Formal methods, security and privacy, and machine learning

Gyumin Lim

* CSRC, KAIST

* AI, cybersecurity

## 6. Reference & Additional Materials

* Github code: None
* Reference:
  * [Unsupervised Detection of Adversarial Examples with Model Explanations](https://arxiv.org/abs/2107.10480)
  * [Adversarial Attack](https://arxiv.org/abs/1412.6572)
  * [Saliency Map](https://arxiv.org/abs/1512.04150)
  * [Unsupervised learning can detect unknown adversarial attacks](https://bdtechtalks.com/2021/08/30/unsupervised-learning-adversarial-attacks-detection/)