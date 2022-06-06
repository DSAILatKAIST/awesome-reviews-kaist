---
description: Li et al. / Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning / CVPR-2021
---



# Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning

> Li, B., Li, Y., & Eliceiri, K. W. (2021). Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14318-14328).

## 1. Problem Definition  

### Histopathology
Human tissue samples are extracted and visualized for the purpose of diagnosing diseases through histopathology. Slide scanners are now capable of converting glass slide mounted samples into high-resolution images with multiple magnifications for examination and analysis.

### Tumor detection in Whole Slide Image (WSI)
Digital histopathology includes tumor detection as a major topic. The challenges arise due to the vastly high resolution of the whole slide images and the lack of localized annotations. One Whole Slide Image (WSI) can have up to 100k x 100k pixels which requires prohibitive memory and multiple magnification for analysis. Most clinic slides in clinics are unannotated and drawing annotations itself is difficult, resulting in hassle getting localized annotations. Researchers have long sought to develop methods of detecting tumors in WSIs using only slide-level labels. Due to the large image dimensions, patch extraction is often necessary for processing WSI. An unannotated tumor slide, however, cannot tell which of the extracted patches are positive.

## 2. Motivation  

### WSI Classification and Multiple Instance Learning (MIL)
In conditions where there is no unannotated slide, WSI classification can be cast as a multiple instance learning (MIL) problem. This is because it only requires slide level labels and that there are large amounts of data collected by clinics. This article proposed a system for WSI classification that was derived from MIL, where the model is trained on a large number of unannotated slides. In the MIL setting, patches are known as instances and a set is referred to as a bag. Using a set of images and only a global label, the classifier is trained to predict the label of the image set and distinguish positive patches. The bag label is positive if the bag contains at least one positive instance. 

However, there are two challenges in building deep MIL models for weakly-supervised WSI classification:

1. Data imbalance  
   Because only a small percentage of patches are positive, models that use a simple aggregation method like max-pooling are prone to misclassify those positive cases. Max-pooling, when compared to fully-supervised training, can cause a shift in the decision boundary under MIL assumptions. Because of the limited supervisory signal, the model is prone to overfitting and is unable to learn rich feature representations.

   <figure>
   <img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Figure1.png"><figcaption align = "center">
   
   <i>Figure 1. Decision boundary learned in MIL. Left: Max-pooling delineates the decision boundary according to the highest-score instances in each bag Right: DSMIL measures the distance between each instance and the highest-score instance.</i>
   </figcaption>
   </figure>

2. Fixed patch features  
   Because end-to-end training of the feature extractor and aggregator is extremely expensive for large bags, current models either use fixed patch features derived by a CNN or only update the feature extractor using a few high score patches. This could result in patch features that aren't ideal for WSI categorization.

The study proposed a new deep MIL model, named dual-stream multiple instance learning network, to address the aforementioned issues (DSMIL). DSMIL uses a two-stream architecture to concurrently learn a patch (instance) and an image (bag) classifier.

### Related Work
The proposed model was built based on the existing research on attention-based MIL. In order to pretrain the embedded network, the authors designed and integrated a nonlocal network into the model. Additionally, the work was motivated by the recent success of weakly supervised learning methods for WSI tumor detection.

## 3. Method  

### Method Overview
In the suggested method, patches are first extracted from WSIs at multiple magnifications. Using self-supervised contrastive learning, an embedder network is trained on the patches for each magnification. Afterwards, the patches are projected into embeddings, concatenated, and arranged into feature pyramids. After the embeddings are generated, they are fed into the MIL network. 

<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Figure2.png"><figcaption align = "center">
   
<i>Figure 2. Overview of the DSMIL model. Patches extracted from each magnification of the WSIs are used for self-supervised contrastive learning separately. The trained feature extractors are used to compute embeddings of patches. Embeddings of different scales of a WSI are concatenated to form feature pyramids to train the MIL aggregator. The figure shows an example of two magnifications (20× and 5×). The 5× feature vector is duplicated and concatenated with each of the 20× feature vectors of the sub-images within this 5× patch.</i>
</figcaption>
</figure>


### Dual-Stream MIL Network
The MIL network is made up of two branches. The first branch utilizes max-pooling to select the important instance with the highest score from the scores obtained by a linear classification head acting on the examples. The second branch of the algorithm learns a bag representation, which is a weighted sum of the instances. The distances between the instances and the critical instance determine the weights. The two branches combine to generate a masked nonlocal block in which only attentions between the critical instance and all other instances are calculated. The suggested approach can therefore lead to a better decision boundary than existing operators such as max-pooling by using similarity to the critical instance as a regularization.

#### First stream - critical instance identification  
Let $B = \{x_1, ..., x_n\}$ denote a bag of patches of a WSI. Given a feature extractor $f$, each instance $x_i$ can be projected into an embedding $h_i = f(x_i) ∈ R^{L×1}$. The first stream uses an instance classifier on each instance embedding, followed by max-pooling on the scores to determine the critical instance with $W_0$ as a weight vector.  
<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/eq1.png">
</figure>
   
#### Second stream - instance embeddings aggregation  
The second stream aggregates the above instance embeddings into a bag embedding which is further scored by a bag classifier. We obtain the embedding $h_m$ of the critical instance, and transform each instance embedding $h_i$ (including $h_m$) into two vectors, query $q_i ∈ R^{L×1}$ and information $v_i ∈ R^{L×1}$, with $W_q$ and $W_v$ each is a weight matrix. A distance measurement $U$ between an arbitrary instance to the critical instance is then defined by taking inner products of two query vectors.  

<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/eq2.png">
</figure>

The bag embedding $b$ is the weighted element-wise sum of the information vectors $v_i$ of all instances, using the distances to the critical instance as the weights.   
<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/eq3.png">
</figure>  
   
With $W_b$ as a weight vector for binary classification, the bag score $c_b$ is then calculated as follow.  
<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/eq4.png">
</figure>

The final bag score is the average of the scores of the two streams.  
<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/eq5.png">
</figure>

### Self-Supervised Contrastive Learning for Embeddings
Training the MIL and embedding networks end-to-end might be challenging due to the massive prohibitive memory requirements to see the enormous bags at once. In addition to that, the occurrence of imbalanced bags where a positive bag only includes a limited number of positive instances makes it hard to converge. As a result, the article proposed that the embedder network be pre-trained using self-supervised contrastive learning. This allows for the learning of good representations from a large number of unlabeled patches while also reducing the memory requirements for large bags by precomputing the patch embeddings. 

### Feature Pyramid for Locally-Constrained Attention
Because of the large image dimensions, it is typical to examine features from several magnifications for WSI analysis. The authors recommended that feature pyramids be built utilizing embeddings from several magnifications, with lower magnification embeddings being repeated and concatenated with higher magnification embeddings that belong to the same lower magnification patch. If the embeddings are spatially adjacent to each other, they will have the same parts. This technique imposes a spatial constraint on the attention scores, which are generated using similarity measurements and also include multiscale information.

<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Figure4.png"><figcaption align = "center">
   
<i>Figure 3. Pyramidal concatenation of multiscale features in WSI. Feature vector from a lower magnification patch is duplicated and concatenated to feature vectors of its higher magnification patches.</i>
</figcaption>
</figure>

## 4. Experiment

### Experiment setup
* Dataset  
The suggested technique was tested on two publicly accessible WSI datasets, namely the Camelyon16 and TCGA lung cancer datasets. The Camelyon16 dataset contains 400 WSI and was intended to identify breast cancer. There are 1053 slides in the TCGA lung cancer dataset, including two subtypes of lung cancer. 

* Baseline  
The proposed DSMIL model was evaluated and compared to deep models using traditional MIL pooling operators such as max-pooling and mean-pooling. For the tasks of WSI classification and tumor localization, the model was compared to recent deep MIL models (i.e. MIL-RNN, ABMIL, MS-ABMIL).

* Evaluation Metrics  
The authors used the classification area under the curve (AUC) and localization free-response receiver operating characteristics (FROC) as evaluation metrics to compare the method to several recent models. 

### Result

#### Results on WSI Datasets
1. Tumor detection in WSI datasets  
   Based on tumor detection experiments on WSI datasets, it was known that the proposed method outperforms previous state-of-the-art methods with an average of 2%.  
   
   <figure>
   <img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Table0.png"><figcaption align = "center">
   
   <i>Table 1. Classification results on Camelyon16 and TCGA datasets. DSMIL/DSMIL-LC denote DSMIL model with/without the proposed multiscale attention mechanism. Instance embeddings are produced by the feature extractor trained using SimCLR for all MIL models.</i>
   </figcaption>
   </figure>  

2. Detection map for detection localization  
   The authors also further visualize the attention map generated by the models. The results showed that the proposed model performed better in delineating tumor regions in WSIs.  
   
   <figure>
   <img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Figure5.png"><figcaption align = "center">
   
   <i>Figure 4: Tumor localization in WSI using different MIL models. (a) A WSI from Camelyon16 testing set. (b)-(e) zoomed in area in the orange box of (a). (b) Max-pooling. (c) ABMIL. (d) DSMIL. (e) DSMIL-LC Note: for (b), classifier confidence scores are used for patch intensities; for (c) (d) and (e), attention weights are re-scaled from min-max to [0, 1] and used for patch intensities.</i>
   </figcaption>
   </figure>  
   
3. Effect of using self-supervised contrastive learning  
   The paper compared the features learned using different pretraining approaches to highlight the effect of utilizing self-supervised learning features. Self-supervised learning produced superior features when compared to ImageNet features, and was even better than features learned using end-to-end training with max-pooling.  
   
   <figure>
   <img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Table3.png"><figcaption align = "center">
   
   <i>Table 2. Comparison of features learned by different methods for a fixed MIL aggregator.</i>
   </figcaption>
   </figure>  
   
#### Results on General MIL Benchmark Datasets

At last, the study tested the model on several MIL benchmark datasets and compared the performance to several recent MIL models. The proposed model showed a clear-cut improvement in classification accuracy over the recent models and demonstrates state-of-the-art performance. 

<figure>
<img src="/.gitbook/2022-spring-assets/NabilahMuallifah/Table5.png"><figcaption align = "center">
   
<i>Table 3. Performance comparison on classical MIL dataset (MUSK1, MUSK2, Fox, TIGER, and ELEPHANT). Experiments were run 5 times each with a 10-fold cross-validation. The mean and standard deviation of the classification accuracy is reported (mean ± std).</i>
</figcaption>
</figure>

## 5. Conclusion

In conclusion, the study proposed a novel MIL network for tumor detection in WSIs. The authors used self-supervised contrastive learning to learn features for the MIL network. A multiscale feature pyramid based on the attention mechanism was implemented for WSI data. Experiment results showed clear-cut improvements over existing models on WSI datasets. On top of that, the additional benchmark results on standard datasets further demonstrated the superior performance of the proposed network on general MIL problems. 

## Author Information

Nabilah Muallifah

* Korea Advanced Institute of Technology (KAIST)
* Lab. of Knowledge Innovation Research Center (KIRC)

## 6. Reference & Additional Materials

* Github code: https://github.com/binli123/dsmil-wsi

* Reference:
  * [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939)
