| Description |
| --- |
| Ankur Mali / Neural JPEG: End-to-End Image Compression Leveraging a Standard JPEG Encoder-Decoder / 2022(DCC) |

# **Review paper Neural JPEG** 

Neural JPEG: End-to-End Image Compression Leveraging a Standard JPEG Encoder-Decoder


## **1. Problem Definition**  



However, current methods
either use additional post-processing blocks on the decoder end to improve compression
or propose an end-to-end compression scheme based on heuristics. For the majority of
these, the trained deep neural networks (DNNs) are not compatible with standard encoders
and would be difficult to deply on personal computers and cellphones.




## **2. Motivation**  

In this paper, the authors want to improve xx

### **Existing works**

Recent advances in deep learning have led to superhuman performance across a variety
of applications. Recently, these methods have been successfully employed to improve the
rate-distortion performance in the task of image compression.

* Previous works xx
* research in deep neural networks (DNNs)
has begun to turn its attention to improving the rate-distortion performance of im-
age compression frameworks.

Though this set of approaches has yielded impressive results, they
unfortunately require a speciffcally trained decoder during the post-processing stage
or a complex DNN-based decoder. As such they are not supported by the commonly
used image viewers in most computers and smartphones. In addition, there is no
guarantee that the compression results uncovered would hold when the input data
distribution shifts, e.g., images of a completely different kind are presented to the sys-
tem, presenting a challenge to DNN-driven approaches.


* xx

 ### **Significance**
Most of the current works have xx

In light of this,
we propose a system that learns to improve the encoding performance by enhancing its
internal neural representations on both the encoder and decoder ends, an approach we call
Neural JPEG. We propose frequency domain pre-editing and post-editing methods to optimize
the distribution of the DCT coecients at both encoder and decoder ends in order to
improve the standard compression (JPEG) method. Moreover, we design and integrate a
scheme for jointly learning quantization tables within this hybrid neural compression framework.

## **3. Method**  

In this method, xx

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Figure 1, proposed method.* |



### **Encoding method**
In the paper the following optimization problem is applied:

<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">

?

### **Decoding method**
?


## **4. Experiment**  
The overall experiment results show that xx

### **Experiment setup**  
* For this experiment, the researchers chose to use xx
* xx
* XX



### **Result**  

The results of the previously mentioned procedure xx


## **5. Conclusion**  

Experiments demonstrate that our approach successfully improves the rate-distortion
performance over JPEG across various quality metrics, such as PSNR and MS-SSIM, and
generate visually appealing images with better color retention quality.




### **Summary** 

In the table below you can find a summary of all the pros and cons of this newly proposed method gives.

| Pros | Cons |
| --- | --- |
| xx | xx |
| xx | xx |
| xx | xx |


Ideas on pros & cons:
* xx


### **Future ideas** 

* xx
*	xx


---  
## **Author Information**  

* Esmée Henrieke Anne de Haas (20214798)
    * Master student researcher ARM lab KAIST  
    * Research topics: Human Computer Interaction, Metaverse, Augmented Reality, Dark Patterns

## **6. Reference & Additional materials**  

* Reference of paper: Dupont, E., Goliński, A., Alizadeh, M., Teh, Y. W., & Doucet, A. (2021). Coin: Compression with implicit neural representations. arXiv preprint arXiv:2103.03123.
* Link to public materials: https://github.com/EmilienDupont/coin  
