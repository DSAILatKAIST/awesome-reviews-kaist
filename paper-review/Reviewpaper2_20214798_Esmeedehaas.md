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


Recent approaches that craft hybrid decoders [6, 7, 8] have presented a very promising direction, yet they struggle
to operate well at the lowest bit rate, given that the quantized signals they must work
with to reconstruct the original input signal are extremely sparse.

Another promising
alternative is to design a hybrid encoder that enhances encoder signals resulting in
better compression even at the lowest bit rates [9, 10]. However, these methods fail
to remove artifacts at the decoder end, thus compromising compression quality in
various situations.

* xx

 ### **Significance**
Most of the current works have xx

In light of this,
we propose a system that learns to improve the encoding performance by enhancing its
internal neural representations on both the encoder and decoder ends, an approach we call
Neural JPEG. We propose frequency domain pre-editing and post-editing methods to optimize
the distribution of the DCT coefficients at both encoder and decoder ends in order to
improve the standard compression (JPEG) method. Moreover, we design and integrate a
scheme for jointly learning quantization tables within this hybrid neural compression framework.

In summary, our contributions are as follows:
* We extend on prior work and improve system rate-distortion performance by optimizing the JPEG encoder in the frequency domain.
* We facilitate better coefficient construction at the decoder end by optimizing the JPEG decoder.
* A sparse recurrent network (Neural JPEG) is adapted to learn how to edit the DCT coefficients at both decoder and encoder ends.
* A learnable quantization table that is optimized jointly with the sparse recurrent encoder/decoder to improve rate-distortion performance, yielding an end-to-end, differentiable JPEG compression system.

## **3. Method**  

In this method, xx

Speciffically, we construct a system that leverages an encoder and decoder
that are each driven by sparse recurrent neural networks (SMRNNs) trained within
the effective framework of neural iterative refinement [6] { the recurrent encoder learns
to \pre-edit" an input image in the frequency domain (producing values that serve as
the necessary DCT coefficients) while the recurrent decoder learns to reduce artifacts
in the reconstructed image.

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Figure 1, proposed method.* |


### **JPEG algorithm & architecture**

Standard JPEG algorithm:
<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">

Standard JPEG architexture:
<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">

Modifications:
* Recurrent neural networks (RNN):
  1. Reduce channel dimension from 256 to 128
  2. Reshape channels to a 8x8x2 tensor for each block which will be split in two matrices of 8x8 afterwards.
* The sparse multiplicative RNN (SM-RNN) component will be used to process both Hl and Hc for K steps. So the activation map in the case of Hl would look like this:
  <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">
  The same algorithm can be used replacing Hc with Hl. A final set of sparse 'edit' values is conducted as followed:
  <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">
  Each discrete cosine transform (DCT) coefficient is multiplied by the original JPEG encoder with corresponding edit score as retrieved in previous mentioned process.
* Decoding method: For the decoder end of our system, we utilize a SM-RNN similar to the encoder
process described above but tie its weights to those of the SM-RNN encoder module
(both models would have the same dimensionalities for their parameters given they
operate in the same internal latent space). In short, after applying the inverse DCT
transform to the outputs of the quantization table and rounding modules (which
produce ^ Z), the SM-RNN decoder takes in the ^Z, processes it K times (in a process
similar to the one depicted above) and nally produces the reconstructed image ^ I.
* Quantitzation table and rounding module:
  
* x


### **Loss formulation**

* Disortion loss:
* Rate loss:
* Alignment loss:


<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">




## **4. Experiment**  
The overall experiment results show that xx

Experiments demonstrate that our approach successfully improves the rate-distortion
performance over JPEG across various quality metrics, such as PSNR and MS-SSIM, and
generate visually appealing images with better color retention quality.


### **Experiment setup**  
* Evaluation metrics:
* Datasets and training procedure:


### **Results**  

The results of the previously mentioned procedure xx

Graph:
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Figure 1, proposed method.* |

Table:
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Table 1, proposed method.* |

Table:
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Table 1, proposed method.* |

## **5. Conclusion**  

Our experiments show that our approach, Neural JPEG, improves JPEG encoding
and decoding through sparse RNN smoothing and learned quantization tables that
are trained end-to-end in an dierentiable framework. The proposed model leads to
better compression/reconstruction at lowest bit rates when evluated using metrics

such as MSE, PSNR and also using perceptual metrics (LPIPS, MS-SSIM) that are
known to be much closer to human perception. Most importantly, the improved
encoder-decoder remains entirely compatible with any standard JPEG algorithm but
produces signicantly better colors than standard JPEG. We have shown that we can
achieve improvement without directly estimating the entropy of the DCT coecients,
only regularizing the sparse maps and quantization tables.



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
