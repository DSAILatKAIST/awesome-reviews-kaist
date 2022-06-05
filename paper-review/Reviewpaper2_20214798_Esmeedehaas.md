| Description |
| --- |
| Ankur Mali / Neural JPEG: End-to-End Image Compression Leveraging a Standard JPEG Encoder-Decoder / 2022(DCC) |

# **Review paper Neural JPEG** 

Neural JPEG: End-to-End Image Compression Leveraging a Standard JPEG Encoder-Decoder


## **1. Problem Definition**  

The usage off the internet has been increasing over the years resulting in a need for improvement of compressing visual data. Current methods use either use additional post-processing blocks on the decoder end to improve compression or propose an end-to-end compression scheme based on heuristics. For most of these methods the trained deep neural networks (DNNs) are not compatible with standard encoders which makes it difficult to extend these methods to personal computers (PC) and smartphones. Later in the realted works the shortcomings and problems of the current works will be mentioned.


## **2. Motivation**  

In this paper, the authors want to create a system that learns to improve the encoding performance by enhancing the
internal neural representations on both encoder and decoder ends. This approach is a Neural JPEG. The authors propose frequency domain pre-editing and post-editing methods to optimize the distribution of the DCT coefficients at both encoder and decoder ends in order to improve the standard compression (JPEG) method.


### **Existing works**

The works related to this paper are mostly about end-to-end image compression systems using deep neural networks (DNN), hybrid decoders and hybrid encoders that enhances encoder signals resulting in an improvement of compression even at low bit rates.

* Deep neural networks (DNNs): this is an approach that seems to improve the rate-disortion performance of image compression framworks.
However, this method requires a speciffcally trained decoder during the post-processing stage or a complex DNN-based decoder. This makes it hard to apply this method for images that have to be compressed on the PC and smartphones. Besides that, there is no
guarantee that the uncovered results would hold when the input data distribution shifts, e.g., images of a completely different kind are presented to the system.

* Hybrid decoders: this appraoch gives good results but it does not operate well at the lowest bit rate. This is because the quantized signals that are used to reconstruct the original input signal are extremely sparse.

* Hybrid encoders: these encoders enhance encoder signals resulting in an improvement of compression even at low bit rates. But these tupes of methods fail
to remove artifacts at the decoder end, thus compromising compression quality in various situations.


 ### **Significance**
The related works as mentioned before are still lacking some features, which this paper will make up for.

IThe main contributions of this paper are as followed:
* The authors extend their prior work and improve system rate-distortion performance by optimizing the JPEG encoder in the frequency domain.
* The authors facilitate better coefficient construction at the decoder end by optimizing the JPEG decoder.
* The neural JPEG is adapted to learn how to edit the discrete cosine transform (DCT) coefficients at both decoder and encoder ends.
* A learnable quantization table that is optimized jointly with the sparse recurrent encoder/decoder to improve rate-distortion performance, yielding an end-to-end, differentiable JPEG compression system.

## **3. Method**  

In this method, the authors construct a system that leverages an encoder and decoder
that are each driven by sparse recurrent neural networks (SMRNNs) trained within
the effective framework of neural iterative refinement while the recurrent decoder learns to reduce artifacts
in the reconstructed image.


### **JPEG algorithm & architecture**

Standard JPEG algorithm:

<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig1.png">

Standard JPEG architexture:

<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig2.png">

And standard JPEG architecture summazrized:

<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig3.png">

Modifications:
* Recurrent neural networks (RNN):
  1. Reduce channel dimension from 256 to 128
  2. Reshape channels to a 8x8x2 tensor for each block which will be split in two matrices of 8x8 afterwards.

* The sparse multiplicative RNN (SM-RNN) component will be used to process both Hl and Hc for K steps. So the activation map in the case of Hl would look like this:
  
  <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig4.png">
  
  The same algorithm can be used replacing Hc with Hl. A final set of sparse 'edit' values is conducted as followed:
  
  <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig5.png">
  
  Each discrete cosine transform (DCT) coefficient is multiplied by the original JPEG encoder with corresponding edit score as retrieved in previous mentioned process.

* Decoding method: utilize a SM-RNN similar to the encoder is utilized to process described above but tie its weights to those of the SM-RNN encoder module.

* Quantitzation table: After applying the inverse DCT transform to the outputs of the quantization table and rounding modules, the SM-RNN decoder takes this in, processes it K times and produces the reconstructed image. This overall proces is visualized as followed:
  
  <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig6.png">
  
  Here you can see that the authors use the differentiable JPEG pipeline to learn the quantization tables. They replace the attention mechanism with sparse RNN to better capture the importance of each representation associated with each channel. You can see that they use Q0 to the power of L and Q0 to power of C as optimization variables for luminance and chrominance. The range of quantization table values are clipped to [1; 255] which should help optimize the process.

* Rounding module: the authors removed the entropy encoding that is being used in JPEG and replaced the hard rounding operation with a differentiable third order approximation as shown below:

 <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig7.png">

### **Loss formulation**

* Disortion loss:

 <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig8.png">


* Rate loss:

 <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig9.png">


* Alignment loss:

 <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig10.png">





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
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig14.png"> |
|:--:| 
| *Figure 1, performance of various compression model with various bit rate setting.* |

Table:
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig15.png"> |
|:--:| 
| *Table 1, test results for Kodak (bpp 0.38), 8-bit compression benchmark (CB, bpp, 0.371).* |

Table:
| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/fig16.png"> |
|:--:| 
| *Table 2, test results for Kodak (bpp 0.37), 8-bit compression benchmark (CB, bpp, 0.341)* |

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
