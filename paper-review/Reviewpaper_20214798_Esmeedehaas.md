| Description |
| --- |
| Emilien Dupont / COIN: COmpression with Implicit Neural Networks / 2021(description) |

# **Review paper COIN** 

COIN: COmpression with Implicit Neural Networks


## **1. Problem Definition**  

Current compression methods are still lacking some important improvements. These current methods still have quite a high distortion rate per bit rate. This is not effective if you want to have efficient compression for images. The problem with current methods for image compression like JPEG is that even though it seems to work fine, this is still a very basic way of compression. In the future the preference would go to image compression methods that give better quality, encode faster, and that is more simple to use.


## **2. Motivation**  

In this paper, the authors want to improve the existing image compression methods. They want to make a difference by introducing a new type of method for the compression of an image. This method could help improve the current methods and make them easier and more approachable. The authors hope to contribute to a more novel class for neural data compression methods. It might help improve the current methods and solve the problems like lack of quality, slow encoding, and compression that does not need entropy coding.

### **Existing works**

* Previous works on image neural representations already proposed methods using mapping pixel locations (MLP), but these works often use relatively small numbers of parameters. In this work, the researchers want to change this by making specific decisions on the architecture of the MLP and by quantizing the weights of the model. This will help fit the images more efficiently compared to storing RGB values.

* Most commonly, neural data compression methods are based on hierarchical variational autoencoders, to later be discretized for entropy coding. But the quantization of the latent variables creates a gap in the discretization. This gap is also a problem but in this work, the researchers propose a new method in which they can use MLPs to overfit the image and use the weights as a compressed description of an image to transmit it. This would make the entropy encoding unnecessary and would prevent a big gap in the discretization.

* Data and model compression have very similar problems, and so the researchers cast the date compression issue into a model compression problem to have a broader set of resources available to solve this kind of problem.


 ### **Significance**
Most of the current works have neural image compression that often operates in an autoencoder setup. The sender then needs an encoder to map data that is input to a discretized latent code, and according to a learned latent distribution, this is then entropy coded into a bitstream. The receiver will then get the transmitted bitstream that is decoded into a latent code that then has to be passed through the decoder to make the reconstruction of the initial image.


## **3. Method**  

In this method, they store the weights of the NN's instead of the traditional methods in which the RGB values are stored for each pixel of an image. The researchers do this by encoding an image by overfitting mapping pixel locations (MLP) over the RGB values of the image and transmitting the weights θ of the MLP as code. The encoding process starts with overfitting an MLP to the image, after that the weights that have to be transmitted have to be quantized. When it is transmitted, the decoding process starts and the MLP will be evaluated at the pixel locations to reconstruct the image. See figure 1 below. 

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding1.png"> |
|:--:| 
| *Figure 1, proposed method.* |

This method does not need entropy coding or learning a distribution over weights so it is very likely to outperform compression methods like JPEG. 

### **Encoding method**
In the paper the following optimization problem is applied:

<img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding6.png">

In which I the image is that has to be encoded, with I[x, y] returning RGB values at pixel location (x, y). The function f0 : R^2 → R^3 with parameters 0 MLP to RGB values in the image, fθ(x, y) = (r, g, b). Then the image can be encoded by overfitting f0 to the image under some distortion measure.

However in this case there are three important things to consider:
* The choice of f0 is very important. Parameterizing f0 by an MLP with standard activation functions will result in underfitting (also with large parameters). To tackle this problem, the researchers tackled this problem by using sine activation functions.
* It is also important that the equation as shown in figure xx is minimized because of the large MLP. But, since the parameters 0 of the MLP are stored as the compressed description of the image, restricting the number of weights should improve the compression rate. So in this case, it is best to fit f0 to I using the fewest parameters as is possible.
* And lastly, it is also important to reduce the model size. The researchers do this by performing a hyperparameter sweep over the width and number of layers of the MLP while quantizing the weights from 32-bit to 16-bit precision. This should be sufficient to outperform the JPEG standard for low bit rates.

### **Decoding method**
The decoding process consists in evaluating the function fθ at every pixel location to reconstruct the image. This should make the decoding process more flexible. The image can progressively be decoded by decoding parts of the image/low-resolution parts of the image first and evaluating the previously mentioned function at various pixel locations. With other methods, partially decoding is considered hard, but in this case, it can easily be done. simply by evaluating the function at various pixel locations.


## **4. Experiment**  
The overall experiment results show that COIN outperforms other methods in rate-distortion plots on the Kodak dataset. Results also show that COIN has a smaller model size per bit-rate (bpp).

### **Experiment setup**  
* For this experiment, the researchers chose to use the Kodak image dataset consisting of 24 images of size 768 x 512. The researchers do not mention any specific reason for choosing this dataset.
* The researchers compare their model against 3 autoencoder-based neural compression baselines which they later refer to in their graphs as BMS, MBT, and CST. The model is also compared with JPEG, JPEG2000, BPG, and VTM image codecs. The researchers used the CompressAI library and the pre-trained model this library contains to benchmark their model. The model was implemented in PyTorch and all experiments were performed on a RTX2080Ti GPU.
* The model was evaluated by using rate-distortion plots. The best model architectures for a given parameter budget (which were measured in bits per pixel or bpp) were determined by firstly finding valid combinations of depth and width for the MLPs that represent the image. After that, the best architecture is determined by running a hyperparameter search over learning rates and valid architectures on a single image by using Bayesian optimization. The resulting model is trained on each image in the dataset at 32-bit precision and converted to 16-bit precision after training. The researchers also found that while doing this the decrease in the weights' precision almost gave no distortion increase. But any case of 8-bit or lower gave a significant amount of distortion, making halving the bpp no longer beneficial.

Note that in this experiment:
* This experiment only requires the weights of a (very small) MLP on the decoder side, leading to memory requirements that are a lot smaller.
* COIN outperforms JPEG after 15k iterations and can keep being improved even more. See figure 2.

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding4.png"> |
|:--:| 
| *Figure 2, model training on the 15th image in the Kodak dataset.* |

### **Result**  

The results of the previously mentioned procedure for various bpp levels are shown in figure xx. As shown in the image, at low bit rates the model improves upon JPEG. While this model is still not close to the quality of the newest compression methods, the performance and simplicity of this specific approach could be very promising for future works. See figures 3 & 4 for the results.

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding2.png"> <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding3.png"> |
|:--:| 
| *Figure 3, rate-distortion plots on the Kodak dataset & figure 4, model sizes at 0.3bpp.* |


## **5. Conclusion**  

COIN is proposed as a new method for compressing images by fitting neural networks to pixels and storing the weights of the resulting models. Experiments showed that this simple approach can outperform JPEG at low bit rates, and this while not needing entropy coding.

See this example below to see the performance of COIN.

| <img src="/.gitbook/2022-spring-assets/Esmeedehaas1/Afbeelding5.png"> |
|:--:| 
| *Figure 5, example of COIN compared to JPEG.* |


### **Summary** 

In the table below you can find a summary of all the pros and cons of this newly proposed method gives.

| Pros | Cons |
| --- | --- |
| This neural data compression algorithm does not require a decoder at test time and so doesn’t take much memory | The encoding process is slow |
| COIN can partially decode, while other methods cannot do this | The methods used for decreasing model size are not sophisticated |
| COIN outperforms JPEG after 15k iterations and continues improving beyond that | When decoding, the network at every pixel location has to be evaluated to decode the full image |
| An interesting method to apply to different types of data | The method performs worse than the state of the art compression methods |

Ideas on pros & cons:
* Maybe with the help of meta-learning, the encoding process could be sped up.
*	The decoding process could be improved by embarrassingly parallelizing it to the point of a single forward pass for all pixels.


### **Future ideas** 

* To reduce the model size, the researchers use specific methods (architecture search and weight quantization) to outperform the JPEG standard for low bit rates. But this method is not that nice, in future works more sophisticated approaches should be found to further improve results.
*	Recent work in generative modeling of implicit representations suggests that learning a distribution over the function weights could translate to significant compression gains for the approach. In addition, exploring meta-learning or other amortization approaches for faster encoding could be an important direction for future work.
*	Refining the architectures of the functions representing the images (through neural architecture search or pruning for example) is another promising avenue. While in this paper they converted weights to half-precision, large gains in performance could likely be made by using more advanced model compression.


---  
## **Author Information**  

* Esmée Henrieke Anne de Haas (20214798)
    * Master student researcher ARM lab KAIST  
    * Research topics: Human Computer Interaction, Metaverse, Augmented Reality, Dark Patterns

## **6. Reference & Additional materials**  

* Reference of paper: Dupont, E., Goliński, A., Alizadeh, M., Teh, Y. W., & Doucet, A. (2021). Coin: Compression with implicit neural representations. arXiv preprint arXiv:2103.03123.
* Link to public materials: https://github.com/EmilienDupont/coin  


