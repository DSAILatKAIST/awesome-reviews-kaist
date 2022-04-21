---
description : Guansong Pang / Deep Anomaly Detection with Deviation Network / 25th 2019 ACM SIGKDD international conference on knowledge discovery & data mining  
---

# **Deep Anomaly Detection with Deviation Network** 

The paper that I will review this time is Pang, G. et. al의 Deep Anomaly Detection with Deviation Network. This paper was published in the 2019 SIGKDD. 

> Pang, G., Shen, C., & van den Hengel, A. (2019, July). Deep anomaly detection with deviation networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 353-362).


Please note that this review has been posted on [**personal blog post(1)**](https://yscho.tistory.com/97) [**personal blog post(2)**](https://yscho.tistory.com/98) and [**personal Youtube Review**](https://www.youtube.com/watch?v=1lEtPCn-lcY).

## **0. What is Anomaly Detection**  

**Anomaly Detection (AD) is the task of detecting samples and events which rarely appear or even do not exist in the available training data**

This literally means detecting samples that do not have any unusual, common characteristics that are different from the events that occur normally. Then you'll have this question.

<br>

"What is the difference from general classification?"

<br>

If you look closely at the above sentence, the phrase 'really appear or even do not exist in training data' is the key. This means detecting anomaly data that is rarely or even never present in the learning data. In general, a Supervisor-based classification model is based on having enough data for the class to classify. However, AD task differs from normal classification because there is very little data classified as anomaly, and the distribution between anomaly data cannot be guaranteed to be similar. 

Let's take a look at the figure below. 

<br>

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOAIob%2Fbtry4T85ixy%2F0gNRdGJJ4JLFaKR83upaK1%2Fimg.png" width="500"/></center>

#####  reference : Mohammadi, B., Fathy, M., & Sabokrou, M. (2021). Image/Video Deep anomaly detection: A survey. arXiv preprint arXiv:2103.01739

<br>

If you look at this figure, the data in the blue circle is normally normal data. We can plot these data using F1, F2 through feature presentation called F. In the case of green dots, you can usually see images of motorcycles as data. But in the case of a red car, it's a little bit different from the motorcycle that we've seen in general. In this case, if you do a representation through F, the data usually exists in a different location, and that 'gap' is the basis for measuring this data as abnormal. In this way, anomaly detection is usually performed.



<br>

Now, in order to do Anomaly Detection, it can be done on different data sets, and it can be categorized into different cases. It is largely divided into **'Supervised'**, **'Unsupervised'**, and **'Semi-supervised'** cases. 

<br>

And for the sake of simplicity, let's define each symbol as follows. 

U : Unlabeled data

N : Normal labeled data

A : Abnormal labeled data


U stands for unlabeled data, and N and A stand for labeled normal and abnormal data, respectively.
In this situation, the above three cases can be summarized as follows.

[1] Supervised Lerning ( N + A )

In the case of Supervisor Learning, if you have enough data, you will have the strongest accuracy of any of the three cases. Predictions in the presence of data are more accurate than those in the absence of data. However, the problem is that there is very little data. In fact, there is not much labeled data in the real world, and even if you have labeled data, the abnormal data is often extremely rare. In this case, supervised learning faces a problem of data unbalance. In addition, generalized prediction cannot be performed. In fact, anomaly data can exist in many other forms besides the case we observed, and the model makes predictions based on the distribution of anomaly data that we observed, so it lacks the ability to respond to unseen anomaly that we haven't seen in learning. As a result, there are usually limitations in using supervised learning in AD.

<br> 

[2] Unsupervised Learning ( U )

Therefore, learning is usually performed based on data that is not labeled. Because this is a little bit more like the real world situation. Unsupervised learning is the most similar to real world, based on the fact that actual labeled data is difficult to obtain and that absolute data rarely occurs. In addition, the generalizability, which is a limitation in the Supervised Learning above, is guaranteed, so it can be said to be a way to compensate for many shortcomings.

<br>
 
[3] Semi-supervised Learning ( N + A + U, N + A << U )

But is there no any downside to unsupervised learning? The biggest problem with Unsupervised is the lack of 'pre-knowledge' of anomaly data. If there's already little anomaly data, and there's no information about the characteristics, it's probably difficult for the model to fully capture the characteristics of the anomaly. Then, there is an unbalance problem in using labeled data, and there is a lack of prior knowledge in using unsupervised learning, so semi-supervised learning is intended to utilize both.

In this case, the purpose of this method is to enhance learning by using limited number of labeled data as prior knowledge

<br>

Based on this background, let's take a look at the contents of the paper

## **1. Problem Definition**  

Traditional methods for performing AD tasks (such as SVMs) face two limitations:

* high dimensionality
* highly non-linear feature relation

First, there was a problem with the curse of the dimension, and secondly, it was difficult to set up a complete model due to the non-linear relationship between features to detect anomaly. This problem was solved by the introduction of a neural network method based on the non-linear method. 

However, there were two major problems encountered after the neural network was applied. 

* **Scarcity of anomaly data**

_it is very difficult to obtain large-scale labeled data to train anomaly detectors due to the prohibitive cost of collecting such data in many anomaly detection application domains_

* **No similarity between anomaly data**

_anomalies often demonstrate different anomalous behaviors, and as a result, they are dissimilar to each other, which poses significant challenges to widely-used optimization objectives that generally assume the data objects within each class are similar to each other_

First of all, the number of labeled anomaly data is very small, and the cost of obtaining them is high.

In addition, although the model typically performs learning based on training data, there is a high possibility that there will be anomaly data with a different form of distribution than the anomaly data learned.

So, the modern deep learning-based AD model tried to solve this problem by applying the unsupervised method, not the supervised method. 

Specifically, **'Representation learning'** is used, and the following two-step approach is applied.


1. They first learn to represent data with new-representation

In other words, it is a step to learn how to extract key features that can express data well. 

2. They use the learned representations to define anomaly scores using reconstruction error or distance metrics space

And with the development of presentation learning, AD applies two concepts of metric, and the representative ones are 'Reconstruction Error' and 'Distance-based measures'.

ex) Intermediate Representation in AE, Latent Space in GAN, Distance metric space in DeepSVDD

## **2. Motivation**  

The author points out that the above two steps are carried out.

> _However, it most of these methods, the representation learning is **separate** from anomaly detection methods, so it may yield representations that are suboptimal or even irrelevant w.r.t specific anomaly detection methods._ 

> _In addition, when anomaly detection is performed based on unsupervised learning due to lack of prior-knowledge, data noise or uninteresting data may be recognized as anomaly data._

**For this second problem, we propose that we can use a limited number of labeled data to compensate for the lack of prior knowledge**

Therefore, as a way to overcome the problems with the above method, the author proposes an end-to-end learning method to learn 'Anomaly Scores'.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FM56LS%2FbtrzadrcSgk%2F5q7PD9FVlcGeocnN8PFYSk%2Fimg.png height="500"/></center>

<br>

Let's take a look at the picture above. (a) is a schematic diagram of how to perform anomaly detection based on the existing representation, and (b) is a schematic diagram of the method proposed in this paper. As you can see from the picture above, the existing method extracts features from the data and applies several metrics (reconstruction error, distance metric) to detect them. However, the author points out that this is an **indirect** optimization of the model. However, in the case of (b), when the data is entered, the anomaly score is derived **directly** from the end-to-end, and the detection is performed. This refers to a structure that allows the model to be directly optimized. 

Another nobelty of the method proposed in this paper is that we have defined a reference score to determine the degree of anomaly. In other words, it is different from the existing method in that it makes judgments based on the average anomaly score extracted from normal data and the deviated degree between the current input.

<br> 

Therefore, there are two ways to summarize the nobelties of this paper.

1. **_With the original data as inputs, we directly learn and output the anomaly scores rather than the feataure representations._**

2. **_Define the mean of anomaly scores of some normal data objects based on a prior probability to serve as a reference score for guiding the subsequent anomaly score learning._** 


## **3. Method**  

Let's take a look at the methodology.

### **End-to-End Anomaly Score Learning**

To address the lack of prior knowledge mentioned above, we use a dataset with a mix of unlabeled and limited labeled data.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpz3nl%2FbtryZHVg9s6%2FSW2PC0KyVud89mPvFQnu3k%2Fimg.png width="350"/></center>

N means the number of labeled data, K means very small labeled anomaly data.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcpGeJV%2Fbtry7SVewZt%2F881Rh4aPrNNFagSXzt7N50%2Fimg.png width="350"/></center>

So when we set up the data like this,

Our main purpose is to learn this ∅ function, which derives the anomaly score, and to make the anomaly scoring difference between the anomaly and normal.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3ERIN%2FbtrzaclALeh%2FZhkwGvlYAaQ7VrMNW8KcWK%2Fimg.png width="200"/></center>
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUbCcl%2Fbtry7vmlCJ9%2Fb982kWn8jGAKbzGDlyXJw0%2Fimg.png width="500"/></center>

So let's start with the macro framework.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBSZ9D%2Fbtry8PxBOB6%2FA8EWYWSv5qd28q89YThFGk%2Fimg.png width="400"/></center>

<br>

First of all, there is a network that derives an anomaly score called the 'Anomaly Scoring Network', and the part that generates a reference score, which shows that it has two architecture.

It can be seen that the network that derives the Anomaly score has two detailed structures, and once the input comes in, it consists of an 'Intermediate representation' layer that creates a representation and a layer that derives the anomaly score immediately. The specific structure of this will be covered in detail later.

And the reference score is R = {x1, x2, .. xl} and we're going to take l random samples from normal data, and the model works by using 
µ_r calculating their average to determine the degree of anomaly. 

So let's look at how it's implemented specifically. 

### **Deviation Network**

If summarize the method proposed by the author in one sentence, we can organize it into the below sentence.

> _"The proposed framework is instantiated into a method called **Deviation Networks (DevNet)**, which defines a **Gaussian prior** and a **Z Score-based deviation loss** to enable the direct optimization anomaly scores with an end-to-end neural anomaly score learner"_

Bold words form the core architecture. First, let's look at the Anomaly Scoring Network, which is the backborn of the first Deviation Network. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJGHpz%2FbtrzprQH0qZ%2FmpTQRo3I7WBjBfMRKKX9rK%2Fimg.png width="300"/></center>

The Anomaly Scoring Network can be represented as a ∅ function, which largely consists of a ψ network that creates Q, which is an intermediate presentation space, and a η network that represents an anomaly score. 

* Intermediate representation space (Q)
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdnn4IU%2FbtrzrK9BwBG%2FKpWM3ceCIbPKVKPOqKaOD0%2Fimg.png width="100"/></center>

* Total anomaly scoring network
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F9PoPw%2Fbtrzprwoj8U%2FtWKEGfRvtREZK9re3UqA7K%2Fimg.png width="150"/></center>

The two sub-networks that make up this ∅ network are:

[1] ψ is feature learner, which make Q ( feature learner )

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOsAcu%2FbtrzpV48e5d%2FMO5vdFH47lnoYDtwROkY4K%2Fimg.png width="150"/></center>

[2] η is extracted anomaly score from Q( anomaly score learner )

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F0bO5p%2Fbtrzn9cbQ0Q%2FYjMJxJhtU87hvKI0TVQgH0%2Fimg.png width="150"/></center>

And ψ network consists of H개의 hidden layer.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtprzG%2FbtrzpVKN0gs%2FMc0MkOb4BNKJt6QFKgw2M0%2Fimg.png width="150"/></center>

Therefore, the entire network can be expressed simply as follows:

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbaelvj%2FbtrzpsWq4zT%2FKgy1DJkq52dL7FBTyvvmc1%2Fimg.png width="150"/></center>

The hidden layers that make up the feature learner are configured differently depending on the incoming input and the task we want to perform. For example, if image data needs to be feature presentation, it uses a CNN network, and for sequence data, it uses an RNN network.

In the case of η network, the network was configured to calculate the score using simple linear unit.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoehTT%2FbtrzpVKOfiu%2F3XPHaenP2jtYAkYiI1KarK%2Fimg.png width="400"/></center>

As a result, we can configure the entire anomaly scoring network as follows:

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcmhY8U%2FbtrzrbGozM2%2F4i4haskM90WJez4gJKYtlK%2Fimg.png width="250"/></center>

The author insists that we can directly maps data inputs to scalar anomaly scores and can be trained in an end-to-end fashion.

<br>

In addition, the author suggests a way to build a reference score that can be used to refer to whether or not. 

Basically, this reference score is randomly picked from R, which is normal objects, and used for optimization based on that score.

There are two ways to do this:

1. Data-driven approach

2. Prior-driven approach

First of all, the Data-driven method is a method of deriving and utilizing the average calculated µ_r by deriving the anonymous score based on the learning data X. However, this method has many limitations because it has a feature that changes little by little each time this µ_r changes the X value.

So in this paper **we adopt a method of calculating µ_r based on reference scores extracted from the prior probability F**. For that reason, the author explains two reasons.

1. The chosen prior allows us to achieve good interpretability of the predicted anomaly scores

2. It can generate µ_r constantly, which is substantially more efficient than the data-driven approach

First, it is said that the following methods have good interpretability when predicting anonymous scores. It also has the advantage of constant fixed µ_r.

But we are not sure. Because we don't have a clue what the prior probability F is. 

In this paper, we borrow this prior probability F from the Gaussian distribution. 

Does the Gaussian distribution really represent the anomaly score of normal data?

The author argues that: 

> _Gaussian distribution fits the anomaly scores very well in a range of data sets. This may be due to that the most general distribution for fitting values derived from Gaussian or non-Gaussian variables is the Gaussian distribution according to the central limit theorem._

In other words, in AD task, most of the data is normal. And then the anonymous scores for these normal data will follow any distribution. These scores will also follow the Gaussian distribution if we get enough samples. It's because of the **Central Limit Theorem**.

We can easily understand the following illustration.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBTIGr%2FbtrzkpM7bbF%2FMM2ppNm1rXkgD1zZesLmkk%2Fimg.png width="400"/></center>

That is, the more normal you get, the closer you get to µ_r, but the more normal you get, the more distant you get from µ_r. This degree of deviation is used to define the loss function.

Therefore, the data that can be used as a reference score are derived as follows.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdIpinw%2FbtrzrM0GfIk%2FI2IGMisxbwgx833klrwmh1%2Fimg.png width="400"/></center>

Each ri is derived from a normal distribution, and that ri means the anomaly score of a random normal data object. In this study, we set µ_r to 0 and σ_r to 1, and state that the random sample is available in any amount sufficient to satisfy the CLT. The author used 5,000 samples.

So let's take a look at how to define loss functions specifically based on these reference scores.  In this paper, we use the Z-score method to express the indicators that measure the degree of deviation. The loss below is named by the author as contrastive loss.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdTpvEM%2FbtrzraOizYQ%2FUmqJrNaprfQTmBBtCwlfyK%2Fimg.png width="400"/></center>

#####  <center>Contrastive Loss</center>

<br>

For example, if it's normal data, that ∅ value is approximated to µ_r, and if it's anomaly data, it's not.

And based on that contrast loss, the final deviation loss is defined.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbTgy9y%2Fbtrzqz1QfLG%2FIUp2skbkoYxINpU7MFc4N0%2Fimg.png width="550"/></center>

#####  <center>Deviation Loss</center>

<br>

If x is anomaly, y = 1, and if x is normal, y = 0.

And in the case of 'a' above, it becomes the confidence interval parameter of Z-score.

<br>

What does this mean?

Let's look at the term once again specifically.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb0i88Z%2Fbtrzn8En8Rx%2FBjaTc6qznSKzfgH0wU9GRk%2Fimg.png width="200"/></center>

> _Note that if x is an anomaly and it has a negative dev(x), the loss is particularly large, which encourages large positive derivations for all anomalies._ 

If x is the anomaly and deviation is negative, the total loss value will increase. So the model tries to make the deviation of this anomaly data have a large positive value. Almost to the point where it approximates the value of a.

The meaning of this word can be easily understood in the next picture.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbjVLGR%2FbtrznzoamSa%2FHasjnvVlfhx0oEwQIhnPv0%2Fimg.png width="400"/></center>

That is, if the 'dev' value comes near 0, it is normal data, but given a=5, you can apply it as an approximation of anomaly to that place. In this paper, we used a=5.

The original text states exactly as follows: 
> _Therefore, the deviation loss is equivalent to enforcing a statistically significant deviation of the anomaly score of all anomalies from that of normal objects in the upper tail. We use a = 5 to achieve a very high significane level for all labeled anomalies._

But I'm sure there are people here who put a question mark. You may be wondering why we are mentioning normal data when we don't know which normal labeled data except for limited animals. In fact, it solves this problem as follows.

> _We address this problem by simply treating the unlabeled training data objects in U as normal objects._

So it's just considered normal data even though the unlabeled data is not guaranteed to be all normal (the actual unlabeled data contains abnormal, it is expressed as contaminated). 

It's very strange...! Why would do that?

In many semi-supervised learning, the model is fitting by considering all unlabeled data as normal. In fact, there are two reasons for that, because first it's a very similar situation to the real world. In fact, we have a lot of unlabeled data. But as we know, anomaly situations happen very scarce. In other words, most of the data we have is based on the premise that it's normal. Therefore, it is a measure that considers this real world situation as it is. It is also assumed that this very small amount of anomaly data is not very influential for SGD-based optimization when performing actual backpropagation. That means it won't affect the performance of the model that much. Therefore, semi-supervised learning uses the unlabeled data as normal, almost rule of thumb (and of course, this should be studied more). 

Therefore, the DevNet algorithm can be organized into pseudo-code as follows: 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmYXPN%2Fbtrzra1PoFF%2F2TCqBjTtEebqG6s0NrdSB0%2Fimg.png width="600"/></center>

Now that we're ready to design and train the loss function, the next thing we need to check is Interpretability, which means when do we decide it's normal or abnormal? 

This study uses the following Propositions:

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb6pwta%2FbtrzmjzHdLP%2F89HskRtyfRlfu2ycz2dKr0%2Fimg.png width="500"/></center>

<br>

Let's think about what this means.

A typical standard normal distribution has the following properties:

If there is a normal distribution where µ_r is 0 and σ_r is 1, and p=0.95, then z(0.95)=1.96.

Based on µ_r = 0, the interval ( µ_r - z to µ_r + z) is eventually the confidence interval.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FepoY3i%2Fbtrzprcerkk%2FmEGSMnkea8IeIzR42ambp0%2Fimg.png width="500"/></center>

<br>

But what if the newly received anomaly score is mapped across this boundary? The implications can be thought of as follows.

> _The object only has a probability of 0.05 generated from the same machanism as the normal data objects._

This means that the probability of being normal is very low.

The reason why set the threshold for judging anomaly based on this form is because

> _This proposition of DevNet is due to the Gaussian prior and Z-Score-based deviation loss._

This is because we defined a Z-score-based deviation loss.

This is how learning is done.

## **4. Experiment**  

This paper uses nine **real-world datasets**. Specifically, use the following datasets:

- Fraud Detection ( fraudulent credit card transaction )

- Malicious URLs in URL

- The thyroid disease detection

- ...

For more information, please refer to the link below that the author added to Appendix. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fs5wZ9%2FbtrzMJXGKyh%2F6X5vfPXude8Vjjdll2tnF1%2Fimg.png width="500"/></center>

<br>

The models used as comparison methods are as follows.


[1] REPEN
<br> 

[2] adaptive DeepSVDD 
<br> 

[3] Prototypical Network (FSNet) 
<br> 

[4] iForest
<br>

In the case of REPEN, it is a neural network based on limited labeled data, and in the case of FSNet, it is a network that performs few show classification. Both networks are characterized by the use of limited labeled data. Same conditions as DevNet. On the other hand, the ensemble-based model iForest, which performs AD using the Unsupervised method, was also used as a comparison group.

DeepSVDD is an algorithm that performs a very famous AD, where the author does modifying to make DeepSVDD comparable to DevNet. It's making semi-supervised learning into a form that you can do.

> _We modified DSVDD to fully leverage the labeled anomalies by adding an additional term into its objective function to guarantee a large margin between normal objects and anomalies in the new space while minimizing the c-based hypershere's volume._

In other words, author mentioned that adjusted the loss function a little bit in the form of using labeled animals. And actually, the performance of the adjusted one is better than original SVDD as a result of these modifications.

As some of you may know, the semi-supervised learning version of DeepSVDD has a method called DeepSAD written by the same author in 2019-2020. However, when DevNet came out, it was still before DeepSAD came out, so I think the DevNet author applied this heuristic at the time.

A total of four comparison groups were used to compare performance. 

<br>

How about 'metric'? 

Typically, the metrics used by AD use AUROC and AUC-PR.

Let's briefly summarize these two.

<br>

* AUROC( Area Under Receiver Operating Characteristics )

This is the ROC curve that many of you know. To understand this ROC curve, you first need to understand the Confusion Matrix. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbfou5g%2FbtrzL9WAdnq%2F3PQIBWSzyR8qx6DfeSihsk%2Fimg.png width="500"/></center>

<br>

Actual is literally the actual label value and pred is the result of prediction. In the following situations, we can define True Positive, True Negative, False Positive, and False Negative and use these numbers to summarize the following characteristics.

<br>

Sensitivity ( True Positive Rate ) = ( True Positive ) / ( True Positive + False Negative )

Specificity ( True Negative Rate ) = ( True Negative ) / ( False Positive + True Negative )

False Positive Rate = ( False Positive ) / ( False Positive + True Negative ) 

Precision = ( True Positive ) / ( True Positive + False Positive ) 

Recall = ( True Positive ) / ( True Positive + False Negative ) 

Accuracy = ( True Positive + True Negative ) / ( True Positive + True Negative + False Negative + False Positive )

<br>

Among these concepts, True Positive Rate (TPR) and False Positive Rate (FPR) allows you to draw the following curves:

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc19T4r%2FbtrzQidNtZp%2FddKzsh1Hj4svLdey9vrWAK%2Fimg.png width="500"/></center>

<br>

If a line is formed like a red dotted line, it is the same as a random classifier. And the area below the red dot is 0.5. For the purple line, on the other hand, if the model is the strongest, that is, if all cases are correct, the area below is 1.

AUROC measures performance based on the size of this area.

* AUC-PR

However, this approach has its limitations. This is because the proportion of errors in the minor class is set low.

Let's look at an example for understanding.

For example, let's say you see data with 30,000 normal data and 100 abnormal data.

Let's say you got 50 wrong in normal. In normal, there are relatively fewer 50 mistakes. Whereas in Abnormal, let's say 50 are wrong. Then, it's not a small number because one-half of the total is wrong. In other words, the error in normal and the error in abnormal should not be considered the same.

It is the combination of Precision and Recall that reflects this.

For a clearer understanding of these two, let's take the following example. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd9MBAs%2FbtrzKKbmH9S%2FJeTMJZpQVte7xfDeMBgPH1%2Fimg.png width="500"/></center>

<br>

Take the task of performing Object Detection, for example. Let's say that the model detects only a one person in a picture with two people and predicts that it is a person. From a Precision point of view, I think all the cases I tried to make predictions were right, so I say that I have 100% accuracy. But the Recall view is a little different. There's actually one more person, but this person didn't get it right, so we're saying that the accuracy is 50%.

If we combine these two different perspectives and create a metric, we can overcome these problems, which means that we can give more appropriate weights to minor classes. And that's where AUC-PR comes in.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F75oHD%2FbtrzNATLZdb%2FFOBpQBYbZ7kBnVnBDTktDk%2Fimg.png width="500"/></center>

<br>

So let's get back to the subject, and let's go into the specifics of the experiment.

First of all, the experimental environment setting is as follows.

One hidden layer was used for the depth of the neural network, and the specific parameter settings were set as follows

- 20 neural units

- RMSProp Optimizer

- 20 mini batch

- ReLU

- L2 Norm Regularize

The network also used Multilayer perceptron, considering that most of the data is unordered multidimensional data. 

Based on these settings, the performance comparison between models is shown below.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqsUaH%2FbtrzGhgYfLN%2FKAIoj79Hv0GgtCV5qrDyj0%2Fimg.png width="700"/></center>

<br>

Except for the census data, both AUROC and AUC-PR showed the best performance of DevNet. 

In this figure, the meaning of the notation from 'Data Characteristic' is as follows.

* '# obj' -> Number of observation

* 'D' -> Data dimension

* 'f1' -> Percentage of anomaly data in training data

* 'f2' -> Percentage of anomaly data in total data


<br> 

The author validates DevNet's efficiency through several specific experiments.

<br>

### [1] Data Efficiency

First, the author tried to solve the following questions.

> * _How data efficient are the DevNet and other deep methods?_
> * _How much improvement can the deep methods gain from the labeled anomalies compared to the unsupervisd iForest?_

**In other words, we wanted to measure how well the model utilizes the anomaly with the addition of labels, and how well we utilize the prior knowledge.**

In the case of iForest, which was used as the base line, there is no label, so there is no difference in the performance of the model, whether given label or not.

Let's look at the next picture.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Ft0saz%2FbtrzKQwytqF%2FzL4211l0OALkcWfJKrKdCK%2Fimg.png width="600"/></center>

<br>

We can see that DevNet shows the highest performance for the number of additional animals. In particular, it was confirmed that there was a significant improvement in performance for the additional anomaly labels for campaign, census, news20, and thyroid data.


### [2] Robustness w.r.t. Anomaly Contamination

The second is how strong a response is to anomaly contamination. 

As I mentioned earlier, what exactly is 'Contamination'? Let's refer to the next sentence.

> _To confuse data by sampling anomaly and adding it to the unreliable training data or removing some anomaly_

In other words, when unlabeled data is considered normal, it represents the ratio of anomaly to the unlabeled data. It's a measure of how sensitive the model is as increase this amount.

The author measured the change in performance by controlling 0-20%.

Let's look at the next figure.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZ21uH%2FbtrzQoSEvNJ%2FPPtgkXAIMtrm5V1DQjsgJ1%2Fimg.png width="600"/></center>

<br>

DevNet generally maintains robust performance against attacks.

However, the question was that DevNet showed a dramatic decrease in the 'news20' dataset. For news20 consisting of text data, we found that there was a relatively larger reduction for contamination.

### [3] Ablation Study

Third, perform ablation study.

The author verified that the various components used in DevNet's architecture, such as {intermediate presentation, FC layer, and one hidden layer}, actually play their respective roles. we wanted to make sure that all of those elements were necessary.

So we're going to create three comparison groups. A simpler schematic is as follows.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcqCafv%2FbtrzKugeIL7%2FQQDGN7Uum8jv33nqwIOz40%2Fimg.png width="800"/></center>

<br>

In other words, if the existing 'Def' is DevNet, we will either remove each component one by one or increase the layer to three. 

For 'DevNet-Rep', we removed the FC layer that derives the anomaly score in scalar form at the end. So we're going to derive performance based on a vector with 20 dimensions. (It's not clearly stated here how the anomaly score was derived. ) For 'DevNet-Linear', remove the network that performs feature presentation and derive the anomaly score through the linear function. Finally, 'DevNet-3HL' consists of three hidden layers with 1000-250-20 ReLUs instead of one layer with 20 ReLUs.

The performance results are as follows:

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBo4XO%2FbtrzLILcDiC%2FIJt3BFq2HKebK9jV7qubkK%2Fimg.png width="600"/></center>

<br>

For most datasets, AUROC was performed best by DevNet. However, in the case of AUC-PR, there were some cases where Rep or 3HL were better. Especially for the census data, we also found that Rep had the best performance.

However, in general, Def's performance was the best, so we explain that each element that builds end-to-end learning has an appropriate contribution.

It also explains why applying deeper neural networks like 3HL doesn't work better, and for this reason, it's easy to miss the feature if you build very deep neural networks with very few labeled anomalies. In other words, in the context of most normal data, deep neural networks lose the characteristics of anomalies. Therefore, the author explains that one-hidden layer is the most fit.

### [4] Scalability

Finally, we conducted an additional experiment to check the performance time according to the size and dimension of the data, that is, the complexity. The size was fixed and the execution time according to the change of dimension was measured, and the execution time according to the change of size was measured with the dimension fixed. The results are shown in the following figure.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fctk3aV%2FbtrzMJKjFsF%2F0r6c9G6kpyhPhYz7KD3PS0%2Fimg.png width="600"/></center>

<br>

Although linear time has been consumed in most models, DevNet has been shown to be consistent with data size in areas over 100,000 in size. We also found that dimension is slower than FSNet, but faster than other models.

# **5.Conclusion** 

Therefore, the contribution of this paper can be summarized into three categories:


> ### 1. _This paper introduces a novel framework and its instantiation DevNet for leveraging a few labeled anomalies with a prior to fulfill and end-to-end differentiable learning of anomaly scores_

To overcome the limitations of most unsupervised learning, we used a small amount of limited anonymous data

> ### 2. _By a direct optimization of anomaly scores, DevNet can be trained much more data-efficiency, and performs significantly better in terms of both AUROC and AUC-PR compared other two-step deep anomaly detectors that focus on optimizing feature representations_

Since the existing feature representation and the anomaly detecting step are divided into two-steps, we propose a method of optimizing them directly rather than indirectly optimizing them.

> ### 3. _Deep anomaly detectors can be well trained by randomly sampling negative examples from the anomaly contaminated unlabeled data and positive examples from the small labeled anomaly set._

Another aspect was that the attempt to measure the degree of anomaly using a reference score based on normal distribution was also a unique approach.

# **6.Code Review** 

As for the code review, I don't think it's meaningful to organize it by posting, so I leave the coordinates of the YouTube video I recorded for those who want to refer to it additionally. Please understand that I did not summarize it in this posting...!

https://www.youtube.com/watch?v=1lEtPCn-lcY 

Code Review : 55:30 ~ end

Thank you for reading this long article.

---  
## **Author Information**  

* Guansong Pang (Prior Author)
    *  Assistant Professor of Computer Science at School of Computing and Information Systems, Singapore Management University (SMU)
    * Abnormality and rarity learning
    * Outlier detection and feature selection

## **Reference & Additional materials**  

### Github code

[1] This paper gives code from github, made by 'keras'

github code : 
https://github.com/GuansongPang/deviation-network

[2] And also, there are some more improved version of DevNet (same author), you can refer below paper and github link, which made by 'Pytorch'

[Pang, G., Ding, C., Shen, C., & Hengel, A. V. D. (2021). Explainable Deep Few-shot Anomaly Detection with Deviation Networks. arXiv preprint arXiv:2108.00462.](https://arxiv.org/abs/2108.00462)

github code : 
https://arxiv.org/abs/2108.00462
(this is what I reviewed in youtube)

<br>

### Other materials

And I referred other materials to make this posting

[3] Mohammadi, B., M., & Sabokrou, M. (2021). Image/Video Deep anomaly detection: A survey. arXiv preprint arXiv:2103.01739

[4] Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E., Müller, K. R., & Kloft, M. (2019). Deep semi-supervised anomaly detection. arXiv preprint arXiv:1906.02694.

[5] Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., ... & Kloft, M. (2018, July). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.

[6] Shi, P., Li, G., Yuan, Y., & Kuang, L. (2019). Outlier Detection Using Improved Support Vector Data Description in Wireless Sensor Networks. Sensors, 19(21), 4712.


