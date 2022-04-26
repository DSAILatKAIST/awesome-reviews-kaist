---
Ioana Bica / Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders / ICML-2020
---

# **Title** 

[Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders](https://arxiv.org/abs/1902.00450)

## **1. Problem Definition**  

In the domain of medical studies, doctors or researchers need to know the time-varying individual treatment effect over time for patients. For example, if a patient's health condition gets better, the doctor needs to know which treatment they give to the patient causes better health condition. The challenge is that treatment effects may vary across time and we need to know the causal effect at different time steps in a time series of patient data.

There have been many related studies in this field. Rubin et al. and his followers develop methods such as propensity score matching, G-method, and inverse probability weighting to estimate the causal effect of treatment from time series [1] [2] [3].

However, in previous methods, people need to make a strong assumption that there are no hidden confounding variables. [Confounding variables](https://en.wikipedia.org/wiki/Confounding) denote variables that influence both the dependent variable and independent variable, causing a spurious association. This is almost impossible in reality since there are always variables that can not be measured. So the above methods are impractical. The problem is how to relax this assumption.

<img src=".gitbook/2022-spring-assets/panyu_1/Causal_structure.jpg" style="zoom:80%;" >

## **2. Motivation**  

There have been previous efforts to relax the assumption. Wang et al. observed that the dependencies in the assignment of multiple causes can be used to infer latent variables that render the causes independent and act as substitutes for the hidden confounders[4]. 

However, their method is focused on static settings which are not applicable to time series data. This paper aims to relax the no hidden confounding variable assumption which is applicable to time-varying causal effect estimation. This idea is first proposed in this paper to the knowledge of the author.


## **3. Method**  

This paper proposes a method called Time Series Deconfounder, a method that enables the unbiased estimation of treatment responses over time in the presence of hidden confounders, by taking advantage of the dependencies in the sequential assignment of multiple treatments.

The Time Series Deconfounder relies on building a factor model over time to obtain latent variables ***Z*** which, together with the observed variables render the assigned causes conditionally independent.

<img src=".gitbook/2022-spring-assets/panyu_1/Latent_variable.jpg" style="zoom:80%;" >

A denotes treatment. Z denotes latent variables. X denotes confounding variables.

<img src=".gitbook/2022-spring-assets/panyu_1/Factor_model_1.jpg" style="zoom:80%;" >

Graphical factor model. Each Zt is built as a function of the history, such that, with Xt, it renders the assigned causes conditionally independent. The variables can be connected to Y(a≥t) in any way.

The author uses RNN to learn the latent variables. The proposed framework is in the following graphs.

<img src=".gitbook/2022-spring-assets/panyu_1/RNN.jpg" style="zoom:80%;" >



## **4. Experiment**  

The author uses the MIMIC dataset to check the validity of the method. If the distribution of treatment assignment learned from the factor model is similar to the real treatment assignment, we can say that the Deconfounder works well which means p value is closer to 0.5.

The proposed multi-task RNN method outperforms MLPs and without multi-task model.

<img src=".gitbook/2022-spring-assets/panyu_1/result.jpg" style="zoom:80%;" >



## **5. Results & Conclusion**  

This method is robust enough in real-world settings. However, this method is only applicable to the multi-cause confounder settings. This method may not be applicable given single-cause confounders. In reality, if there are more and more treatment options, it is likely that the confounding variable will affect more than one single treatment.

---
## **Author Information**  

* Panyu ZHANG  
    * KAIST ICLab
    * Causal inference

## **6. Reference & Additional materials**  

[1]Robins, J. M., Rotnitzky, A., and Scharfstein, D. O. Sen- sitivity analysis for selection bias and unmeasured confounding in missing data and causal inference models. In Statistical models in epidemiology, the environment, and clinical trials, pp. 1–94. Springer, 2000b.

[2]Robins, J. M. Correcting for non-compliance in randomized trials using structural nested mean models. Communications in Statistics-Theory and methods, 23(8):2379–2412, 1994.

[3]Robins, J. M., and Hernan, M. A. Estimation of the causal effects of time-varying exposures. In Longitudinal data analysis, pp. 547–593. Chapman and Hall/CRC, 2008.

[4]Wang, Y. and Blei, D. M. The blessings of multiple causes. Journal of the American Statistical Association, (justaccepted):1–71, 2019a.
