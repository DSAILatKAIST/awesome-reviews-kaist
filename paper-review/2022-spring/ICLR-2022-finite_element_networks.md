---
description : Marten Lienen and Stephan Günnemann / Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks / ICLR 2022
---

# Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks

We will present a blog post on [_"Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks"_](https://arxiv.org/abs/2203.08852) from Marten Lienen and Stephan Günnemann **[NEEDREF]**, which has been accepted as a Spotlight presentation in [ICLR 2022](https://iclr.cc/). 

## **1. Problem Definition**  

We will firstly introduce the problem in a brief, yet somewhat lengthy, necessary background about differential equations and the finite element method that constitutes a backbone of the paper.

### Modeling Complex Systems with Differential Equations
Differential Equations are regarded by many as the _language of nature_. Many complex systems can be modeled by describing each single variable as a relation with others in both _space_ and _time_: Partial Differential Equations (PDEs) describe such processes. A quite general formulation can be written as following:

$$
\partial_t u  = F (t, x, u, \partial_x u, \partial_{x^2} u, \dots)
$$

where $u$ is a solution of the equation and $F$ are the _dynamics_ which can be a function of time, space, $u$ itself and its derivatives. PDEs are generally either very expensive to compute if not intractable altogether. For these reason, multiple algorithms have been developed over the centuries to try and solve this extremely complex endeavor. In particular, computers are very capable of handling _discretized_ data, in the form of digital bits instead of their continuous, analog counterparts. Can we apply some algorithm which is well suitable to computers?

### The Finite Element Method

The Finite Element Method (FEM) is a way to _divide and conquer_ the realm of PDEs.

<p align="center">
  <img src=".gitbook/../../../.gitbook/2022-spring-assets/FedericoBerto/FEN/fem-example.png" width = 60% alt="Image">

<figcaption align = "center">Figure 1.<i> An example of Finite Element Method (FEM) applied to a magnetical shield. </i></figcaption>
</p>

In particular, the domain with set of points $\mathcal{X}$ is divided into a set of simplices (i.e., $ n $ -dimensional triangles) which is called _triangulation_. Triangulations, such as the Delaunay triangulation, are also referred to as _meshes_ and are frequently used in many other areas such as movie CGI, gaming and most 3D graphics. This can be seen on the left of Figure 1. Then, operations are performed on this discretized domain to obtain a solution, as shown on the right of Figure 1.

#### Basis Functions

In general, the solution $u$ would lie in an infinite-dimensional space $\mathcal{U}$. What if, however, we cannot have infinite dimensions? Then, we need to approximate $u$ with a finite-dimensional subspace $\mathcal{\tilde{U}}$. To do so we employ _basis functions_ $\varphi$, which map points from $\mathcal{U}$ to $\mathcal{\tilde{U}}$. The simplest choice, which the authors use, is the P1 piecewise linear functions which map

$$  
\varphi^{(j)} (x^{(i)}) = \begin{cases}
  1 & \text{if }x^{(i)} = x^{(j)}\\
  0 & \text{otherwise}
  \end{cases} \quad \forall x^{(i)} \in \mathcal{X}.
$$
that is basically to simply map each point in $\mathcal{U}$ to the same values in $\mathcal{\tilde{U}}$ as in Figure 2 left.

<p align="center">
  <img src=".gitbook/../../../.gitbook/2022-spring-assets/FedericoBerto/FEN/basis-function-choice.png" width = 90% alt="Image">

<figcaption align = "center">Figure 2.<i> Solving a PDE with the Galerkin method and method of lines consists of three steps. </i></figcaption>
</p>

#### Galerkin Method

The piecewise linear approximation above is not differentiable everywhere. We can constrain the residual, i.e. the difference between $\partial_t u$ and $F$ to be orthogonal to the approximation space:

$$
\langle R(u), \varphi^{(i)} \rangle = 0 \quad \forall i \in 1, \dots, N
$$

In simpler terms, we are asking for the _best possible_ approximation. Given this, we can now reconstruct the equation as following

$$
\langle \partial_t u, \varphi^{(i)}\rangle =  \langle F (t, x, u, \partial_x u, \partial_{x^2} u, \dots), \varphi^{(i)}, \forall i \in 1, \dots, N
$$

By stacking the equations above we obtain the following linear system
$$
A \partial_t c = m
$$

where $A$ is the so-called mass matrix, $c$ is the vector of basis coefficients of $u$ and $m$ captures the effects of dynamics $F$.

#### Method of Lines

If we can evaluate the right hand side $m$, then the equations is easily solvable with time derivatives. In particular, we can consider a _stacked_ version of multiple scalar fields instead of vector ones as

$$
A \partial_t C = M
$$

where $C$ and $M$ are $m$-dimensional matrices due to $m$-scalar fields. In practice, we have transformed a PDE into a matrix ODE (ordinary differential equation) by discretizing in space; we managed to obtain a much simpler way of solving our problem by only needing to _integrate_ over time: a much simpler task!


## **2. Motivation**  

PDEs are the _language of nature_ and as such they are incredibly important for the scientific community. However, many hand-crafted models either take too long to compute solutions or do not have enough expressibility. Therefore, it is necessary to include at least partial, _data-driven_ terms that can learn from past experiences.

Machine and Deep Learning have proven incredibly powerful tools for solving real-world complex phenomena: they can accelerate simulations by orders of magnitude enabling faster predictions, design and control and even describe previously unknown dynamics which cannot be derived by equations.

There are mainly two lines of research in the area of PDEs and Deep Learning: either constraining PDE solution learning with a cost function, or learning directly from data to obtain a simulator via inductive biases.

In this work, the authors follow the second path and derive a model which sprouts from research on numerical methods for differential equations and can incorporate knowledge of dynamics (such as transport terms).

## **3. Method**  



## **4. Experiment**

The authors experiment on three datasets - one synthetic, and two real. We fi

### Cylinder Flow
The following dataset consists of simulated flow fields around a cylinder as collected by **[NEEDREF]**.

<p align="center">
  <img src=".gitbook/../../../.gitbook/2022-spring-assets/FedericoBerto/FEN/cylinder-flow.png" width = 80% alt="Image">

<figcaption align = "center">Figure .<i> CylinderFlow snapshot. </i></figcaption>
</p>

### Black Sea 
This dataset is composed data on daily mean sea surface temperature and water velocities on the Black Sea over several years. The training data is made of frames from 2012 to 2017, validation is on frames from 2018 and testing is done with frames from the year 2019. The time resolution $\Delta t$ is of 1 day. 

<p align="center">
  <img src=".gitbook/../../../.gitbook/2022-spring-assets/FedericoBerto/FEN/black-sea.png" width = 80% alt="Image">

<figcaption align = "center">Figure .<i> Learned flow fields of water velocities on the Black Sea dataset: T-FEN recognized the relationships between features. </i></figcaption>
</p>

### ScalarFlow
The **[NEEDREF]**


<p align="center">
  <img src=".gitbook/../../../.gitbook/2022-spring-assets/FedericoBerto/FEN/scalarflow-comparison.png" width = 80% alt="Image">

<figcaption align = "center">Figure .<i> Long-range extrapolations on the ScalarFlow dataset (60 time steps). FEN models perform better than the strongest baseline by also better modeling of sources and sinks. </i></figcaption>
</p>

## **5. Conclusion**  

We have reviewed _Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks_, a novel graph paradigm for learning dynamics on graphs based on inductive biases from differential equations. The authors provided a detailed analysis of the method from the ground up - starting from the theory of Finite Element analysis - and then devised two main models variations. While the first one learns directly the solution derivative in time of the physical system, the second separates learning with a _transport term_ which is shown to improve learning under many conditions. The experiments were conducted in one syntethic and two real-world high-dimensional datasets. Results demonstrated that the proposed models either perform competitively or outperform state-of-the-art baselines. This work represents and important contribution to the scientific machine learning community by tightly integrating the theory of Finite Element Method and Graph Neural Networks.

#### Limitations
The proposed model uses a simple basis - namely, linear piecewise basis function. If higher order derivatives were used, such as second order, these basis functions would evaluate to $0$, which is thus a current limitation of the model. Another limitation is the number of function evaluations: it is shown that the models can take more than 300 evaluations, while other non-continuous models may require just one. This is due to the adaptive ODE solvers used. Although the model can theoretically describe continuous dynamics, this practically makes it way slower than _one-step-prediction_ counterparts that do not need to evaluate an ODE.

---
## **Author Information**  

**Federico Berto**
[Personal Website](https://fedebotu.github.io/)


Affiliation: KAIST, Industrial & Systems Engineering Department
MSc students at [SILAB](http://silab.kaist.ac.kr/)
Member of the open research group [DiffEqML](https://github.com/DiffEqML)


## **6. Reference & Additional materials**  

#### Github Implementation: [https://github.com/martenlienen/finite-element-networks](https://github.com/martenlienen/finite-element-networks).


### References

[1] TODO!!