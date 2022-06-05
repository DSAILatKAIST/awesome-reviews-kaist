---
description : Shitong Luo /  A 3D Molecule Generative Model for Structure-Based Drug Design / NeurIPS 2021  
---

# **A 3D Molecule Generative Model for Structure-Based Drug Design** 

## **1. Problem Definition**  

"Drug design is the process of finding new medications based on the knowledge of a biological target"[1]. Since drug molecules takes effect by binding to some specific proteins, the designed molecules have to be compatible in shape and charge with the target binding site.  

The goal seems relatively simple but the difficulty lies in the underlying complexity of molecul structures in space. As a reminder, a molecule can be defined as a group of atoms held together by chemical bonds. The primary structure can be described as an amino acid sequence with its own chemical characteristics. However the folding of the protein can change those characteristics by creating new interactions between atoms. As a result, proteins with the same primary structure but different conformations in space leads to totally different biological effects. Designing molecules that bind to a specific protein binding site is one of the most challenging tasks in structure-based drug discovery.  


![Figure 1: Understanding the complexity of protein structures](../../.gitbook/assets/2022-spring-assets/EmelineBagoris2/levels-of-protein-structure-1.jpg)


## **2. Motivation**  

The challenge is basically to fold enormous space of synthetically feasible chemicals and conformational degree of freedom. The vast majority of existing works relies on string (1D) and graph (2D) representation of molecules. However, these methods are unable to perceive the 3D structures.
Recently, new algorithms called G-SchNet and MolGym were created to generate molecules in the 3D space. Unfortunately, they are also unable to target a specific structure and the results aren't scalable. Overall, the related works aren't suitable to find molecules able to fit to a specific binding site.

The authors goal is to design a model able to capture the 3d structure of specific protein binding site and directly generate molecules in the binding site to fit it. Their generative model estimates the probability density of atoms in space with the binding site as context. Then it uses auto-regressive sampling algorithm for generating molecules from the learned density of atoms. 

## **3. Method**  

The model proposed by the authors is split into two parts. First, they predict the probability of atom occurence in 3D space of the binding site. Second, they use an auto-regressive sampling algorithm to generate multi-model molecules. 

In order to represent molecules in the context of proteins, they decided to learn the probability density of atom occurences in the 3D space of the binding site $C$. This probability can be written $p(e|r, C)$, where $r ∈ R^3$ is a random coordinate in 3D space and $e$ is the type of the atom. To implement this probabilty, they used a context encoder to learn the representation of each atom in the context of the binding site $C$ as well as a spatial classifier to predict the probability density. The context encoder use an invariant graph neural netwok to encode the atoms using the distances as edge feature. Then the spatial classifier takes an arbitrary coordinate $r$, aggregate the features of the atom nearby $r$ and feed the aggregated feature to an MLP to predict the atom occupation. 

![Figure 2: Representation of molecules in a context of proteins](../../.gitbook/assets/2022-spring-assets/EmelineBagoris2/method_1.png)

Now that we have the density map, how can we use it to generate 3D molecules ? The answer is to place atom one-by-one auto-regressively. After placing each atom, we can feed the protein structure along with the previously placed atoms to update the density map. This allows the model to capture dependencies between generated atoms. Finally, they used OpenBabel to construct bonds between atoms.

![Figure 3: Representation of molecules in a context of proteins](../../.gitbook/assets/2022-spring-assets/EmelineBagoris2/method_3.png)

## **4. Experiment**  

### **Experiment setup**  

The authors performed an experiment on molecule design to test their model. The dataset used is a subset of CrossDocket composed of 100k protein-ligand pairs. The model was trained with the Adam optimizer at learning rate 0.0001. 

**Baseline :** The state-of-the-art of Drug Design is called liGAN. It's a 3D CNN relying on post-processing to reconstruct the molecule from a voxelized image. 

**Evaluation Metric :** 
- Binding Affinity of the generated molecules with the target site using the Vina Score. 
- Drug Likeness of the generated molecules using the QED Score. 
- Synthesizability of the generated molecules using the SA score.
- Diversity of the generated molecules

### **Result**  

The task here is to generate molecules for given binding sites. Table 1 shows the average and madian of the evaluation metrics for the authors model against the baseline. Overall, the authors model achieves greater performance on all metrics than the liGAN baseline. In fact, their model got higher scores on QED and SA which demonstrates realistic drug-like molecules. The lower VINA energy of their model demonstrates its ability to generate diverse molecules that have higher binding affinity for their target.  

![Figure 4: Results of the Molecule Design experiment](../../.gitbook/assets/2022-spring-assets/EmelineBagoris2/results_1.png)

We can visualize an example of generated molecules in the Figure 4. Each row contains six generated molecules plus the molecule used as reference for the binding site. The generated molecules possess very similar patterns to the reference. Furthermore, some of the generated molecules achieves a lower Vina and higher SA and QED than the reference. Those results demonstrates the ability of the generated molecules to fit well in the binding site while retaining high drug-like quality. 

![Figure 5: Example of molecules generated](../../.gitbook/assets/2022-spring-assets/EmelineBagoris2/results_2.png)


## **5. Conclusion**  

In this paper, the authors proposed a new approach to generate structure-based molecules. After experimenting on drug design, they obtained superior results than the state-of-the-art model for drug design. To conclude, their model is able to generate drug-like molecules with high binding affinity for designated targets. However, their model don't really take into account the effect of the molecule other than for it to be able to fit to the binding site. While it can easily be used for drug discovery as it is, introducting information for specific effect could give interesting perspective. 

---  
## **Author Information**  

* Emeline BAGORIS 
    * KAIST, Graduate School of AI  

## **6. Reference & Additional materials**  

* Github Implementation : https://github.com/luost26/3D-Generative-SBDD
* Original Paper : https://arxiv.org/abs/2203.10446
* [1] Shu-Feng Zhou and Wei-Zhu Zhong. Drug Design and Discovery: Principles and Applications. PMC, 2017. 