Comparing latent spaces of different Variational Autoencoders (VAEs) involves several techniques and methodologies to understand how well the latent representations capture the underlying data distribution and how similar or dissimilar these representations are between models. Here are some steps and methods you can use:

### 1. Visualization Techniques
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** This is a popular technique to visualize high-dimensional data by projecting it into two or three dimensions. Apply t-SNE on the latent vectors from different VAEs and compare the resulting scatter plots.
- **UMAP (Uniform Manifold Approximation and Projection):** Similar to t-SNE, UMAP is used to visualize high-dimensional data. It's known for preserving more of the global structure.
- **PCA (Principal Component Analysis):** Use PCA to reduce the dimensionality of latent spaces and visualize the principal components. This can help in understanding the variance captured by the latent dimensions.

NICE READY VISUALISATION SOLUTIONS: https://scanpy.readthedocs.io/en/stable/ecosystem.html


### 2. Statistical Comparison
- **KL Divergence:** Measure the Kullback-Leibler divergence between the latent distributions of different VAEs to quantify how much one distribution diverges from another.
- **Wasserstein Distance:** Also known as the Earth Mover's Distance, this metric can be used to compare the distributions of latent spaces.

### 3. Latent Space Traversal
- **Interpolate Between Points:** Generate samples by interpolating between points in the latent space and observe how smoothly the decoder generates data. This helps in understanding the continuity and smoothness of the latent space.
- **Random Sampling:** Sample random points from the latent space and generate outputs to see the diversity and quality of samples produced by different VAEs.

### 4. Reconstruction Quality
- **Reconstruction Error:** Compare the reconstruction errors (e.g., mean squared error, mean absolute error) for the same input data across different VAEs.
- **Latent Space Density:** Analyze the density and spread of the latent representations. Overlapping densities might indicate similarity in how the models capture the data distribution.

### 5. Clustering and Classification
- **Clustering Analysis:** Apply clustering algorithms (e.g., K-means, DBSCAN) on the latent space and compare the clustering results. Evaluate cluster purity and other clustering metrics.
- **Latent Space Labeling:** If the data has labels, train a classifier on the latent space representations and compare the classification accuracies.

### 6. Transferability
- **Transfer Learning:** Use the latent representations learned by one VAE as input features for a downstream task and measure the performance. Compare this across different VAEs to see which latent space provides more useful features.

### 7. Model-Specific Metrics
- **Latent Space Regularization:** Evaluate how well the latent space adheres to the prior distribution (usually a Gaussian) by checking the regularization term in the VAE loss function.

### Example Workflow
1. **Extract Latent Vectors:**
   - Pass the same dataset through each VAE and collect the latent vectors.
   
2. **Visualization:**
   - Apply t-SNE or UMAP on the latent vectors and create scatter plots for visual comparison.

3. **Statistical Metrics:**
   - Calculate KL Divergence and Wasserstein Distance between the latent distributions of the different VAEs.

4. **Reconstruction and Sampling:**
   - Compute reconstruction errors and visually inspect generated samples from interpolated latent points.

5. **Clustering and Classification:**
   - Perform clustering on the latent vectors and evaluate cluster metrics. Train classifiers using the latent representations and compare their accuracies.

### Conclusion
Comparing latent spaces of different VAEs is a multifaceted process involving visualization, statistical analysis, and empirical evaluation. By combining these methods, you can gain a comprehensive understanding of how different VAEs capture and represent the underlying data distribution.

Evaluating the learned manifold qualities in the latent spaces of different VAEs involves examining how well the latent space captures the intrinsic structure of the data. Here are some methods to assess the quality of the learned manifold:

### 1. Manifold Continuity and Smoothness
- **Interpolation Quality:** Perform linear interpolation between pairs of points in the latent space and decode them to observe the resulting samples. High-quality manifolds will generate smooth and coherent transitions in the data space.
- **Geodesic Interpolation:** Instead of linear interpolation, use geodesic paths on the manifold. This can be more representative of the true structure in cases where the manifold is non-linear.

### 2. Topological Properties
- **Homology Analysis:** Use tools like persistent homology to study the topological features of the manifold. This can reveal the number of connected components, holes, and voids, providing insight into the manifold's shape.
- **Dimension Estimation:** Estimate the intrinsic dimensionality of the manifold using techniques like Maximum Likelihood Estimation (MLE) or the correlation dimension. Compare these estimates to the true dimensionality of the data.

### 3. Manifold Density and Coverage
- **Density Estimation:** Analyze the density of points in the latent space. Tools like Gaussian Mixture Models (GMM) can help visualize and quantify density. A good manifold should avoid overly sparse or overly dense regions.
- **Coverage:** Assess how well the latent space covers the data distribution. This can be done by sampling from the latent space and checking if the generated samples span the entire data distribution.

### 4. Local Linearity
- **Locally Linear Embedding (LLE):** Check how well local neighborhoods in the data space are preserved in the latent space. This involves comparing distances and angles between neighboring points in both spaces.
- **Jacobian Analysis:** Analyze the Jacobian matrix of the decoder to understand how local perturbations in the latent space map to changes in the data space. This can reveal local linearity and smoothness.

### 5. Reconstruction Fidelity
- **Reconstruction Error Distribution:** Instead of just looking at mean reconstruction error, analyze the distribution of errors across different regions of the latent space. High fidelity should be consistent across the manifold.
- **Class-Specific Reconstruction:** If the data has categorical labels, evaluate reconstruction errors separately for each class. This can reveal if the manifold is biased towards certain regions of the data space.

### 6. Robustness to Noise
- **Latent Space Perturbation:** Add noise to the latent vectors and decode them to see if the generated samples remain plausible. A robust manifold should tolerate small perturbations without significant degradation in output quality.

### 7. Overlap and Separation
- **Latent Space Clustering:** Use clustering algorithms on the latent space to see if different classes of data are well separated. Good manifolds should naturally cluster similar data points together.
- **Class Overlap Analysis:** For labeled data, analyze the overlap between classes in the latent space. Minimal overlap indicates a well-structured manifold.

### 8. Inverse Mapping and Reconstruction
- **Cycle Consistency:** Encode and then decode samples multiple times to see if the results remain consistent. This can test the stability of the manifold.
- **Inverse Mapping Accuracy:** Evaluate how well you can reconstruct latent vectors from generated samples, if you have an inverse mapping function or use methods like latent variable inference.

### Practical Steps to Evaluate Learned Manifold Qualities
1. **Visualization of Interpolations:**
   - Perform both linear and geodesic interpolations between random latent points and visualize the decoded outputs.

2. **Topological and Density Analysis:**
   - Use persistent homology and density estimation techniques to study the structure and distribution of the latent space.

3. **Local Linearity and Reconstruction Analysis:**
   - Analyze the Jacobian matrix of the decoder and evaluate the reconstruction error distribution across different regions of the latent space.

4. **Robustness Tests:**
   - Add noise to the latent vectors and decode to observe the robustness of the manifold.

5. **Clustering and Overlap Evaluation:**
   - Perform clustering in the latent space and analyze class separation and overlap.


<!-- - classification quality based on the latent space -->
- local dimensionality
- clustering???
- 