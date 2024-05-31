## **Setup**
- $p(x)$ - true data distribution (unknown analytical form)
- $p(z)$ - prior distribution of the latent variable (known analytical form)
- $f_\phi$ - encoder
- $q_\phi(z | x) \approx p(z | x)$ - approximate posterior on the latent distribution
- $h_\theta$ - decoder 
- $p_\theta(z | x) \approx p(x | z)$ - true posterior on data distribution

### IWAE loss derivation - data likelihood perspective

$\log p(x) = \log \mathbb{E}_{(z_1, ..., z_K) \stackrel{\text{iid}}{\sim} q_\phi(z | x)} \left[ \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q_\phi(z_k | x)} \right] \stackrel{\text{by Jensen inequality}}{\geq} = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q_\phi(z_k | x)} \right) \right] = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\log \frac{p(x, z_k)}{q_\phi(z_k | x)}} \right) \right] = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right) \right]$

Where $\hat{\text{ELBO}}$ stands for one sample estimate of $\text{ELBO}$.

### Comparison with VAE objective

Let's compare the objective to the VAE and IWAE objective estimates in case of $k$ samples from the posterior approximation. In VAE we are estimating the $\text{ELBO}$

$\mathbb{E}_{q_\phi(z | x)}\left[ \log \frac{p(x, z_k)}{q_\phi(z_k | x)} \right] \approx \frac{1}{K} \sum_{k = 1}^K \log \frac{p(x, z_k)}{q_\phi(z_k | x)} = \frac{1}{K} \sum_{k = 1}^K \hat{\text{ELBO}}$

On the other hand in the IWAE we are are estimating the objective with $\log \text{sum} \exp$ of the $\text{ELBO}$ estimates for the individual samples.

$\mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right) \right] \approx \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right)$

Note, that in IWAE we may increase the $K$ to get more samples or we could increase the number of $K$-sized random vectors being the samples indexing the expected value in IWAE. In the second case we would multiply the $\text{sum} \exp$ of the $\text{ELBO}$'s for each $K$-sized random vector under the logarithm and effectively get the $\log \text{sum} \exp$ of the $\text{ELBO}$ over the parwise products of both $K$-sized random vectors resulting in the loss focusing on the least "fitting in" examples but is a different way.

The analytical work is done as in the case of VAE.

It may be shown that the IWAE provides better lower bound on the log likelihood of the data and is an asymptotically i $K$ unbiased estimator of the it.

### Question
Why we do not benefit from more samples in VAE and do in IWAE? - in VAE we stay with the ELBO, but better estimated and in IWAE if we are increasing the number of samples by increasing $K$ we get closer to unbiased estimator of data log likelihood.

### Comments
We may choose some $K$ and try to estimate asymptotic behaviour of the bound

# GPT has sthg to say as always... but I asked wrong question (there is no relaxation of the requirement of some analitical use of the assumed prior and posterior family of distributions as it is still used in formuls, but we adhere to it less strictly)

# Q: Why KL divergence does not need to be analytical in the IWAE case?

In the Importance Weighted AutoEncoder (IWAE), the requirement for an analytical KL divergence is relaxed compared to the traditional Variational Autoencoder (VAE) due to the use of importance sampling to estimate the Evidence Lower Bound (ELBO). Here’s a detailed explanation:

### Traditional VAE

In a traditional VAE, the objective is to maximize the Evidence Lower Bound (ELBO), which typically consists of two terms:
1. The reconstruction term: \(\mathbb{E}_{q(z|x)}[\log p(x|z)]\)
2. The KL divergence term: \(D_{KL}(q(z|x) || p(z))\)

The KL divergence term \(D_{KL}(q(z|x) || p(z))\) often needs to be computed analytically to ensure efficient training. This requirement usually constrains \(q(z|x)\) and \(p(z)\) to be distributions for which the KL divergence can be computed in closed form, such as Gaussians.

### Importance Weighted AutoEncoder (IWAE)

In IWAE, the ELBO is tightened using an importance-weighted approximation, which allows for a better estimation of the log marginal likelihood of the data. The IWAE objective, or the importance-weighted ELBO (IW-ELBO), is given by:

\[ \log p(x) \geq \text{IW-ELBO} = \mathbb{E}_{z_{1:K} \sim q(z|x)} \left[ \log \left( \frac{1}{K} \sum_{k=1}^{K} \frac{p(x, z_k)}{q(z_k|x)} \right) \right] \]

where \(K\) is the number of importance samples.

### Relaxation of Analytical KL Divergence

In IWAE, the need for an analytical form of the KL divergence is relaxed because:

1. **Importance Sampling:**
   The IW-ELBO uses importance sampling to approximate the expectation over the posterior distribution. By drawing multiple samples \(z_{1:K}\) from the variational distribution \(q(z|x)\) and computing the weighted average, IWAE can handle more complex variational distributions that do not require an analytical KL divergence.

2. **Monte Carlo Estimation:**
   The expectation \(\mathbb{E}_{z_{1:K} \sim q(z|x)}\) is estimated using Monte Carlo methods. This means that the ELBO is approximated based on samples rather than requiring a closed-form solution for the KL divergence. The Monte Carlo approach inherently deals with the intractability by providing a sampled estimate of the required expectations.

3. **Flexible Variational Distributions:**
   Since IWAE does not rely on an analytical KL divergence, it allows for the use of more flexible and expressive variational distributions \(q(z|x)\). These distributions can be complex and tailored to better approximate the true posterior, potentially leading to better performance and tighter bounds.

4. **Tighter Bound:**
   The importance-weighted bound is generally tighter than the standard ELBO. As the number of importance samples \(K\) increases, the IW-ELBO approaches the true log marginal likelihood. This tighter bound reduces the need for exact analytical forms and instead relies on empirical estimates through sampling.

### Practical Implications

In practice, this relaxation means that IWAE can:
- Utilize variational distributions \(q(z|x)\) that are more complex and potentially non-Gaussian.
- Avoid the constraints of needing closed-form KL divergences, making it more flexible in modeling complex posterior distributions.
- Achieve better performance with tighter bounds on the log likelihood through the use of multiple importance samples.

### Summary

The IWAE's use of importance sampling to estimate the ELBO allows it to bypass the requirement for an analytical KL divergence. This flexibility enables the use of more complex and expressive variational distributions, potentially leading to better approximation of the true posterior and improved generative modeling performance.


# Q: So I may just plug in the density formula to the KL Divergence?

Yes, in the context of the Importance Weighted Autoencoder (IWAE), you can indeed work with the density formula directly without needing an analytical form for the KL divergence. This is because the IWAE objective leverages importance sampling to approximate the ELBO. Here’s how you can proceed:

### IWAE Objective

The IWAE objective (IW-ELBO) is given by:

\[ \log p(x) \geq \text{IW-ELBO} = \mathbb{E}_{z_{1:K} \sim q(z|x)} \left[ \log \left( \frac{1}{K} \sum_{k=1}^{K} \frac{p(x, z_k)}{q(z_k|x)} \right) \right] \]

where:
- \( K \) is the number of importance samples.
- \( z_k \) are samples from the variational distribution \( q(z|x) \).

### Steps to Implement IWAE with Non-Analytical KL Divergence

1. **Sample from the Variational Distribution:**
   Draw \( K \) samples \( z_1, z_2, \ldots, z_K \) from the variational distribution \( q(z|x) \).

2. **Compute Importance Weights:**
   For each sample \( z_k \), compute the importance weight:

   \[ w_k = \frac{p(x, z_k)}{q(z_k|x)} = \frac{p(x|z_k) p(z_k)}{q(z_k|x)} \]

3. **Approximate IW-ELBO:**
   The IW-ELBO is then approximated as:

   \[ \text{IW-ELBO} \approx \log \left( \frac{1}{K} \sum_{k=1}^{K} w_k \right) \]

4. **Optimization:**
   Optimize the parameters of the VAE (both the encoder \( q(z|x) \) and the decoder \( p(x|z) \)) using gradient-based methods, with the IW-ELBO as the objective function.

### Practical Example

Let's break down a practical example where we assume specific forms for \( p(z) \), \( p(x|z) \), and \( q(z|x) \):

- **Prior \( p(z) \)**: A standard Gaussian \( \mathcal{N}(0, I) \).
- **Likelihood \( p(x|z) \)**: Typically Gaussian with mean given by a neural network and fixed variance.
- **Variational Distribution \( q(z|x) \)**: A Gaussian with parameters (mean and variance) output by the encoder network.

1. **Sample from \( q(z|x) \):**

   \[ z_k \sim q(z|x) \quad \text{for} \quad k = 1, 2, \ldots, K \]

2. **Compute \( p(x, z_k) \):**

   \[ p(x, z_k) = p(x|z_k) p(z_k) \]

   Here, \( p(x|z_k) \) is computed using the decoder network, and \( p(z_k) \) is the Gaussian prior.

3. **Compute \( q(z_k|x) \):**

   \[ q(z_k|x) \] is the output of the encoder network.

4. **Compute Importance Weights:**

   \[ w_k = \frac{p(x|z_k) p(z_k)}{q(z_k|x)} \]

5. **Estimate IW-ELBO:**

   \[ \text{IW-ELBO} \approx \log \left( \frac{1}{K} \sum_{k=1}^{K} w_k \right) \]

### Summary

In IWAE, you avoid the need for an analytical KL divergence by using the importance-weighted approximation. This allows you to directly use the density functions \( p(z) \), \( p(x|z) \), and \( q(z|x) \) to compute the importance weights and thus the IW-ELBO. This method provides a more flexible approach, accommodating more complex variational distributions and priors.