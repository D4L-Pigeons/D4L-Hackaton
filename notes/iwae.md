## **Setup**
- $p(x)$ - true data distribution (unknown analytical form)
- $p(z)$ - prior distribution of the latent variable (known analytical form)
- $f_\phi$ - encoder
- $q_\phi(z | x) \approx p(z | x)$ - approximate posterior on the latent distribution
- $h_\theta$ - decoder 
- $p_\theta(z | x) \approx p(x | z)$ - true posterior on data distribution

### IWAE loss derivation - data likelihood perspective
$\log p(x) = \mathbb{E}_{q_\phi(z | x)}[\log p(x)] = \mathbb{E}_{q_\phi(z | x)}\left[ \frac{\log p_\theta(x, z)}{\log p(z | x)} \cdot \frac{\log q_\phi(z | x)}{\log q_\phi(z | x)} \right] = \underbrace{\mathbb{D}_{KL}[q_\phi(z | x) || p(z | x)]}_{\geq 0} + \overbrace{\mathbb{E}_{q_\phi(z | x)}\left[ \frac{\log p_\theta(x, z)}{\log q_\phi(z | x)} \right]}^{\text{ELBO}}$

$\log p(x) = \log \mathbb{E}_{(z_1, ..., z_K) \stackrel{\text{iid}}{\sim} q_\phi(z | x)} \left[ \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q_\phi(z_k | x)} \right] \stackrel{\text{by Jensen inequality}}{\geq} = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q_\phi(z_k | x)} \right) \right] = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\log \frac{p(x, z_k)}{q_\phi(z_k | x)}} \right) \right] = \mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right) \right]$

Where $\hat{\text{ELBO}}$ stands for one sample estimate of $\text{ELBO}$.

Let's compare the objective to the VAE and IWAE objective estimates in case of $k$ samples from the posterior approximation. In VAE we are estimating the $\text{ELBO}$

$\mathbb{E}_{q_\phi(z | x)}\left[ \log \frac{p(x, z_k)}{q_\phi(z_k | x)} \right] \approx \frac{1}{K} \sum_{k = 1}^K \log \frac{p(x, z_k)}{q_\phi(z_k | x)} = \frac{1}{K} \sum_{k = 1}^K \hat{\text{ELBO}}$

On the other hand in the IWAE we are are estimating the objective with $\log \text{sum} \exp$ of the $\text{ELBO}$ estimates for the individual samples.

$\mathbb{E}_{q_\phi(z | x)}\left[ \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right) \right] \approx \log \left( \frac{1}{K}\sum_{k=1}^K \mathrm{e}^{\hat{\text{ELBO}}} \right)$

Note, that in IWAE we may increase the $K$ to get more samples or we could increase the number of $K$-sized random vectors being the samples indexing the expected value in IWAE. In the second case we would multiply the $\text{sum} \exp$ of the $\text{ELBO}$'s for each $K$-sized random vector under the logarithm and effectively get the $\log \text{sum} \exp$ of the $\text{ELBO}$ over the parwise products of both $K$-sized random vectors resulting in the loss focusing on the least "fitting in" examples but is a different way.

The analytical work is done as in the case of VAE.

It may be shown that the IWAE provides better lower bound on the log likelihood of the data and is an asymptotically i $K$ unbiased estimator of the it.

### Question
Why we do not benefit from more samples in VAE and do in IWAE? - in VAE we stay with the ELBO, but better estimated and in IWAE if we are increasing the number of samples by increasing $K$ we get closer to unbiased estimator of data log likelihood.