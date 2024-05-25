## **Setup**
- $p(x)$ - true data distribution (unknown analytical form)
- $p(z)$ - prior distribution of the latent variable (known analytical form)
- $f_\phi$ - encoder
- $q_\phi(z | x) \approx p(z | x)$ - approximate posterior on the latent distribution
- $h_\theta$ - decoder 
- $p_\theta(z | x) \approx p(x | z)$ - true posterior on data distribution

### Modelling $p(x, z)$
#### Clarification: Assume there is some true latent distribution of $z$. Because of that we use $\approx$ symbol below
$p(x, z) = \underbrace{p(z | x)}_{\text{unknown}} \cdot \underbrace{p(x)}_{\text{we can sample from it (we already sampled)}} \approx \underbrace{q_\phi(z | x)}_{\text{trained}} \cdot \underbrace{p(x)}_{\text{we can sample from it (we already sampled)}} \rightarrow \text{we can sample from } \approx p(x, z)$

$p(x, z) = \underbrace{p(x | z)}_{\text{unknown}} \cdot \underbrace{p(z)}_{\text{we can sample from it}} \approx \underbrace{p_\theta(x | z)}_{\text{trained}} \cdot \underbrace{p(z)}_{\text{we can sample from it}} \rightarrow \text{we can sample from } \approx p(x, z)$

### Modelling $p(x)$
$p(x) = \int_z \underbrace{p_\theta(x | z)}_{\text{unknown }} \cdot p(z) dz = \mathbb{E}_{p(z)}[p_\theta(x | z)] \leftarrow \text{MC approximation}$
We will use it as part of the training objective, although we won't be able to sample from it.

### VAE loss derivation - data likelihood perspective
$\log p(x) = \mathbb{E}_{q_\phi(z | x)}[\log p(x)] = \mathbb{E}_{q_\phi(z | x)}\left[ \frac{\log p_\theta(x, z)}{\log p(z | x)} \cdot \frac{\log q_\phi(z | x)}{\log q_\phi(z | x)} \right] = \underbrace{\mathbb{D}_{KL}[q_\phi(z | x) || p(z | x)]}_{\geq 0} + \overbrace{\mathbb{E}_{q_\phi(z | x)}\left[ \frac{\log p_\theta(x, z)}{\log q_\phi(z | x)} \right]}^{\text{ELBO}}$

<!-- $\mathcal{L}_\text{ELBO}(x;\phi, \theta) = \mathbb{E}_{q_\phi(z | x)}[\log p_\theta(x, z) - \log q_\phi(z | x)] \approx \log p_\theta(x, z) - \log q_\phi(z | x)$ -->

$\mathcal{L}_\text{ELBO}(\mathcal{D};\phi, \theta) = \sum_{x \in \mathcal{D}} \mathcal{L}_\text{ELBO}(x;\phi, \theta)$

Calculating $\mathcal{L}_\text{ELBO}(x;\phi, \theta)$ is intractable as it requirese integrating over $z$ and we estimate it with sampling (typically one sample).

$\mathcal{L}_{\text{ELBO}}(x; \phi, \theta) = \underbrace{\mathbb{D}_{\text{KL}}[q_\phi(z | x) || p(z)]}_{\text{this is where we use analytical form}} + \underbrace{\mathbb{E}_{q_\phi(z | x)}[\log p_\theta(x | z)]}_{\log \text{likelihood } \approx \text{ reconstructions loss}}$

Second term is approximated with one sample.

$\nabla_\theta\mathcal{L}_{\text{ELBO}}(x; \phi, \theta) = \nabla_\theta\mathbb{E}_{q_\phi(z | x)}[\log p_\theta(x | z)] = \mathbb{E}_{q_\phi(z | x)}[\nabla_\theta \log p_\theta(x | z)]$

$\nabla_\phi\mathcal{L}_{\text{ELBO}}(x; \phi, \theta) = \nabla_\phi \underbrace{\mathbb{D}_{\text{KL}}[q_\phi(z | x) || p(z)]}_{\text{this is where we use analytical form}} + \underbrace{\nabla_\phi\mathbb{E}_{q_\phi(z | x)}[\log p(x | z)]}_{\text{problem with finding an unbiased estimator}}$

## Gaussian Prior $p(z)$ and posterior $q_\phi(z | x)$

$q_\phi(z | x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x)) = \frac{1}{(2\pi)^{d/2} |\Sigma_\phi|^{1/2}} \exp \left( -\frac{1}{2} (z - \mu_\phi(x))^\top \Sigma^{-1}_\phi (z - \mu_\phi(x)) \right) \leftarrow \text{with } \Sigma_\phi = \text{diag}(\sigma^2_\phi(x))$

$p(z) = \mathcal{N}(0, \mathbb{I}) = \frac{1}{(2\pi)^{d/2}} \exp \left( -\frac{1}{2} z^\top z \right)$

$\mathbb{D}_{\text{KL}}[q_\phi(z | x) || p(z)] = \underbrace{\int_z q_\phi(z | x) \cdot \log q_\phi(z | x) dz}_{ -\mathcal{H}(q_\phi(z | x)) } - \int_z q_\phi(z | x) \cdot \log p(z) dz$

The second term may be simplified

$\int_z q_\phi(z | x) \cdot \log p(z) dz = \int_z q_\phi(z | x) \cdot \log \left( \frac{1}{(2\pi)^{d/2}} \exp \left( -\frac{1}{2} z^\top z \right) \right) = \log \left( \frac{1}{(2\pi)^{d/2}} \right) -\frac{1}{2} \int_z q_\phi(z | x) \cdot z^\top z dz = -\frac{d}{2}\log(2\pi) - \frac{1}{2} \sum_i \left( \sigma^2_{\phi, i}(x) + \mu^2_{\phi, i}(x) \right) \leftarrow \text{Are we assuming marginal Gaussians or is it implicit?}$

The first term may be simplified with calculations similar to the ones above

$\mathcal{H}(q_\phi(z | x)) = -\int_z q_\phi(z | x) \cdot \log q_\phi(z | x) dz = \frac{d}{2}\log(2\pi) + \frac{1}{2} \sum_i \log \sigma^2_{\phi, i}(x) + \frac{1}{2} \int_z (z - \mu_\phi(x))^\top \text{diag}(\sigma^2_\phi(x))^{-1} (z - \mu_\phi(x)) dz = \frac{d}{2}\log(2\pi) + \frac{1}{2}  \sum_i \log \sigma^2_{\phi, i}(x) + \frac{d}{2}$

Finally combining the results

$\mathbb{D}_{\text{KL}}[q_\phi(z | x) || p(z)] = -\mathcal{H}(q_\phi(z | x)) - \int_z q_\phi(z | x) \cdot \log p(z) dz = -\frac{d}{2}\log(2\pi) - \frac{1}{2} \sum_i \log \sigma^2_{\phi, i}(x) - \frac{d}{2} + \frac{d}{2}\log(2\pi) + \sum_i \left( \sigma^2_{\phi, i}(x) + \mu^2_{\phi, i}(x) \right) = -\frac{1}{2}\sum_i \left( 1 + \sigma^2_{\phi, i}(x) - \mu^2_{\phi, i}(x) - \log \sigma^2_{\phi, i}(x) \right)$

### VAE Loss

$\mathcal{L}_{\text{ELBO}}(x; \phi, \theta) = \mathcal{L}_{\text{KL}}(x; \phi, \theta) + \overbrace{d(x,\underbrace{\mathbb{E}_{q_\phi(z | x)}[h_\theta(z)]}_{\approx \text{ with one sample}})}^{\text{reconstruction loss}}$

For simplicity we will write $\hat{x} \approx \mathbb{E}_{q_\phi(z | x)}[h_\theta(z)]$