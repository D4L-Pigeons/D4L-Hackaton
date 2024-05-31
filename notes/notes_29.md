## How to Compare Latents
- Custom (contest) metrics
- Downstream metrics
- Comparison of dimensionalities?

## Items for Comparison
- OmniVAE (cross-modality concatenation)
- BABEL (similar to cycle GAN). Reconstruction loss depends on the modality. In our case, both are Negative Binomial.

## Differences in Modalities
- ADT: 134
- GEX: 13k

## Ideas
- Clustering: silhouette score
- UMAP, t-SNE: color by cell types, batch, patients
- Batch effect per patient/site
- Downstream metrics
- All clustering metrics -> lift over original data
- Standardization (regular and log)
- Read any relevant paper working on similar data!

## What has Marcin done?
- GMM VAE
- Mode seeking
- Add IWAE to all weights
- Hyper prior?

## IWAE

\[ \text{ELBO} = - \frac{1}{N} \sum [\text{Recloss}(x, \text{Dec}(x)) - \text{KL}(q(z|x)\|p(z))] \]

Dec(z) - distribution over X

\[ \frac{1}{N} \sum \log \text{prob} + \log \text{prior} - \log \text{posterior} \]
\[ \log \left((\sum \text{prob}) \text{prior} - \text{posterior}\right) \]

Normally, this would be an average over all A + B + C, but in IWAE we use log-sum-exp, which favors the best samples. Log-sum-exp is unbiased.

In practice:

\[ \text{rec loss} + \log \text{prior} + \log \text{hierarchical prior} - \log \text{prob}(z | q(z|x)) \]

torch.distribution

If there are no cell types, we can use torch.distribution: \(\pi_c\) - logits, \(\mu_c\) - vectors. Then -> MixtureModel(\(\pi_c, \mu_c, \sigma_c\)), logprob. The mixture parameters are shared between both models.

- What are cell types?
- There should be a predefined hierarchy. Including just one level is enough. There are four levels.

## Most Important Outcome
- Large batch \(N = 16\)
- IWAE
- All mathematical details are described in VAADERS

## Additional Topics to Address
### Geneformer
Transformer that uses only non-zero genes to create rankings. Encoding can be done using Geneformer.

- **Gene embedding**: We have a dictionary \(G_0\) (size = 128/256): \(E_0\), \(G_1: E_1\) (32), ...
  1. For each G, calculate the median non-zero expression.
  2. Normalize gene embeddings.
  3. For each \(M_g > 0\): \(e_g / M_{e_g}\), where \(e\) is the embedding.
  4. Select top 256.
  5. Randomly select 50 genes with zero expression and add 50 to the window.
  6. Additional statistic: Calculate the % of patients for which this gene is zero.
  7. Ensure the zero element in the window is added to \(G_1, G_2, ...\) to prevent loss.

### Prediction of Modality from Modality
If GEX is known, ADT can be inferred.
- Stick-breaking prior
  \[ z \rightarrow (N,) (N, \text{latent_dim}) \rightarrow \mu \]
  \[ \text{Mixture}(\log \pi, \mu) \text{ log prob} \]