## **Methods**

(1) **OmiVAE** *Integrated Multi-omics Analysis Using Variational Autoencoders: Application to Pan-cancer Classification*

OmiVAE requires paired samples from two modalities. It creadtes joint embedding by concatenating the Encoder layers and deconcatenating the Decocer layers to allow for separately imputting and outputting two modalities.

The sample $x$ is given by a paired samples $x_1$, $x_2$ for two modalities written as $x = (x_1, x_2)$. The loss incorporates KL divergence, two reconstruction losses for different modalities and classification loss. The VAE part of the loss is given by the equation (1)
$\begin{equation}
\mathcal{L}_{\text{ELBO}}(x; \phi, \theta) = \mathcal{L}_{\text{KL}}(x; \phi, \theta) + d_1(x_1,\hat{x_1}) + d_2(x_2,\hat{x_2})
\end{equation}$
Where reconstruction losses $d_1$, $d_2$ correspond to two different modalities. One may add weight hyperparameters to the loss definition. The classification part of the loss is crossentropy loss
$\begin{equation}
\mathcal{L}_{\text{class}}(x,y;\psi) = -\sum_{i=1}^N \log p_{y_i}(x_i)
\end{equation}$
The total loss is expressed by equation (3)
$\begin{equation}
\mathcal{L}_{} = \alpha \cdot \mathcal{L}_{\text{ELBO}}(x; \phi, \theta) + \beta \cdot \mathcal{L}_{\text{class}}(x,y;\psi)
\end{equation}$

![alt text](OmiVAE-schema.png)

(2) **BABEL** *BABEL enables cross-modality translation between multiomic profiles at single-cell resolution*

BABEL requires paired samples from two modalities. It uses pair of autoencoders, which output parameters of the input data distribution. The joint embedding space is learned by CycleGAN-like training procedure (it may be made even more CycleGAN-like), which exchganges the embedding and reconstruction betwwen modality Encoders and Decoders of modality specific autoencoders.

BABEL relaxes the assumption of the Gaussian distribution when formulating the reconstruction loss part from the log likelihood perspective and instead of using MSE as a derivative of log likelihook in the Gaussian case to estimate mean parameter of the output distribution it allows to specify the parametrised distibution at the output and adjust the log likelihood loss accordingly.



![alt text](BABEL-schema.png)

(3) **(noname) Adversarial VAE** *Multi-domain translation between single-cell
imaging and sequencing data using autoencoders*

![alt text](noname-adv-VAE-schema.png)