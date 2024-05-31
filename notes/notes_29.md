## Jak porownac latents
- custom (contest) metrics,
- downstream metrics,
- czy jakos porownac wymiarowosci?

## Co do porownania:
- OmniVAE (cross-modality concatenation)
- BABEL (podobne do cycle GAN). Do lossu się dodaje rekonstrukcje w zaleznosci od modalnosci. W naszym przypadku są oba Negative Binomial
- 

## Roznice w modalities
- ADT - 134
- GEX - 13k

## Ideas
- clustering - sillhoutte score
- umap, tsne - color by cell types, batch, patiens
- batch effect per patient/site
- downstream
- all clustering metrics -> lift ponad oryginalne dane
- zwykla i log standaryzacja
- warto przeczytac jakikolwiek paper, ktory pracuje nad danymi!!

## Co marcin zrobil?
- GMM VAE
- mode seeking
- do wszystkich wag dodac IWAE
- hyper prior? 

## IWAE

$$ ELBO = - \frac{1}{N} \sum [Recloss(x, Dec(x)) - KL(q(z|x)|p(z))] $$

Dec(z) - rozklad nad X

$$ \frac{1}{N} \sum \log prob + \log prior - \log posterior $$
$$ \log ((\sum prob ) prior - posterior) $$

normalnie bylaby to srednia po wszystkich A + B + C. ale w IWAE bierzemy log sum exp, co premiuje najlepsze sample. log sum exp jest nieobciąone


czyli w praktyce:

$$rec loss + \log prior + \log hierarchical prior -\log prob(z | q(z|x))$$

torch.distribution

Jak nie ma cell types to mozna z torch.distribution: $\pi_c$ - logits, $\mi_c$ - vectors. Potem -> MixtureModel($\pi_c, \mi_c, \sigma_c$), logprob. 
Parametry mikstury są wspólne dla obu modeli. 

- co to są cell typy?
- powinna byc zadana z gory hierarchia. wystarczy ze uwzglednimy jeden poziom. sa 4 poziomy

# Najwazniejszy outcome
- duzy batch N = 16
- IWAE
- wszystkie matematyczne rzeczy są opisane w VAADERS

## Co jeszcze poruszyc
### Geneformer
Transformer ktory uzywa tylko niezerowych genow, tworzy rankingi.
Mozna zrobic encodingi za pomocą geneformera


- gene embedding: mamy dict $G_0 (size = 128/256): E_0, G_1: E_1 (32), ...$
1) dla kazdego G policz median non-zero expression
2) ) fajnie by bylo znormalizowac gene embeddings
3) x foreach $M_g>0: e_g/M_{e_g}$ gdzie $e$ to embedding
4) wez top 256, ...
5) * wylosuj 50 genow o zerowej ekspresji. doloz 50 do window
6) mozna miec dodatkowa statystyke: policz % pacjentow, dla ktorych ten gen jest zerowy
7) trzeba zerowy element windowu do $G_1, G_2, ...$ zeby on sie nie zgubil
### predykcja modalnosci z modalnosci
jak znam GEX to znam ADT
-stick breaking prior
$z -> (N,) (N, latent_dim) -> \mi$

$Mixture(\log \pi, \mi)$ log prob