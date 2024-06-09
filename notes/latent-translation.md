predykcja niedeteministycznej mapy między latentatmi do sprawdzenia czy to mapowanie jest niedeterministyczne

VAE dla GEX daje $\mu$, $\sigma$ wtedy możemy wziąć mapę do latentu, która jako loss bierze logprob z rozkładu danego przez encoder. Można tak mapować w obie strony (ogólnie nie wymiary nie muszą się przy takim mapowaniu zgadzać idealnie. jak się zgadzają i założymy odwracalność mapy to można  robić Normalizing Flow i od razu mieć odwrotność)

można też zrobić dodatkowy latent wspólny dla modalności, lub kontrolowany przez jedną, do modelowania niepewności (czyli mogą być zwykłe AE + mapa dla każdego do tego wspólneog latentu tylko do modelowania niepewności przynależności do klastra, wtedy ten wspólny latent może mieć mniejszą wymiarowość niż latenty nawet dla obu AE)
- dla jednej modalności też to można robić i traktować to jako klasyfikację/fuzzy clustering