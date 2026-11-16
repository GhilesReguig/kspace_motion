# Webapp de simulation de mouvement IRM

```bash
conda activate kspace_motion_test
python webapp/app.py            # http://127.0.0.1:8000  (--port / --host pour changer)
```

1. **Trajectoire** : les paramètres (même noms que `RandomMotionGPU`) génèrent les courbes de
   translation et de rotation. Elles se mettent à jour à chaque modification, avec une graine fixe.
2. **Simulation** : le bouton applique exactement la trajectoire affichée à la T1w de Colin27
   (`data/colin27_t1.nii.gz`, version 1998, 1 mm) avec `MotionGPU`.

3. **Comparaison** : chaque simulation est enregistrée dans l'un des trois emplacements A, B, C
   (paramètres, trajectoire, volumes et métriques). On change les paramètres entre deux lancements,
   puis on compare les coupes côte à côte (original, A, B, C) avec leur différence à l'original.
   Chaque emplacement peut être affiché ou masqué, remplacé, supprimé, ou « rechargé » (ses paramètres
   et ses courbes reviennent dans le formulaire). Les résolutions peuvent différer d'un emplacement à l'autre :
   les curseurs de coupe sont en position relative.

4. **Corégistration (option)** : après la simulation, l'image bougée peut être recalée sur l'image
   originale, avec Elastix (rigide, `coregister_itk`) ou torchreg (affine, `coregister_torchreg`).
   L'image brute est conservée : un sélecteur « Image affichée » bascule entre brute et corégistrée, et
   les cartes montrent les métriques avant → après. Si la corégistration échoue, la simulation est quand
   même renvoyée avec un message d'erreur.

5. **Suréchantillonnage (padding)** : avant la simulation, le volume est entouré de zéros
   (`oversampling_pct`, 30 % par défaut, comme l'option de la version CPU), puis recadré. Le mouvement
   agit dans le Fourier, qui est circulaire : sans marge, ce qui sort d'un côté du champ de vue réapparaît
   de l'autre, et le recalage laisse des bandes noires sur les bords. La corégistration est faite sur le
   volume paddé et le recadrage n'intervient qu'après. Coût : à 1 mm, environ 11 s et 3 Go de mémoire
   avec 30 %, contre 5 s et 1,6 Go sans padding.

6. **Artefacts, modes Simple et Avancé** : le formulaire a deux niveaux. En mode *Simple*, chaque artefact
   est une carte à cocher avec un ou deux réglages ; en mode *Avancé*, tous les paramètres sont visibles
   (mouvement détaillé, axes, unités, graines...). Les listes « Intensité » (mouvement), « Source »
   (ghosting) et « Fréquence » (pics) sont des préréglages des champs avancés : les choisir remplit ces
   champs, les modifier à la main bascule la liste sur « Personnalisé ».

   | Carte | Principe (module `kspace_motion/kspace_artifacts.py`) |
   |---|---|
   | Mouvement du patient | trajectoire aléatoire (ou « Aucun ») |
   | Ghosting | translation sinusoïdale ajoutée à la trajectoire : répliques espacées de FOV / cycles le long de l'axe de phase (l'espacement est affiché) |
   | Bruit (Ricien) | bruit gaussien complexe sur le k-space, module de l'image Ricien ; réglé par le SNR (dB) |
   | Pics | quelques échantillons de k-space très forts, de phase aléatoire : franges (« spike noise ») |
   | Gibbs, Flou, T2\* | troncature du k-space, filtre passe-bas, décroissance en lecture |
   | Lignes manquantes | plans ou lignes du k-space mis à zéro (ou atténués) |
   | Fourier partiel, Repliement | k-space partiellement acquis ; une ligne sur N |
   | Zipper, Offset, Saturation | fuite RF (ligne pointillée), constante (voxel central), écrêtage du centre du k-space |

   Les artefacts sont appliqués au k-space acquis, juste avant la reconstruction, dans un ordre fixe
   (géométrie, lignes, interférences, récepteur, bruit). Sans mouvement, le k-space est calculé par FFT au
   lieu de la NUFFT. La trajectoire est en mm (translations) et en degrés (rotations) : le serveur divise les
   translations par la résolution avant d'appeler `MotionGPU`, qui compte en voxels. Les amplitudes
   d'artefacts sont relatives au signal des tissus.

Le formulaire et la validation viennent d'une seule définition (`CONFIG` dans `app.py`,
servie par `/api/config`). Le serveur fixe `OMP_NUM_THREADS=1` : avec plusieurs threads, la NUFFT d'un
volume complet à 1 mm se bloque dans cet environnement (torch pip + finufft conda).

Tests : `pytest tests/test_webapp.py`.
