# Guide des concepts (pour comprendre et défendre le projet)

## 1. Comment Stable Diffusion génère une image
Stable Diffusion a trois parties :
1. **Encodeur de texte (CLIP)** : il transforme le prompt en 77 vecteurs.
2. **VAE** : il compresse une image 512×512×3 en une « image latente » 64×64×4, soit 48 fois moins de valeurs. Toute la diffusion se passe dans cet espace, ce qui la rend abordable.
3. **U-Net** : on lui donne un latent bruité, l'instant *t* et le texte. Il prédit le bruit présent dans le latent. À la génération, on part d'un bruit pur et on retire le bruit prédit, étape par étape.

Le U-Net « lit » le texte grâce à la **cross-attention**. Les requêtes (Q) viennent de l'image, les clés (K) et valeurs (V) viennent du texte. C'est donc là que le lien entre les mots et ce qu'on voit est appris, et c'est là qu'on place la LoRA.

## 2. LoRA : l'idée
Fine-tuner tout le U-Net, c'est modifier environ 860 millions de poids. Il faut beaucoup de VRAM, on obtient un fichier de 3,4 Go, et avec 30 images le modèle apprend tout par cœur.

LoRA fait l'hypothèse que **la modification utile a un rang faible**. Au lieu d'apprendre une grosse matrice ΔW (d×d), on la factorise :

```
W' = W + (alpha / r) · B · A        A : r×d,  B : d×r,   r ≪ d  (ex : r = 8, d = 320…1280)
```

- On n'apprend que A et B. Le script affiche le nombre de paramètres entraînables : il représente **moins de 1 %** du U-Net.
- B est initialisé à zéro. Au début de l'entraînement, le modèle est donc exactement le modèle de base.
- Le fichier produit ne pèse que quelques Mo. On le charge par-dessus le modèle d'origine, qui n'est jamais modifié.

### Rang `r`
C'est la « capacité » de l'adaptation.
- Un **r petit** (4) donne peu de paramètres et une adaptation douce. Le risque est de ne pas capturer assez de détails.
- Un **r grand** (32) apprend plus de détails, mais apprend aussi plus vite par cœur. Le fichier est plus gros et la diversité baisse.
- C'est exactement ce que montre l'expérience `configs/rank*.yaml`.

### `alpha`
C'est un facteur d'échelle. Avec `alpha = r`, l'échelle vaut 1, donc changer le rang ne change pas la force de la mise à jour. On compare alors vraiment la *capacité* et pas la force.

### Intensité (`scale`) à l'inférence
Elle multiplie la contribution de la LoRA : 0 donne le modèle de base, 1 la LoRA entraînée. Une valeur intermédiaire, souvent entre 0.7 et 0.8, garde la flexibilité du modèle de base tout en ajoutant le style.

## 3. Le mot déclencheur `mrcdoor`
On utilise un mot **inventé** que le modèle ne connaît pas, pour que le concept s'y attache. Si on utilisait « door », on modifierait toutes les portes du modèle : c'est le problème de la **fuite**, ou *language drift*. Le test `leakage_prompts` vérifie que les prompts sans `mrcdoor` donnent toujours la même image qu'avant.

## 4. Pourquoi décrire la couleur dans les légendes ?
Ce qui n'est **pas** écrit dans la légende est attribué au mot `mrcdoor`. Si 80 % des portes sont bleues et que la légende ne le dit jamais, « mrcdoor » finira par vouloir dire « porte *bleue* ». En écrivant « blue door », la couleur devient un attribut qu'on peut changer avec le prompt. C'est ce que teste le prompt « a red mrcdoor ».

## 5. Les astuces mémoire (pour tenir dans moins de 12 Go)
| Astuce | Effet |
|--------|------|
| Poids gelés en fp16 | Le modèle prend deux fois moins de mémoire |
| Seuls A et B en fp32 + mixed precision | Les mises à jour restent stables |
| Gradient checkpointing | On recalcule les activations au lieu de les stocker. C'est plus lent (environ 20 %) mais beaucoup plus léger |
| Pré-calcul des latents et des embeddings de texte | Le VAE et CLIP sortent du GPU pendant l'entraînement |
| Batch 1 × accumulation 4 | Équivaut à un batch de 4 sans la mémoire d'un batch de 4 |

## 6. Les métriques (et leurs limites)
| Métrique | Ce qu'elle mesure | Limite |
|----------|------------------|--------|
| CLIPScore | L'image correspond-elle au texte ? | CLIP ne connaît pas `mrcdoor`, d'où le remplacement par « moroccan door » |
| domain_sim | L'image ressemble-t-elle aux vraies portes *jamais vues* ? | Seulement 5 à 10 images de référence |
| train_nn_sim / copy_rate | Le modèle recopie-t-il ses images d'entraînement ? | Le seuil 0.95 est un choix : on le justifie avec des exemples visuels |
| diversity | Les seeds différentes donnent-elles des images différentes ? | Elle baisse quand le modèle surapprend |
| leakage_sim | Le reste du modèle est-il intact ? | — |
| Temps, VRAM, taille | Le coût | Dépend du GPU (préciser T4) |

**Pourquoi pas la FID ?** Elle compare des distributions et demande des milliers d'images. Avec 8 images de test, elle n'a pas de sens.

## 7. Reconnaître le surapprentissage
- La perte d'entraînement continue de baisser, mais les images de **généralisation** (« in the snow », « watercolor ») redeviennent des photos de la médina : le contexte demandé est ignoré.
- `train_nn_sim` monte et `diversity` baisse.
- Des images quasi identiques à une photo d'entraînement apparaissent. Il faut les montrer côte à côte dans le rapport.

## 8. Questions probables en soutenance
- Pourquoi LoRA plutôt qu'un fine-tuning complet ou DreamBooth ?
- Pourquoi placer la LoRA dans l'attention et pas ailleurs ?
- Comment avez-vous choisi le nombre d'étapes ?
- Comment savez-vous que le modèle ne recopie pas les images ?
- Pourquoi SD 1.5 et pas SDXL ? (VRAM : SDXL dépasse facilement 12 Go en entraînement, et sa résolution native est 1024)
