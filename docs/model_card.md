# Fiche du modèle : LoRA « mrcdoor »

| Élément | Valeur |
|--------|--------|
| Modèle de base | stable-diffusion-v1-5/stable-diffusion-v1-5 (licence CreativeML OpenRAIL-M) |
| Type d'adaptation | LoRA sur les couches d'attention du U-Net (to_q, to_k, to_v, to_out) |
| Rang retenu | 16 (alpha 16), 1 000 étapes ; intensité conseillée 0,8 |
| Données | 29 images (24 entraînement, 5 test), Wikimedia Commons + Openverse, voir `data/sources.csv` |
| Entraînement | 1 000 étapes, lr 1e-4, batch effectif 4, fp16, GPU Tesla T4 (Kaggle) |
| Coût | 25,8 min, 2,08 Go de VRAM max, fichier de 12,8 Mo |
| Mot déclencheur | `mrcdoor` |

## Usage prévu
Pédagogique : illustrer l'adaptation légère d'un modèle texte-image à un domaine culturel précis.

## Limites connues
- Le jeu est très petit et vient surtout de photos touristiques, donc biaisé vers quelques villes (Fès, Chefchaouen…) et vers les portes « photogéniques ».
- Le modèle peut produire des motifs et une calligraphie **inventés** qui ressemblent à de vrais motifs sans en être. Les images ne doivent pas servir de référence historique.
- Héritées de SD 1.5 : mains, texte illisible, biais du jeu LAION.

## Usages déconseillés
Faire passer une image générée pour une photo réelle d'un lieu précis, ou produire de faux documents patrimoniaux.

## Évaluation (intensité 1,0, prompts du domaine)
| | CLIPScore | Proximité domaine | Copies | Fuite |
|---|---|---|---|---|
| Modèle de base | 31,5 | 0,731 | – | 1,00 |
| LoRA rang 16 | 35,2 | 0,788 | 0 % | 0,92 |

Échecs connus : contexte « bâtiment de verre moderne » (le concept disparaît), neige (porte générique), zellige qui envahit la porte. Détails : `results/eval/metrics.csv` et le rapport.
