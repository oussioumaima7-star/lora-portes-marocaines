# Fiche du modèle : LoRA « mrcdoor »

| Élément | Valeur |
|--------|--------|
| Modèle de base | stable-diffusion-v1-5/stable-diffusion-v1-5 (licence CreativeML OpenRAIL-M) |
| Type d'adaptation | LoRA sur les couches d'attention du U-Net (to_q, to_k, to_v, to_out) |
| Rang retenu | … |
| Données | … images de portes marocaines, Wikimedia Commons (voir `data/sources.csv`) |
| Entraînement | … étapes, lr 1e-4, batch effectif 4, fp16, GPU T4 |
| Coût | … min, … Go de VRAM max, fichier de … Mo |
| Mot déclencheur | `mrcdoor` |

## Usage prévu
Pédagogique : illustrer l'adaptation légère d'un modèle texte-image à un domaine culturel précis.

## Limites connues
- Le jeu est très petit et vient surtout de photos touristiques, donc biaisé vers quelques villes (Fès, Chefchaouen…) et vers les portes « photogéniques ».
- Le modèle peut produire des motifs et une calligraphie **inventés** qui ressemblent à de vrais motifs sans en être. Les images ne doivent pas servir de référence historique.
- Héritées de SD 1.5 : mains, texte illisible, biais du jeu LAION.

## Usages déconseillés
Faire passer une image générée pour une photo réelle d'un lieu précis, ou produire de faux documents patrimoniaux.

## Évaluation
Voir `results/eval/metrics.csv` et `docs/guide_concepts.md` §6.
