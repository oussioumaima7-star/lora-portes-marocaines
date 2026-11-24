# Plan du rapport (6 à 8 pages)

1. **Introduction** (½ page) : le problème (spécialiser un grand modèle avec peu de données et peu de calcul), le domaine choisi et pourquoi, les questions de recherche :
   - Q1. Quel rang offre le meilleur compromis entre qualité, coût et surapprentissage ?
   - Q2. À partir de combien d'étapes le modèle surapprend-il ?
   - Q3. Le concept appris se combine-t-il avec des contextes jamais vus ?
2. **Notions** (1 page) : diffusion latente, cross-attention, LoRA (équation, rang, alpha), mot déclencheur. Inclure un schéma.
3. **Données** (¾ page) : sources, licences, tri, préparation, stratégie de légendes, statistiques et une mosaïque d'exemples.
4. **Méthode** (1 page) : baseline, configuration d'entraînement, optimisations mémoire (tableau), plan d'expériences, métriques et justification (pourquoi pas la FID).
5. **Résultats** (2 pages)
   - Baseline vs LoRA : grille avant/après.
   - Rangs : tableau des métriques + `comparaison_rangs.png`.
   - Étapes : `loss_curves.png` + `surapprentissage_etapes.png`. La perte baisse alors que la généralisation se dégrade.
   - Intensité 0.5, 0.8 et 1.0.
   - Coût : temps, VRAM, taille.
6. **Analyse des échecs** (1 page) : quasi-copies (images côte à côte avec leur voisin d'entraînement), prompts ignorés, biais de couleur, fuite de style.
7. **Éthique et limites** (½ page) : résumé de `docs/ethique.md`.
8. **Conclusion** (¼ page) : réponses à Q1, Q2 et Q3, pistes (LoRA sur l'encodeur de texte, SDXL, DoRA, légendes par VLM).

**Soutenance (10 min)** : 1 min contexte, 2 min LoRA, 1 min données, 3 min résultats, 1 min échecs, 2 min démo en direct.
