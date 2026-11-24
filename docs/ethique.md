# Réflexion éthique

## Droits sur les images
Seules des images sous licence libre compatible (CC0, domaine public, CC BY, CC BY-SA) sont utilisées, et chacune est tracée dans `data/sources.csv`. Les licences BY et BY-SA imposent de **créditer les auteurs** : la liste figure dans `docs/licences.md`. Aucune œuvre d'un artiste vivant n'est ciblée, et le nom d'aucun artiste n'apparaît dans les prompts.

## Patrimoine culturel
Le zellige et la menuiserie traditionnelle sont un savoir-faire artisanal vivant. Un générateur de « style marocain » pose des questions :
- **Appropriation et dévalorisation** : des images générées en masse peuvent concurrencer le travail des artisans ou des photographes locaux.
- **Authenticité** : le modèle mélange des motifs de régions et d'époques différentes. Le résultat peut sembler authentique sans l'être.
- **Mesures prises** : la démo marque clairement les images comme générées, la fiche du modèle déconseille l'usage comme référence historique, et le projet reste pédagogique.

## Personnes
Les photos où des personnes sont reconnaissables sont écartées au tri, pour le droit à l'image et pour éviter que la LoRA n'apprenne des visages.

## Coût énergétique
Chaque entraînement dure quelques dizaines de minutes sur un T4. Le temps GPU total du projet est indiqué dans le rapport (somme des `train_time_min`).
