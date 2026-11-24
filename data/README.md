# Fiche des données

**Domaine** : portes traditionnelles marocaines (bois sculpté, arc outrepassé, zellige, heurtoirs).
**Un seul concept** : on ne mélange pas portes, plafonds, fontaines, etc.

| Élément | Valeur |
|--------|--------|
| Source | Wikimedia Commons (catégorie *Doors in Morocco* et sous-catégories) + photos de l'équipe |
| Licences acceptées | CC0, domaine public, CC BY, CC BY-SA (pas de NC/ND) |
| Traçabilité | `sources.csv` : fichier, titre, auteur, licence, URL de licence, URL source |
| Taille brute | … candidates téléchargées, … gardées après tri manuel |
| Après préparation | … train / … heldout (voir `prepare_report.txt`) |
| Format | PNG 512×512, recadrage centré (`--mode crop`) ou fond flou (`--mode pad`) |
| Légendes | `captions.csv`, pré-remplies par BLIP puis corrigées à la main |
| Mot déclencheur | `mrcdoor` |

## Critères de tri manuel
- Garder : une porte bien visible, vue de face ou de trois quarts, image nette.
- Écarter : plusieurs portes, personnes reconnaissables, filigranes, textes, porte coupée, très sombre.
- Écarter les quasi-doublons (détectés aussi par `prepare.py`).

## Ce qui n'est PAS dans Git
Les images (`raw/`, `processed/`). Elles sont recréées par `download_commons.py` + `prepare.py`
à partir de `sources.csv`.

## Photos de l'équipe
Les ajouter dans `raw/` sous le nom `equipe_XXX.jpg` et ajouter une ligne dans `sources.csv`
(auteur = membre, licence = CC BY 4.0, source_url vide).
