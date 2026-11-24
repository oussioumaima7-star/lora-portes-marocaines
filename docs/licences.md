# Licences et crédits

- **Code du projet** : MIT (à adapter selon les consignes).
- **Stable Diffusion 1.5** : CreativeML OpenRAIL-M. L'usage doit respecter ses restrictions.
- **BLIP, CLIP** : licences de leurs auteurs (Salesforce, OpenAI), utilisés uniquement en inférence.
- **Images d'entraînement** : voir `data/sources.csv`. Générer la liste des crédits avec :

```python
import pandas as pd
df = pd.read_csv("data/sources.csv")
for r in df.itertuples():
    print(f"- {r.title} — {r.author} — {r.license} — {r.source_url}")
```

Coller la sortie ci-dessous avant le rendu.

## Crédits des images
…
