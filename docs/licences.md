# Licences et crédits

- **Code du projet** : MIT.
- **LoRA entraînées** : CC BY-SA 4.0 (20 des 29 images d'entraînement sont sous CC BY-SA).
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

## Crédits des images (29)

| Fichier | Œuvre | Auteur | Licence |
|---|---|---|---|
| commons_0022.jpg | [Oualidia, Casablanca-Settat region, Morocco November 2023 - Blue door and window shutters.jpg](https://commons.wikimedia.org/wiki/File:Oualidia,_Casablanca-Settat_region,_Morocco_November_2023_-_Blue_door_and_window_shutters.jpg) | Sharon Hahn Darlin | CC BY 2.0 |
| commons_0025.jpg | [ضريح الشيخ الكامل 24.jpg](https://commons.wikimedia.org/wiki/File:%D8%B6%D8%B1%D9%8A%D8%AD_%D8%A7%D9%84%D8%B4%D9%8A%D8%AE_%D8%A7%D9%84%D9%83%D8%A7%D9%85%D9%84_24.jpg) | ZAINEB HACHAMI | CC BY-SA 4.0 |
| commons_0028.jpg | [Door in Essaouira (ancienne Mogador) 10.jpg](https://commons.wikimedia.org/wiki/File:Door_in_Essaouira_(ancienne_Mogador)_10.jpg) | Mounir Neddi | CC0 |
| commons_0032.jpg | [Essaouira , Morocco - panoramio (7).jpg](https://commons.wikimedia.org/wiki/File:Essaouira_,_Morocco_-_panoramio_(7).jpg) | Patrick Nouhailler's… | CC BY-SA 3.0 |
| commons_0038.jpg | [Essaouira le matin - panoramio (58).jpg](https://commons.wikimedia.org/wiki/File:Essaouira_le_matin_-_panoramio_(58).jpg) | Patrick Nouhailler's… | CC BY-SA 3.0 |
| commons_0039.jpg | [Essaouira le matin - panoramio (62).jpg](https://commons.wikimedia.org/wiki/File:Essaouira_le_matin_-_panoramio_(62).jpg) | Patrick Nouhailler's… | CC BY-SA 3.0 |
| commons_0040.jpg | [Essaouira le matin - panoramio (65).jpg](https://commons.wikimedia.org/wiki/File:Essaouira_le_matin_-_panoramio_(65).jpg) | Patrick Nouhailler's… | CC BY-SA 3.0 |
| commons_0042.jpg | [Essaouira Medina - panoramio (10).jpg](https://commons.wikimedia.org/wiki/File:Essaouira_Medina_-_panoramio_(10).jpg) | Patrick Nouhailler's… | CC BY-SA 3.0 |
| commons_0044.jpg | [Maroc Essaouira Luc Viatour 5.JPG](https://commons.wikimedia.org/wiki/File:Maroc_Essaouira_Luc_Viatour_5.JPG) | Luc Viatour (Lviatour) | CC BY-SA 3.0 |
| commons_0045.jpg | [Maroc Essaouira Luc Viatour 6.JPG](https://commons.wikimedia.org/wiki/File:Maroc_Essaouira_Luc_Viatour_6.JPG) | Luc Viatour (Lviatour) | CC BY-SA 3.0 |
| commons_0046.jpg | [Maroc Essaouira Luc Viatour 7.JPG](https://commons.wikimedia.org/wiki/File:Maroc_Essaouira_Luc_Viatour_7.JPG) | Luc Viatour (Lviatour) | CC BY-SA 3.0 |
| commons_0048.jpg | [Bab Sbitriyyin.jpg](https://commons.wikimedia.org/wiki/File:Bab_Sbitriyyin.jpg) | Robert Prazeres | CC BY-SA 4.0 |
| commons_0049.jpg | [Qarawiyyin Mosque Bab Sbitriyine DSCF2935.jpg](https://commons.wikimedia.org/wiki/File:Qarawiyyin_Mosque_Bab_Sbitriyine_DSCF2935.jpg) | R Prazeres | CC BY-SA 4.0 |
| commons_0052.jpg | [Bab Shama'in.jpg](https://commons.wikimedia.org/wiki/File:Bab_Shama%27in.jpg) | Robert Prazeres | CC BY-SA 4.0 |
| commons_0060.jpg | [Qarawiyyin Bab al-Ganaiz 01.jpg](https://commons.wikimedia.org/wiki/File:Qarawiyyin_Bab_al-Ganaiz_01.jpg) | Robert Prazeres | CC BY-SA 4.0 |
| commons_0062.jpg | [Qarawiyyin Bab al-Ganaiz 03.jpg](https://commons.wikimedia.org/wiki/File:Qarawiyyin_Bab_al-Ganaiz_03.jpg) | Robert Prazeres | CC BY-SA 4.0 |
| commons_0064.jpg | [Qarawiyyin Mosque Bab al-Ganaiz DSCF2932.jpg](https://commons.wikimedia.org/wiki/File:Qarawiyyin_Mosque_Bab_al-Ganaiz_DSCF2932.jpg) | R Prazeres | CC BY-SA 4.0 |
| commons_0074.jpg | [Bab al-Ward doors.jpg](https://commons.wikimedia.org/wiki/File:Bab_al-Ward_doors.jpg) | Robert Prazeres | CC BY-SA 4.0 |
| openverse_0000.jpg | [Moroccan Door 1](https://www.flickr.com/photos/87091938@N00/983241684) | Liamfm . | CC BY-SA 2.0 |
| openverse_0001.jpg | [Moroccan Door 1](https://www.flickr.com/photos/64886991@N00/100118771) | icelight | CC BY 2.0 |
| openverse_0002.jpg | [Moroccan Door 2](https://www.flickr.com/photos/64886991@N00/100118772) | icelight | CC BY 2.0 |
| openverse_0003.jpg | [Moroccan Door 3](https://www.flickr.com/photos/64886991@N00/100118773) | icelight | CC BY 2.0 |
| openverse_0004.jpg | [Moroccan Door 2](https://www.flickr.com/photos/87091938@N00/983241364) | Liamfm . | CC BY-SA 2.0 |
| openverse_0011.jpg | [Moroccan Design](https://www.flickr.com/photos/143850343@N06/36260016012) | soomness | CC BY 2.0 |
| openverse_0019.jpg | [Moroccan Fleur de Lis on a Door-2](https://www.flickr.com/photos/29001414@N00/5116965351) | zeevveez | CC BY 2.0 |
| openverse_0023.jpg | [231ab6e7-1ec2-4b05-b03d-dc73e80d606e](https://www.flickr.com/photos/7605418@N07/2720161381) | Alkan de Beaumont Chaglar | CC BY-SA 2.0 |
| openverse_0028.jpg | [2fdae048-621e-4e4a-b9f9-20c7ddb562c3](https://www.flickr.com/photos/7605418@N07/2720155861) | Alkan de Beaumont Chaglar | CC BY-SA 2.0 |
| openverse_0035.jpg | [Colorful aesthetic Moroccan architecture](https://www.rawpixel.com/image/3388640/free-photo-image-morocco-pattern) | Peace Corps | CC0 1.0 |
| openverse_0037.jpg | [Blue City, Chefchaouene, Morocco， 摩洛哥 - Explore](https://www.flickr.com/photos/68166820@N08/49319209351) | cattan2011 | CC BY 2.0 |
