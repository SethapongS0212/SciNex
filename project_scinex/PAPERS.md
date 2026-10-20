# The corpus — 20 papers

Every paper the C1–C6 experiment was run on. All twenty come from **R. Chawuthai's group**
and span applied machine learning across ophthalmology, chemistry, materials science,
traffic and ITS, cloud systems, computer vision and NLP — deliberately diverse, so the
evaluation is not measuring one narrow domain.

**The PDFs are not in this repository.** Ten of the twenty are paywalled (IEEE, Springer,
Elsevier, ACM), so redistributing them here would not be lawful. Every DOI is listed below;
`fetch_corpus.sh` downloads the open-access subset automatically.

`paper_id` is the PDF stem and propagates through every output path —
`papers/strabismus2026.pdf` → `output/strabismus2026/`. Do not rename a PDF after parsing.

---

## Summary

| | |
|---|---|
| Papers | **20** |
| Paragraphs extracted from | **528** |
| Extractors | 4 (Qwen3-235B, GPT-OSS-120B, Gemma-4-31B, Ministral-14B) |
| Ontologies | 2 (CEO, scinex) |
| Triples produced | **6,014** (3,065 CEO + 2,949 scinex) |
| Judged | 408 items, single rater (claude-opus-5, in session) |

---

## The papers

Triple counts are the sum across all four extractors, over the 20-paper corpus.

| # | paper_id | Year | Venue | Access | CEO | scinex |
|---|---|---|---|---|---|---|
| 1 | `osmotic2026` | 2026 | Scientific Reports | Open access | 142 | 172 |
| 2 | `strabismus2026` | 2026 | Scientific Reports | Open access | 347 | 344 |
| 3 | `gamlprop2025` | 2025 | Chemical Engineering Transactions | Open access | 72 | 76 |
| 4 | `llamacorrupt2025` | 2025 | ASSE (ACM) | Author-supplied | 94 | 103 |
| 5 | `oxidecrack2025` | 2025 | Scientific Reports | Open access | 170 | 188 |
| 6 | `pesticide2025` | 2025 | Measurement | Author-supplied | 175 | 202 |
| 7 | `vehiclemake2025` | 2025 | IEEE Access | Author-supplied | 367 | 342 |
| 8 | `videoseg2025` | 2025 | ECTI-CON | Author-supplied | 145 | 123 |
| 9 | `csysguard2024` | 2024 | IEEE Access | Author-supplied | 280 | 242 |
| 10 | `eyelandmark2024` | 2024 | ECTI-CON | Author-supplied | 98 | 92 |
| 11 | `ugmo2024` | 2024 | ITC-CSCC | Author-supplied | 57 | 41 |
| 12 | `parkingyolo2023` | 2023 | Sensors | Open access | 219 | 199 |
| 13 | `textaug2023` | 2023 | ECTI-CON | Author-supplied | 81 | 59 |
| 14 | `traveltime2022` | 2022 | Applied Sciences | Open access | 334 | 305 |
| 15 | `trafficspeed2021` | 2021 | ICEAST | Author-supplied | 52 | 51 |
| 16 | `tripplanner2020` | 2020 | ACIIDS (Springer) | Author-supplied | 95 | 87 |
| 17 | `microwave2018` | 2018 | ICEAST | Author-supplied | 58 | 52 |
| 18 | `reststop2018` | 2018 | Transportation Research Procedia | Author-supplied | 58 | 48 |
| 19 | `roadwaylight2018` | 2018 | Sensors & Materials | Open access | 79 | 59 |
| 20 | `linkpred2015` | 2015 | JIST (Springer) | Author-supplied | 142 | 164 |

### Titles and DOIs

1. **`osmotic2026`** — A machine learning approach for predicting osmotic coefficients and deriving activity coefficients in alkyl ammonium salts  
   *Scientific Reports*, 2026. DOI: [10.1038/s41598-026-36758-x](https://doi.org/10.1038/s41598-026-36758-x)

2. **`strabismus2026`** — Utilizing deep learning from mobile phone photos for early detection of horizontal strabismus: a screening approach  
   *Scientific Reports*, 2026. DOI: [10.1038/s41598-026-48893-6](https://doi.org/10.1038/s41598-026-48893-6)

3. **`gamlprop2025`** — Integration of Genetic Algorithm with Machine Learning for Properties Prediction  
   *Chemical Engineering Transactions*, 2025. DOI: [10.3303/CET25117170](https://doi.org/10.3303/CET25117170)

4. **`llamacorrupt2025`** — Assessing the Effects of Corrupted Parameters in a Large Language Model: A Case Study of LLAMA 3.2 1B  
   *ASSE (ACM)*, 2025. DOI: [10.1145/3775030.3775040](https://doi.org/10.1145/3775030.3775040)

5. **`oxidecrack2025`** — Novel method for predicting the cracks of oxide scales during high temperature oxidation of metals and alloys by using machine learning  
   *Scientific Reports*, 2025. DOI: [10.1038/s41598-025-91449-3](https://doi.org/10.1038/s41598-025-91449-3)

6. **`pesticide2025`** — Development of machine learning enhanced low-cost spectrophotometer for pesticide prediction  
   *Measurement*, 2025. DOI: [10.1016/j.measurement.2025.114890](https://doi.org/10.1016/j.measurement.2025.114890)

7. **`vehiclemake2025`** — Minimizing model size of CNN-based Vehicle Make Recognition for Frontal Vehicle Images  
   *IEEE Access*, 2025. DOI: [10.1109/ACCESS.2025.3574187](https://doi.org/10.1109/ACCESS.2025.3574187)

8. **`videoseg2025`** — A Comparative Study of Video Segmentation Techniques for Graduate Detection in Thai Graduation Ceremonies  
   *ECTI-CON*, 2025. DOI: [10.1109/ECTI-CON64996.2025.11100560](https://doi.org/10.1109/ECTI-CON64996.2025.11100560)

9. **`csysguard2024`** — Stateless System Performance Prediction and Health Assessment in Cloud Environments: Introducing cSysGuard, an Ensemble Modeling Approach  
   *IEEE Access*, 2024. DOI: [10.1109/ACCESS.2024.3406670](https://doi.org/10.1109/ACCESS.2024.3406670)

10. **`eyelandmark2024`** — Eye Landmarks Detection using RT-DETR with Rules  
   *ECTI-CON*, 2024. DOI: [10.1109/ECTI-CON60892.2024.10594871](https://doi.org/10.1109/ECTI-CON60892.2024.10594871)

11. **`ugmo2024`** — U-GMo: Individual Clip Detection from a Graduation Ceremony Video  
   *ITC-CSCC*, 2024. DOI: [10.1109/ITC-CSCC62988.2024.10628160](https://doi.org/10.1109/ITC-CSCC62988.2024.10628160)

12. **`parkingyolo2023`** — Parking time violation tracking using YOLOv8 and tracking algorithms  
   *Sensors*, 2023. DOI: [10.3390/s23135843](https://doi.org/10.3390/s23135843)

13. **`textaug2023`** — Improving a text classifier using text augmentation: road traffic content from Twitter  
   *ECTI-CON*, 2023. DOI: [10.1109/ECTI-CON58255.2023.10153191](https://doi.org/10.1109/ECTI-CON58255.2023.10153191)

14. **`traveltime2022`** — Travel time prediction on long-distance road segments in Thailand  
   *Applied Sciences*, 2022. DOI: [10.3390/app12115681](https://doi.org/10.3390/app12115681)

15. **`trafficspeed2021`** — Spatial-temporal traffic speed prediction on Thailand roads  
   *ICEAST*, 2021. DOI: [10.1109/ICEAST52143.2021.9426257](https://doi.org/10.1109/ICEAST52143.2021.9426257)

16. **`tripplanner2020`** — A Recommender System for Trip Planners  
   *ACIIDS (Springer)*, 2020. DOI: [10.1007/978-981-15-3380-8_43](https://doi.org/10.1007/978-981-15-3380-8_43)

17. **`microwave2018`** — The analysis of a microwave sensor signal for detecting a kick gesture  
   *ICEAST*, 2018. DOI: [10.1109/ICEAST.2018.8434455](https://doi.org/10.1109/ICEAST.2018.8434455)

18. **`reststop2018`** — A Hybrid Method for Predicting a Potential Next Rest Stop of Commercial Vehicles  
   *Transportation Research Procedia*, 2018. DOI: [10.1016/j.trpro.2018.11.011](https://doi.org/10.1016/j.trpro.2018.11.011)

19. **`roadwaylight2018`** — Monitoring Roadway Lights and Pavement Defects for Nighttime Street Safety Assessment by Sensor Data Analysis and Visualization  
   *Sensors & Materials*, 2018. DOI: [10.18494/SAM.2018.1842](https://doi.org/10.18494/SAM.2018.1842)

20. **`linkpred2015`** — Link prediction in linked data of interspecies interactions using hybrid recommendation approach  
   *JIST (Springer)*, 2015. DOI: [10.1007/978-3-319-15615-6_9](https://doi.org/10.1007/978-3-319-15615-6_9)

---

## Two papers that are NOT in the corpus

Worth recording, because an earlier write-up mistakenly described the corpus as 22 papers.

- **`aiabstract2025`** — ISCON (IEEE), 2025. RETIRED SUBSTITUTE for tripplanner2020; stood in only while that PDF was unobtainable. The original was recovered 2026-09-03, so this paper is NOT part of the 20-paper corpus. Parsed output kept on disk.
- **`routepred2023`** — ICCAE (IEEE), 2023. RETIRED SUBSTITUTE for linkpred2015; stood in only while that PDF was unobtainable. The original was recovered 2026-09-03, so this paper is NOT part of the 20-paper corpus. Parsed output kept on disk.

`aiabstract2025` and `routepred2023` were added as **substitutes** while `tripplanner2020` and
`linkpred2015` were unobtainable. Those two PDFs were later supplied, so the substitutes were
**retired** — but for a while both sets were kept, which is how the corpus was briefly and
wrongly counted as 22. The canonical list is `papers_20.txt`; regenerate it by excluding
manifest rows whose `note` **starts with** `SUBSTITUTE for`. Filtering on the substring
`SUBSTITUTE` drops the wrong two, because the recovered papers say `SUBSTITUTED` in their own
notes.

---

## Getting the PDFs

```bash
bash fetch_corpus.sh      # downloads the open-access subset, verifies each is a real PDF
```

The rest must be obtained through your own institutional access and saved as
`papers/<paper_id>.pdf`, using **exactly** the id in the table above — a wrong filename
silently forks the output tree.

> **Verify what you download.** A DOI suffix is not the publisher's PDF number: guessing
> `SM1842.pdf` from DOI `10.18494/SAM.2018.1842` returns a valid PDF of a *different paper*
> (the real one is `SM1674.pdf`). PDF text streams are compressed, so `grep` cannot confirm a
> title — check `main.py`'s parsed `Title:` line against this table on the first parse run.
