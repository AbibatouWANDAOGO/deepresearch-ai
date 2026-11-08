# Evaluation `baseline-rules`

10 of 10 researches completed, depth `quick`, at most 12 sources each.

| Metric | Value |
|---|---|
| Key points in the report (coverage) | 32 % |
| Key points found anywhere in the research | 64 % |
| Sub-questions answered with evidence | 75 % |
| Pages fetched successfully | 73 % |
| Sources cited per report | 3.2 |
| Claims checked / supported | 23 / 23 |
| Retrieval: origin chunk among passages checked (recall) | 96 % |
| Retrieval: mean reciprocal rank of the origin chunk | 0.786 |
| Citations: every number of the claim is in the passage | 96 % |
| Citations: share of the claim's words in the passage | 88 % |
| Citations: supported by another source than the one it came from | 39 % |
| Citations: passages cited per claim | 3.1 |
| Citations: cited passages that contain most of the claim's words | 84 % |
| Citation precision, model judge (23 claims) | 76 % |
| Citation precision, manual annotation (23 claims) | 87 % |
| Judge agrees with the annotation | 52 % |
| Fact Checker status agrees with the annotation | 56 % |
| Duration per research, mean / max | 16.5 min / 19.5 min |
| Model calls, tokens in / out | 82, 141,139 / 25,306 |
| Cost | $0.00 |

## Time per step (mean)

| Step | Time |
|---|---|
| analyze | 5.5 min |
| index | 2.3 min |
| plan | 0.6 min |
| fact_check | 4.5 min |
| write | 0.7 min |
| collect | 0.6 min |
| research | 1.8 min |

## Per question

| Question | Coverage | Sources cited | Checked / supported | Citation precision | Duration | Missing |
|---|---|---|---|---|---|---|
| battery-prices | 20 % | 1 | 1 / 1 | 100 % | 17.0 min | economies of scale, energy density, raw material costs, size of the fall |
| heat-pump | 20 % | 5 | 1 / 1 | 50 % | 16.5 min | fluide frigorigène, compresseur, coefficient de performance, 3 à 4 fois plus d'énergie restituée |
| mrna-vaccines | 60 % | 4 | 4 / 4 | 75 % | 15.2 min | lipid nanoparticles, immune response |
| sea-level | 20 % | 3 | 3 / 3 | 83 % | 16.1 min | dilatation thermique, fonte des glaciers, calottes polaires, rythme de la hausse |
| crispr | 60 % | 2 | 4 / 4 | 75 % | 11.7 min | DNA repair, bacterial origin |
| nuclear-power | 60 % | 5 | 3 / 3 | 50 % | 19.5 min | coût et délais, production pilotable |
| transformer | 60 % | 3 | 3 / 3 | 67 % | 16.6 min | 2017 paper, parallel processing |
| gdpr | 0 % | 3 | 1 / 1 | 100 % | 17.0 min | consentement et base légale, délégué à la protection des données, notification des violations, droits des personnes, sanctions |
| svb-collapse | 0 % | 4 | 1 / 1 | 100 % | 17.7 min | rising interest rates, losses on bonds, bank run, uninsured deposits, concentrated tech clientele |
| inflation-france-2022 | 20 % | 2 | 2 / 2 | 100 % | 17.9 min | guerre en Ukraine, prix alimentaires, reprise après la pandémie, niveau de l'inflation |
