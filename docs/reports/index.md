# Flexible-scoring fits: reports

State of the fits on 2026-11-25. Three reports with the current findings (earlier iterations that were superseded are not included):

1. [The US flexible score: the final form](us_flex_final_score.md): what the score is, what it is fitted on, its curves and how well it
   predicts transit use in metros it has not seen.
2. [The US flexible score: what the studies found](us_flex_studies.md): the studies behind the final form (speed and mode, the
   controls, the shape of the curves, carrying over to unseen metros).
3. [The flexible score in the city science cities](city_science_flex.md): whether the US form carries over to Boston, San Francisco,
   Toronto, Concepción and Guadalajara, one shared score, the five small cities, and the level of the survey in Guadalajara.

## The two sets of fits side by side

The US fit and the city fits use the same rules (transit against car, speed curve first and then the mode factor, walking curve flat below its
floor, headway curve flat below 3 min, no literature values). The city fits are the revised, smooth ones (modes in order, the unit's score the
mean of its cells' scores). Their scores of fit are not comparable (the US skill is a beta-binomial improvement over no service effect; the cities
report the R² of the held-out transit share), but their curves are:

| Curve value | US (238,422 block groups) | Boston | San Francisco | Toronto | Concepción | Shared by the three large cities |
|---|---:|---:|---:|---:|---:|---:|
| headway 10 min | 0.62 | 0.61 | 0.09 | 0.16 | 0.25 | 0.15 |
| headway 20 min | 0.42 | 0.16 | 0.01 | 0.03 | 0.12 | 0.02 |
| walk 400 m | 0.86 | 0.50 | 0.56 | 0.95 | 0.71 | 0.87 |
| walk 800 m | 0.39 | 0.31 | 0.36 | 0.88 | 0.30 | 0.74 |
| bus relative to rail | 0.58 | 0.51 | 0.67 | 0.96 | 0.60 | 0.94 |

* **Headway.** The cities lose the value of a service much faster than the US fit: at 20 minutes a US service keeps 0.42 of its value, a
  city's 0.01 to 0.16, and the three large cities pooled 0.02. The disagreement is not an artefact of the cities' bound on the curve's steepness
  (see the city report), and it is the main open question: the US takes the best station per cell and averages over block groups, the cities
  average the scores of cells, and the cities have little service between 10 and 30 minutes.
* **Walking.** The least stable curve, and the walking floors differ (303.5 m in the US fit, 115 m in the cities): from a steep fall (the US,
  Boston, San Francisco, Concepción) to almost none (Toronto, and the three large cities pooled).
* **Mode.** The mode factor of a bus ranges from 0.51 to 0.96 of rail in the cities and is 0.94 pooled; the US fit has 0.58. (It is the value at
  equal headway, speed and distance, fitted after the speed curve took its share.)
* **What is decided from this.** Shared curves with the city's own level, slope and controls describe the cities as well as curves of their own (mean
  held-out R² 0.574 against 0.580) and are the basis for ranking them; only cities with thousands of units can fit curves of their own, and a city's
  survey has to be coarse enough to carry the service effect (Guadalajara: centralidades, not AGEBs). The resulting ranking is of service within
  reach, not of how much people ride.

## Where each is built

| | Code | Data |
|---|---|---|
| US fit | `usa_flex/fit/torch_fit4.py`, studies in `usa_flex/fit/score_study/` | `usa_flex/fit/data_buf/` |
| City fits | `city_science_network/flex/fit_tests.py`, model `tools/studytools/flexfit.py` | `city_science_network/flex/results/`, `FIT_RESULTS.md` |
