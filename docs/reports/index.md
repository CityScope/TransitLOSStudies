# Flexible-scoring fits: reports

State of the fits on 2026-11-25. Three reports with the current findings (earlier iterations that were superseded are not included):

1. [The US flexible score: the final form](us_flex_final_score.md): what the score is, what it is fitted on, its curves and how well it
   predicts transit use in metros it has not seen.
2. [The US flexible score: what the studies found](us_flex_studies.md): the studies behind the final form (speed and mode, the
   controls, the shape of the curves, carrying over to unseen metros).
3. [The flexible score in the city science cities](city_science_flex.md): whether the US form carries over to Boston, San Francisco,
   Toronto, Concepción and Guadalajara, one shared score, the five small cities, and the level of the survey in Guadalajara.

## The two sets of fits side by side

The US fit and the city fits use the same form and rules (transit against car, speed curve first and then the mode factor, walking
curve flat below 303.5 m, headway curve flat below 3 min, no literature values). Their scores of fit are not comparable (the US skill
is a beta-binomial improvement over no service effect; the cities report the R² of the held-out transit share), but their curves are:

| Curve value | US (238,422 block groups) | Boston | San Francisco | Toronto | Concepción | Shared by the three large cities |
|---|---:|---:|---:|---:|---:|---:|
| headway 10 min | 0.62 | 0.46 | 0.38 | 0.35 | 0.27 | 0.48 |
| headway 20 min | 0.42 | 0.11 | 0.04 | 0.16 | 0.13 | 0.18 |
| walk 400 m | 0.86 | 0.62 | 0.28 | 0.87 | 0.68 | 0.46 |
| walk 800 m | 0.39 | 0.38 | 0.00 | 0.63 | 0.47 | 0.23 |
| bus relative to rail | 0.58 | 0.97 | 0.92 | 0.82 | 0.86 | 0.94 |

* **Headway.** The cities, fitted on their own, lose the value of a service faster than the US fit: at 20 minutes a US service keeps
  0.42 of its value, a city's 0.04 to 0.16. The three large cities pooled (0.18) sit between. The cities' curves are alike one another
  (10 minutes: 0.27 to 0.46), so the difference is between the US pooled over 383 metros and single cities.
* **Walking.** The least stable curve: San Francisco loses the service by 400 m (0.28 at 400 m, 0.00 at 800 m), while the US fit and the other cities keep 0.38 to 0.63 of it at 800 m.
* **Mode.** In the cities a bus is worth 0.82 to 0.97 of a rail line at equal service; the US fit has 0.58. (The US mode factor is the
  value at equal headway, speed and distance, fitted after the speed curve took its share.)
* **What is decided from this.** Shared curves with the city's own level, slope and controls are the way to score the cities, because
  only a city with thousands of units can fit curves of its own, and a city's survey has to be coarse enough to carry the service effect
  (Guadalajara: centralidades, not AGEBs).

## Where each is built

| | Code | Data |
|---|---|---|
| US fit | `usa_flex/fit/torch_fit4.py`, studies in `usa_flex/fit/score_study/` | `usa_flex/fit/data_buf/` |
| City fits | `city_science_network/flex/fit_tests.py`, model `tools/studytools/flexfit.py` | `city_science_network/flex/results/`, `FIT_RESULTS.md` |
