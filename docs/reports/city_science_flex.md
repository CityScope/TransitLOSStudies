# The flexible score in the city science cities

State of the fits on 2026-11-25 (second revision). The question: can one smooth score, fitted on observed transit use inside the cities, be
used to rank all the cities, and how well does it describe each city? The code is `city_science_network/flex/fit_cities.py` and the model
`tools/studytools/flexfit.py` (`SmoothModel`); every table is in `city_science_network/flex/FIT_RESULTS.md`. Only current findings are kept.

## The model, as revised

* **The score is computed per H3 resolution-11 cell**, `s = log(1 + A / eps)` with `A` the best station value of the cell, and **a unit takes
  the population-weighted mean of the scores of the cells inside it**. Any geometry works the same way: block group, AGEB, centralidad, district.
* **Modes cannot be out of order**: rail = 1, tram = rail × a factor below 1, bus = tram × a factor below 1.
* **Smooth curves**: headway, speed and walking are log-logistic in the log of their argument with a bounded steepness; the walking curve is
  flat below two cell diameters (115 m at resolution 11) and the headway curve flat below 3 minutes. The walking decay does not depend on the
  service (a free dependence added nothing in the US studies).
* **Speed first, then mode**: the speed curve is fitted with every mode factor equal to 1, so that it captures as much as it can; it is then
  frozen and the mode factors are fitted on what is left. No literature value enters the fit.
* **Shared fits**: the curves and eps are common; each city has its own level, slope of the score (geometric mean 1) and control
  coefficients (controls standardised within the city). **The score used to rank cities contains none of the city's own terms**: it is
  `100 × s / P95` with the shared curves and eps only.
* Every number is held out: within-city 5-fold spatial cross-validation, the curves refitted without the fold. A city that is not a donor of the
  curves only has its level, slope and controls fitted on its own training folds. **R²** is the people-weighted share of the variance of the units'
  transit share explained; **controls only** is the same model with no service term.

The data are as before: commuters by transit against car by residence in the finest geography each city publishes (Boston and San Francisco
block groups, Toronto dissemination areas, Concepción blocks, Guadalajara **centralidades** of the origin-destination survey, 60 of 71 reached by
feeds), and the five small cities (7 to 30 units) only predicted.

## 1. Each city on its own

Its own curves (same model and bounds), its usual controls (density, income, poverty, education, age; the left vote for Guadalajara):

| Held-out R² | Boston | San Francisco | Toronto | Concepción | Guadalajara |
|---|---:|---:|---:|---:|---:|
| units | 3,252 | 4,328 | 9,681 | 10,266 | 60 |
| own curves | 0.655 | 0.553 | 0.646 | 0.566 | 0.482 |
| controls only | 0.538 | 0.515 | 0.510 | 0.541 | 0.397 |
| added by the service | +0.117 | +0.038 | +0.135 | +0.026 | +0.085 |

* **Service matters most in Boston and Toronto** (+0.12 and +0.14), little in San Francisco and Concepción (+0.04, +0.03).
* **The curves fitted on their own differ a lot between cities** (headway 10 min worth 0.61 in Boston, 0.09 in San Francisco, 0.16 in
  Toronto, 0.25 in Concepción), and Guadalajara's, from 60 units, are not interpretable (walking steepness at its bound, speed curve near
  zero). Own curves are good for describing a city, not for comparing cities.

## 2. Curves shared by the cities

| Held-out R² | Boston | San Francisco | Toronto | Concepción | Guadalajara | mean |
|---|---:|---:|---:|---:|---:|---:|
| curves from Boston, San Francisco, Toronto (`large`) | 0.646 | 0.566 | 0.647 | 0.542 | 0.470 | **0.574** |
| + Concepción (`four`) | 0.650 | 0.573 | 0.634 | 0.545 | 0.461 | 0.572 |
| + Guadalajara (`five`) | 0.614 | 0.565 | 0.594 | 0.549 | 0.500 | 0.565 |
| `large`, headway steepness allowed up to 6 | 0.648 | 0.570 | 0.643 | 0.542 | 0.475 | 0.576 |
| `large`, headway curve of the US fit | 0.636 | 0.516 | 0.619 | 0.541 | 0.465 | 0.556 |
| `large`, headway and walking curves of the US fit | 0.613 | 0.537 | 0.590 | 0.544 | 0.386 | 0.534 |
| controls only | 0.535 | 0.516 | 0.511 | 0.542 | 0.392 | 0.499 |
| each city on its own curves | 0.655 | 0.553 | 0.646 | 0.566 | 0.482 | 0.580 |

* **One shared set of curves is as good as each city's own**: the mean falls from 0.580 to 0.574 (San Francisco is even better with shared
  curves, 0.566 against 0.553). Beyond what controls explain (0.499), the shared service term adds 0.075 on average (Boston +0.11, San
  Francisco +0.05, Toronto +0.14, Guadalajara +0.08, Concepción 0.00).
* **Which cities set the curves matters little among the large ones, and adding Guadalajara hurts**: its 60 units pull the speed curve to
  almost nothing and the bus factor to 0.71, and lower Boston (0.614) and Toronto (0.594). The curves of the three large cities are kept.
* **The steep headway curve is what these cities' data prefer, not a limit of the bound**: allowing steepness up to 6 moves it only from 3.0
  to 3.9 with no gain in R². Fixing the headway curve of the US fit (gentle: 20 minutes still worth 0.42) costs 0.018 of mean R², mostly in
  San Francisco (0.566 to 0.516) and Toronto (0.647 to 0.619); fixing the US walking curve as well costs 0.022 more.

### The curves chosen for the ranking (`large`)

| | |
|---|---|
| headway (min) | 3: 1.00 · 5: 0.64 · 10: 0.15 · 20: 0.02 · 30: 0.01 (half value 5.2 min, steepness 3.0) |
| speed (km/h) | 5: 0.53 · 10: 0.65 · 20: 0.78 · 50: 0.92 (half value 7.9 km/h, steepness 0.6) |
| walking (m) | 200: 0.96 · 400: 0.87 · 800: 0.74 · 1,200: 0.65 · 1,600: 0.58 (half distance 1.9 km, steepness 0.93) |
| mode factors | bus 0.94 · tram 0.98 · rail 1 |
| eps | 0.101 |
| city slopes of the score | Boston 1.10 · San Francisco 0.87 · Toronto 1.04 |

The headway curve is a steep, smooth decay: a service every 5 minutes keeps about two thirds of its value, every 10 minutes 15%, and
beyond 20 minutes almost nothing. The speed and walking curves are gentle: speed and walking distance (within 1.5 km) change the value of a
service much less than its frequency does.

## 3. The ranking of the cities

The score of a city is the population-weighted mean of its cells' scores, `100 × s / P95` (P95: the 95th percentile of all the cities' cells),
with the shared curves and nothing of a city's own:

| City | Score | People with a stop within 800 m | Score where served | Best headway within 800 m ≤ 5 min / ≤ 10 min |
|---|---:|---:|---:|---|
| Guadalajara | 75.7 | 0.80 | 87.8 | 0.65 / 0.75 |
| Concepción | 62.8 | 0.63 | 90.2 | 0.50 / 0.60 |
| Taipei | 50.9 | 0.78 | 64.5 | 0.22 / 0.44 |
| Beersheba | 44.9 | 0.73 | 59.2 | 0.25 / 0.52 |
| Toronto | 40.5 | 0.70 | 56.6 | 0.22 / 0.40 |
| Hamburg | 35.7 | 0.73 | 46.4 | 0.20 / 0.35 |
| Gipuzkoa | 29.6 | 0.87 | 33.2 | 0.10 / 0.35 |
| San Francisco | 21.9 | 0.63 | 29.8 | 0.07 / 0.20 |
| Boston | 19.6 | 0.53 | 32.4 | 0.08 / 0.19 |
| Andorra | 17.2 | 0.44 | 27.7 | 0.00 / 0.24 |

* **The ranking is stable**: its rank correlation with the variants is 1.00 (steepness bound 6), 1.00 (curves also from Concepción), 0.95 (US
  headway curve), 0.94 (curves also from Guadalajara) and 0.81 (US headway and walking curves). The bottom three (San Francisco, Boston, Andorra)
  are the same in every variant. The top two are Guadalajara and Concepción in every variant except `five`, where Guadalajara's own curves pull it down
  to the middle (47); with the US headway curve Gipuzkoa rises from seventh to fifth or higher.
* **Read it as the service within reach, not as how much people ride.** The headway of a stop combines every route that serves it, so a dense
  network of overlapping bus routes gives very short headways: in Guadalajara two thirds of the people have a stop within 800 m whose combined
  headway is 5 minutes or less, in Boston 8%. Cities with many overlapping routes (Guadalajara, Concepción) therefore top the ranking, and
  Boston and San Francisco, where about half and over a third of the people have no stop within 800 m, are at the bottom.
* **The ranking depends on the headway curve**: the curve that the cities' own transit use prefers is steep, so frequent overlapping service
  weighs heavily. With the gentler US curve the middle of the ranking is more compressed (Boston 31, Toronto 51, Guadalajara 76).
* The feeds differ in how they describe service (several are built by hand or from OpenStreetMap, with assumed headways: see each city's
  `gtfs_notes`), so a difference of a few points between neighbouring cities should not be read as a difference in service.
* Shanghai has no units to fit on and is not in the ranking.

## 4. The small cities (predicted only)

R² of the logit of the unit's share after the city's level (rank correlation of the score with the share), curves of `large`:

| Beersheba | Gipuzkoa | Taipei | Hamburg | Andorra |
|---:|---:|---:|---:|---:|
| 0.56 (0.76) | 0.81 (0.90) | 0.52 (0.66) | 0.80 (0.93) | 0.31 (0.63) |

The score ranks the units of the small cities well (rank correlations 0.63 to 0.93). The measures differ (trips, vehicles, commuters), so each
city is compared with itself.

## 5. Guadalajara: what level can be fitted

At the AGEB level the survey has about 16 to 23 respondents per unit, so the share of a unit is noisy and the service effect is not
detectable (in a test with the previous model, service added 0.02 over controls at AGEBs, 0.01 at microzones and 0.07 at centralidades). The
centralidades (60 of 71 are reached by feeds; the units no feed reaches are left out) carry it: with the shared curves the held-out R² is
0.470 against 0.392 from the controls (density, education, age and the left vote) alone. The left vote is a Guadalajara control only: its
meaning differs between countries. A full own fit on 60 units is not trustworthy (section 1).

## What is settled and what is open

**Settled**
* One smooth set of curves, with modes in order, fitted on Boston, San Francisco and Toronto, describes all five fitted cities almost as well
  as their own curves, and predicts the small cities.
* The cities' transit use prefers a steep headway curve: frequent service matters, infrequent service almost not, and walking and speed
  matter little within 1.5 km.
* A survey city is fitted on the coarsest survey zones with enough units.

**Open**
* **The cities and the US disagree on the headway curve**: the US fit (238,422 block groups) keeps most of the value of a 20-minute service
  (0.42), the cities almost none (0.02). The US fit and the cities measure differently (the US takes the best station value per cell and
  averages over block groups; the cities average the scores of cells), and the cities have little service between 10 and 30 minutes. A joint fit
  of the US and the cities would show which is right for a ranking; it has not been done.
* The ranking reflects how feeds describe service; a check of the feeds of the top-ranked cities (Guadalajara, Concepción) against their
  real timetables would show how much of their lead is real.
* Guadalajara's suburban feeds are missing (Ixtlahuacán, Zapotlanejo, Juanacatlán, Tlajomulco), so those units are left out rather than explained.
* Making this score the default on the city maps is agreed, not yet applied.
