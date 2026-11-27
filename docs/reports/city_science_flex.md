# The flexible score in the city science cities

State of the fits on 2026-11-27 (third revision: all fits rerun on the library `transitlos.flex.fit`, headway steepness at most 2). The question: can one smooth score, fitted on observed transit use inside the cities, be
used to rank all the cities, and how well does it describe each city? The code is `city_science_network/flex/fit_cities.py` and the model is
the library `transitlos.flex.fit` (`ScoreModel`); every table is in `city_science_network/flex/FIT_RESULTS.md`. Only current findings are kept.

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

Its own curves (same model and bounds, headway steepness at most 2), its usual controls (density, income, poverty, education, age; the left vote for Guadalajara only):

| Held-out R² | Boston | San Francisco | Toronto | Concepción | Guadalajara |
|---|---:|---:|---:|---:|---:|
| units | 3,249 | 4,328 | 9,681 | 10,266 | 59 |
| own curves | 0.635 | 0.561 | 0.646 | 0.566 | 0.479 |
| controls only | 0.539 | 0.509 | 0.509 | 0.540 | 0.409 |
| added by the service | +0.096 | +0.052 | +0.137 | +0.026 | +0.070 |

* **Service matters most in Toronto and Boston** (+0.14 and +0.10), little in San Francisco and Concepción (+0.05, +0.03).
* **The curves fitted on their own differ a lot between cities** (headway 10 min worth 0.34 in Boston, 0.25 in San Francisco, 0.18 in
  Toronto, 0.25 in Concepción), and Guadalajara's, from 59 units, are not interpretable (walking curve flat at 1.00, speed curve near
  zero). Own curves are good for describing a city, not for comparing cities.

## 2. Curves shared by the cities

| Held-out R² | Boston | San Francisco | Toronto | Concepción | Guadalajara | mean |
|---|---:|---:|---:|---:|---:|---:|
| curves from Boston, San Francisco, Toronto (`large`) | 0.626 | 0.550 | 0.646 | 0.541 | 0.451 | **0.563** |
| + Concepción (`four`) | 0.633 | 0.557 | 0.631 | 0.545 | 0.440 | 0.561 |
| + Guadalajara (`five`) | 0.604 | 0.557 | 0.592 | 0.549 | 0.481 | 0.557 |
| `large`, headway steepness at most 1.5 | 0.622 | 0.546 | 0.645 | 0.541 | 0.449 | 0.561 |
| `large`, headway steepness at most 1 | 0.618 | 0.538 | 0.640 | 0.541 | 0.450 | 0.557 |
| `large`, headway steepness allowed up to 6 | 0.633 | 0.559 | 0.643 | 0.542 | 0.459 | 0.567 |
| `large`, headway curve of the US fit | 0.614 | 0.510 | 0.623 | 0.540 | 0.449 | 0.547 |
| `large`, headway and walking curves of the US fit | 0.593 | 0.526 | 0.588 | 0.545 | 0.382 | 0.527 |
| controls only | 0.537 | 0.510 | 0.509 | 0.542 | 0.404 | 0.501 |
| each city on its own curves | 0.635 | 0.561 | 0.646 | 0.566 | 0.479 | 0.578 |

* **One shared set of curves is almost as good as each city's own**: the mean falls from 0.578 to 0.563. Beyond what controls explain (0.501),
  the shared service term adds 0.062 on average (Boston +0.09, San Francisco +0.04, Toronto +0.14, Guadalajara +0.05, Concepción 0.00).
* **Which cities set the curves matters little among the large ones, and adding Guadalajara hurts**: its 59 units pull the speed curve to
  almost nothing (a 10 km/h service keeps 0.08) and the bus factor to 0.84, and lower Boston (0.604) and Toronto (0.592). The curves of the three
  large cities are kept.
* **The steep headway curve is what these cities' data prefer, and the bound matters little**: with the steepness limited to 2 the fit sits at the
  bound (2.0); allowed up to 6 it goes to 3.9 and the mean R² rises by 0.004 only; limited to 1.5 or 1 it falls by 0.002 or 0.006. The city default
  is 2 (a curve that cannot become a cliff). Fixing the headway curve of the US fit (gentle: 20 minutes still worth 0.42) costs 0.016 of mean R²,
  mostly in San Francisco (0.550 to 0.510); fixing the US walking curve as well costs 0.020 more.

### The curves chosen for the ranking (`large`, headway steepness at most 2)

| | |
|---|---|
| headway (min) | 3: 1.00 · 5: 0.59 · 10: 0.20 · 20: 0.06 · 30: 0.03 (half value 3.8 min, steepness 2.0) |
| speed (km/h) | 5: 0.54 · 10: 0.66 · 20: 0.78 · 50: 0.92 (half value 7.6 km/h, steepness 0.57) |
| walking (m) | 200: 0.96 · 400: 0.88 · 800: 0.77 · 1,200: 0.68 · 1,600: 0.62 (half distance 2.3 km, steepness 0.91) |
| mode factors | bus 0.96 · tram 0.98 · rail 1 |
| eps | 0.121 |
| city slopes of the score | Boston 1.17 · San Francisco 0.72 · Toronto 1.19 |

The headway curve is a steep, smooth decay: a service every 5 minutes keeps about three fifths of its value, every 10 minutes a fifth, and
beyond 20 minutes very little. The speed and walking curves are gentle: speed and walking distance (within 1.5 km) change the value of a
service much less than its frequency does. The three modes are almost equal once headway, speed and distance are accounted for.

## 3. The ranking of the cities

The score of a city is the population-weighted mean of its cells' scores, `100 × s / P95` (P95: the 95th percentile of all the cities' cells),
with the shared curves and nothing of a city's own:

| City | Score | People with a stop within 800 m | Score where served | Best headway within 800 m ≤ 5 min / ≤ 10 min |
|---|---:|---:|---:|---|
| Guadalajara | 74.3 | 0.79 | 87.1 | 0.64 / 0.73 |
| Concepción | 63.5 | 0.64 | 90.2 | 0.51 / 0.61 |
| Taipei | 51.7 | 0.78 | 65.4 | 0.22 / 0.44 |
| Beersheba | 45.2 | 0.73 | 59.5 | 0.25 / 0.52 |
| Toronto | 41.9 | 0.70 | 58.3 | 0.22 / 0.40 |
| Hamburg | 37.4 | 0.73 | 48.6 | 0.20 / 0.35 |
| Gipuzkoa | 31.3 | 0.81 | 37.9 | 0.10 / 0.34 |
| Shanghai | 26.1 | – | 42.9 | – |
| San Francisco | 22.2 | 0.59 | 32.2 | 0.06 / 0.18 |
| Boston | 20.4 | 0.52 | 34.2 | 0.07 / 0.18 |
| Andorra | 20.2 | 0.44 | 32.5 | 0.00 / 0.24 |

* **The ranking is stable**: its rank correlation with the variants is 1.00 (steepness up to 6), 0.99 (curves also from Concepción; steepness at most
  1.5 or 1), 0.95 (curves also from Guadalajara; US headway curve) and 0.93 (US headway and walking curves). San Francisco, Boston and Andorra are the
  bottom three in every variant, except the two with the US headway curve, where Shanghai joins them in place of San Francisco. The top two are
  Guadalajara and Concepción in every variant except `five`, where Guadalajara's own curves pull it to third (49; Taipei leads at 51).
* **Read it as the service within reach, not as how much people ride.** The headway of a stop combines every route that serves it, so a dense
  network of overlapping bus routes gives very short headways: in Guadalajara almost two thirds of the people have a stop within 800 m whose combined
  headway is 5 minutes or less, in Boston 7%. Cities with many overlapping routes (Guadalajara, Concepción) therefore top the ranking, and
  Boston, San Francisco and Andorra, where about half of the people (48% in Boston, 41% in San Francisco, 56% in Andorra) have no stop within 800 m, are at the bottom.
* **The ranking depends on the headway curve**: the curve that the cities' own transit use prefers is steep, so frequent overlapping service
  weighs heavily. With the gentler US curve the middle of the ranking is more compressed (Boston 30, Toronto 50, Guadalajara 74).
* The feeds differ in how they describe service (several are built by hand or from OpenStreetMap, with assumed headways: see each city's
  `gtfs_notes`), so a difference of a few points between neighbouring cities should not be read as a difference in service.
* Shanghai has no units to fit on: it is scored with the shared curves and shown for the service it has, not tested against transit use.

## 4. The small cities (predicted only)

R² of the logit of the unit's share after the city's level (rank correlation of the score with the share), curves of `large`:

| Beersheba | Gipuzkoa | Taipei | Hamburg | Andorra |
|---:|---:|---:|---:|---:|
| 0.55 (0.76) | 0.75 (0.91) | 0.52 (0.65) | 0.75 (0.92) | 0.28 (0.63) |

The score ranks the units of the small cities well (rank correlations 0.63 to 0.92). The measures differ (trips, vehicles, commuters), so each
city is compared with itself.

## 5. Guadalajara: what level can be fitted

At the AGEB level the survey has about 16 to 23 respondents per unit, so the share of a unit is noisy and the service effect is not
detectable (in a test with the previous model, service added 0.02 over controls at AGEBs, 0.01 at microzones and 0.07 at centralidades). The
centralidades (59 of 71 are used: the units no feed reaches are left out) carry it: with the shared curves the held-out R² is
0.451 against 0.404 from the controls (density, education, age and the left vote) alone. The left vote is a Guadalajara control only: its
meaning differs between countries. A full own fit on 59 units is not trustworthy (section 1).

## What is settled and what is open

**Settled**
* One smooth set of curves, with modes in order, fitted on Boston, San Francisco and Toronto, describes all five fitted cities almost as well
  as their own curves, and predicts the small cities.
* The cities' transit use prefers a steep headway curve: frequent service matters, infrequent service almost not, and walking and speed
  matter little within 1.5 km.
* A survey city is fitted on the coarsest survey zones with enough units.

**Open**
* **The cities and the US disagree on the headway curve**: the US fit (238,422 block groups) keeps most of the value of a 20-minute service
  (0.42), the cities very little (0.06). The US fit and the cities measure differently (the US takes the best station value per cell and
  averages over block groups; the cities average the scores of cells), and the cities have little service between 10 and 30 minutes. A joint fit
  of the US and the cities would show which is right for a ranking; it has not been done.
* The ranking reflects how feeds describe service; a check of the feeds of the top-ranked cities (Guadalajara, Concepción) against their
  real timetables would show how much of their lead is real.
* Guadalajara's suburban feeds are missing (Ixtlahuacán, Zapotlanejo, Juanacatlán, Tlajomulco), so those units are left out rather than explained.
* Making this score the default on the city maps is agreed, not yet applied.
