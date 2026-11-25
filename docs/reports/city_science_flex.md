# The flexible score in the city science cities

State of the fits on 2026-11-25. The question: does the transit score fitted on US block groups (see [the US final score](us_flex_final_score.md))
describe transit use inside other cities, and what do the cities say when they are fitted themselves? The code is
`city_science_network/flex/` (`outcomes.py`, `build_fit_data.py`, `fit_tests.py`; the model is `tools/studytools/flexfit.py`) and its
tables are in `city_science_network/flex/FIT_RESULTS.md`. This report keeps the current findings.

## What is measured

Observed transit use below the municipality, in the finest geography each city publishes. The outcome is transit divided by transit plus
car (the US definition), for commuters by residence, except where the source measures trips.

| City | Unit | Units | Source |
|---|---|---:|---|
| Boston | block group | 3,510 | ACS 5-year |
| San Francisco | block group | 4,332 | ACS 5-year |
| Toronto | dissemination area | 9,712 | Statistics Canada census 2021 |
| Concepción | block (manzana) / rural entity | 10,470 | INE Censo 2024 |
| Guadalajara | **centralidad** of the origin-destination survey (60 of 71 reached by feeds) | 71 | IMEPLAN Encuesta Origen-Destino 2023 |
| Beersheba, Gipuzkoa, Taipei, Hamburg, Andorra | sub-quarters, neighbourhoods, districts, boroughs, parishes | 30, 26, 19, 20, 7 | local surveys and censuses |
| Shanghai | nothing below the city is published | – | – |

The first four cities and Guadalajara are **fitted**; the five small cities are only **predicted** (too few units to fit). Every city has
its own **level** (some cities use transit more at the same service); controls enter standardised within each city (density, income,
poverty, education and age, those the city has, with the left vote added for Guadalajara). Everything is held out: within a city by
5-fold spatial cross-validation, across cities by fitting on some and predicting another. **R²** is the people-weighted share of the
variance of the units' transit share explained.

## 1. Do the US curves carry over?

With the US final curves and each city's own level refitted, and then with also the city's own slope (how strongly service counts there):

| Held-out R² | Boston | San Francisco | Toronto | Concepción |
|---|---:|---:|---:|---:|
| US curves, level refitted | 0.44 | 0.22 | 0.42 | −0.05 |
| US curves, level and slope refitted | 0.60 | 0.32 | 0.55 | 0.06 |

The US curves describe the three large cities reasonably once each city's slope is allowed. Concepción's transit use is almost
unrelated to service on its own (0.06): see section 2.

## 2. Each city fitted on its own

| Held-out R² | Boston | San Francisco | Toronto | Concepción |
|---|---:|---:|---:|---:|
| service only | 0.65 | 0.46 | 0.66 | 0.01 |
| service and the usual controls | 0.66 | 0.58 | 0.69 | 0.57 |

* In the three large cities **service alone explains most** of what service and controls explain together. In **Concepción** it explains
  nothing alone and the controls (density, education, age) carry the whole 0.57: the service curve there is not identified apart from them.
* Fitted on their own, with controls, the **curves are alike in headway and differ in walking**:

| Curve value | Boston | San Francisco | Toronto | Concepción |
|---|---:|---:|---:|---:|
| headway 5 min | 0.88 | 0.91 | 0.69 | 0.58 |
| headway 10 min | 0.46 | 0.38 | 0.35 | 0.27 |
| headway 20 min | 0.11 | 0.04 | 0.16 | 0.13 |
| headway 30 min | 0.04 | 0.01 | 0.09 | 0.08 |
| walk 400 m | 0.62 | 0.28 | 0.87 | 0.68 |
| walk 800 m | 0.38 | 0.00 | 0.63 | 0.47 |
| mode factor bus (rail = 1) | 0.97 | 0.92 | 0.82 | 0.86 |
| mode factor tram | 0.97 | 1.80 | 1.11 | 0.82 |

All four cities lose most of the value of a service between 5 and 20 minutes of headway. The walking curve is the least stable: San
Francisco reaches almost nothing beyond 400 m, Toronto still values 800 m. The mode factors are close to 1 (a bus is worth 0.8 to 0.97 of
rail at equal service) except for San Francisco's trams.

## 3. One city's curves predicting another

Held-out R² when the curves (and controls' form) come from the city in the row and are used in the city of the column, with the
column's own level, dispersion and control coefficients refitted:

| From \ to | Boston | San Francisco | Toronto | Concepción |
|---|---:|---:|---:|---:|
| Boston | 0.66 | 0.55 | 0.67 | 0.30 |
| San Francisco | 0.64 | 0.58 | 0.63 | 0.39 |
| Toronto | 0.63 | 0.50 | 0.69 | 0.08 |
| Concepción | 0.56 | 0.53 | 0.54 | 0.57 |

Curves carry between the three large cities with a loss of 0.02 to 0.08 against fitting the city itself. They carry poorly into Concepción
(0.08 to 0.39), where the service effect is not identified.

## 4. One score for the cities

The curves (headway, walking, speed, mode, eps) shared; each city its own level, slope of the score and control coefficients. The
version retained takes the curves from the three large cities (Boston, San Francisco, Toronto), puts the tram halfway between bus and
rail, and keeps the city slopes around 1. Held-out R², the other cities with the curves fitted on the three:

| | Boston | San Francisco | Toronto | Concepción | Guadalajara (centralidades) |
|---|---:|---:|---:|---:|---:|
| one score, with controls | 0.654 | 0.553 | 0.642 | 0.542 | **0.455** |
| the city fitted on its own, with controls | 0.664 | 0.575 | 0.693 | 0.567 | – |
| one score, no controls | 0.633 | 0.357 | 0.606 | −0.002 | −0.060 |

* **One shared score is almost as good as fitting each city**: the four cities lose 0.01 to 0.05 of R² (mean 0.598 against 0.625).
* The curves are sensible: headway 5 min 0.84, 10 min 0.48, 20 min 0.18, 30 min 0.09; walking 200 m 0.77, 400 m 0.46, 800 m 0.23,
  1,200 m 0.15; speed 0.38 at 5 km/h to 0.78 at 50 km/h; bus 0.94, tram 0.97, rail 1. Slopes: Boston 1.13, San Francisco 0.82, Toronto 1.08.
* **Fitting all curves freely on the cities gives a step headway curve** (1 at 5 min, 0 at 10 min), so those variants are not used: the
  cities are too few and too different to pin the curve shapes freely.
* Without controls the score alone is good in Boston and Toronto (0.63, 0.61), weaker in San Francisco (0.36), and unrelated to transit
  use in Concepción and Guadalajara: there the controls carry the fit.

### The small cities (predicted only)

R² of the logit of the unit's share after the city's level (and, in brackets, the rank correlation of the predicted service with the share):

| Beersheba | Gipuzkoa | Taipei | Hamburg | Andorra |
|---:|---:|---:|---:|---:|
| 0.53 (0.78) | 0.76 (0.92) | 0.48 (0.67) | 0.87 (0.92) | 0.42 (0.67) |

The score ranks the units of the small cities well (rank correlations 0.67 to 0.92) and explains between 0.42 and 0.87 of the variance of
their transit shares. The measures differ (trips, vehicles, commuters), so each city is compared with itself.

## 5. Guadalajara: the level of the survey decides what can be fitted

The survey has about 16 to 23 respondents per AGEB (a median of about 280 per centralidad), so the transit share of a fine unit is noisy:
a model that predicted the true shares perfectly would still reach an R² of only about 0.81 to 0.84 against the observed ones. Guadalajara
fitted at three levels, held-out R² (5 spatial folds, all units):

| Level | Units | Controls only | + US curves, level | + slope | + left vote | Full own fit |
|---|---:|---:|---:|---:|---:|---:|
| AGEB | 1,045 | 0.12 | 0.12 | 0.14 | 0.24 | 0.10 |
| Microzone | 512 | 0.16 | 0.13 | 0.17 | 0.28 | 0.19 |
| **Centralidad** | **71** | 0.34 | **0.42** | 0.40 | **0.46** | 0.52 |

* **What service adds over the controls**: +0.02 at AGEBs, +0.01 at microzones, **+0.07 at centralidades**. Part of the rise of R² at
  coarser levels is just less sampling noise; the service effect itself is only visible at the coarsest level.
* **Headway at the centralidad level is identified**: the best curves have a half value of 5 to 10 minutes (0.40 to 0.42); half values of
  30 to 60 minutes are clearly worse (0.36 to 0.37); steepness is barely identified. The North American curve (half value 8.7 min) is within
  0.01 of the best. At AGEBs and microzones every curve is within 0.01 of the others: no information.
* **A full own fit on 71 units is not trustworthy**: it reaches 0.52 held out, but refitting on all 71 with three starts gives a headway
  step of 27 to 60, walking curves that are flat or a cliff, and buses worth 1.6 times rail. The few-parameter fit (shared curves, only the
  level, the slope and the controls fitted) is stable at every level.
* **The left vote is a Guadalajara control**: it adds 0.06 to 0.11 at every level (not a shared control: its meaning differs between countries).
* **Units the feeds do not reach are left out.** Municipalities with no feed at all (Ixtlahuacán, and others) have a transit share of
  about 0.35 and a score of 0; 104 AGEBs score below 15, two thirds of which reach no stop. Transit share rises from 0.35 at a score near 2
  to 0.55 at about 50 and is flat above. Wealth proxies add 0 to 0.03.
* **The retained fit**: centralidades with a score of 5 or more (60 units), the curves from the three large cities, Guadalajara's own level and
  slope, the usual controls and the left vote: held-out **R² 0.455** (0.148 on AGEBs before).
* **A data bug was found and fixed on the way**: feeds that write a timetable as frequency windows exactly one headway wide (6,094 of 9,166
  windows in Guadalajara) had them dropped, leaving some trips at the template's time. Fixed in pyGTFSHandler; only 9% of Guadalajara's
  stations changed headway and the units' scores are almost unchanged. Shanghai (16,580 windows) and Taipei (150) have the same pattern
  and were rebuilt.

## What is settled and what is open

**Settled**
* Curves fitted on the three large cities carry to the others with controls, almost as well as each city's own fit.
* The headway curve is similar across cities; the walking curve is not.
* A city's curves can be fitted only where there are many units with little noise (Boston, San Francisco, Toronto); the others get shared
  curves and their own level, slope and controls.
* Survey cities are evaluated on the coarsest survey zones with enough units (centralidades for Guadalajara).

**Open**
* Guadalajara's suburban feeds (Ixtlahuacán, Zapotlanejo, Juanacatlán, Tlajomulco) are missing: the units there are left out, not explained.
* The other Guadalajara tests (its own fit, transfer, pooled) have not been rerun at the centralidad level; only the one-score test has.
* The shared score as the default on the city maps (the curves from Boston, San Francisco and Toronto) is agreed but not yet applied to them.
