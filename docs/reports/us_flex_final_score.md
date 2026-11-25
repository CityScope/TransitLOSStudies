# The US flexible score: the final form

State of the fits on 2026-11-25. This report describes the form that is deployed on the US flexible-scoring map, what it was fitted
on, its curves and how well it predicts transit use in places it has not seen. The studies that led to it, and what each tested, are
in [the US studies report](us_flex_studies.md).

## What the score is

The score of a place says how much the transit service reaching it raises the chance that a commuter takes transit instead of the
car. It is fitted on observed commuting, not set from the literature: **no literature value enters the fit**.

* **Outcome.** Commuters by transit divided by commuters by transit plus car (walking, cycling, working from home and other modes
  are ignored), per census block group, from the American Community Survey. Four outcomes share the curves: all transit, bus, rail
  and tram, each against the same transit-plus-car denominator.
* **Data.** 238,422 block groups in 383 metro areas. Each block group is the population-weighted mean of its H3 resolution-10 cells.
* **Service of a cell.** For every station reachable from the cell, the product of four curves: a **mode factor**, a **headway
  curve** (service frequency over the 6 to 22 h window), a **speed curve** (the station's speed: the straight-line distance covered
  in the 15 minutes before and after) and a **walking-distance curve**. The cell's service is the best station's value
  (the maximum), which is the "one distance curve" form: one continuous distance curve for all distances, with no separate
  walking and driving channels.
* **Link to behaviour.** `logit p = level + log(1 + A / eps) + controls`, with `A` the cell's service, so the odds of choosing
  transit are proportional to the service and controls shift them. The part that is not service (controls and the level) is fitted
  together with the curves and is **not** in the score: the score is `100 × log(1 + A / eps) / P95`, capped at 100, so nothing regional
  or socio-economic enters it.
* **The work end.** The service at the workplace (where the residents work, from LODES flows) enters the same way as the service at
  home. It is worth keeping: without it the held-out skill falls from 0.1106 to 0.1051.

### Fitting rules

* The **speed curve is fitted first**, without a mode factor, and then **frozen**; the **mode factor is fitted afterwards** and so
  only takes what speed leaves. The speed curve therefore captures as much as it can and the mode factor does not absorb speed.
* The **walking curve is 1 below 303.5 m** (two H3 cell diameters) and the **headway curve is 1 below 3 minutes**: flat below the
  floor, because the data cannot separate service that is already at its best.
* **Skill** is the held-out improvement over a model with no service effect: metro-grouped 5-fold cross-validation, so every metro
  is predicted by a model fitted without it. Beta-binomial likelihood. Training weights are a 50/50 blend of equal weight per metro
  and weight by population (the blend is itself tested in the studies report).
* Nothing is tuned on the held-out data. A change to the form is kept only when it improves the held-out skill significantly (paired
  bootstrap over metros), and a simplification when it loses very little (under 0.0007) and is not significantly worse.

## The curves (final lean form, "V1")

Values of each curve (1 is the best service; the other curves held at their own value):

| Headway (min) | 2 | 5 | 10 | 15 | 20 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|---|---|
| Headway curve | 1.00 | 0.84 | 0.62 | 0.50 | 0.42 | 0.33 | 0.20 | 0.12 |

| Speed (km/h) | 5 | 10 | 15 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|---|---|
| Speed curve | 0.19 | 0.28 | 0.28 | 0.31 | 0.38 | 0.50 | 0.82 |

| Walking distance (m) | 300 | 400 | 600 | 800 | 1,200 | 1,600 | 2,000 |
|---|---|---|---|---|---|---|---|
| Walking curve | 1.00 | 0.86 | 0.59 | 0.39 | 0.19 | 0.10 | 0.06 |

| Mode | Bus | Tram | Rail |
|---|---|---|---|
| Mode factor (rail = 1) | 0.58 | 0.81 | 1.00 |

* The headway curve is gentle: half value at about 6 minutes, steepness 0.83, so service keeps counting well beyond 10 or 20 minutes.
* The speed curve is low and flat until about 20 km/h and rises towards fast service.
* The walking curve is the same for the three modes in this form (a mode only changes the factor).
* A bus is worth 0.58 of a rail line at equal headway, speed and distance; a tram 0.81.

## The controls kept (8, in 10 columns with their missing-value flags)

Income, poverty share, bachelor's-degree share, and the housing stock: vacancy share, mobile-home share, small multi-family share,
attached-house share and median rooms (income and rooms each have a flag for missing values). Density, jobs, distance to the centre, age,
race, household composition and fuel price were tested and left out, because they add no held-out skill (see the studies report).

## How well it predicts

| Form | Control columns | Held-out skill |
|---|---:|---:|
| **V1, lean** (income, poverty, education, extra housing) | 10 | **0.1106** |
| V2 = V1 + fuel price | 11 | 0.1112 |
| V3, the 36 controls of the control study (car speed included) | 40 | 0.1117 |

The three differ by 0.001, which is within what the tests can separate, so the lean form V1 is the one kept: fewer controls, and
car speed (too close to the outcome) is not among them.

Generalisation to metros not seen by the fit (the binomial skill per worker; 178 metros and counties have 1,000 or more transit
commuters):

| | Pooled held-out | Median metro | Within a metro |
|---|---:|---:|---:|
| V1 | 0.256 | 0.077 | 0.122 |

* **Pooled** is dominated by a few very large metros (New York alone carries about an eighth of the gain).
* The **median metro** gains 0.077 over its own constant share.
* **Within a metro** (each unseen metro's level calibrated away) the score still ranks the places inside it, with skill 0.122: the
  service term carries information about where inside a metro people use transit, not only between metros.

## What it does not explain

* 43% of the pooled gain lies in counties with fewer than 100 transit commuters, mostly "no transit here": a large part of the
  skill is telling where there is no service at all.
* Miami (0.005) and Phoenix (0.015) are almost not explained, against 0.06 for metros with 100 to 100,000 transit commuters.
* The outcome is a commute share by residence: it says nothing about trips that are not commutes, and block-group estimates carry
  the sampling error of the American Community Survey.
