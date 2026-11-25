# The US flexible score: what the studies found

State of the fits on 2026-11-25. These are the studies behind the [final form](us_flex_final_score.md) (studies 7 to 10 of the US
score fits; earlier generations are superseded and not reported). The scripts are `usa_flex/fit/score_study/ctrl_study*.sh`, the
model `usa_flex/fit/torch_fit4.py`.

## How every study is run

* One change at a time to a reference form, refitted on all 238,422 block groups of 383 metro areas.
* **Held-out skill**: metro-grouped 5-fold cross-validation, so each metro is predicted by a fit that never saw it. Skill is the
  improvement over a model with no service effect.
* Differences are tested with a **paired bootstrap over metros** (`ctrl_rank.py`). A change that adds something is kept only when the
  gain is significant; what adds nothing, or very little, is left out ("what adds nothing is not included").
* Fixed by decision, not tested: walking curve flat below 303.5 m, headway curve flat below 3 min, no literature values in the fit,
  raw speed (car speed is not a control), the speed curve fitted first and then frozen, and only then the mode factor.
* A difference of about 0.001 in skill is within what the tests separate: read tables with that in mind.

## 1. Speed and mode (study 7)

All runs keep the same controls (the base plus car commute speed); only the speed and mode terms differ.

| Speed term | No mode factor | With mode factor |
|---|---:|---:|
| none | 0.1046 | 0.1113 |
| speed curve | 0.1058 | 0.1115 |
| speed curve, monotone | 0.1073 | 0.1112 |
| one log-linear exponent (power) | 0.1054 | 0.1114 |
| an exponent per mode | 0.1062 | 0.1117 |
| exponent by the length of the commute | – | 0.1115 |
| speed relative to the block group's car speed | 0.1051 | 0.1111 |

* **The mode factor carries most of the gain** (0.1046 to 0.1113). **Speed without a mode factor recovers only part of it**
  (0.1054 to 0.1073): in these data the mode of a service and its speed are largely the same information.
* **Speed adds almost nothing on top of the mode factor** (at most +0.0004: 0.1113 to 0.1117).
* The fits that put as much as possible into speed first and only then fit the mode factor on what is left (sequential, speed terms
  frozen) reach 0.1115 to 0.1117, so the speed curve can be made to take its share without losing skill. That is the form kept.
* In the final forms (study 8) the same comparison, with identical controls, is: speed alone 0.1081 / 0.1073 / 0.1073 against speed
  and then mode 0.1106 / 0.1112 / 0.1117 for the three control sets: the mode factor still adds 0.0025 to 0.0044 after speed.

## 2. Which controls (studies 7 and 8)

Controls are fitted with the curves, so that a place's service is not credited with what wealth or housing explain; they are not part
of the score. Skill with one group left out (reference 0.1113, all groups):

| Group left out | Skill | | Group left out | Skill |
|---|---:|---|---|---:|
| extra housing (vacancy, mobile homes, small multi-family, attached, rooms) | 0.1065 | | age | 0.1116 |
| income, poverty, bachelor's degree | 0.1092 | | density (7 radii) | 0.1117 |
| fuel price | 0.1102 | | household composition | 0.1117 |
| housing type and age | 0.1104 | | race, foreign-born | 0.1121 |
| jobs density | 0.1113 | | distance to the centre, metro size | 0.1122 |

* **Only some groups matter** (loss on removal, with the study's paired tests): extra housing 0.0048 (p 0.004), income, poverty and
  education 0.0022 (p 0.009), fuel price 0.0011 (p 0.044), housing type and age 0.0009 (not significant). Car speed as a control
  0.0000. Removing density, age, race, household composition, distance to the centre or metro size changes nothing, or slightly
  improves the held-out skill.
* **Density in place of the housing controls loses**: 0.1076 against 0.1111 (p 0.002). Of the single density radii, 1 km is the best
  (0.1123); the others are about 0.1113.
* **Fuel price** added +0.0006 (not significant) once income, poverty, education and the extra housing are in, and the speed curve
  **relative to car speed** added +0.0004 (not significant).
* The three forms kept: **V1 lean** (income, poverty, education, extra housing) 0.1106; **V2** = V1 + fuel price 0.1112; **V3** all
  36 controls of study 7, car speed included, 0.1117. The three are within 0.001.

## 3. The shape of the curves (study 9)

Each open question as one change to V1 (0.1106), speed curve fitted first, then the mode factor:

| Change | Skill | Against V1 |
|---|---:|---:|
| speed curve: log-logistic / power | 0.1105 / 0.1105 | −0.0001 |
| headway curve: free-form monotone / wait-time form | 0.1101 / 0.1106 | −0.0005 / 0.0000 |
| **walking curve: free-form decreasing** | **0.1121** | **+0.0015** |
| walking curve: exponential | 0.1107 | +0.0001 |
| **walk decay no longer depends on service (simpler)** | **0.1112** | **+0.0006** |
| station values combined by the sum / by a PTAL-like sum | 0.1044 / 0.1066 | −0.0062 / −0.0040 |
| without the work end (jobs side) | 0.1051 | −0.0055 |
| speed not clipped | 0.1106 | 0.0000 |

* **Combining stations by the best one** (the maximum) is clearly better than summing them (−0.006) or a PTAL-like sum (−0.004).
* **The service at the workplace matters**: dropping it costs 0.0055.
* The curve shapes do not matter much: the parametric curves are as good as free-form ones, except that a free-form decreasing walking
  curve gains 0.0015.
* The two changes that did not lose (the free walking curve and the flat decay) together give V1 0.1112, V2 0.1111 and V3 0.1107,
  against 0.1106, 0.1112 and 0.1117 before: **no real change**, so the deployed curves are unchanged.

## 4. Carrying over to metros the fit has not seen (study 10)

Binomial skill per worker, from the held-out predictions of the 5-fold fit, for the lean form V1 (178 metros and counties with
1,000 or more transit commuters in the last two columns):

| Variant | Pooled held-out | Median metro | Within a metro |
|---|---:|---:|---:|
| **V1** | **0.2558** | **0.0772** | **0.1219** |
| training weight: equal per metro | 0.2394 | 0.0523 | 0.1172 |
| training weight: plain population | 0.2624 | 0.0879 | 0.1196 |
| + metro mean of the access term | 0.2555 | 0.0796 | 0.1205 |
| + metro mean of every control | 0.2561 | 0.0802 | 0.1203 |
| + both | 0.2557 | 0.0821 | 0.1208 |

* **Within a metro** the score ranks places inside metros the fit never saw (0.122), so the service term carries information about
  where people use transit, not only about which metros do.
* **No variant improves the within-metro skill.** Giving every metro equal training weight loses everywhere. Weighting by plain
  population raises the pooled skill (+0.0066, p 0.000) and the median metro, but lowers the within-metro skill: it learns the big
  metros. Adding metro-level context (the metro's mean access or mean controls) does not help: the pooled and within-metro
  skills change by under 0.002 (not significant) and the median metro moves by 0.002 to 0.005 (no test).
* **Where the skill is**: 43% of the pooled gain is in counties with fewer than 100 transit commuters (mostly "no transit here");
  metros with 100 to 100,000 transit commuters gain about 0.06; Miami gains 0.005 and Phoenix 0.015; New York alone is 12% of the gain.

## What is settled and what is open

**Settled**: the form (best-station maximum, mode × headway × speed × walking curves, workplace end, link `log(1 + A / eps)`); speed
first and mode after; controls income, poverty, education and the extra housing group; the curves above.

**Open**: why Miami and Phoenix are not explained (service data, or a different way of using transit); whether a free-form walking
curve is worth its freedom (+0.0015); a metro-level structure that improves the within-metro skill (none of the variants of study 10
did); and the carry-over to other countries, in [the city science report](city_science_flex.md).
