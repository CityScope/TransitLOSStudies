# City science flex study

Fits the flexible transit score (mode x headway curve x speed curve x distance curve, the form fitted on the US block groups in
`usa_flex`) to the observed transit use of the city science cities, below the municipality. Two questions:

1. Does the US form carry over? The US curves are scored on each city's own transit use (held-out, nothing refitted but the city's level).
2. What do the cities say themselves? The same form fitted to the cities pooled, each city with its own level, held out city by city.

The fitting rules are those of the US study: the speed curve is fitted first without a mode factor and then frozen, the mode factor
after it; the curves are flat below two H3 cell diameters (walk) and the minimum headway (3 min); no literature values enter the fit;
only controls that add held-out skill are kept.

## Transit use below the municipality, per city

`flex/outcomes.py` writes `<city>/results/flex_outcomes.parquet`: one row per unit of the finest published geography, with its transit
share, the commuters or trips behind it, and the surveyed sample where the source gives it.

| City | Source | Unit (count) | Measure |
|---|---|---|---|
| Boston | ACS 5-year | block group (3,510) | commuters by mode, residence |
| San Francisco | ACS 5-year | block group (4,332) | commuters by mode, residence |
| Toronto | Statistics Canada census 2021 | dissemination area (9,712) | commuters by main mode, residence |
| Concepcion | INE Censo 2024 | block (manzana) / rural entity (10,470) | commuters by mode, residence |
| Beersheba | CBS census 2022 public-use file per sub-quarter | sub-quarter / locality (30) | commuters by main mode, residence |
| Guadalajara | IMEPLAN Encuesta Origen-Destino 2023 | centralidad of residence (71; the 60 that feeds reach are fitted; the survey's AGEBs and microzones are too noisy to fit, see `docs/reports/city_science_flex.md`) | surveyed workers by main mode of the first work trip (24,152 surveyed) |
| Gipuzkoa | PMUS San Sebastian 2024-2029 neighbourhood sheets (EM CAPV 2021 + 2022 oversample); outside Donostia the Estudio de la Movilidad de la CAPV 2021 origin-destination matrices | neighbourhood of Donostia (15), comarca elsewhere | weekday trips by mode, all purposes, by origin |
| Taipei | Taipei DOT daily mode-use survey 2018; in New Taipei the MOTC survey 2010 (Taipei DOT report 102-03); Keelung and Taoyuan publish city totals only | district of Taipei City (12), traffic zone of New Taipei (7) | vehicles used on commuting / school trips |
| Hamburg | MiD 2023 (the city's modal-split WFS); outside the city the MiD 2017 Metropolregion report | Bezirk (7), Kreis (13) | residents' trips by main mode |
| Andorra | ENM 2050 citizen survey 2021 | parish (7) | respondents' modes the day before |
| Shanghai | -- | -- | nothing below the city is published |

Notes on the measures:

* Commuters by residence (Boston, San Francisco, Toronto, Concepcion, Beersheba, Guadalajara) are the outcome of the US fit. The
  trip-based units (Donostia by origin, Hamburg and Andorra by residence) and Taipei's vehicle counts measure something else: each
  city keeps its own level and is compared on its own, not pooled with the commuters.
* Guadalajara has about 16 to 23 surveyed commuters per AGEB (about 280 per centralidad): `outcomes.py` therefore takes the centralidad, the
  coarsest survey level with enough units; a unit weighs the people it stands for (`n_tc`, expanded) and the likelihood counts its surveyed sample (`n_sample_tc`). The left vote is an extra control for Guadalajara only
  (`CITY_EXTRA` in `fit_tests.py`) and units with a default score under 5 (no feed reaches them) are left out (`NO_FEED_BELOW`).
  Its AGEBs are the survey's own 2020 urban AGEBs, then INEGI's urban and rural AGEB layers.
* Finer data exist only on request: MobiHam 2022 microdata with the Stadtteil of residence (Hamburg, 104 Stadtteile), the MOTC survey
  microdata with the district (Taipei, later years than 2018), the EM CAPV microdata (Gipuzkoa, municipalities above 10,000
  inhabitants), the TTS 2022 by traffic zone (Toronto, trips next to the census commuters).

The fits use transit against the car only (`share_tc` = transit / (transit + car)), as the US fit.

## Steps

1. `flex/run_city_flex.py`: station profiles, street-network search and the res-11 cell / station table of `transitlos.flex` on each city's
   feeds and streets (20 service options per station, 15 stations per cell); `flex/build_city_flex_web.py`: the flex map.
   The flex map has an editor (the "Edit" button; `tools/flex_web/flex_edit.js` on `transitlos/flex/frontend/edit.js`): routes and stations
   are changed in the browser and the score recomputed there. Every change is a factor on whole routes: speed (all travel times divided),
   headway (the headway at every station multiplied; the new timetable runs from the departure closest to the middle of the 6-22 h window at
   the new interval), mode, deletion; a station's table edits one route or all its routes (of one mode) together; new lines add stations (a
   point within 150 m of a station joins it; a new station reaches the cells within 1.5 km of street distance, estimated as the straight
   distance times the city's detour factor). The stations an edit touches get their profiles rebuilt from their departures exactly as
   `transitlos.flex.profiles` builds them; only the cells listing those stations are rescored. `run_city_flex.py` saves the departures per
   station and route, `build_city_flex_web.py` exports them (`transitlos.flex.export.export_routes`).
2. `flex/outcomes.py`: the outcome table per city (above).
3. `flex/build_fit_data.py`: per city the outcome units, their populated res-11 cells (the cell centre in the unit; within 100 m for units
   drawn without the streets), the cells' service items and the units' controls (population-weighted over the cells) -> `<city>/flex/fit/`.
4. `flex/fit_cities.py` (model: `SmoothModel` in `tools/studytools/flexfit.py`): each fitted city on its own, the curves shared by the cities
   (pools of donor cities and sensitivity variants), the ranking of the cities from the shared score, and the service each city has;
   `fit_cities.py report` writes `flex/FIT_RESULTS.md`. The score is computed per H3 res-11 cell and a unit takes the population-weighted mean of
   its cells' scores; modes are in order (rail >= tram >= bus); the curves are smooth with bounded steepness; speed is fitted first, then the modes.
   `flex/fit_tests.py` supplies the data loading and the scoring helpers (and the earlier, unsmoothed tests).