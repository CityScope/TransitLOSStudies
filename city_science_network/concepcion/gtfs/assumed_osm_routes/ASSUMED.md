# ASSUMED feed: Gran Concepción routes from OpenStreetMap + fleet register

Not official data. Services that the DTPR Gran Concepción feed does not contain: it has 6 stops in Coronel and none in Lota, Tomé or Santa Juana, and no line 44.
Routes written: 300 ida: Lota → Concepción, 300 regreso: Concepción → Lota, 44A: Hualpencillo → Lomas, 44B: Lomas → Hualpencillo, 44E: Hualpén → Lomas, Caleta Chome → Talcahuano, Coronel-Lota (Vuelta), Hualqui → Talcamávida, Talcahuano → Caleta Chome, Talcamávida → Hualqui.

GEOMETRY: OSM route relations (ways + stop members; one stop every 600 m when the relation has no stops). Relations that are letter variants of a DTPR line already in the feed are skipped; line 44 (Flota Centauro, 44A/44B/44E) is not in the feed, so it is kept.
The OSM 'Coronel-Lota (Vuelta)' relations are 0.5-1.9 km fragments and are not used.

HEADWAY_BASIS (never the feed's medians):
- 300 Lota-Concepción: Seremi de Transportes guide, 3-5 min after the first departure (Lota 05:07, Concepción 05:20; 2023): 5 min, 05:00-22:00. The fleet value (39 buses) would give 8 min.
- 302, 303, 304: same corridor as the 300 (Wikipedia, 'Transporte público en el Gran Concepción': Parque Lota/Lota - Coronel - San Pedro de la Paz - Collao); geometry cloned from the 300. Headway = round-trip time / buses in service, from the fleet register (Flota P.E. Gran Concepción, Lota - Coronel y Tomé 2024 v3, 2024-08-10): 302 Sotral 88 buses -> 3 min, 303 Playa Blanca 52 -> 6 min, 304 Expresos del Carbón 21 -> 15 min. 8% of the fleet is in the workshop and 9% is spare; layover 10% of running time; 25 km/h.
- Local Coronel-Lota (Sotral 44, Nueva Takora 39, Nueva Ruta 160 43, Playa Colcura 28 = 154 buses): the Lota-Coronel stretch of the 300 geometry (22 km round trip) at 20 km/h; the fleet alone gives 2 min, floored at 3 min because these operators also run neighbourhood branches and part of the fleet goes on to Concepción. 06:00-22:00 (Sotral night run 22:00 from Lota Bajo, 2026; Playa Colcura 07:00-20:00, Subtrans 2023).
- 44 Flota Centauro: 38 buses over the three OSM variants (88 km of round trips) at 20 km/h -> 9 min, 05:30-22:00.
- Rural CTR0222 Talcahuano-Caleta Chome: MTT, the subsidised bus reaches the cove twice a week -> one trip every 12 h. CTR0201 Hualqui-Talcamávida: no timetable found, 3 trips a day (assumption).
Not covered (no route or fleet-to-line data found): Los Tilos 10 buses, Nueva Los Tilos 49, Ruta Costa 4.
