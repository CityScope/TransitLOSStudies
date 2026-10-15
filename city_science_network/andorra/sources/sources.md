https://www.busos.ad/ (This web shows realtime gtfs data. Maybe you could reconstruct a gtfs from this web and the paper timetables)
HEre you can find the timetables https://bus.ad/ 
TREBALL FI DE GRAU
Grau en Enginyeria de l’Energia
ANÀLISI DELS COMPONENTS BÀSICS PER AL DISSENY D’UN
MODEL DE MOBILITAT TERRESTRE AL PRINCIPAT
D’ANDORRA
Memòria i Annexos
Autor: Claudi Bové Gresa Director: Pablo Buenestado Caballero Co-Director: Oriol Travesset Baró Convocatòria: Juny 2019

--- 2026-08-10 GTFS build ---
No official GTFS exists for Andorra's E.N.A. network, and busos.ad's live
data is behind a client-rendered Next.js app with no discoverable public
API for stops/timetables (checked the JS bundles directly, no fetch
endpoints found). bus.ad publishes per-line operating-hours + headway text
(no stop-by-stop timetable, no stop list) for lines L1-L7, fetched
2026-08-10.

Built a schematic GTFS by hand from that headway/hours data plus known
parish-capital geography for stop placement: `andorra/sources/build_andorra_gtfs.py`
generates `andorra/gtfs/ena_manual_reconstruction/` using GTFS
`frequencies.txt` (headways/windows straight from bus.ad, accurate) with
running times between stops estimated from great-circle distance at an
assumed 18 km/h (approximate, not measured -- real stop-by-stop timetables
were not available). Loads successfully via `pyGTFSHandler.feed.Feed`.

--- 2026-08-10 follow-up: found the official timetable PDF ---
Found the actual official source: Govern d'Andorra publishes a leaflet PDF
at https://www.govern.ad/documents/d/guest/bus_linies_nacionals (found via
https://www.govern.ad/ca/tematiques/accio-climatica/transports/linies-nacionals/horaris),
"El temps es teu" / Agost 2026 edition -- one page per line (Le, L1-L7) with
the exact official headway schedule per direction and day-type, plus a
stop-sequence diagram with real numeric stop codes per line. This confirmed
the bus.ad headway text was accurate, and resolved the "L1-L8" naming seen
on busos.ad's title: the 8th line is "Le", a Fener <-> Pont de Fontaneda
shuttle not listed among L1-L7 on bus.ad's menu -- there is no separate L8.
(Also checked interurbana.ad, which hosts a stale 2019 summer timetable for
L1 only -- not used, superseded by the Aug-2026 official PDF.)

Rebuilt `build_andorra_gtfs.py` on this: frequencies.txt now uses the exact
official per-direction headway windows (previously the same window was
applied to both directions; the PDF shows they differ slightly), added the
Le line, and the ~11 representative stops per line now carry real official
stop codes/names from the PDF (e.g. 5555 = Estacio Nacional, 720 = Caldea)
instead of invented ones -- though still only the terminals/major
interchanges, not the full 20-30-stop official sequence per line. Stop
coordinates remain approximate town-centre positions (PDF has no
coordinates). One-off reinforcement departures outside the regular headway
pattern are not modelled.