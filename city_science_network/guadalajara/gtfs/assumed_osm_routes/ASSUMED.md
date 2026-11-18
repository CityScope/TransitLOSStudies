# ASSUMED feed: Guadalajara routes from OpenStreetMap

Not official data. OSM bus route relations of the Mi Transporte network whose ref is not in the official feed (mdb-2366): 175-Roca, C11, C12, C14, MP-A05-2, MP-T04, ST-L1B, T13A-1, T13B-C01-2, T13B-C02, T13C-C04, T18-Lopez, T18-Terranova, T19-C06.
Route and stops come from the OSM relation; there is no timetable, so every route has an assumed 20-minute headway from 06:00 to 21:00.

HEADWAY_BASIS: these lines are far less important than the lines of the official feed and must never look better than them, so the value is NOT the feed's own class median (complementary lines 4.8 min, trunk 2.9 min). 20 min is the headway the state itself announces for its newer outer routes: T21 Zapotlanejo-Guadalajara, 12 buses, 20 min (El Ciudadano Jalisco 2023) and a new feeder in north Zapopan, 20 min (Informador 2026-11-04); routes already running at 10-12 min (Tren Ligero feeders) are the better-served ones.
