# ASSUMED feed: Transportes Tomé 401 and 411

Not official data.
Build the ASSUMED supplementary GTFS feeds of the city-science cities (lines that no official feed we could download contains).

Every feed written here is marked so that it can always be told apart from official data:
  * the feed folder is named `assumed_*`; it contains `ASSUMED.md` (sources, method, what is an assumption) and is listed in `assumed_feeds/registry.json`;
  * `feed_info.txt`: feed_publisher_name starts with "ASSUMED";
  * `agency.txt`: agency_name starts with "ASSUMED";
  * `routes.txt`: route_id starts with "assumed_" and route_desc starts with "ASSUMED:".

Feeds
  guadalajara/gtfs/assumed_t21_moovit_osm: Mi Transporte T21/C01 (Central de Autobuses - Tonala - Puente Grande - Zapotlanejo). Stop NAMES and order from the Moovit
      line PDF (65 + 72 stops); stop LOCATIONS = the OpenStreetMap bus stop with the same name chosen by dynamic programming so the stops follow one corridor
      (names that no OSM stop carries are left out); the path between stops follows the OSM road network; headway 20 min, 05:00-21:00 / 06:00-22:00 (El Ciudadano
      Jalisco, launch of the route, 2023).
  concepcion/gtfs/assumed_tome_401_411: Transportes Tome lines 401 (Alto Tome - Concepcion) and 411 (Dichato - Tome - Concepcion), DTPR exclusion-perimeter services
      that the DTPR Gran Concepcion feed does not contain. Terminals from OpenStreetMap, path = shortest road path, stops = OSM bus stops within 35 m of the path.
      Headways: 401 every 5 min at peak / 7 min otherwise (Seremi de Transportes note; Tome bus federation 2020), 411 hourly (Rome2Rio listing).

Usage (from city_science_network/): `python assumed_feeds/build_assumed_feeds.py [guadalajara] [concepcion]`
