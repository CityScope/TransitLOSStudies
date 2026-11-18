#!/bin/bash
# The 11 city maps (city_science_network): every city from its feeds, census and OSM (streets cached), then the modal-split table and the
# checks. Memory-aware: at most 2 cities at once, 40 GB kept free.   ./run.sh [city ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd); C=$HERE; LOGS=${LOGS:-$HERE/logs}; mkdir -p $LOGS
CITIES=${*:-"andorra beerseba boston concepcion guadalajara hamburg san_francisco shanghai taipei toronto gipuzkoa"}
cd $C && LANG=C.UTF-8 python3 -u run_all.py --reserve-gb 40 --max-parallel 2 $CITIES > $LOGS/cities.log 2>&1 || echo "run_all exit $?"
LANG=C.UTF-8 python3 -u code/modal_split.py >> $LOGS/cities.log 2>&1
for c in $CITIES; do
  python3 $HERE/../checks/check_map_data.py city $C/$c > $LOGS/check_$c.log 2>&1 && python3 $HERE/../checks/test_map.py $C $c/map.html $LOGS/$c.png --wait 30 >> $LOGS/check_$c.log 2>&1 \
    && echo "$c ok" || echo "$c CHECK FAILED (see $LOGS/check_$c.log)"
done
