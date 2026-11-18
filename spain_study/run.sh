#!/bin/bash
# The nationwide Spain map (spain_study): the 52 provinces (NAP feeds only), the MITMA mobility, densities and 1 km census densities, the map,
# the rescale factors, the checks. One memory-capped process at a time.   ./run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ES=$HERE; LOGS=${LOGS:-$HERE/logs}; mkdir -p $LOGS
CAP="systemd-run --user --scope -q -p MemoryMax=70G -p MemorySwapMax=4G"
cd $ES
for k in 1 2 3; do env LANG=C.UTF-8 $CAP python3 -u main.py ALL --skip-map >> $LOGS/spain.log 2>&1 && break; done   # a run killed for memory resumes
for s in add_mitma_mobility add_densities add_density_1km_census; do python3 -u $s.py >> $LOGS/spain.log 2>&1 || echo "$s exit $?"; done
LANG=C.UTF-8 systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=2G python3 -u build_spain_map.py > $LOGS/spain_map.log 2>&1 || echo "map exit $?"
python3 rescale_factors_from_logs.py >> $LOGS/spain_map.log 2>&1
python3 $HERE/../checks/check_map_data.py spain $ES/data > $LOGS/check_spain.log 2>&1 && python3 $HERE/../checks/test_map.py $ES/data/_spain map.html $LOGS/spain.png --wait 40 >> $LOGS/check_spain.log 2>&1 \
  && echo "spain ok" || echo "spain CHECK FAILED"
