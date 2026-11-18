#!/bin/bash
# The nationwide US map (usa_study): every county, the census add-on columns (ACS, housing, commute, OD level of service, voting districts and
# precinct elections, densities), the equity thresholds, the map, the checks. One memory-capped process at a time.   ./run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); US=$HERE; LOGS=${LOGS:-$HERE/logs}; mkdir -p $LOGS
CAP="systemd-run --user --scope -q -p MemoryMax=70G -p MemorySwapMax=4G"
cd $US
for k in 1 2 3; do env LANG=C.UTF-8 $CAP python3 -u main.py ALL --skip-map >> $LOGS/usa.log 2>&1 && break; done
for s in add_acs_columns add_housing_columns add_commute_columns add_od_los_columns add_voting_districts add_precinct_elections add_densities add_density_1km_census; do
  $CAP python3 -u $s.py >> $LOGS/usa.log 2>&1 || echo "$s exit $?"
done
python3 check_elections.py >> $LOGS/usa.log 2>&1 || echo "election totals off (see usa.log)"
python3 -c "from code.equity import write_all_thresholds as w; print(w())" >> $LOGS/usa.log 2>&1
LANG=C.UTF-8 $CAP python3 -u build_usa_map.py --out data/_usa > $LOGS/usa_map.log 2>&1 || echo "map exit $?"
python3 rescale_factors_from_logs.py >> $LOGS/usa_map.log 2>&1
python3 $HERE/../checks/check_map_data.py us $US/data > $LOGS/check_us.log 2>&1 && python3 $HERE/../checks/test_map.py $US/data/_usa map.html $LOGS/us.png --wait 45 >> $LOGS/check_us.log 2>&1 \
  && echo "us ok" || echo "us CHECK FAILED"
