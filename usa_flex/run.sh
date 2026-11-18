#!/bin/bash
# The US flexible-scoring map and the data of the score fits (usa_flex, stop-buffer variant). Needs the US map (usa_study/run.sh) first: it reads its
# counties, census blocks and street networks.   ./run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); F=$HERE; LOGS=${LOGS:-$HERE/logs}; mkdir -p $LOGS
cd $F
python3 -u run_us_buffered.py > $LOGS/flex.log 2>&1 || echo "flex states exit $?"            # every state (memory scheduler), web_buf/
python3 -u build_flex_page.py ../usa_study/data/_usa >> $LOGS/flex.log 2>&1                    # the flex tab of the US map
# fit inputs: the national block-group table (ACS, LODES), then the fit data from the flex build
python3 -u analysis/fetch_acs_bg.py >> $LOGS/fitdata.log 2>&1 && python3 -u analysis/fetch_lodes_od.py >> $LOGS/fitdata.log 2>&1
python3 -u analysis/build_bg_table.py >> $LOGS/fitdata.log 2>&1 && python3 -u analysis/enrich_bg_table.py >> $LOGS/fitdata.log 2>&1
export FLEX_VARIANT=buf TQDM_DISABLE=1
for s in build_dataset save_flows build_items build_blocks build_bg_extra build_extra_controls build_extra_controls3; do
  systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=4G python3 -u fit/$s.py > $LOGS/fd_$s.log 2>&1 || { echo "fit data failed at $s"; exit 1; }
done
echo "fit data done: run fit/score_study/ctrl_study7.sh then fit/score_study/ctrl_study7b.sh"
