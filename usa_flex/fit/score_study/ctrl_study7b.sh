#!/bin/bash
# 2026-11-17 study 7 (replaces the rest of study 6): the control + speed study rerun on the fit data rebuilt with the new station speed (user: speed = straight-line distance 15 min before / after; "find ways of integrating the speed and properly separating the effect of mode and speed"). Earlier stages: torch_runs/cs_tripspeed/.
# "stability and generalisability across metro areas is important", "look at pycensus ... more controls", the drop "has to be significant",
# "possible too close to outcome car columns and housing price and rent").
# Final form (B, one distance curve, 3-min headway floor, no metro intercepts), warm start, 1 start, 6000 steps; metro-grouped 5-fold held-out
# skill, per-metro held-out losses (paired bootstrap tests: ctrl_rank.py) and each fold's curves. Waits for the map rebuild (US build: 95 GB).
# Study 7 (user 2026-11-17): "I want car speed as control" -- every stage keeps the car commute speed (base = sp_ref = B0 + car speed);
# "see if speed and no mode works well" -- speed forms without a mode factor, against no speed / no mode and against the mode factor alone;
# "make speed capture the most effect possible and mode avoid capturing any speed effect" -- sequential fits: the speed terms of the no-mode
# fit frozen, the mode factor then only takes what speed leaves (seq_*); the exact split of each fitted mode+speed form (mode factor
# projected on log speed, ctrl_split.py) is computed afterwards from the saved fits.
HERE=$(cd "$(dirname "$0")" && pwd); S=${LOGS:-$HERE/logs}; mkdir -p $S; L=$S/ctrl_study7b.log; F=$(cd "$HERE/../.." && pwd)
INIT=$F/fit/init/g5_controls_w_Bf_one_plus_nore.pt   # the warm start (the final form fitted before this study), kept in the repository
# the base: the 25 controls without car commute speed (too close: tested apart), + 2 / 4 / 5 km density, + housing age 1980 / 2000 / median year
BASE=log_pop_density_2km,log_pop_density_4km,log_pop_density_5km,built_pre1980_share,built_post2000_share,year_built_median
NOCAR=log_car_speed
log() { echo "$* $(date '+%F %T')" >> $L; }
log "study 7b (speed-first form: car-relative speed, cold-start stability) started"
fit() {   # fit <tag> [VAR=value ...]
  tag=$1; shift
  [ -f $F/fit/data_buf/torch_runs/g5_controls_w_cs_$tag.json ] && { log "$tag exists"; return; }
  cd $F && env FLEX_VARIANT=buf FIT4_SUFFIX=_cs_$tag FIT4_RUN_STARTS=1 FIT4_STEPS=6000 FIT4_INIT=$INIT FIT4_DECAY=service FIT4_H_FLOOR=3 FIT4_ONECURVE=1 \
    FIT4_CONTROLS_PLUS=1 FIT4_NO_METRO_RE=1 FIT4_NO_SPEED=1 FIT4_CTRL_STUDY=1 FIT4_CTRL_ADD=$BASE FIT4_CTRL_DROP= "$@" \
    systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=2G ${TORCH_PYTHON:-python3} -u fit/torch_fit4.py national g5_controls_w > $S/cs_$tag.log 2>&1 < /dev/null
  log "$tag exit $? $(grep -o 'cv {.*}' $S/cs_$tag.log)"
}
run() {   # run <tag> [VAR=value ...]: fit in the background, at most $(cat ctrl_par) at once
  while [ $(jobs -rp | wc -l) -ge $(cat $S/ctrl_par 2>/dev/null || echo 1) ]; do sleep 20; done
  fit "$@" &
  sleep 90   # stagger the data loading
}
G='{"density":["log_pop_density","log_pop_density_500m","log_pop_density_1km","log_pop_density_2km","log_pop_density_3km","log_pop_density_4km","log_pop_density_5km"],
"jobs":["log_job_density","log_jobs_5km"],"geography":["log_dist_cbd","log_metro_pop"],"class":["log_income","poverty_share","bachelor_share"],
"age":["under18_share","over65_share"],"race":["black_share","hispanic_share","asian_share","foreign_born_share"],
"household":["household_size","children_hh_share","living_alone_share"],
"housing":["multifamily10_share","detached_share","built_pre1950_share","built_pre1980_share","built_post2000_share","year_built_median"],"fuel":["log_fuel_price"],
"income_dist":["income_gini","lowinc_share","highinc_share","snap_share"],
"occupation":["service_occ_share","production_occ_share","management_occ_share","female_worker_share"],
"vulnerability":["disability_share","uninsured_share","limited_english_share","noncitizen_share","veteran_share","no_computer_share"],
"housing2":["vacancy_share","mobile_home_share","small_multi_share","attached_share","rooms_median"]}'
grp() { python3 -c "import json;print(','.join(json.loads('''$G''')['$1']))"; }
ALL0=$(python3 -c "import json;g=json.loads('''$G''');print(','.join(c for k in ['density','jobs','geography','class','age','race','household','housing','fuel'] for c in g[k]))")
ADD=log_pop_density_2km,log_pop_density_4km,log_pop_density_5km,built_pre1980_share,built_post2000_share,year_built_median,vacancy_share,mobile_home_share,small_multi_share,attached_share,rooms_median   # the stage-B base chosen by study 5 (base + housing2), kept
C="FIT4_CTRL_ADD=$ADD"   # sp_ref: the stage-B base + car commute speed (kept in every run of this study)
R=$F/fit/data_buf/torch_runs
export CTRL_BASE=sp_ref
# Study 7b (user 2026-11-18): the speed-first form is preferred (seq_mono_clip: speed fitted without a mode factor and frozen, then the mode
# factor); "what about controlling speed fit with car speed?" -- the same with transit speed relative to the block group's car commute speed
# (FIT4_SPEED_CAR: 20 x v / car speed), against the raw-speed version; then the winner's stability: 3 cold starts x 12000 steps of the
# no-mode speed fit, then the mode factor on top (seq stays warm: its speed terms are the cold fit's, frozen).
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do sleep 60; done
MONO="FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono FIT4_SPEED_CLIP=1"
fit spv_nomode_mono_car_clip $C $MONO FIT4_SPEED_CAR=1 FIT4_NO_MODE=1
[ -f $R/g5_controls_w_cs_spv_nomode_mono_car_clip.pt ] && fit seq_mono_car_clip $C $MONO FIT4_SPEED_CAR=1 FIT4_INIT=$R/g5_controls_w_cs_spv_nomode_mono_car_clip.pt FIT4_FREEZE=th_s,c_v
# the car-relative version replaces the raw one only when it is significantly better (held-out, paired bootstrap over metros)
W=$(cd $HERE && python3 -c "
import ctrl_rank as c
d, p = c.test('seq_mono_clip', 'seq_mono_car_clip') if (c.R / 'g5_controls_w_cs_seq_mono_car_clip.json').exists() else (0, 1)
print('car' if d > 0 and p < 0.05 else 'raw')" 2>> $L)
log "speed-first form: $W"
CAR=""; [ "$W" = car ] && CAR="FIT4_SPEED_CAR=1"
COLD="FIT4_RUN_STARTS=3 FIT4_STEPS=12000 FIT4_INIT="
fit final_nomode_mono $C $MONO $CAR FIT4_NO_MODE=1 $COLD
[ -f $R/g5_controls_w_cs_final_nomode_mono.pt ] && fit final_seq_mono $C $MONO $CAR FIT4_INIT=$R/g5_controls_w_cs_final_nomode_mono.pt FIT4_FREEZE=th_s,c_v
python3 $HERE/ctrl_rank.py table > $S/ctrl_table7b.txt 2>&1
log "study 7b done"
