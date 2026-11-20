#!/bin/bash
# Study 8 (user 2026-11-19): the control set of the final form. "If car speed or any other variable doesn't add anything or very little do not
# include it" -- only the groups whose drop cost held-out skill in study 7 are kept: the extra housing group (0.0048, p 0.004) and income /
# poverty / education (0.0022, p 0.009); fuel price (0.0011, p 0.044) is tested; car speed (0.0000), building type and age (0.0009, n.s.) and
# the rest are left out. "I prefer population density than housing columns if similar skill" -- density (1 km, the best single radius) is
# tried in place of the extra housing group. Every set is fitted exactly as the final form of study 7b: the no-mode speed fit from 3 cold
# starts x 12000 steps, then the mode factor on top (speed terms frozen). Compared by held-out skill (paired bootstrap over metros:
# ctrl_rank.py) and by the shift of the mode factor.
HERE=$(cd "$(dirname "$0")" && pwd); S=${LOGS:-$HERE/logs}; mkdir -p $S; L=$S/ctrl_study8.log; F=$(cd "$HERE/../.." && pwd)
R=$F/fit/data_buf/torch_runs
log() { echo "$* $(date '+%F %T')" >> $L; }
fit() {   # fit <tag> [VAR=value ...]
  tag=$1; shift
  [ -f $R/g5_controls_w_cs_$tag.json ] && { log "$tag exists"; return; }
  cd $F && env FLEX_VARIANT=buf FIT4_SUFFIX=_cs_$tag FIT4_RUN_STARTS=1 FIT4_STEPS=6000 FIT4_DECAY=service FIT4_H_FLOOR=3 FIT4_ONECURVE=1 \
    FIT4_CONTROLS_PLUS=1 FIT4_NO_METRO_RE=1 FIT4_NO_SPEED=1 FIT4_CTRL_STUDY=1 "$@" \
    systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=2G ${TORCH_PYTHON:-python3} -u fit/torch_fit4.py national g5_controls_w > $S/cs_$tag.log 2>&1 < /dev/null
  log "$tag exit $? $(grep -o 'cv {.*}' $S/cs_$tag.log)"
}
# the 36 controls of the final form (study 7b), by group
DENSITY=log_pop_density,log_pop_density_500m,log_pop_density_1km,log_pop_density_2km,log_pop_density_3km,log_pop_density_4km,log_pop_density_5km
JOBS=log_job_density,log_jobs_5km; GEOGRAPHY=log_dist_cbd,log_metro_pop; CLASS=log_income,poverty_share,bachelor_share; AGE=under18_share,over65_share
RACE=black_share,hispanic_share,asian_share,foreign_born_share; HOUSEHOLD=household_size,children_hh_share,living_alone_share; FUEL=log_fuel_price
HOUSING=multifamily10_share,detached_share,built_pre1950_share,built_pre1980_share,built_post2000_share,year_built_median
HOUSING2=vacancy_share,mobile_home_share,small_multi_share,attached_share,rooms_median; CAR=log_car_speed
ADD=log_pop_density_2km,log_pop_density_4km,log_pop_density_5km,built_pre1980_share,built_post2000_share,year_built_median,$HOUSING2   # as study 7b
ALL=$DENSITY,$JOBS,$GEOGRAPHY,$CLASS,$AGE,$RACE,$HOUSEHOLD,$FUEL,$HOUSING,$HOUSING2,$CAR
drop_all_but() { python3 -c "import sys; k=set(sys.argv[2].split(',')); print(','.join(c for c in sys.argv[1].split(',') if c not in k))" "$ALL" "$1"; }
MONO="FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono FIT4_SPEED_CLIP=1"
COLD="FIT4_RUN_STARTS=3 FIT4_STEPS=12000 FIT4_INIT="
final() {   # final <tag> <kept controls>: the final form with that control set
  D="FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$(drop_all_but "$2")"
  fit nomode_$1 $D $MONO FIT4_NO_MODE=1 $COLD
  [ -f $R/g5_controls_w_cs_nomode_$1.pt ] && fit seq_$1 $D $MONO FIT4_INIT=$R/g5_controls_w_cs_nomode_$1.pt FIT4_FREEZE=th_s,c_v
}
log "study 8 (minimal control set of the final form, no car speed) started"
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do sleep 60; done
final s8_min      "$HOUSING2,$CLASS,$FUEL"                  # A: the groups that add skill (+ fuel)
final s8_dens     "log_pop_density_1km,$CLASS,$FUEL"        # B: density in place of the extra housing group
final s8_nofuel   "$HOUSING2,$CLASS"                        # D: A without fuel price
python3 $HERE/ctrl_rank.py table > $S/ctrl_table8.txt 2>&1
log "study 8 done"
