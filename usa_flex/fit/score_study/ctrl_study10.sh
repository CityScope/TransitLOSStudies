#!/bin/bash
# Study 10 (user 2026-11-20): "explore ways of making the score more generalizable across different US metro areas".
# Per-metro held-out skill of the lean final form (seq_s8_nofuel) shows 43% of the gain in counties with < 100 transit commuters (mostly
# "no transit here"), 0.06 in metros of 100-100k transit commuters, Miami 0.005, Phoenix 0.015, New York 12% of the gain on its own.
# Each run is the lean final form V1 (income / poverty / education + extra housing, raw speed, speed curve first without a mode factor, then
# frozen, then the mode factor) with one change (study10_variants.txt), against g_base (V1 refitted so its held-out predictions are saved):
#   training weight: every metro equal (WLAM 1) / plain population (WLAM 0)   (0.5 before; the evaluation weight stays 0.5)
#   metro network context: + the metro mean of the access term (how much of the metro the network reaches)
#   context controls: + the metro mean of every control
# Every run saves its held-out block-group predictions (<run>_ho.npz); metro_diag.py turns them into the held-out skill, the skill of the
# median metro, and the within-metro skill (each held-out metro's level calibrated away: does the score rank places inside a metro it never saw).
HERE=$(cd "$(dirname "$0")" && pwd); S=${LOGS:-$HERE/logs}; mkdir -p $S; L=$S/ctrl_study10.log; F=$(cd "$HERE/../.." && pwd)
R=$F/fit/data_buf/torch_runs
log() { echo "$* $(date '+%F %T')" >> $L; }
fit() {   # fit <tag> [VAR=value ...]
  tag=$1; shift
  [ -f $R/g5_controls_w_cs_$tag.json ] && { log "$tag exists"; return; }
  cd $F && env FLEX_VARIANT=buf FIT4_SUFFIX=_cs_$tag FIT4_RUN_STARTS=1 FIT4_STEPS=6000 FIT4_DECAY=service FIT4_H_FLOOR=3 FIT4_ONECURVE=1 \
    FIT4_CONTROLS_PLUS=1 FIT4_NO_METRO_RE=1 FIT4_NO_SPEED=1 FIT4_CTRL_STUDY=1 "$@" ${SMOKE:+FIT4_STEPS=20 FIT4_RUN_STARTS=1 FIT4_CV_STEPS=5} \
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
final() {   # final <tag> <kept controls> [variant VAR=value ...]: speed curve first (no mode, 3 cold starts), frozen, then the mode factor
  local t=${SMOKE:+smoke_}$1 keep=$2; shift 2
  local D="FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$(drop_all_but "$keep")"
  local FRZ
  case " $* " in *" FIT4_SHAPE_S=power "*) FRZ=b_v;; *" FIT4_SHAPE_S=loglogistic "*) FRZ=lv50,lkv;; *) FRZ=th_s;; esac
  case " $* " in *" FIT4_DECAY= "*) ;; *) FRZ=$FRZ,c_v;; esac       # c_v: the speed scaling of the distance curve (absent when it is not scaled)
  fit nomode_$t $D $MONO "$@" FIT4_NO_MODE=1 $COLD
  [ -f $R/g5_controls_w_cs_nomode_$t.pt ] && fit seq_$t $D $MONO "$@" FIT4_INIT=$R/g5_controls_w_cs_nomode_$t.pt FIT4_FREEZE=$FRZ
}
log "study 10 (generalisation across metros) started"
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do sleep 60; done
V1="$HOUSING2,$CLASS"
grep -v '^#' $HERE/study10_variants.txt | while read -r tag env; do
  [ -n "$tag" ] && final $tag "$V1" $env
done
[ -n "$SMOKE" ] && { log "smoke done"; exit 0; }
python3 $HERE/metro_diag.py > $S/study10_diag.txt 2>&1
log "study 10 done"
