#!/bin/bash
# Study 9 (user 2026-11-20): "have a few final versions and use them to better try different curve shapes and things that are still open".
# Final versions (studies 8-8c; raw speed, speed curve fitted first without a mode factor, then frozen, then the mode factor):
#   V1 lean       income / poverty / education + extra housing (seq_s8_nofuel, 0.1106) -- the chosen form, every test runs on it
#   V2 lean+fuel  V1 + fuel price (seq_s8_min, 0.1112)
#   V3 all        the 36 controls of study 7b, car speed included (final_seq_mono, 0.1117)
# Stage 1: each open question as one change to V1 (study9_variants.txt), compared with V1 by held-out skill (paired bootstrap over metros).
#   shape changes are adopted when significantly better (Holm over the shape tests); simplifications when they lose under 0.0007 and are not
#   significantly worse (the rule of the control studies: what adds nothing or very little is left out). One setting per family.
# Stage 2: the adopted settings together on V1, V2 and V3 (skipped when nothing is adopted: V1-V3 are then the forms above).
# Fixed by the user, not tested: walk curve flat below 303.5 m, headway curve flat below 3 min, no literature priors, raw speed (no car speed).
HERE=$(cd "$(dirname "$0")" && pwd); S=${LOGS:-$HERE/logs}; mkdir -p $S; L=$S/ctrl_study9.log; F=$(cd "$HERE/../.." && pwd)
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
log "study 9 (curve shapes and open questions on the final versions) started"
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do sleep 60; done
V1="$HOUSING2,$CLASS"; V2="$HOUSING2,$CLASS,$FUEL"
grep -v '^#' $HERE/study9_variants.txt | while read -r tag fam kind env; do
  [ -n "$tag" ] && final $tag "$V1" $env
done
[ -n "$SMOKE" ] && { log "smoke done"; exit 0; }   # SMOKE=1: every variant runs 20 steps under smoke_ tags (crash check), nothing else
python3 $HERE/ctrl_rank.py pick9 $HERE/study9_variants.txt > $S/study9_pick.txt 2> $S/study9_pick.log
COMBO=$(cat $S/study9_pick.txt); log "stage 1 done; adopted: ${COMBO:-nothing}"
if [ -n "$COMBO" ]; then
  [ $(echo $COMBO | wc -w) -gt 1 ] && final s9_v1_best "$V1" $COMBO   # one setting: V1 with it is its stage-1 fit
  final s9_v2_best "$V2" $COMBO
  final s9_v3_best "$ALL" $COMBO
fi
python3 $HERE/ctrl_rank.py table > $S/ctrl_table9.txt 2>&1
log "study 9 done"
