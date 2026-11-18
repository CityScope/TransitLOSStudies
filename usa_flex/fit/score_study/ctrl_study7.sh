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
HERE=$(cd "$(dirname "$0")" && pwd); S=${LOGS:-$HERE/logs}; mkdir -p $S; L=$S/ctrl_study7.log; F=$(cd "$HERE/../.." && pwd)
INIT=$F/fit/init/g5_controls_w_Bf_one_plus_nore.pt   # the warm start (the final form fitted before this study), kept in the repository
# the base: the 25 controls without car commute speed (too close: tested apart), + 2 / 4 / 5 km density, + housing age 1980 / 2000 / median year
BASE=log_pop_density_2km,log_pop_density_4km,log_pop_density_5km,built_pre1980_share,built_post2000_share,year_built_median
NOCAR=log_car_speed
log() { echo "$* $(date '+%F %T')" >> $L; }
log "study 7 (car speed in every stage; speed without mode; sequential speed / mode) started"
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
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do sleep 60; done   # each fit needs ~22 GB of GPU memory
# 0. the references (study 6): A0 = the 25 controls + density / housing age, B0 = + housing2 (both without car speed), sp_ref = B0 + car commute
#    speed, and the study-6 speed forms
fit A0 FIT4_CTRL_DROP=log_car_speed
fit B0 FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=log_car_speed
fit sp_ref $C
fit sp_mode $C FIT4_NO_SPEED=0
fit sp_mode_mono $C FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono
fit spv_nodecay $C FIT4_SPEED_VAR=nodecay
fit spv_far_clip $C FIT4_SPEED_VAR=far FIT4_SPEED_CLIP=1
# 1. speed and mode (car speed control in all): the no-mode forms first
declare -A FORM=(
 [nosp_nomode]="FIT4_NO_MODE=1"
 [sp_nomode]="FIT4_NO_SPEED=0 FIT4_NO_MODE=1"
 [spv_nomode_power_clip]="FIT4_SPEED_VAR=power FIT4_SPEED_CLIP=1 FIT4_NO_MODE=1"
 [spv_nomode_mono_clip]="FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono FIT4_SPEED_CLIP=1 FIT4_NO_MODE=1"
 [spv_nomode_permode_clip]="FIT4_SPEED_VAR=permode FIT4_SPEED_CLIP=1 FIT4_NO_MODE=1"
 [sp_nomode_car]="FIT4_NO_SPEED=0 FIT4_SPEED_CAR=1 FIT4_NO_MODE=1"
 [spv_power]="FIT4_SPEED_VAR=power" [spv_power_clip]="FIT4_SPEED_VAR=power FIT4_SPEED_CLIP=1"
 [spv_mono_clip]="FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono FIT4_SPEED_CLIP=1"
 [spv_permode]="FIT4_SPEED_VAR=permode" [spv_permode_clip]="FIT4_SPEED_VAR=permode FIT4_SPEED_CLIP=1"
 [spv_commute_clip]="FIT4_SPEED_VAR=commute FIT4_SPEED_CLIP=1" [sp_mode_car]="FIT4_NO_SPEED=0 FIT4_SPEED_CAR=1")
ORDER="nosp_nomode sp_nomode spv_nomode_power_clip spv_nomode_mono_clip spv_power_clip spv_mono_clip spv_nomode_permode_clip spv_permode_clip sp_nomode_car sp_mode_car spv_power spv_permode spv_commute_clip"
for t in $ORDER; do run $t $C ${FORM[$t]}; done
wait
# 2. sequential: speed first (frozen from the no-mode fit), the mode factor only on what is left
[ -f $R/g5_controls_w_cs_spv_nomode_power_clip.pt ] && run seq_power_clip $C FIT4_SPEED_VAR=power FIT4_SPEED_CLIP=1 FIT4_INIT=$R/g5_controls_w_cs_spv_nomode_power_clip.pt FIT4_FREEZE=b_v,c_v
wait
[ -f $R/g5_controls_w_cs_spv_nomode_mono_clip.pt ] && run seq_mono_clip $C FIT4_NO_SPEED=0 FIT4_SHAPE_S=mono FIT4_SPEED_CLIP=1 FIT4_INIT=$R/g5_controls_w_cs_spv_nomode_mono_clip.pt FIT4_FREEZE=th_s,c_v
wait
python3 $HERE/ctrl_rank.py table > $S/ctrl_table7_speed.txt 2>&1; python3 $HERE/ctrl_rank.py speed >> $S/ctrl_table7_speed.txt 2>&1
log "speed part done"
# 3. D: which density radius (car speed kept)
for r in 500m 1km 2km 3km 4km 5km; do
  run dens_only_$r FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$(python3 -c "print(','.join(f'log_pop_density_{x}' for x in ['500m','1km','2km','3km','4km','5km'] if x != '$r'))")
done
# B: drop one group from the base (car speed kept)
for g in density jobs geography class age race household housing fuel income_dist occupation vulnerability housing2; do
  first=$(python3 -c "import json;print(json.loads('''$G''')['$g'][0])")
  case "$g" in income_dist|occupation|vulnerability|housing2) [[ ",$ADD," == *",$first,"* ]] || continue;; esac
  run drop_$g FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$(grp $g)
done
wait
# C: the smallest sets (top-k groups of stage B)
python3 $HERE/ctrl_rank.py sets "$G" "$ADD" > $S/ctrl_sets7.txt 2>> $L
while read k drop; do run top$k FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$drop; done < $S/ctrl_sets7.txt
wait
# E: stability of the finalists -- 3 cold starts, 12000 steps
PICK=$(python3 $HERE/ctrl_rank.py pick $S/ctrl_sets7.txt 2>> $L); SP=$(python3 $HERE/ctrl_rank.py speed 2>> $L); log "picked drop [$PICK] speed [$SP]"
COLD="FIT4_RUN_STARTS=3 FIT4_STEPS=12000 FIT4_INIT="
run final_ref $C $COLD
[ -n "$PICK" ] && run final_small FIT4_CTRL_ADD=$ADD FIT4_CTRL_DROP=$PICK $COLD
[ -n "$SP" ] && [ -n "${FORM[$SP]}" ] && run final_speed $C ${FORM[$SP]} $COLD
wait
fit coldcv_sp_ref $C FIT4_CV_COLD=1
python3 $HERE/ctrl_rank.py table > $S/ctrl_table7.txt 2>&1
python3 $HERE/ctrl_rank.py speed >> $S/ctrl_table7.txt 2>&1
log "study 7 done"
