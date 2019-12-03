#!/usr/bin/env bash
for ff in `cat progs_run`; do
cwm=$(last_wm=2000+; for wm in `ls ${ff}_res*.txt | cut -d_ -f7 | sort -nr | uniq`; do if grep -q "^[^,]*,False" ${ff}_res_1000_${wm}_10_80_metrics*.txt; then break; else last_wm=$wm; fi; done; echo $last_wm)
#ctd=$(last_wm=14+; for wm in `ls ${ff}_res*.txt | cut -d_ -f8 | sort -nr | uniq`; do if grep -q "^[^,]*,False" ${ff}_res_1000_1000_${wm}_80_metrics*.txt; then break; else last_wm=$wm; fi; done; echo $last_wm)
ctd=$(last_wm=10+; for wm in `ls ${ff}_res*.txt | cut -d_ -f8 | sort -nr | uniq | tail -7`; do if grep -q "^[^,]*,False" ${ff}_res_1000_1000_${wm}_80_metrics*.txt; then break; else last_wm=$wm; fi; done; if [ "$last_wm" = "10+" ] ; then for wm in 12 14 14+ ; do if grep -q "^[^,]*,True" ${ff}_res_1000_1000_${wm}_80_metrics*.txt; then break; fi; done; echo $wm; else echo $last_wm; fi)
cdelta=$(last_wm=80+; for wm in `ls ${ff}_res*.txt | cut -d_ -f9 | sort -nr | uniq | tail -8`; do if grep -q "^[^,]*,False" ${ff}_res_1000_1000_10_${wm}_metrics*.txt; then break; else last_wm=$wm; fi; done; if [ "$last_wm" = "80+" ] ; then for wm in 90 95 99 99+ ; do if grep -q "^[^,]*,True" ${ff}_res_1000_1000_10_${wm}_metrics*.txt; then break; fi; done; echo $wm; else echo $last_wm; fi)
echo $ff,$cwm,$ctd,$cdelta
done
