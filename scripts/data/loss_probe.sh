#!/bin/bash
# 单次探测:输出 VERDICT|详情
L=/storage/yukaichengLab/mazijian/wth/ImageWAM/slurm_logs/ap_patch_aug4n_90459.out
st=$(bash -lc "squeue -j 90459 -h -o %T" 2>/dev/null)
laststep=$(grep -oE "step=[0-9]+/[0-9]+" "$L" 2>/dev/null | tail -1)
recent=$(grep -E "loss=" "$L" 2>/dev/null | tail -10 | grep -oE "loss=[0-9.naN]+" | head -10)
if [ -z "$st" ]; then echo "ENDED|$laststep"; exit 0; fi
if echo "$recent" | grep -qi nan; then echo "NAN|$laststep"; exit 0; fi
spike=$(echo "$recent" | sed "s/loss=//" | awk '$1>0.3{c++} END{print c+0}')
if [ "$spike" -ge 5 ]; then echo "SPIKE|$laststep|$(echo $recent|tr "\n" " ")"; exit 0; fi
echo "OK|$st|$laststep"
