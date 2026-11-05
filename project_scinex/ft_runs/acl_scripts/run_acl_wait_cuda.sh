#!/usr/bin/env bash
# Wait until a new process can use CUDA again, then continue the ACL run.
# Since GPU1 dropped off the bus (2026-10-28 16:39) no new CUDA context can start on this server
# (hands_off §37.7). This covers a driver reset/reload WITHOUT a reboot; a reboot ends this screen,
# and then `screen -dmS scinexacl2 /home/dev1/scinex_runs/run_acl_after_reboot.sh` is run by hand.
set -u
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/acl_driver.log"; }

say "wait-cuda: polling every 5 min for a usable GPU"
until CUDA_VISIBLE_DEVICES=0 timeout 120 "$PY" -c "import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()" \
      > /dev/null 2>&1; do
  sleep 300
done
say "wait-cuda: CUDA is usable again - starting run_acl_after_reboot.sh"
exec /home/dev1/scinex_runs/run_acl_after_reboot.sh
