#!/usr/bin/env bash
# Session 40: when run_g3_qa.sh logs "=== g3 done", commit + push the guards-v3 Q&A report so the
# results reach GitHub even if the chat session has ended. Docs get a one-line pointer only.
LOGS=/home/dev1/scinex_runs
until tail -3 $LOGS/scale_driver.log | grep -q "=== g3 done"; do sleep 60; done
cd /home/dev1/SciNex || exit 1
cp $LOGS/g3_report.log project_scinex/ft_runs/scale/g3_report.log
printf '\n- **%s (auto):** guards-v3 Q&A pass finished; report in `qa_eval/scale/report_g3.json` and `ft_runs/scale/g3_report.log`. Table 27 Q&A column still to be written into results.md.\n' "$(date '+%F %H:%M')" >> project_scinex/HANDOFF_NEXT_SESSION.md
git add project_scinex/qa_eval/scale/report_g3.json project_scinex/ft_runs/scale/g3_report.log project_scinex/ft_runs/scale/compare_g3.json project_scinex/HANDOFF_NEXT_SESSION.md project_scinex/ft_runs/acl_scripts/run_g3_qa_gpu0.sh project_scinex/ft_runs/acl_scripts/run_g3_autopush.sh
git -c user.name=SethapongS0212 -c user.email=samaruek2004@gmail.com commit -q -m "Guards v3: Q&A pass finished (auto-push); report_g3.json

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
GIT_SSH_COMMAND="ssh -i ~/.ssh/scinex_deploy_ed25519 -o IdentitiesOnly=yes" git push -q git@github.com:SethapongS0212/SciNex.git main >> $LOGS/g3_autopush.log 2>&1
echo "[$(date '+%F %T')] g3 autopush: rc=$?" >> $LOGS/scale_driver.log
