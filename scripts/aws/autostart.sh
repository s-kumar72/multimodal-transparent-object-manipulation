#!/bin/bash
# Resumes the sweep after the instance is stopped and started again (that wipes the NVMe disk and every process).
# Register it once yourself (it adds a cron @reboot line for your user):
#   bash scripts/aws/autostart.sh install        remove it with: bash scripts/aws/autostart.sh uninstall
# At boot it does nothing unless the job file results/checkpoint/sweep_job exists. Each line of the job file is one
# sweep call, run in order, e.g.
#   L4b rgb rgbd rgbdt
#   L0 rgb rgbd rgbdt
# Finished runs are skipped by sweep.sh, so the same file is safe to rerun. Delete the file to stop resuming.
# Log: results/checkpoint/autostart.log
REPO=$(cd "$(dirname "$0")/../.." && pwd)
JOB=$REPO/results/checkpoint/sweep_job
LOG=$REPO/results/checkpoint/autostart.log
LINE="@reboot bash $REPO/scripts/aws/autostart.sh"
case "$1" in
    install) (crontab -l 2>/dev/null | grep -vF "$LINE"; echo "$LINE") | crontab - && echo "installed: $LINE"; exit ;;
    uninstall) crontab -l 2>/dev/null | grep -vF "$LINE" | crontab - && echo "removed"; exit ;;
esac
[ -f $JOB ] || exit 0
cd $REPO
{
    echo "=== $(date) autostart"
    sleep 30                                    # let the GPU driver and the network come up
    [ -f /opt/dlami/nvme/work/env.sh ] || bash scripts/aws/setup_nvme.sh
    source /opt/dlami/nvme/work/env.sh
    if pgrep -f "^bash scripts/aws/sweep.sh" > /dev/null; then echo "a sweep is already running"; exit 0; fi
    while read -r level variants; do
        [ -n "$level" ] && bash scripts/aws/sweep.sh "$level" "$variants" || break
    done < $JOB
    echo "=== $(date) autostart finished"
} >> $LOG 2>&1
