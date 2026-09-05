#!/bin/bash
set -Eeuo pipefail
export FULL3203_TIME_POOL_COUNT="${FULL3203_TIME_POOL_COUNT:-128}"
exec bash "$(dirname "$0")/run_helmholtz_full3203_time96_continue_ddp4.sh"
