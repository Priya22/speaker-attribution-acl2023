#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
python pdnc_run_pipeline.py