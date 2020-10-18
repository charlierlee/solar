#!/bin/bash
sleep 30
source ~/.bash_profile
source /media/tb2/solar/env/bin/activate
cd /media/tb2/solar
mate3_pg -H 192.168.1.63 --definitions /media/tb2/solar/pg_config.yaml --database-url postgres://solaruser:sqlsa@127.0.0.1:5432/solardb
