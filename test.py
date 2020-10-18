#!/usr/bin/python2.4
#
import psycopg2
try:
    db = psycopg2.connect("dbname='solardb' user='solaruser' host='192.168.1.86' password='sqlsa'")
except:
    exit(1)

exit(0)
