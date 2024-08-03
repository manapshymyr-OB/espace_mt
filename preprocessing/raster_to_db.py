"""
Used to dump raster data into PostgreSQL database.
raster2pgsql has to be set up
"""

import os
import subprocess
from multiprocessing.pool import ThreadPool


DB_USER = 'postgres'
DB_NAME = 'postgres'
DB_HOST = 'localhost'
raster_dir = '../raster_data'
rasters = os.listdir(raster_dir)
total = len(rasters)


def raster_to_db(r):
    global total
    raster_filename = os.path.join(raster_dir, r)
    cmd = 'raster2pgsql -I -M -F -n filename -a {} {}.{} | psql -U {} -d {} -h {} -p 5432'.format(raster_filename, 'raster', 'dom', DB_USER, DB_NAME, DB_HOST)
    subprocess.call(cmd, shell=True)
    total -= 1
    print(total)

p = ThreadPool(1)
xs = p.map(raster_to_db, rasters)