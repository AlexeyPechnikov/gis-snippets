#!/bin/bash
# "Ubuntu 18.04 LTS - Bionic" EC2 init scipt
#sudo less /var/lib/cloud/instance/scripts/part-001
# to check availability zone (should be allowed for the EFS)
#curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone

# EFS system id
EFS="fs-xxx"
# timeout in seconds between launch of single dask-scheduler and multiple dask-worker instances
TIMEOUT=120
# number of cores
CORES=$(nproc)

cd /tmp

# make and mount swapfile
dd if=/dev/zero of=/swapfile bs=1GB count=2
chmod 0600 /swapfile
mkswap /swapfile
swapon /swapfile

##########################################################################################
# System installation
##########################################################################################
# aptitude
apt-get -y install aptitude
aptitude update

# upgrade Ubuntu
DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade
#aptitude install --reinstall build-essential
# base work tools
aptitude -y install mc htop rsync zip parallel bc git netcdf-bin
# to install from deb files locally
apt-get install -y gdebi

# for parcels python package
aptitude install -y ffmpeg cdo nco

##########################################################################################
# EFS installation and initialization
##########################################################################################
# to use Amazone FS
aptitude -y install nfs-common

# build EFS helper
#https://docs.aws.amazon.com/efs/latest/ug/using-amazon-efs-utils.html
apt-get -y install binutils
git clone https://github.com/aws/efs-utils
cd efs-utils
./build-deb.sh
#apt-get -y install ./build/amazon-efs-utils*deb
yes | sudo gdebi ./build/amazon-efs-utils*deb
cd ..

# mount EFS
mkdir -p /mnt/efs
chown ubuntu /mnt/efs
mount -t efs "$EFS:/" /mnt/efs

##########################################################################################
# Python + matplotlib installation
##########################################################################################
# install python modules
aptitude -y install python3-pip python3-matplotlib python3-mpltoolkits.basemap
#pip install --upgrade pip
pip3 install --upgrade pip==9.0.3
pip3 install --upgrade setuptools

##########################################################################################
# Python science packages installation
##########################################################################################

# required for rasterio library
aptitude -y install libgdal-dev 

# python science packages
pip3 install --upgrade numpy
pip3 install --upgrade pandas
pip3 install --upgrade scipy
pip3 install --upgrade xarray
pip3 install --upgrade cython
pip3 install --upgrade h5py
pip3 install --upgrade h5netcdf
pip3 install --upgrade netCDF4
pip3 install --upgrade toolz
pip3 install --upgrade statsmodels
pip3 install --upgrade sklearn
pip3 install --upgrade psycopg2
pip3 install --upgrade zarr
pip3 install --upgrade pyresample

# this downgrade is very important because
# in modern pandas version we have problem:
# ValueError: cannot convert float NaN to integer
#pip3 install pandas==0.19.0

# libgdal-dev required
pip3 install --upgrade rasterio
pip3 install gsw==3.0.6
pip3 install --upgrade Shapely
pip3 install --upgrade geopandas descartes rtree
pip3 install --upgrade salem
pip3 install --upgrade joblib
pip3 install --upgrade gis

# Jupyter notebook installation
pip3 install --upgrade jupyter

# will be upgraded from git later
pip3 install datashader

#http://holoviews.org/install.html
#install a larger set of packages that provide additional functionality in HoloViews:
pip3 install "holoviews[all]"

# dask cluster
#pip3 install --upgrade dask distributed bokeh
pip3 install --upgrade distributed bokeh

pip3 install --upgrade fastparquet
pip3 install --upgrade tables
pip3 install --upgrade pyarrow

#https://github.com/OceanParcels/parcels
pip3 install --upgrade py cgen progressbar pymbolic cartopy netcdftime
#pip3 install git+https://github.com/OceanParcels/parcels.git
pip3 install git+https://github.com/OceanParcels/parcels.git@v1.1.1

##########################################################################################
# xmitgcm & xgcm installation
##########################################################################################

# xmitgcm
git clone https://github.com/xgcm/xmitgcm.git
cd xmitgcm
python3 setup.py install
cd ..

# xgcm
git clone https://github.com/xgcm/xgcm.git
cd xgcm
python3 setup.py install
cd ..

#https://github.com/dask/dask/blob/master/docs/source/install.rst
#https://github.com/pyviz/hvplot/issues/41
git clone https://github.com/dask/dask.git
cd dask
pip3 install -e ".[complete]"
cd ..

#https://github.com/bokeh/datashader
pip3 uninstall --yes datashader
git clone https://github.com/bokeh/datashader.git
cd datashader
pip3 install -e .
cd ..

##########################################################################################
# dask cluster initialization
# I think all required Python libraries should be configured before!
# Note: with default config value --nprocs=1 we have only 1 core utilized
##########################################################################################

#http://distributed.readthedocs.io/en/latest/quickstart.html
#http://distributed.readthedocs.io/en/latest/setup.html
#https://media.readthedocs.org/pdf/distributed/stable/distributed.pdf
INDEX=$(ec2metadata --ami-launch-index)
if [ $INDEX -eq 0 ]
then
	#Launch dask-scheduler on one node
	dask-scheduler --scheduler-file /mnt/efs/dask-scheduler.json &
else
	#launch dask-worker on the rest of the nodes
	sleep "$TIMEOUT"
	#http://distributed.readthedocs.io/en/latest/efficiency.html
	dask-worker --nprocs "$CORES" --nthreads 1 --memory-limit=auto --scheduler-file /mnt/efs/dask-scheduler.json &
	#dask-worker --memory-limit=auto --scheduler-file /mnt/efs/dask-scheduler.json &
fi

#  --nthreads INTEGER            Number of threads per process.
#  --nprocs INTEGER              Number of worker processes.  Defaults to one.

##########################################################################################
# PostgreSQL/PostGIS installation
##########################################################################################
# install and configure modern PostgreSQL + PostGIS
add-apt-repository "deb http://apt.postgresql.org/pub/repos/apt/ bionic-pgdg main"
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
aptitude update
aptitude -y install postgresql-10 postgresql-client-10 postgresql-10-postgis-2.4

runuser -l postgres -c 'psql -c "create role ubuntu superuser;" '
runuser -l postgres -c 'psql -c "create database ubuntu owner ubuntu;" '
runuser -l postgres -c 'psql -c "ALTER ROLE ubuntu WITH LOGIN;" '
runuser -l postgres -c 'psql -d ubuntu -c "CREATE EXTENSION postgis;" '

runuser -l postgres -c 'psql -c "create role jet superuser;" '
runuser -l postgres -c 'psql -c "create database jet owner jet;" '
runuser -l postgres -c 'psql -c "ALTER ROLE jet WITH LOGIN;" '
runuser -l postgres -c 'psql -d jet -c "CREATE EXTENSION postgis;" '

aptitude clean
