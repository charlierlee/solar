sudo apt install python3.7-venv postgresql postgresql-contrib
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

/etc/postgresql/10/main/postgresql.conf

sudo -u postgres psql postgres


# \password postgres

Enter new password: ...
\q

sudo su - postgres
$ create database solardb
\q
$ psql -s solardb
# create user someuser password 'somepassword';
# GRANT ALL PRIVILEGES ON DATABASE mydb TO someuser;

RUNNING:
#maybe?
sudo su - postgres

#then
mate3_pg -H 192.168.1.63 --definitions /media/tb2/solar/pg_config.yaml --database-url postgres://solaruser:sqlsa@127.0.0.1:5432/solardb --debug


sudo snap install dbeaver-ce

