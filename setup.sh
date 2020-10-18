sudo apt install python3.7-venv postgresql postgresql-contrib
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt


/etc/postgresql/10/main/postgresql.conf
#Edit ^ and add the following line
listen_addresses = '*'

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

#To Test:
sudo su - postgres
cd /media/tb2/solar/
source env/bin/activate
mate3_pg -H 192.168.1.63 --definitions pg_config.yaml --database-url postgres://solaruser:sqlsa@127.0.0.1:5432/solardb --debug

#install service
sudo chmod +x ./solar.sh
sudo cp solar.service /lib/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable solar.service
sudo systemctl start solar.service

#to debug service
sudo journalctl -u solar

#on personal laptop
sudo snap install dbeaver-ce
#and Run
SELECT * FROM public.device_data_logs3;
