cd /home/alice/git3/solar
python3 -m venv ../venv
source ../venv/bin/activate
sudo apt-get install libpq-dev
pip install -r requirements.txt

cd /home/alice/git3/solar
source ../venv/bin/activate
python test.py --database-url postgres://zalando:OMXpgNuneVRCnxQcMbohQOZSzntJTVAWDUOdhJkMYqL8rtMgmOLROgEm7a2mvnKI@192.168.1.151:30001/foo?sslmode=require
