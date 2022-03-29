cd /home/alice/git3/solar
python3 -m venv ./venv
source venv/bin/activate
sudo apt-get install libpq-dev
pip install -r requirements.txt

cd /home/alice/git3/solar
source venv/bin/activate
python main.py --database-url postgres://zalando:fya8ciOW87Bbunjhz43SqWgMGrESK8c7Dj0AqfiX5JuP33GIJDjlWCcOBvZCKc57@192.168.1.151:30001/foo?sslmode=require
