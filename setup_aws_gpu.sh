sudo mkdir /data
sudo mount /dev/xvdf /data

sudo apt-get update
sudo apt-get -y install git
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
sudo pip install Lasagne==0.1
sudo apt-get install python-sklearn
# needs to install (simple)spearmint
