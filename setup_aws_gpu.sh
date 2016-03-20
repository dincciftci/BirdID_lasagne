sudo mkdir /data
sudo mount /dev/xvdf /data

sudo apt-get update
sudo apt-get -y install git
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
sudo pip install Lasagne==0.1
sudo apt-get  -y install python-sklearn
git clone https://github.com/HIPS/Spearmint.git
cd Spea*
python setup.py install --user
cd ..
git clone https://github.com/craffel/simple_spearmint.git
cd simpl*
python setup.py install --user
