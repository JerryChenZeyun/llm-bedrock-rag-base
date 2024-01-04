pip install virtualenv
python -m virtualenv venv
source ./venv/bin/activate

pip install -q -U pip --root-user-action=ignore
pip3 install -q -r requirements.txt --root-user-action=ignore