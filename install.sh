python3 -m venv env
source env/bin/activate
pip install -r require.txt
pip install -e .
pip install 'numpy<=1.20'
pip install pyGDM2
pip install ipykernel
python -m ipykernel install --user --name head --display-name "head"

