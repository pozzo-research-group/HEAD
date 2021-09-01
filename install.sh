python3 -m venv env
source env/bin/activate
python3 -m pip install -r require.txt
pip install -e .
pip install ipykernel
python3 -m ipykernel install --user --name head --display-name "head"

