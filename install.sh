python3 -m venv env
source env/bin/activate
python -m pip install -r require.txt
pip install ipykernel
python -m ipykernel install --user --name head --display-name "head"

