python3 -m venv env
source env/bin/activate
python3 -m pip install -r require.txt
pip install -e .
mkdir ext
git clone https://github.com/GAA-UAM/scikit-fda.git
pip install ./scikit-fda
git clone https://gitlab.com/wiechapeter/pyGDM2.git
pip install ./pyGDM2
git clone https://github.com/pytorch/botorch.git
pip install ./botorch
pip install ipykernel
python3 -m ipykernel install --user --name head --display-name "head"

