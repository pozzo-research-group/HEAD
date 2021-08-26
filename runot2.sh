source env/bin/activate
cd OT2
python3 initialize.py
python3 generate_random_batch.py
python3 ot2_platereader.py

for i in 0 1 2 3 4 5
do 
	python3 run_bo.py
	python3 ot2_platereader.py 
	python3 teach_bo.py
done

python3 make_plots.py