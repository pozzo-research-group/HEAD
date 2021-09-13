python3 initialize.py
python3 generate_random_batch.py
python3 ot2_platereader.py
python3 read_uvvis.py --xlsx output/uvvis_0.xlsx
python3 read_saxs.py --xlsx output/saxs_0.xlsx
python3 teach_bo.py

for i in 1 2
do 
	python3 run_bo.py
	python3 ot2_platereader.py 
	python3 read_uvvis.py --xlsx output/uvvis_$i.xlsx
	python3 read_saxs.py --xlsx output/saxs_$i.xlsx
	python3 teach_bo.py
done

python3 make_plots.py
python3 run_bo.py