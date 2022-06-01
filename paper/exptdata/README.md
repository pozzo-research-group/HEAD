## Data Folders (e.g., data_AP_Hard)

The folders contain the experimental files from the optimization campaigns. 
Readers can find synthesis procedure used in this study (using a liquid handling robot) in the `synthesis` section of the [Electronic Supplementary Information]() of the accompanying paper.
They are structured with 'data' + metric_name + target_difficulty where 'metric_name' refers to:

- peakwv: the metric that compares the wavelengths of the peaks
- euclidean: euclidean distance 
- SRSF: Square-root slope function metric that was introduced in this paper 
- AP: Amplitude Phase metric that was introduced in this paper

target_difficulty represents which target was used as the objective of the optimization. The 'Easy' target represents the one that was chosen in the design space. This data was used to create the contour plots shown in the paper. The 'Hard' target represents the simulated target. 

## Folder Contents (e.g., data_AP_Hard)

Each folder contains data on the volumes used to create the samples, the spectra collected from these samples, and information on the surrogate model of the bayesian optimization. 


### Excel files (e.g., 1.xlsx, 2.xlsx)

The biotek plate reader that we used exports data in excel files so every excel file in the folder contains information on the Uv-vis spectra of the samples. The files that have a numeber in its name (e.g., 1.xlsx, 2.xlsx) contain all the samples in an iteration (i.e., 1.xlsx contains all the spectra from the first iteration). All excel files are formatted in the same way. The first column in the wavelength in nanometers that ranges from 400 to 900 nm in increments of 5 nm. All the other columns are the Uv-vis spectrum of the samples in the wellplate. The header of the column contains some combination of a letter (A-H) and number (1-12) which represents the sample's location in the 96 wellplate (e.g., A1, B1, C6, D12).  

In addition to the numbered excel files, there are also files containing 'Best_Estimate' + number. These files represent the Uv-vis spectrum of the best estimate of the Bayesian Optimization's surrogate model after each iteration. 


### Numbered Subfolder Contents (e.g., 1, 2) 

In each folder there are subfolders labeled with numbers. These folders contain information from the bayesian optimization at each iteration. 

The volumes of different components (in microliters) used to create the samples (total volume of 350 uL each in the well plate)  are included in this folder. The contents of the folder are:

- best_estimate.npy: a 1-D numpy array of the volumes (in microliters) that will generate the closest spectra to the target according to the  Bayesian Optimization's surrogate
- model.pth: file containing information on the surrogate model 
- new_obj.npy: a 1-D numpy array of the scores given by the similarity metric (in arbitrary units) of the samples from the previous iteration
- new_x.npy: a 2-D numpy array of the volumes (in microliters) suggested by the algorithm for the current iteration 
- spectra.npy: a 2-D numpy array of the spectra that was generated from the volumes from the previous iteration 
- storage.pkl: file containing information on the surrogate model 
- train_obj.npy: a 1-D numpy array of the scores (in arbitrary units) given by the similarity metric of each sample 
- train_x.npy: a 2-D numpy array of all the volumes (in microliters) suggested by the algorithm so far
- wavelengths.npy: a 1-D numpy array of the wavelengths (400-900) in increments of 5. 

### Additional Numpy Files

The folder also contains .npy files. These files contain infomation on the Bayesian Optimization's surrogate model and were used to generate the contour plots shown in the paper. The files are:

- confidence.npy: confidence of the posterior 
- lower.npy: lower bound
- posterior_mean.npy: mean of the posterior 
- upper.npy: upper bound 

### PDF file

The pdf file 'summary.pdf' is a picture of the contour plot generated with each optimization campaign using a distance metric. 


