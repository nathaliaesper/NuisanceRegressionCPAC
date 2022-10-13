#!/usr/bin/env python

"""

Nuisance Regression

Connectomes generated for Schaefer200
- Correlation figures for each nuisance regressor strategy
- Two pipelines used: preproc and fmriprep
- Strategies:
	- aCompCor
	- aCompCor with GSR
	- 36p
	- linear detrend
	- scrubbing - Jenkinson's method
	- scrubbing - Power's method
	- spike regression - Jenkinson's method
	- spike regression - Power's method

Created by: Nathalia Bianchini Esper, 2022

"""

from nilearn import plotting, datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import plot_epi, show, plot_roi
from nilearn.image import resample_to_img
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import nibabel as nib


basedir = '/ocean/projects/med220004p/nesper/NuisanceRegression/'
pipeline = 'preproc'
pip_path = basedir + 'results_new/' + pipeline + '/output/cpac_cpac_' + pipeline
file_name = '_task-rest_run-1_space-template_desc-preproc-'

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)

for regressor in range (1,9):

	subj_file_list = glob(pip_path + '/*/func/*_task-rest_run-1_space-template_desc-preproc-' + str(regressor) + '_bold.nii.gz')
	#print(subj_file_list)

	all_corr_matrix = np.zeros((200,200,len(subj_file_list)))
	#print("all_corr_matrix shape: ", all_corr_matrix.shape)

	for subj in range(len(subj_file_list)):
		open_image = nib.load(subj_file_list[subj])
		#print("Functional image shape: ", open_image.shape)

		# Extract signals
		masker = NiftiLabelsMasker(atlas.maps,
								   labels=atlas.labels,
								   standardize=True)
		masker.fit(open_image)
		signals = masker.transform(open_image)
		#print(signals.shape)

		# Compute and display a correlation matrix
		correlation_measure = ConnectivityMeasure(kind='correlation')
		correlation_matrix = correlation_measure.fit_transform([signals])[0]
		np.fill_diagonal(correlation_matrix, 0)

		#print(correlation_matrix.shape)

		all_corr_matrix[:,:,subj-1] = correlation_matrix
		#print(all_corr_matrix.shape)

	all_corr_mean = np.mean(all_corr_matrix, axis=2)
	#print("test shape: ", test.shape)

	plt.imshow(all_corr_mean)
	plt.clim([-1, 1])
	#plt.show()

	if regressor == 1:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - aCompCor' 
	elif regressor == 2:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - aCompCor with GSR'
	elif regressor == 3: 
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - 36p'
	elif regressor == 4:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - linear detrend'
	elif regressor == 5:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - scrubbing Jenkinson'
	elif regressor == 6:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - scrubbing Power'
	elif regressor == 7:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - spike regression Jenkinson'
	elif regressor == 8:
		title_name = 'Connectomes for ' + pipeline + ' - Schaefer200 - spike regression Power'
	
	plt.title(title_name)
	fig_name = basedir + 'figures/all_subjs_' + pipeline + '_Schaefer200_template_space_reg' + str(regressor)
	plt.savefig(fig_name)


