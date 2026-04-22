parameters={}

parameters['misallignment_error_parameters_quads'] =  {
'error_element_familys' : ['qf[23]a.*|qd1a.*|Q[FD][0-5][CDIJ].*|Q[FD][0-4]F.*'],  
'error_class' : 'random', #  'random', 'systimatic'
'misallignment_shift_x' :[50*1E-6],
'misallignment_shift_y' : [50*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' :[50*1E-6],
'error_seed' : seed, 
'switch' : ['misalignment_quads']
}




parameters['misallignment_error_parameters_dipoles'] =  {
'error_element_familys' : arc_dipoles, 
'error_class' : 'random', # 'random', 'systimatic'
'misallignment_shift_x' : [1000*1E-6],
'misallignment_shift_y' : [1000*1E-6],
'misallignment_shift_s' : [500*1E-6], #
'misallignment_rot_s_rad' : [1000*1E-6],
'error_seed' : seed, 
}
parameters['misallignment_error_parameters_sextupoles'] =  {
'error_element_familys' : arc_sext, 
'error_class' : 'random', # 'random', 'systimatic'
'misallignment_shift_x' : [50*1E-6],
'misallignment_shift_y' : [50*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' : [50*1E-6],
'error_seed' : seed, 
}
parameters['misallignment_error_parameters_bpm'] =  {
'error_element_familys' : 'bpm', 
'error_class' : 'random', # 'random', 'systimatic'
'misallignment_shift_x' : [10*1E-6],
'misallignment_shift_y' : [10*1E-6],
'misallignment_shift_s' : [0], #,
'misallignment_rot_s_rad' : [10*1E-6],
'error_seed' : seed, 
}
parameters['misallignment_error_parameters_girder'] =  {
'error_element_familys' : arc_quads, 
'error_class' : 'random', # 'random', 'systimatic'
'misallignment_shift_x' : [150*1E-6],
'misallignment_shift_y' : [150*1E-6],
'misallignment_shift_s' : [500*1E-6],
'misallignment_rot_s_rad' : [150*1E-6],
'error_seed' :seed, 
}
parameters['misallignment_error_parameters_SS_quads'] =  {
'error_element_familys' : "Q[FD][0-6]M.*|Q[FD]5F.*|Q[FD][6-9][CFDIJ].*|Q[FD]1[0-8][CFDIJM].*|Q[FD][7-9]M.*",  
'error_class' : 'random', #  'random', 'systimatic'
'misallignment_shift_x' :[100*1E-6],
'misallignment_shift_y' : [100*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' :[100*1E-6],
'error_seed' : seed, 
}
#non arc dipoles
parameters['misallignment_error_non_arc_dipoles'] =  {
'error_element_familys' : 'd[SF]1a.*|DOG[LR]_COLL.*|DOG[LR]_DIAG.*|DOG[LR]_RF.*|DL[089][LR]_RF.*|DS[123][LR]_RF.*|DI[012][LR].*|B[0-7].*',  
#'error_element_familys' : 'd[SF]1a.*|VSEI[12].*|DOG[LR]_COLL.*|DOG[LR]_DIAG.*|DOG[LR]_RF.*|DL[089][LR]_RF.*|DS[123][LR]_RF.*|DI[012][LR].*|B[0-7].*',  
'error_class' : 'random', #  'random', 'systimatic'
'misallignment_shift_x' :[1000*1E-6],
'misallignment_shift_y' : [1000*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' :[1000*1E-6],
'error_seed' : seed, 
}
#elements between final doublet and crab sextupole
parameters['misallignment_error_parameters_fd_to_cs'] =  {
'error_element_familys' : 'Q[XY][0-4][LR].*|Q[FD][2-9][LR].*|Q[FD]1[0-9][LR].*|Q[FD]20[LR].*|SCRAB[LR].*|S[FD][MXY][12][LR].*',  
'error_class' : 'random', #  'random', 'systimatic'
'misallignment_shift_x' :[30*1E-6],
'misallignment_shift_y' : [30*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' :[30*1E-6],
'error_seed' : seed, 
}
#final doublet errors 
parameters['misallignment_error_parameters_final_doublet'] =  {
'error_element_familys' : 'Q[FD][01][ABCD][LR].*',  
'error_class' : 'random', #  'random', 'systimatic'
'misallignment_shift_x' :[10*1E-6],
'misallignment_shift_y' : [10*1E-6],
'misallignment_shift_s' : [100*1E-6], 
'misallignment_rot_s_rad' :[10*1E-6],
'error_seed' : seed, 
}
