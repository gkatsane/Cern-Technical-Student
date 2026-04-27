import numpy as np
from collections import Counter
from mpmath import *
import pandas as pd
import xtrack as xt
from scipy.special import *
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
import re
from scipy.stats import uniform, truncnorm, gamma


def ensure_list(value):
    # Check if the value is a single string
    if isinstance(value, (str, int, float)):
        return [value]
    # Return as-is if it's already a list or another iterable
    return value

def add_misalignment_error (line, parameters):
    """
    Applies misalignment and rotation errors to elements in the lattice based on a parameter dictionary.
    Supports both direct element misalignments (dipoles, quadrupoles, sextupoles etc.), BPM-specific alignment errors and Girder structures.
    A girder contains quadrupoles, sextupoles, orbit/optics correctors and markers in between two consecutive arc dipoles.

    line: xtrack Line object to which misalignments will be applied.

    parameters: dictionary containing configuration for the misalignments. Expected keys:
        - 'error_element_familys': element types or regex patterns (e.g. 'Quadrupole', 'sf.*', 'bpm')
        - 'error_class': 'systematic' or 'random' (default: 'systematic')
        - 'error_seed': seed for reproducibility (default: 201)
        - 'misalignment_shift_x': horizontal shift STD
        - 'misalignment_shift_y': vertical shift STD
        - 'misalignment_shift_s': longitudinal shift STD
        - 'misalignment_rot_s_rad': rotation around s-axis
        - 'switch': optional knob name(s) to control activation of misalignment
        - 'is_girder': if True, applies girder-type misalignments (shared offsets across elements)

    Behavior:
        - For BPMs:
            Returns a dictionary of BPM misalignments from the element they are attached to, and the alignment error with said element. 
            Assumes the naming of BPM's follows "bpm_{element}", with element being the element the BPM is attached to
        - For other elements:
            Applies misalignments directly to the line elements and optionally creates a knob
            to scale/turn off the applied errors.

    returns:
        misalignment_dict: dictionary containing applied misalignments for each affected element.
    """
    element_familys = parameters.get('error_element_familys')
    error_class = parameters.get('error_class', 'systematic')
    seeds = parameters.get('error_seed', 201)

    shift_x = parameters.get('misalignment_shift_x', 0)
    shift_y = parameters.get('misalignment_shift_y', 0)
    shift_s = parameters.get('misalignment_shift_s', 0)
    rot_s_rad = parameters.get('misalignment_rot_s_rad', 0)

    knob_name = parameters.get('switch', None)
    is_girder = parameters.get('is_girder', None)

    #make a dictionary with the errors for BPM's, to be given to the orbit correction function
    if element_familys== 'bpm':

        tt = line.get_table(attr=True)

        misalignment_dict = {}
        BPMs=tt.rows[f"{element_familys}.*"].name
        n_bpm = len(BPMs)
        x1_array = np.zeros(n_bpm)
        y1_array = np.zeros(n_bpm)
        r1_array = np.zeros(n_bpm)
        for jj, BPM in enumerate(BPMs):
            match = re.search(rf"^{element_familys}_(.*)$", BPM)

            element_name = match.group(1)
            #the bpm: bpm_qd12f is placed at the s coordinate that match with qd12fa. qd12f is not present
            #and qd12fa, qd12fb, qd12f.0 are all present (different s)
            if element_name =='qd12f':
                element_name = 'qd12fa'


            misalignment_dict[BPM] = {}

            #find the misalignment of the element the BPM is attached to 
            misalignments=tt.rows[element_name].cols['shift_x','shift_y','rot_s_rad']
            mean = 0
            std_dev = 1
            lower_bound = -2.5 # in sigma
            upper_bound = 2.5 # in sigma
            size=len(BPMs)
            seeds2 =[seeds, seeds+1,seeds+2, seeds+3]
            for ii, name in enumerate(['sx','sy','rs']):
                if name == 'sx':
                    np.random.seed(seeds2[ii])
                    x1 = misalignments['shift_x'][0].item()
                    x1_array[jj]=x1
                    x2=shift_x*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev,size=size)
                elif name == 'sy':
                    np.random.seed(seeds2[ii])
                    y1 = misalignments['shift_y'][0].item()
                    y1_array[jj]=y1
                    y2=shift_y*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
		        #the rad_s_no_frame is not available for BPM's (xtrack/trajectory_correction line250)
                elif name == 'rs':
                    np.random.seed(seeds2[ii])
                    el = line[element_name]
                    r1 = float(el.rot_s_rad_no_frame)
                    r1_array[jj]=r1
                    r2=rot_s_rad*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
            if error_class == 'systematic':
                x2 = np.full(shift_x, size)
                y2 = np.full(shift_y, size)
                r2 = np.full(rot_s_rad, size)        
        for jj, BPM in enumerate(BPMs):
                #add the element misalignnamement and the random BPM beam based alignment error to obtain the total BPM position error
                misalignment_dict[BPM]['shift_x']=float(x1_array[jj]+x2[jj])
                misalignment_dict[BPM]['shift_y']=float(y1_array[jj]+y2[jj])
                misalignment_dict[BPM]['rot_s_rad']=float(r1_array[jj]+r2[jj])
        return misalignment_dict


    else:
        #misalignments for dipoles, quadrupoles, sextupoles, girders
        tt = line.get_table(attr=True)
        tt_no_parent_elem = tt.rows[tt.parent_name==None]
        tt_no_marker_no_parent_elem = tt_no_parent_elem.rows[tt_no_parent_elem.element_type!='Marker']

        element_family_list = ensure_list(element_familys)
        elements_with_types = [(name, line.element_dict[name].__class__.__name__) for name in line.element_names]



        for ii,element_family in enumerate(element_family_list):
            #off switch to regulate the misalignments without calling the function
            if knob_name is None:
                mis_switch_name = 'mis_'+element_family+'_'+error_class[:3]+'_'+str(seeds)
                line[mis_switch_name] = 1
            elif np.size(knob_name)== np.size(element_familys):
                mis_switch_name = knob_name[ii]
                line[mis_switch_name] = 1
            else:
                mis_switch_name = knob_name[0]
                line[mis_switch_name] = 1


            ## in order to include the parent elements if the line has thin element
            parent_names = [item for item in tt.parent_name if item is not None]
            if len(parent_names)>0:
                parent_types = [line[parent_names[ii]].__class__.__name__ for ii in range(len(parent_names))]
            else:
                parent_types = []

            if element_family in np.unique(np.append(tt.element_type,parent_types)):
                parent_type_names = [name for ii, name in enumerate(parent_names) if parent_types[ii] == element_family]
                element_names = np.append(tt_no_marker_no_parent_elem.rows[tt_no_marker_no_parent_elem.element_type==element_family].name,parent_type_names)
            else:
                parent_type_names = [name for name in parent_names if re.match(element_family, name)] 
                element_names = np.append(tt_no_marker_no_parent_elem.rows[element_family].name,parent_type_names)

            size = len(element_names)

            if error_class == 'systematic':
                shift_values_x = np.full(shift_x, size)
                shift_values_y = np.full(shift_y, size)
                shift_values_s = np.full(shift_s, size)
                rot_values_s = np.full(rot_s_rad, size)
            elif error_class == 'random':
                mean = 0
                std_dev = 1
                lower_bound = -2.5 # in sigma
                upper_bound = 2.5 # in sigma
                seeds2 =[seeds, seeds+1,seeds+2, seeds+3]

                for ii, name in enumerate(['sx','sy','ss','rs']):
                    if name == 'sx':
                        np.random.seed(seeds2[ii])
                        shift_values_x = shift_x*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'sy':
                        np.random.seed(seeds2[ii])
                        shift_values_y = shift_y*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'ss':
                        np.random.seed(seeds2[ii])
                        shift_values_s = shift_s*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
                    elif name == 'rs':
                        np.random.seed(seeds2[ii])
                        rot_values_s = rot_s_rad*truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)

            if is_girder== None:
                misalignment_dict={}
                #for all other element types the misalignment is simply the value obtained
                for ii, name in enumerate(element_names):
                    all_names = [name]
                    hcor_name = f'hcor_{name}'
                    vcor_name = f'vcor_{name}'
                    if hcor_name in line.element_names:
                        all_names.append(hcor_name)
                    if vcor_name in line.element_names:
                        all_names.append(vcor_name)                    

                    s=tt.rows[name].s
                    misalignment_dict[name]={}
                    misalignment_dict[name]['shift_x']=shift_values_x[ii]
                    misalignment_dict[name]['shift_y']=shift_values_y[ii]
                    misalignment_dict[name]['shift_s']=shift_values_s[ii]
                    misalignment_dict[name]['rot_s_rad_no_frame']=rot_values_s[ii]
                    for elem_name in all_names:
                        line[elem_name].shift_x = line.ref[mis_switch_name]*shift_values_x[ii]
                        line[elem_name].shift_y = line.ref[mis_switch_name]*shift_values_y[ii]
                        line[elem_name].shift_s = line.ref[mis_switch_name]*shift_values_s[ii]
                        if line[elem_name].rot_s_rad !=0:
                            line[elem_name].rot_s_rad_no_frame = line.ref[mis_switch_name]*rot_values_s[ii]
                        elif line[elem_name].rot_s_rad==0:
                            line[elem_name].rot_s_rad = 0.0 
                            line[elem_name].rot_s_rad_no_frame = line.ref[mis_switch_name]*rot_values_s[ii]


            if is_girder is not None:
                misalignment_dict={}
                #for girders the total element missalignment is the misalignment of the element plus the misalignment of the girder on which the element is placed
                for ii, name in enumerate(element_names):
                    idx = line.element_names.index(name)
                    i_left = idx
                    #a girder contains a quadrupole and the nearest sextupole, unless there is a dipole imnbetween, in which case it only contains a quadrupole
                    while i_left >= 0 and elements_with_types[i_left][1] != "RBend":
                        i_left -= 1

                    # Scan right
                    i_right = idx

                    while i_right < len(elements_with_types) and elements_with_types[i_right][1] != "RBend":
                        i_right += 1

                    # all_of_them = [elements_with_types[i][0] for i in range(i_left + 1, i_right)]

                    all_of_them = [elements_with_types[i][0] 
                            for i in range(i_left + 1, i_right) 
                            if elements_with_types[i][1] not in ("Drift", "Marker", 'Multipole')]
                    for aa,el in enumerate(all_of_them):
                        s=tt.rows[el].s
                        # if not any((smin <= s <= smax) for smin, smax in ir_ranges):
                        misalignment_dict[el]={}
                        misalignment_dict[el]['shift_x']=shift_values_x[ii]
                        misalignment_dict[el]['shift_y']=shift_values_y[ii]
                        misalignment_dict[el]['shift_s']=shift_values_s[ii]
                        misalignment_dict[el]['rot_s_rad_no_frame']=rot_values_s[ii]
                        line[el].shift_x = line.ref[mis_switch_name]*line[el].shift_x+line.ref[mis_switch_name]*shift_values_x[ii]
                        line[el].shift_y = line.ref[mis_switch_name]*line[el].shift_y+line.ref[mis_switch_name]*shift_values_y[ii]
                        line[el].shift_s = line.ref[mis_switch_name]*line[el].shift_s+line.ref[mis_switch_name]*shift_values_s[ii]
                        line[el].rot_s_rad = 0.0
                        line[el].rot_s_rad_no_frame =line.ref[mis_switch_name]*line[el].rot_s_rad_no_frame+ line.ref[mis_switch_name]*rot_values_s[ii]
    return misalignment_dict

def add_optics_correctors(line, corrector_names, corector_type=None):
    '''
    Similar to add_steering_correctors. This function adds any order correctors (dipole, quadrupole, sextupole ....) through the {knl,ksl} funcitonality of the elements.
    X and Y planes can have separate correctors.

    line: the line the correctors will be added to
    corrector_names: the names of the elements the correctors will be added to, add as lists (eg. ['Sextupole'])
    corector_type:defines the order and plane of the correctors added. eg. for a corrector names input of ['qf2a:.*','sf1a:.*'] the corrector type input will be 
    [['ksl1','knl1', knl2], ['ksl3','knl4']] which adds quadrupole correctors in x, y plane and sextupole correctors in the x plane, all placed on the 'qf2a:.*' family.
    On the 'sf1a:.*' family octupole correctors are placed on the y plane, and decapole correctors on th x plane.
    '''
    #adds any order correctors (dipole, quadrupole, sextupole ....)
    tt = line.get_table(attr=True)
    element_groups1 = []
    corector_groups1 = []
    knl_elements = []
    ksl_elements = []

    for jj, name in enumerate(corrector_names):
        if name in ['Bend','RBend','Quadrupole','Sextupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        if len(corrector_names) != len(corector_type):
            corector_type = [corector_type[0] for _ in corrector_names]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
        corector_groups1.extend([corector_type[jj]] * len(expanded_names))
    
    for i, element in enumerate(element_groups1):
        types = corector_groups1[i]

        for t in types:
            # extract the number (indicative of the order of the corrector)
            num = int(re.search(r'\d+', t).group())  
            
            if "knl" in t.lower():
                line.vars[f'knl{num}' + str(element)] = 0
                line[element].knl[num] = line.vars[f'knl{num}' + str(element)]
                if element not in knl_elements:
                    knl_elements.append(element)
            
            elif "ksl" in t.lower():
                line.vars[f'ksl{num}' + str(element)] = 0
                line[element].ksl[num] = line.vars[f'ksl{num}' + str(element)]
                if element not in ksl_elements:
                    ksl_elements.append(element)

    return (ksl_elements, knl_elements)

def add_markers(line, marker_placement, marker_name=None):
    '''
    Adds marker elements to the latticel.

    line: the line the markers will be added to
    marker_placement: the elements next to which the markers will be added
    marker_name: the distinguishing word attached to the start of the markers to distinguish them eg. {BPM}_{sf1a:1}, where marker_name=BPM and the element on 
    which the marker is attached is sf1a:1.
    '''
    tt = line.get_table(attr=True)
    # env = xt.Environment()

    added_markers = []
    element_groups1 = []
    for jj, name in enumerate(marker_placement):
        if name in ['Bend','RBend','Dipole','Quadrupole','Sextupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
    # marker_places = []

    for i, element in enumerate(element_groups1): 
        if marker_name is None: 
            marker = f"{element}_marker" 
        else: 
            marker = f"{marker_name}_{element}" 
        line.insert_element(marker, xt.Marker() , at=f"{element}")
        # Insert all markers in one call
        # line.insert(marker_places)

        added_markers.append(marker)
    line.steering_monitors_x =added_markers
    line.steering_monitors_y =added_markers

    return added_markers

def add_steering_correctors(line, corrector_names, corrector_prefix,corrector_plane='hv'):
    '''
    Similar to add_optics_correctors. This function only returns dipole correctors, with an option to have them only in the x or y plane.
    Adds new elements to the line rather than using the {knl,ksl} funcitonality contrary to add_optics_correctors.

    line: the line the correctors will be added to
    corrector_names: the names of the elements the correctors will be added to
    corrector_plane: the plane on which the correctors will act.
    corrector_prefix: list containing the prefix for the horizontal and vertical correctors e.g. ['hcor_', 'vcor_']
    '''
    tt = line.get_table(attr=True)
    element_groups1 = []


    for jj, name in enumerate(corrector_names):
        if name in ['Bend','RBend','Quadrupole','Sextupole']:
            tt_name = tt.rows[tt.element_type==name]
        else:
            tt_name = tt.rows[name]
        expanded_names = tt_name.name.tolist()
        element_groups1.extend(expanded_names)
    
    for i, element in enumerate(element_groups1):
        line.vars['knl' + f'{element}'] = 0
        line.vars['ksl' + f'{element}'] = 0
        vcor_name = f"{corrector_prefix[1]}{element}"
        hcor_name = f"{corrector_prefix[0]}{element}"
        if corrector_plane == 'hv':
            if hcor_name not in line.element_names: 
                line.insert_element(hcor_name, xt.Multipole(knl=np.array([0])), at=f'{element}')
                line[hcor_name].knl = line.vars['knl' + f'{element}']

            if vcor_name not in line.element_names:
                line.insert_element(vcor_name, xt.Multipole(ksl=np.array([0])), at=f'{element}')
                line[vcor_name].ksl = line.vars['ksl' + f'{element}']

        elif corrector_plane == 'h':
            if hcor_name not in line.element_names:
                line.insert_element(hcor_name, xt.Multipole(knl=np.array([0])), at=f'{element}')
                line[hcor_name].knl = line.vars['knl' + f'{element}']

        elif corrector_plane == 'v': 
            if vcor_name not in line.element_names:
                line.insert_element(vcor_name, xt.Multipole(ksl=np.array([0])), at=f'{element}')
                line[vcor_name].ksl = line.vars['ksl' + f'{element}']

    return

def sextupoles_strength_edit2 (line, family_name=all, error_strength=1, custom=False):
    '''
    Modifies the strength of sextupoles in the lattice by scaling their k2 component.
    Pre-determined arc/ir families for LCC_106.2.0. Sextupole names can be user-defined.

    line: the line containing the sextupole elements to be modified

    family_name: 
        - if custom=False:
            (for LCC_106.2.0)
            'all' → scales both arc and IR sextupoles  
            'arc' → scales arc sextupoles only  
            'ir'  → scales IR sextupoles only  
            The strengths of the sextupoles can then be modified by accessing the generated knob. (line.vars['k2n.weight_ir'], line.vars['k2n.weight_arc']=i)


        - if custom=True:
            interpreted as a regex/string selector for sextupole names (e.g. 'S[FD].*')
            The strengths of the sextupoles can then be modified by accessing the generated knob. (line.vars[f'k2n.weight_{family_name}']).


    error_strength: multiplicative factor applied to the sextupole strength (k2)

    custom:
        - False → uses predefined sextupole families (arc/ir/all)
        - True  → applies scaling only to sextupoles matching the user-defined family_name

    The function updates the sextupole strengths through line.vars knobs and directly
    modifies the corresponding element_refs.

    returns: None
    '''
    tt = line.get_table(attr=True)
    if custom==False:
        if family_name == 'all':
            line.vars['k2n.weight_ir'] = error_strength
            line.vars['k2n.weight_arc'] = error_strength

        elif family_name == 'ir':
            line.vars['k2n.weight_ir'] = error_strength

        elif family_name == 'arc':
            line.vars['k2n.weight_arc'] = error_strength

        line.vars['k2n.weight'] = error_strength
        tt_sext = tt.rows[tt.element_type=='Sextupole']
        if family_name == 'all':
            for ii in tt_sext.rows['SCRAB[LR].*|S[FD][MXY][12][LR].*|S[FD][12][AB].*|S[FD][1234][CIJDFM][LR].*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_ir']*line.element_refs[ii].k2._expr
        elif family_name == 'ir':
            for ii in tt_sext.rows['SCRAB[LR].*|S[FD][MXY][12][LR].*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_ir']*line.element_refs[ii].k2._expr

        elif family_name == 'arc':
            for ii in tt_sext.rows['S[FD][12][AB].*|S[FD][1234][CIJDFM][LR].*'].name:
                line.element_refs[ii].k2 = line.vars['k2n.weight_arc']*line.element_refs[ii].k2._expr
    elif custom:
        line.vars[f'k2n.weight_{family_name}'] = error_strength
        tt_sext = tt.rows[tt.element_type=='Sextupole']

        for ii in tt_sext.rows[family_name].name:
            line.element_refs[ii].k2 = line.vars[f'k2n.weight_{family_name}']*line.element_refs[ii].k2._expr
    return

def pseudo_inverse(responce_matrix, Tikhonov_lambda=None):
    '''
    Evaluates the pseudo inverse of the responce matrix using Tikhonov Regularisation.
    '''
    U, S, Vt = np.linalg.svd(responce_matrix, full_matrices=False)

    if Tikhonov_lambda is not None:
        S_reg = np.array([s / (s**2 + Tikhonov_lambda) for s in S])
        S_inv = np.diag(S_reg)
    else:
        S_inv = np.diag([1/s for s in S])

    p_inverse = Vt.T @ S_inv @ U.T
    return p_inverse

def optics_corrections(line, reference_twiss, observables, 
                       observation_points, correctors, p_inverse, 
                       Delta_mu=False, rdt=False, radiation=False,weight=1):
    '''
    Evaluates and applies neccessary corrections.     
    line: Line with misalignments
    reference_twiss: Unperturbed line twiss
    observables: ['mux', 'muy', 'dx], ['c_minus_re', 'c_minus_im', 'dy'] or ['f1001_real', 'f1001_imag','f1010_real', 'f1010_imag', 'dy']
    observation_points: Points at which the responce matrix was evaluated (BPMs)
    correctors: Correctors for provided observables. Quadrupoles for phase, beta and dx and skew quadrupoles for coupling and dy
    p_inverse: Pseudoinverse for responce matrix
    # Delta_mu: If true, the phase observables become the phase difference between consecutive BPMs
    weight: Incase the full solution cannot be applied, the weight defines what part is (eg. weight=0.7 is 70%)
    '''
    if radiation:
        twiss = line.twiss(coupling_edw_teng=True,eneloss_and_damping=True)
    else:
        twiss = line.twiss4d(coupling_edw_teng=True)
    # Prepare ideal and measured values
    if rdt:
        ideal_values = {'f1001_real': reference_twiss.rows['bpm.*']['f1001'].real,
                        'f1001_imag': reference_twiss.rows['bpm.*']['f1001'].imag,
                        'f1010_real': reference_twiss.rows['bpm.*']['f1010'].real,
                        'f1010_imag': reference_twiss.rows['bpm.*']['f1010'].imag,
                        'dy': reference_twiss.rows['bpm.*']['dy']}
        measured_values = {'f1001_real': twiss.rows['bpm.*']['f1001'].real,
                        'f1001_imag': twiss.rows['bpm.*']['f1001'].imag,
                        'f1010_real': twiss.rows['bpm.*']['f1010'].real,
                        'f1010_imag': twiss.rows['bpm.*']['f1010'].imag,
                        'dy': twiss.rows['bpm.*']['dy']}

    else:
        ideal_values = {o: reference_twiss.rows['bpm.*'][o] for o in observables}
        measured_values = {o: twiss.rows['bpm.*'][o] for o in observables}

    # Compute difference vector
    delta_y = {o: np.array(ideal_values[o]) - np.array(measured_values[o]) for o in observables}
    delta_y_vec = np.concatenate([delta_y[o] for o in observables]) 
    # Compute corrections
    delta_p = p_inverse @ delta_y_vec
    dp = delta_p.flatten()             

    # Apply corrections
    assert len(correctors) == len(dp), f"Length mismatch: {len(correctors)} knobs, {len(dp)} deltas"
    for name, magnet_shift in zip(correctors, dp):
        line.vars[name] += weight * magnet_shift

    if radiation:
        tw_corr = line.twiss(coupling_edw_teng=True,eneloss_and_damping=True)
    elif rdt: 
        tw_corr = line.twiss4d(coupling_edw_teng=True)
    else: 
        tw_corr = line.twiss4d()

    return tw_corr

def response_matrix(line,observables, obs_points, corr_elements, dk=1e-5,rdt=None, bipolar=True):
    '''
    line: Unperturbed line
    observables: ['mux', 'muy', 'dx], ['c_minus_re', 'c_minus_im', 'dy'] or ['f1001_real', 'f1001_imag','f1010_real', 'f1010_imag', 'dy']
    observation_points: Points at which the responce matrix will be evaluated (BPMs)

    corr_elements:  Correctors for provided observables. Quadrupoles for phase, beta and dx and skew quadrupoles for coupling and dy
    dk: Step size
    bipolar: Includes postive and negative steps
    '''
    tw_ref=line.twiss4d(coupling_edw_teng=True)
    response = {oo: np.zeros((len(obs_points), len(corr_elements)))
                for oo in observables}

    #positive step
    for ii, cc in enumerate(corr_elements):
        nn = cc
        print(f'Processing Positive {ii}/{len(corr_elements)}')
        line.vars[nn]+= dk
        twp = line.twiss4d(coupling_edw_teng=True)

        line.vars[nn] -= dk
        if rdt:
            response['f1001_real'][:, ii] = (twp.rows[obs_points].f1001.real-tw_ref.rows[obs_points].f1001.real)/dk
            response['f1001_imag'][:, ii] = (twp.rows[obs_points].f1001.imag-tw_ref.rows[obs_points].f1001.imag)/dk
            response['f1010_real'][:, ii] = (twp.rows[obs_points].f1010.real-tw_ref.rows[obs_points].f1010.real)/dk
            response['f1010_imag'][:, ii] = (twp.rows[obs_points].f1010.imag-tw_ref.rows[obs_points].f1010.imag)/dk
            response['dy'][:, ii] = (twp.rows[obs_points]['dy'] - tw_ref.rows[obs_points]['dy'])  / dk            
        else: 
            for observable in observables:
                response[observable][:, ii] = (
                    twp.rows[obs_points][observable] - tw_ref.rows[obs_points][observable])  / dk
    response_array1 = np.vstack([response[obs] for obs in observables])

    if bipolar:
        #negative step
        dk =-dk
        response2 = {oo: np.zeros((len(obs_points), len(corr_elements)))
                    for oo in observables}

        for ii, cc in enumerate(corr_elements):
            nn = cc
            print(f'Processing Negative {ii}/{len(corr_elements)}')
            line.vars[nn]+= dk
            twp = line.twiss4d(coupling_edw_teng=True)

            line.vars[nn] -= dk

            if rdt:
                response2['f1001_real'][:, ii] = (twp.rows[obs_points].f1001.real-tw_ref.rows[obs_points].f1001.real)/dk
                response2['f1001_imag'][:, ii] = (twp.rows[obs_points].f1001.imag-tw_ref.rows[obs_points].f1001.imag)/dk
                response2['f1010_real'][:, ii] = (twp.rows[obs_points].f1010.real-tw_ref.rows[obs_points].f1010.real)/dk
                response2['f1010_imag'][:, ii] = (twp.rows[obs_points].f1010.imag-tw_ref.rows[obs_points].f1010.imag)/dk
                response2['dy'][:, ii] = (twp.rows[obs_points]['dy'] - tw_ref.rows[obs_points]['dy'])  / dk            
            else: 
                for observable in observables:
                    response2[observable][:, ii] = (
                        twp.rows[obs_points][observable] - tw_ref.rows[obs_points][observable])  / dk
        response_array2 = np.vstack([response2[obs] for obs in observables])

    if bipolar:
        response_array=(response_array2+response_array1)/2
    else:
        response_array=response_array1

    return response_array

def phase_advance_between_consecutive_BPMs(twiss, observable):
    '''
    Returns an array with the phase difference between consecutive BPMs.
    '''
    if observable in ['mux', 'muy']:
        mu = np.array(twiss.rows['bpm.*'][observable])
        dmu = []
    twiss.rows['bpm.*']['name']
    N = len(twiss.rows['bpm.*']['name'])

    for ii in range(N):
        mu1 = mu[ii]
        mu2 = mu[(ii + 1) % N]    # wraps around automatically
        dmu.append((mu2 - mu1) % 1.0)
    return np.array(dmu) 

def get_delta_vec(twiss, reference_twiss, observables, rdt=False,Delta_mu=False):
    '''
    Returns the difference between values in two lines. Used between the unperturbed and misaligned lattice.  
    twiss: Twiss of the observed line 
    reference_twiss: Description
    observation_points: Points at which the vector should be evaluated
    observables: For what quantities should it be evaluated
    Delta_mu: If true, the phase observables become the phase difference between consecutive BPMs
    '''
    if rdt:
        ideal_values = {'f1001_real': reference_twiss.rows['bpm.*']['f1001'].real,
                        'f1001_imag': reference_twiss.rows['bpm.*']['f1001'].imag,
                        'f1010_real': reference_twiss.rows['bpm.*']['f1010'].real,
                        'f1010_imag': reference_twiss.rows['bpm.*']['f1010'].imag,
                        'dy': reference_twiss.rows['bpm.*']['dy']}
        measured_values = {'f1001_real': twiss.rows['bpm.*']['f1001'].real,
                        'f1001_imag': twiss.rows['bpm.*']['f1001'].imag,
                        'f1010_real': twiss.rows['bpm.*']['f1010'].real,
                        'f1010_imag': twiss.rows['bpm.*']['f1010'].imag,
                        'dy': twiss.rows['bpm.*']['dy']}

    elif Delta_mu:
        phase_obs = ['mux', 'muy']
        other_obs = [o for o in observables if o not in phase_obs]
        ideal_values = {o: phase_advance_between_consecutive_BPMs(reference_twiss, o) for o in phase_obs}
        measured_values = {o: phase_advance_between_consecutive_BPMs(twiss, o) for o in phase_obs}
        ideal_values.update({o: reference_twiss.rows['bpm.*'][o] for o in other_obs})
        measured_values.update({o: twiss.rows['bpm.*'][o] for o in other_obs})
    # if coupling_rdt:
        
    else:
        ideal_values = {o: reference_twiss.rows['bpm.*'][o] for o in observables}
        measured_values = {o: twiss.rows['bpm.*'][o] for o in observables}

    delta_y = {o: np.array(ideal_values[o]) - np.array(measured_values[o]) for o in observables}
    delta_y_vec = np.concatenate([delta_y[o] for o in observables])
    return delta_y_vec

def tikhonov_lcurve(M, b, min_slope_condition=20, lambdas=None, plot=True, title=None):
    """
    Computes the Tikhonov-regularized solution of a system using the L-curve method
    and automatically selects the optimal regularization parameter.

    M: response matrix.
    b: observable deviations
    min_slope_condition: a minimum slope condition for the selection of the optimal lambda.
    lambdas: array of regularization parameters to scan (default: logspace from 1e-1 to 1e15).
    plot: if True, displays the L-curve and selected corner point.
    title: optional title for the plot.

    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    UTb = U.T @ b

    if lambdas is None:
        lambdas = np.logspace(-1, 15, 2000)

    rnorms = np.empty(len(lambdas))
    snorms = np.empty(len(lambdas))
    solutions = []

    for i, lam in enumerate(lambdas):
        filt = s / (s**2 + lam)
        c = Vt.T @ (filt * UTb)
        solutions.append(c)

        res = M @ c - b
        rnorms[i] = np.linalg.norm(res)
        snorms[i] = np.linalg.norm(c)

    # log-log curve
    log_r = np.log10(rnorms)
    log_s = np.log10(snorms)
    t = np.log10(lambdas)

    # derivatives
    d1r = np.gradient(log_r, t)
    d1s = np.gradient(log_s, t)
    d2r = np.gradient(d1r, t)
    d2s = np.gradient(d1s, t)

    # curvature
    num = np.abs(d1r * d2s - d1s * d2r)
    den = (d1r**2 + d1s**2)**1.5
    curvature = num / (den + 1e-30)

    # ---- secondary slope condition ----
    slope = d1s / (d1r + 1e-30)
    abs_s = np.abs(slope)

    mask = np.zeros_like(curvature, dtype=bool)

    #Applies a minimum slope threshold for the selected optimal lambda. Ensures the elbow
    #point is selected. Value to be adjusted, depends on system optics.
    slope_threshold = min_slope_condition   

    mask = np.zeros_like(curvature, dtype=bool)

    # require large slope BEFORE the candidate point
    mask[1:] = abs_s[:-1] > slope_threshold

    # keep curvature only where condition holds
    curvature_filtered = np.where(mask, curvature, 0)

    idx_corner = np.argmax(curvature_filtered)

    # fallback if nothing passes threshold
    if curvature_filtered[idx_corner] == 0:
        idx_corner = np.argmax(curvature)

    lambda_opt = lambdas[idx_corner]
    c_opt = solutions[idx_corner]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        sc = ax.scatter(log_r, log_s, c=t, cmap='viridis', s=30)
        ax.plot(log_r, log_s, '-', color='black', linewidth=1)

        exp = int(np.floor(np.log10(lambda_opt)))
        mant = lambda_opt / 10**exp

        label = rf'corner $\lambda = {mant:.2f}\times 10^{{{exp}}}$'
        ax.scatter(log_r[idx_corner], log_s[idx_corner], c='red', s=100, label=label)
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\log_{10}(\lambda)$', fontsize=24)
        cbar.ax.tick_params(labelsize=22)

        ax.set_xlabel(r'$\log_{10} \| M c - b \|$', fontsize=24)
        ax.set_ylabel(r'$\log_{10} \| c \|$', fontsize=24)

        # if title:
        #     ax.set_title(f'L-curve (log-log) {title}', fontsize=30)
        # else:
        #     ax.set_title('L-curve (log-log)', fontsize=30)

        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.legend(fontsize=22, loc='best')
        ax.grid()

        fig.tight_layout()
        plt.show()

    return lambda_opt, c_opt

def remove_element_from_list(element_list, switch_rate):
    ''' 
    Given the list of elements affected it will remove elements randomly. Used for correctors and BPM's to see how stable the solution is.

    element_list: list of elements to be treated. Accepts nested lists eg. [[family10, family11],[family20, family21]]
    switch_rate: the rate at which the element will be removed, maximum value 1, minimum 0.

    returns the lists in the order and format provided with the modifications.
    '''
    kept = []
    removed = []

    for e in element_list:
        if random.random() < switch_rate:
            removed.append(e)
        else:
            kept.append(e)

    return kept, removed

def extract_misalignments(line, name, only_nonzero=True):
    '''
    name: name of elements of interest within brackets, eg. for arc quadrupoles ['q[fd].*a.*'] and ['Quadrupole'] for all quadrupoles
    only_nonzero: if True, includes only the elements with non zero misalignment values
    '''
    name=name[0]
    misalignment_dict={}
    tt = line.get_table(attr=True)
    if name in ['Bend','RBend','Quadrupole','Sextupole']:
        tt_name = tt.rows[tt.element_type==name]
    else:
        tt_name = tt.rows[name]
    elements = tt_name.name.tolist()
    for el_name in elements:
        try:
            el = line[el_name]  
        except KeyError:
            continue  

        if hasattr(el, "shift_x") or hasattr(el, "shift_y") \
        or hasattr(el, "shift_s") or hasattr(el, "rot_s_rad_no_frame"):

            sx = float(el.shift_x)
            sy = float(el.shift_y)
            ss = float(el.shift_s)
            rs = float(el.rot_s_rad_no_frame)
            if only_nonzero==True:
                if any([sx, sy, ss, rs]):  # only add if at least one is non-zero
                    misalignment_dict[el_name] = {
                        'shift_x': sx,
                        'shift_y': sy,
                        'rot_s_rad_no_frame': rs,
                        'shift_s': ss
                    }
            else:
                misalignment_dict[el_name] = {
                    'shift_x': sx,
                    'shift_y': sy,
                    'rot_s_rad_no_frame': rs,
                    'shift_s': ss
                }
        else:
            continue
    return misalignment_dict


#matching function from xsuite utilities adapted for chromaticity and tune correction for LCC_106.2.0
def match_tune_chroma (line, target_twiss, match_quantities='tune_chroma', method='6d'):

    if isinstance(target_twiss, dict):
        if 'tune' in match_quantities:
            target_qx = target_twiss['qx']
            target_qy = target_twiss['qy']
        if 'chroma' in match_quantities:
            target_dqx = target_twiss['dqx']
            target_dqy = target_twiss['dqy']
    else:
        if 'tune' in match_quantities:
            target_qx = target_twiss.qx
            target_qy = target_twiss.qy
        if 'chroma' in match_quantities:
            target_dqx = target_twiss.dqx
            target_dqy = target_twiss.dqy

    if 'tune' in match_quantities:
        if 'k1qf2' in line.vars.get_table().name:
            opt_tune = line.match(
                method=method,
                vary=[xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8, tag='quad'),
                    ],
                targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                    ])
        
        elif 'kqf6' in line.vars.get_table().name:
            opt_tune = line.match(
                method=method,
                vary=[xt.VaryList(['kqf2', 'kqf4', 'kqf6', 'kqd1', 'kqd3', 'kqd5',], step=1e-8, tag='quad'),
                    ],
                targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                    ])
        
        elif 'kqf2' in line.vars.get_table().name:
            opt_tune = line.match(
                method=method,
                vary=[xt.VaryList(['kqf2', 'kqd1', ], step=1e-8, tag='quad'),
                    ],
                targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                    ])
        
        opt_tune.target_status()
        opt_tune.vary_status()

    if 'chroma' in match_quantities:
        opt_chroma = line.match(
            method=method,
            vary=[xt.VaryList(['ksd2', 'ksf2', 'ksf1', 'ksd1',], step=1e-3, tag='sext'),
                ],
            targets=[xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                ])
        opt_chroma.target_status()
        opt_chroma.vary_status()

    return

#Plotting functions taken from @Kyriakos Skoufaris (MA_vs_turns, DA_vs_turns) 
def DA_vs_turns(particles, num_r_steps, num_theta_steps, x_norm, y_norm, delta_initial, output_dir=None,delta_plots=False):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    if delta_plots and np.size(delta_initial) > 1:

        for ii in np.unique(delta_initial):
            delta_index = np.where(delta_initial==ii)[0]
            
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
            x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
            y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
            part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
                        
            x_DA = np.full(num_theta_steps, np.nan)
            y_DA = np.full(num_theta_steps, np.nan)
            for jj in range(num_theta_steps):
                for ii in range(num_r_steps):
                    if part_at_turn_2d[ii,jj] != max_turns:
                        x_DA[jj] = x_norm_2d[ii,jj]
                        y_DA[jj] = y_norm_2d[ii,jj]
                        break

            min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
            where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
            
            # Plot DA using scatter and pcolormesh
            fig = plt.subplots()
            plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            #plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_x/1000}$]')
            plt.tick_params(axis='both', labelsize=14)
            cb = plt.colorbar()
            cb.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')

            fig = plt.subplots()
            plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            #plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))    
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_x/1000}$]')
            plt.tick_params(axis='both', labelsize=14)

            ax = plt.colorbar()
            ax.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')
            if output_dir is not None:  # <<< ADDED
                import os
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, "DA.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    else:

        if not delta_plots and np.size(delta_initial) > 1:
            closest_to_zero_delta = delta_initial[(np.abs(delta_initial - 0)).argmin()]
            delta_index = np.where(delta_initial==closest_to_zero_delta)[0]
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
        else:
            x_norm_1d = x_norm
            y_norm_1d = y_norm      
            part_at_turn_1d = part_at_turn

        x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
        y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
        part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
        x_DA = np.full(num_theta_steps, np.nan)
        y_DA = np.full(num_theta_steps, np.nan)
        for jj in range(num_theta_steps):
            for ii in range(num_r_steps):
                if part_at_turn_2d[ii,jj] != max_turns:
                    x_DA[jj] = x_norm_2d[ii,jj]
                    y_DA[jj] = y_norm_2d[ii,jj]
                    break

        min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
        where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
        
        # Plot DA using scatter and pcolormesh
        fig = plt.subplots()
        plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        #plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]', fontsize=16, fontweight='bold')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_x/1000}$]', fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', labelsize=14)
        cb = plt.colorbar()
        cb.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')

        fig = plt.subplots()
        plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        #plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))    
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]', fontsize=16, fontweight='bold')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_x/1000}$]', fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', labelsize=14)
        ax = plt.colorbar()
        ax.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')
        if output_dir is not None:  # <<< ADDED
            import os
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "DA.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    return (x_DA, y_DA, where_min_DA)

def MA_vs_turns(particles, num_r_steps, num_delta_steps, x_norm, y_norm, delta_initial, output_dir=None):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    theta = np.max(np.unique(np.arctan2(y_norm,x_norm)))

    x_norm_2d = x_norm.reshape(num_delta_steps, num_r_steps)
    y_norm_2d = y_norm.reshape(num_delta_steps, num_r_steps)
    delta_norm_2d = delta_initial.reshape(num_delta_steps, num_r_steps)
    part_at_turn = part_at_turn
    part_at_turn_2d = part_at_turn.reshape(num_delta_steps, num_r_steps)
    x_MA = np.full(num_delta_steps, np.nan)
    y_MA = np.full(num_delta_steps, np.nan)
    delta_MA = np.full(num_delta_steps, np.nan)
    for jj in range(num_delta_steps):
        for ii in range(num_r_steps):
            if part_at_turn_2d[jj,ii] != max_turns:
                x_MA[jj] = x_norm_2d[jj,ii]
                y_MA[jj] = y_norm_2d[jj,ii]
                delta_MA[jj] = delta_norm_2d[jj,ii]
                break

    min_MA = np.nanmin(np.round(np.sqrt(x_MA**2+y_MA**2),1)) 
    where_min_MA = np.where(np.round(np.sqrt(x_MA**2+y_MA**2),1) == min_MA)[0]
    
    # Plot MA using scatter and pcolormesh
    fig = plt.subplots()
    plt.scatter(delta_initial*100, x_norm, c=part_at_turn)
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    #plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.1f$\sigma$'%(min_MA))
    plt.xlabel(r'$\delta$ [%]', fontsize=16, fontweight='bold')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_x/1000}$]'%(theta*180/np.pi), fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    cb = plt.colorbar()
    cb.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')

    fig = plt.subplots()
    plt.pcolormesh(delta_norm_2d*100, x_norm_2d, part_at_turn_2d, shading='gouraud')
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    #plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.1f$\sigma$'%(min_MA))    
    plt.xlabel(r'$\delta$ [%]', fontsize=16, fontweight='bold')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_x/1000}$]'%(theta*180/np.pi), fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    ax = plt.colorbar()
    ax.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')
    if output_dir is not None:  # <<< ADDED
        import os
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "MA.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return (x_MA, delta_MA, where_min_MA)


#Plotting functions to analyse single seed optics
def closed_orbit_vs_s(reference_twiss, studied_twiss, outline=None, corr=False):

    orbit_div_x = studied_twiss.x-reference_twiss.x
    orbit_div_y = studied_twiss.y-reference_twiss.y

    fig, ax = plt.subplots(figsize=(10,6))

    if np.max(np.abs([orbit_div_x.max(), orbit_div_x.min()])) > np.max(np.abs([orbit_div_y.max(), orbit_div_y.min()])):
        ax.plot(studied_twiss.s, orbit_div_x, '-', color='crimson', label='x corrected' if corr else 'x', alpha=0.7)
        ax.plot(studied_twiss.s, orbit_div_y, '-', color='y',label='y corrected' if corr else 'y', alpha=0.7)

        ip_names = reference_twiss.rows['ip.[1-8]$'].name
        for ii in ip_names:
            plt.axvline(x = reference_twiss.rows[ii].s, linestyle = '--', color = 'black', linewidth=0.5)
            plt.text(reference_twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))

    else:
        ax.plot(studied_twiss.s, orbit_div_y, '-', color='y', label='y corrected' if corr else 'y', alpha=0.7)
        ax.plot(studied_twiss.s, orbit_div_x, '-', color='crimson', label='x corrected' if corr else 'x', alpha=0.7)
        ip_names = reference_twiss.rows['ip:[1-8]$'].name
        for ii in ip_names:
            plt.axvline(x = reference_twiss.rows[ii].s, linestyle = '--', color = 'black',linewidth=0.5)
            plt.text(reference_twiss.rows[ii].s, 0.8*ax.get_ylim()[1], ii, horizontalalignment='center',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))

    ax.set_xlabel(r'$s\,[\mathrm{m}]$', fontsize=24)
    ax.set_ylabel(r'$(x - x_{nom};\; y - y_{nom})\,[\mathrm{m}]$', fontsize=24)
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = studied_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8 * ax.get_ylim()[1], ip,
            ha='center', va='center', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True),)
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.offsetText.set_size(15)

    ax.tick_params(axis='both', which='major', labelsize=22)  # Increase major ticks

    ax.legend(fontsize=18, loc='best')
    ax.grid()
    # plt.title(f'Closed Orbit Deviation, Seed {135}', fontsize=30)

    plt.tight_layout()
    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return orbit_div_x, orbit_div_y

def beta_beating_vs_s(reference_twiss, studied_twiss, zoom=1, outline=None):

    beta_beat_x = (studied_twiss.betx - reference_twiss.betx) / reference_twiss.betx
    beta_beat_y = (studied_twiss.bety - reference_twiss.bety) / reference_twiss.bety

    fig, ax = plt.subplots(figsize=(10,6))

    # Plot dominant plane first
    if np.max(np.abs([beta_beat_x.max(), beta_beat_x.min()])) > \
       np.max(np.abs([beta_beat_y.max(), beta_beat_y.min()])):

        ax.plot(reference_twiss.s, beta_beat_x*100, '-', color='crimson', label='x', alpha=0.7)
        ax.plot(reference_twiss.s, beta_beat_y*100, '-', color='y', label='y', alpha=0.7)
    else:
        ax.plot(reference_twiss.s, beta_beat_y*100, '-', color='y', label='y', alpha=0.7)
        ax.plot(reference_twiss.s, beta_beat_x*100, '-', color='crimson', label='x', alpha=0.7)

    # IP markers
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = studied_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8 * ax.get_ylim()[1], ip,
            ha='center', va='center', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )

    ax.set_xlabel(r'$s\ [\mathrm{m}]$', fontsize=24)
    ax.set_ylabel(
        r'$(\beta - \beta_{ref}) / \beta_{ref}$ [%]',

        fontsize=24
    )

    if zoom is not None:
        ax.set_ylim(-zoom,zoom)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=18, loc='best')
    ax.grid()
    # plt.title(f'Beta Beating, Seed {135}', fontsize=30)
    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return np.sqrt(np.mean(beta_beat_x**2)), np.sqrt(np.mean(beta_beat_y**2))

def dispersion_deviation_vs_s(reference_twiss, studied_twiss, outline=None):
    # Compute deviations
    disp_div_x = studied_twiss.dx - reference_twiss.dx
    disp_div_y = studied_twiss.dy - reference_twiss.dy

    fig, ax = plt.subplots(figsize=(10,6))

    # Plot dominant plane first
    if np.max(np.abs([disp_div_x.max(), disp_div_x.min()])) > \
       np.max(np.abs([disp_div_y.max(), disp_div_y.min()])):

        ax.plot(reference_twiss.s, disp_div_x, '-', color='crimson', label='x', alpha=0.7)
        ax.plot(reference_twiss.s, disp_div_y, '-', color='y', label='y', alpha=0.7)
    else:
        ax.plot(reference_twiss.s, disp_div_y, '-', color='y', label='y', alpha=0.7)
        ax.plot(reference_twiss.s, disp_div_x, '-', color='crimson', label='x', alpha=0.7)

    # IP markers
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = studied_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8 * ax.get_ylim()[1], ip,
            ha='center', va='center', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )
    ax.set_xlabel('s [m]', fontsize=24)
    ax.set_ylabel(
        r'$(D - D_{ref})$ [m]',
        fontsize=24
    )

    # Scientific notation styling
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.offsetText.set_size(15)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=18, loc='best')
    ax.grid()
    # plt.title(f'Dispersion Deviation, Seed {135}', fontsize=30)

    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return np.sqrt(np.mean(disp_div_x**2)), np.sqrt(np.mean(disp_div_y**2))

def coupling_vs_s(studied_twiss, outline=None):
    """
    Plot deviation of linear coupling components (real & imaginary)
    along the machine.
    """

    # Coupling deviations
    coupling_re = studied_twiss.c_minus_re 
    coupling_im = studied_twiss.c_minus_im 

    fig, ax = plt.subplots(figsize=(10,6))

    # Plot dominant component first
    if np.max(np.abs([coupling_re.max(), coupling_re.min()])) > \
       np.max(np.abs([coupling_im.max(), coupling_im.min()])):

        ax.plot(studied_twiss.s, coupling_re, '-', color='crimson',
                label=r'$\Re\{c_-\}$', alpha=0.7)
        ax.plot(studied_twiss.s, coupling_im, '-', color='royalblue',
                label=r'$\Im\{c_-\}$', alpha=0.7)
    else:
        ax.plot(studied_twiss.s, coupling_im, '-', color='royalblue',
                label=r'$\Im\{c_-\}$', alpha=0.7)
        ax.plot(studied_twiss.s, coupling_re, '-', color='crimson',
                label=r'$\Re\{c_-\}$', alpha=0.7)

    # IP markers
    ip_names = studied_twiss.rows['ip.[1-8]$'].name


    for ip in ip_names:
        s_ip = studied_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8 * ax.get_ylim()[1], ip,
            ha='center', va='center', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )

    # Labels
    ax.set_xlabel('s [m]', fontsize=18, fontweight='bold')
    ax.set_ylabel(
        r'$\mathbf{c_- }$',
        fontsize=18, fontweight='bold'
    )

    # Scientific notation styling
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.offsetText.set_size(15)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18, loc='best')
    ax.grid()
    plt.title(f'C minus Seed {135}', fontsize=22, fontweight='bold')
    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return np.sqrt(np.mean(coupling_re**2)), np.sqrt(np.mean(coupling_im**2))

def coupling_rdt_single(twiss, outline=None):

    fig, ax = plt.subplots(figsize=(10,6))

    s = twiss.s

    f1001_re = np.real(twiss.f1001)
    f1001_im = np.imag(twiss.f1001)
    f1010_re = np.real(twiss.f1010)
    f1010_im = np.imag(twiss.f1010)

    # plot
    ax.plot(s, f1001_re, '-', color='crimson', label=r'$\Re f_{1001}$')
    ax.plot(s, f1001_im, '-', color='darkorange', label=r'$\Im f_{1001}$')
    ax.plot(s, f1010_re, '-', color='royalblue', label=r'$\Re f_{1010}$')
    ax.plot(s, f1010_im, '-', color='limegreen', label=r'$\Im f_{1010}$')

    # BPM / IP markers
    bpm_rows = twiss.rows['bpm.*']
    bpm_s = bpm_rows.s
    ip_names = twiss.rows['ip.[1-8]$'].name
    # Scientific notation styling
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.offsetText.set_size(15)

    for ip in ip_names:
        s_ip = twiss.rows[ip].s
        bpm_idx = np.argmin(np.abs(bpm_s - s_ip))

        bpm_s_closest = bpm_s[bpm_idx]

        ax.axvline(x=bpm_s_closest, linestyle='--', color='black', linewidth=0.7)

        ax.text(
            bpm_s_closest,
            0.8 * ax.get_ylim()[1],
            ip,
            ha='center',
            va='center', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )

    # labels
    ax.set_xlabel('s [m]', fontsize=24)
    ax.set_ylabel('Amplitude', fontsize=24)

    ax.tick_params(axis='both', labelsize=22)
    ax.legend(fontsize=18, loc='best')
    ax.grid(True)

    # ax.set_title('Coupling RDTs, Seed 135', fontsize=30)

    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # RMS along the ring
    rms = {
        'f1001_re': np.sqrt(np.mean(f1001_re**2)),
        'f1001_im': np.sqrt(np.mean(f1001_im**2)),
        'f1010_re': np.sqrt(np.mean(f1010_re**2)),
        'f1010_im': np.sqrt(np.mean(f1010_im**2))
    }

    return rms

#Plotting functions to study many seeds
def beta_beating_vs_s_manyseed(reference_twiss, studied_twisses_dict,
                               zoom=None, outline=None, outlier_sigma=3):

    fig_x, ax_x = plt.subplots(figsize=(10, 6))
    fig_y, ax_y = plt.subplots(figsize=(10, 6))

    beta_x_all = []
    beta_y_all = []

    seed_keys = list(studied_twisses_dict.keys())

    # compute beta beating and plot faint seeds
    for studied_twiss in studied_twisses_dict.values():
        beta_beat_x = (studied_twiss.betx - reference_twiss.betx)*100 / reference_twiss.betx
        beta_beat_y = (studied_twiss.bety - reference_twiss.bety)*100 / reference_twiss.bety

        beta_x_all.append(beta_beat_x)
        beta_y_all.append(beta_beat_y)

        ax_x.plot(reference_twiss.s, beta_beat_x, '-', color='grey', zorder=2,linewidth=1, alpha=0.5)
        ax_y.plot(reference_twiss.s, beta_beat_y, '-', color='grey', zorder=2,linewidth=1, alpha=0.5)

    beta_x_all = np.array(beta_x_all)
    beta_y_all = np.array(beta_y_all)

    # mean and std
    beta_x_mean = beta_x_all.mean(axis=0)
    beta_y_mean = beta_y_all.mean(axis=0)
    beta_x_std = beta_x_all.std(axis=0)
    beta_y_std = beta_y_all.std(axis=0)

    # === X PLANE ===
    ax_x.plot(reference_twiss.s, beta_x_mean, '-', color='crimson',  zorder=10,linewidth=1.5)
    ax_x.fill_between(reference_twiss.s,
                      beta_x_mean - beta_x_std,
                      beta_x_mean + beta_x_std,
                      color='crimson', alpha=0.4, zorder=8, label='x mean ±1σ')

    # === Y PLANE ===
    ax_y.plot(reference_twiss.s, beta_y_mean, '-', color='goldenrod',  zorder=10,linewidth=1.5)
    ax_y.fill_between(reference_twiss.s,
                      beta_y_mean - beta_y_std,
                      beta_y_mean + beta_y_std,
                      color='goldenrod', alpha=0.4, zorder=8, label='y mean ±1σ')

    # IP markers (both plots)
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s
        for ax in [ax_x, ax_y]:
            ax.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.5)
            ax.text(s_ip, 0.8 * ax.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1))

    # labels
    ax_x.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)
    ax_y.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)

    ax_x.set_ylabel(
        r'$(\boldsymbol{\beta}_{\boldsymbol{x}}-\boldsymbol{\beta}_{\boldsymbol{x,ref}})/\boldsymbol{\beta}_{\boldsymbol{x,ref}}\ [\%]$',
        fontsize=24
    )
    ax_y.set_ylabel(
        r'$(\boldsymbol{\beta}_{\boldsymbol{y}}-\boldsymbol{\beta}_{\boldsymbol{y,ref}})/\boldsymbol{\beta}_{\boldsymbol{y,ref}}\ [\%]$',
        fontsize=24
    )

    # styling
    for ax in [ax_x, ax_y]:
        if zoom is not None:
            ax.set_ylim(-zoom, zoom)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=18, loc='lower left')
        ax.grid()

    fig_x.suptitle('Beta Beating X', fontsize=30, fontweight='bold')
    fig_y.suptitle('Beta Beating Y', fontsize=30, fontweight='bold')

    fig_x.tight_layout()
    fig_y.tight_layout()

    if outline is not None:
        fig_x.savefig(outline.replace(".png", "_x.png"), dpi=300, bbox_inches="tight")
        fig_y.savefig(outline.replace(".png", "_y.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_y)
    else:
        plt.show()

    # RMS per seed
    rms_x_seeds = np.sqrt(np.mean(beta_x_all**2, axis=1))
    rms_y_seeds = np.sqrt(np.mean(beta_y_all**2, axis=1))

    # outliers
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
        (rms_x_seeds[i] > rms_x_seeds.mean() + outlier_sigma*rms_x_seeds.std() or
         rms_y_seeds[i] > rms_y_seeds.mean() + outlier_sigma*rms_y_seeds.std())]

    outlier_sigmas = [
        (
            (rms_x_seeds[i] - rms_x_seeds.mean()) / rms_x_seeds.std() if rms_x_seeds.std() != 0 else 0,
            (rms_y_seeds[i] - rms_y_seeds.mean()) / rms_y_seeds.std() if rms_y_seeds.std() != 0 else 0
        )
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    rms_x = np.mean(rms_x_seeds)
    rms_y = np.mean(rms_y_seeds)

    return rms_x, rms_y, np.std(rms_x_seeds), np.std(rms_y_seeds), [outliers, outlier_sigmas]

def closed_orbit_vs_s_manyseed(reference_twiss, studied_twisses_dict,
                               outline=None, corr=False, outlier_sigma=3):

    fig_x, ax_x = plt.subplots(figsize=(10, 6))
    fig_y, ax_y = plt.subplots(figsize=(10, 6))

    orbit_x_all = []
    orbit_y_all = []

    seed_keys = list(studied_twisses_dict.keys())

    # compute orbit deviations and plot faint seeds
    for studied_twiss in studied_twisses_dict.values():
        orbit_x = studied_twiss.x - reference_twiss.x
        orbit_y = studied_twiss.y - reference_twiss.y

        orbit_x_all.append(orbit_x)
        orbit_y_all.append(orbit_y)

        ax_x.plot(reference_twiss.s, orbit_x, '-', color='grey', alpha=0.3)
        ax_y.plot(reference_twiss.s, orbit_y, '-', color='grey', alpha=0.3)

    orbit_x_all = np.array(orbit_x_all)
    orbit_y_all = np.array(orbit_y_all)

    orbit_x_mean = orbit_x_all.mean(axis=0)
    orbit_y_mean = orbit_y_all.mean(axis=0)

    orbit_x_std = orbit_x_all.std(axis=0)
    orbit_y_std = orbit_y_all.std(axis=0)

    # === X PLANE ===
    ax_x.plot(reference_twiss.s, orbit_x_mean, '-', color='crimson')
    ax_x.fill_between(reference_twiss.s,
                      orbit_x_mean - orbit_x_std,
                      orbit_x_mean + orbit_x_std,
                      color='crimson', alpha=0.55,zorder=2,
                      label='x mean ±1σ')

    # === Y PLANE ===
    ax_y.plot(reference_twiss.s, orbit_y_mean, '-', color='goldenrod')
    ax_y.fill_between(reference_twiss.s,
                      orbit_y_mean - orbit_y_std,
                      orbit_y_mean + orbit_y_std,
                      color='goldenrod', alpha=0.55,zorder=2,
                      label='y mean ±1σ')

    # IP markers
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s
        for ax in [ax_x, ax_y]:
            ax.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.7)
            ax.text(s_ip, 0.8*ax.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat',zorder=1, alpha=1))

    # labels
    ax_x.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)
    ax_y.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)

    ax_x.set_ylabel(
        r'$(\boldsymbol{x}-\boldsymbol{x}_{\boldsymbol{nom}})$ [m]',
        fontsize=24,
    fontweight='bold'
    )
    ax_y.set_ylabel(
        r'$(\boldsymbol{y}-\boldsymbol{y}_{\boldsymbol{nom}})$ [m]',
        fontsize=24,
    fontweight='bold'
    )

    # formatting
    for ax in [ax_x, ax_y]:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.offsetText.set_size(24)

        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.legend(fontsize=18, loc='lower left')
        ax.grid()

    fig_x.suptitle('Closed Orbit X', fontsize=30, fontweight='bold')
    fig_y.suptitle('Closed Orbit Y', fontsize=30, fontweight='bold')

    fig_x.tight_layout()
    fig_y.tight_layout()

    if outline is not None:
        fig_x.savefig(outline.replace(".png", "_x.png"), dpi=300, bbox_inches="tight")
        fig_y.savefig(outline.replace(".png", "_y.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_y)
    else:
        plt.show()

    # RMS per seed
    rms_x_seeds = np.sqrt(np.mean(orbit_x_all**2, axis=1))
    rms_y_seeds = np.sqrt(np.mean(orbit_y_all**2, axis=1))

    # detect outliers
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
        (rms_x_seeds[i] > rms_x_seeds.mean() + outlier_sigma*rms_x_seeds.std() or
         rms_x_seeds[i] < rms_x_seeds.mean() - outlier_sigma*rms_x_seeds.std() or
         rms_y_seeds[i] > rms_y_seeds.mean() + outlier_sigma*rms_y_seeds.std() or
         rms_y_seeds[i] < rms_y_seeds.mean() - outlier_sigma*rms_y_seeds.std())]

    outlier_sigmas = [
        (
            (rms_x_seeds[i] - rms_x_seeds.mean()) / rms_x_seeds.std() if rms_x_seeds.std() != 0 else 0,
            (rms_y_seeds[i] - rms_y_seeds.mean()) / rms_y_seeds.std() if rms_y_seeds.std() != 0 else 0
        )
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    rms_x = np.mean(rms_x_seeds)
    rms_y = np.mean(rms_y_seeds)

    return rms_x, rms_y, np.std(rms_x_seeds), np.std(rms_y_seeds), [outliers, outlier_sigmas]

def dispersion_deviation_vs_s_manyseed(reference_twiss, studied_twisses_dict,
                                       outline=None, outlier_sigma=3, zoom=None):

    fig_x, ax_x = plt.subplots(figsize=(10, 6))
    fig_y, ax_y = plt.subplots(figsize=(10, 6))

    disp_x_all = []
    disp_y_all = []

    seed_keys = list(studied_twisses_dict.keys())

    # compute dispersion deviations and plot faint seeds
    for studied_twiss in studied_twisses_dict.values():
        disp_x = studied_twiss.dx - reference_twiss.dx
        disp_y = studied_twiss.dy - reference_twiss.dy

        disp_x_all.append(disp_x)
        disp_y_all.append(disp_y)

        ax_x.plot(reference_twiss.s, disp_x, '-', color='grey',  zorder=2,alpha=0.5)
        ax_y.plot(reference_twiss.s, disp_y, '-', color='grey',  zorder=2,alpha=0.5)

    disp_x_all = np.array(disp_x_all)
    disp_y_all = np.array(disp_y_all)

    disp_x_mean = disp_x_all.mean(axis=0)
    disp_y_mean = disp_y_all.mean(axis=0)

    disp_x_std = disp_x_all.std(axis=0)
    disp_y_std = disp_y_all.std(axis=0)

    # === X PLANE ===
    ax_x.plot(reference_twiss.s, disp_x_mean, '-',  zorder=10,color='crimson')
    ax_x.fill_between(reference_twiss.s,
                      disp_x_mean - disp_x_std,
                      disp_x_mean + disp_x_std,
                      color='crimson', alpha=0.55, zorder=8,
                      label='x mean ±1σ')

    # === Y PLANE ===
    ax_y.plot(reference_twiss.s, disp_y_mean, '-', zorder=10, color='goldenrod')
    ax_y.fill_between(reference_twiss.s,
                      disp_y_mean - disp_y_std,
                      disp_y_mean + disp_y_std,
                      color='goldenrod', alpha=0.55, zorder=8,
                      label='y mean ±1σ')

    # zoom
    if zoom is not None:
        ax_x.set_ylim(-zoom, zoom)
        ax_y.set_ylim(-zoom, zoom)

    # IP markers
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s
        for ax in [ax_x, ax_y]:
            ax.axvline(x=s_ip, linestyle='--', color='black',zorder=1, linewidth=0.5)
            ax.text(
                s_ip, 0.8*ax.get_ylim()[1], ip,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1)
            )

    # labels
    ax_x.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)
    ax_y.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)

    ax_x.set_ylabel(
        r'$(\boldsymbol{D}_{\boldsymbol{x}}-\boldsymbol{D}_{\boldsymbol{ref}})$ [m]',
        fontsize=24,
    fontweight='bold'
    )
    ax_y.set_ylabel(
        r'$(\boldsymbol{D}_{\boldsymbol{y}}-\boldsymbol{D}_{\boldsymbol{ref}})$ [m]',
        fontsize=24,
    fontweight='bold'
    )

    # formatting
    for ax in [ax_x, ax_y]:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.offsetText.set_size(24)

        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.legend(fontsize=18, loc='lower left')
        ax.grid()

    fig_x.suptitle('Dispersion Deviation X', fontsize=30, fontweight='bold')
    fig_y.suptitle('Dispersion Deviation Y', fontsize=30, fontweight='bold')

    fig_x.tight_layout()
    fig_y.tight_layout()

    if outline is not None:
        fig_x.savefig(outline.replace(".png", "_x.png"), dpi=300, bbox_inches="tight")
        fig_y.savefig(outline.replace(".png", "_y.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_y)
    else:
        plt.show()

    # RMS per seed
    rms_x_seeds = np.sqrt(np.mean(disp_x_all**2, axis=1))
    rms_y_seeds = np.sqrt(np.mean(disp_y_all**2, axis=1))

    # detect outliers
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
        (rms_x_seeds[i] > rms_x_seeds.mean() + outlier_sigma*rms_x_seeds.std() or
         rms_x_seeds[i] < rms_x_seeds.mean() - outlier_sigma*rms_x_seeds.std() or
         rms_y_seeds[i] > rms_y_seeds.mean() + outlier_sigma*rms_y_seeds.std() or
         rms_y_seeds[i] < rms_y_seeds.mean() - outlier_sigma*rms_y_seeds.std())]

    outlier_sigmas = [
        (
            (rms_x_seeds[i] - rms_x_seeds.mean()) / rms_x_seeds.std() if rms_x_seeds.std() != 0 else 0,
            (rms_y_seeds[i] - rms_y_seeds.mean()) / rms_y_seeds.std() if rms_y_seeds.std() != 0 else 0
        )
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    rms_x = np.mean(rms_x_seeds)
    rms_y = np.mean(rms_y_seeds)

    return rms_x, rms_y, np.std(rms_x_seeds), np.std(rms_y_seeds), [outliers, outlier_sigmas]

def tune_spread(reference_twiss, studied_twisses_dict, outline=None, outlier_sigma=3):
    """
    Plot overlaid histograms of absolute tune deviations |Qx-Qx_ref| and |Qy-Qy_ref|
    for many seeds, with log-scaled x-axis.
    """

    Qx_devs, Qy_devs = [], []
    seed_keys = list(studied_twisses_dict.keys())
    Qx_seed, Qy_seed = [], []

    for tw in studied_twisses_dict.values():
        Qx_dev = np.abs(tw.qx - reference_twiss.qx)
        Qy_dev = np.abs(tw.qy - reference_twiss.qy)

        Qx_devs.extend(np.ravel(Qx_dev))
        Qy_devs.extend(np.ravel(Qy_dev))

        Qx_seed.append(np.mean(np.abs(Qx_dev)))
        Qy_seed.append(np.mean(np.abs(Qy_dev)))

    Qx_devs = np.array(Qx_devs)
    Qy_devs = np.array(Qy_devs)
    Qx_seed = np.array(Qx_seed)
    Qy_seed = np.array(Qy_seed)

    # Use log bins (avoid zero)
    min_nonzero = min(Qx_devs[Qx_devs>0].min(), Qy_devs[Qy_devs>0].min())
    max_val = max(Qx_devs.max(), Qy_devs.max())
    n_bins = 30
    min_val = min_nonzero * 0.9
    max_val = max_val * 1.1
    bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins+1)


    # Plot histograms
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(Qx_devs, bins=bins, color='crimson', alpha=0.5,
            edgecolor='k', label=r'$|Q_x - Q_{x,ref}|$')
    ax.hist(Qy_devs, bins=bins, color='royalblue', alpha=0.5,
            edgecolor='k', label=r'$|Q_y - Q_{y,ref}|$')

    ax.set_xscale('log')  # log x-axis
    ax.set_xlabel(r'Absolute tune deviation $|Q_{x,y}-Q_{x,y,\mathrm{ref}}|$', fontsize=24, fontweight='bold')
    ax.set_ylabel('Counts', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=18, loc='best')

    plt.title('Tune Spread', fontsize=30, fontweight='bold')
    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Outlier detection per seed
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
                (Qx_seed[i] > Qx_seed.mean() + outlier_sigma*Qx_seed.std() or
                 Qx_seed[i] < Qx_seed.mean() - outlier_sigma*Qx_seed.std() or
                 Qy_seed[i] > Qy_seed.mean() + outlier_sigma*Qy_seed.std() or     
                 Qy_seed[i] < Qy_seed.mean() - outlier_sigma*Qy_seed.std())]

    outlier_sigmas = [
        ((Qx_seed[i] - Qx_seed.mean()) / Qx_seed.std() if Qx_seed.std() != 0 else 0,
         (Qy_seed[i] - Qy_seed.mean()) / Qy_seed.std() if Qy_seed.std() != 0 else 0)
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    Qx_rms_seeds = np.sqrt(Qx_seed**2)
    Qy_rms_seeds = np.sqrt(Qy_seed**2)
    Qx_rms, Qy_rms = np.mean(Qx_rms_seeds), np.mean(Qy_rms_seeds)
    Qx_std, Qy_std = np.std(Qx_rms_seeds), np.std(Qy_rms_seeds)

    outlier = [outliers, outlier_sigmas]
    return Qx_rms, Qy_rms, Qx_std, Qy_std, outlier, Qx_devs,Qy_devs

def chromaticity_spread(reference_twiss, studied_twisses_dict, outline=None, outlier_sigma=3):
    """
    Plot overlaid histograms of absolute chromaticity deviations 
    |dQx-dQx_ref| and |dQy-dQy_ref| for many seeds with log-scaled x-axis.
    """

    dQx_devs, dQy_devs = [], []
    seed_keys = list(studied_twisses_dict.keys())
    dQx_seed, dQy_seed = [], []

    for tw in studied_twisses_dict.values():
        dQx_dev = np.abs(tw.dqx - reference_twiss.dqx)
        dQy_dev = np.abs(tw.dqy - reference_twiss.dqy)

        dQx_devs.extend(np.ravel(dQx_dev))
        dQy_devs.extend(np.ravel(dQy_dev))

        dQx_seed.append(np.mean(np.abs(dQx_dev)))
        dQy_seed.append(np.mean(np.abs(dQy_dev)))

    dQx_devs = np.array(dQx_devs)
    dQy_devs = np.array(dQy_devs)
    dQx_seed = np.array(dQx_seed)
    dQy_seed = np.array(dQy_seed)

    # Logarithmic bins (avoid zero)
    min_nonzero = min(dQx_devs[dQx_devs>0].min(), dQy_devs[dQy_devs>0].min())
    max_val = max(dQx_devs.max(), dQy_devs.max())
    n_bins = 30
    min_val = min_nonzero * 0.9
    max_val = max_val * 1.1
    bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins+1)


    # Plot histograms
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(dQx_devs, bins=bins, color='crimson', alpha=0.5, edgecolor='k', label=r'$|dQ_x - dQ_{x,ref}|$')
    ax.hist(dQy_devs, bins=bins, color='royalblue', alpha=0.5, edgecolor='k', label=r'$|dQ_y - dQ_{y,ref}|$')

    ax.set_xscale('log')  # log x-axis
    ax.set_xlabel(r'Absolute chromaticity deviation $|dQ_{x,y}-dQ_{x,y,\mathrm{ref}}|$', fontsize=24, fontweight='bold')
    ax.set_ylabel('Counts', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=18, loc='best')

    plt.title('Chromaticity Spread', fontsize=30, fontweight='bold')
    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Outlier detection per seed
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
                (dQx_seed[i] > dQx_seed.mean() + outlier_sigma*dQx_seed.std() or
                 dQx_seed[i] < dQx_seed.mean() - outlier_sigma*dQx_seed.std() or
                 dQy_seed[i] > dQy_seed.mean() + outlier_sigma*dQy_seed.std() or
                 dQy_seed[i] < dQy_seed.mean() - outlier_sigma*dQy_seed.std())]

    outlier_sigmas = [
        ((dQx_seed[i] - dQx_seed.mean()) / dQx_seed.std() if dQx_seed.std() != 0 else 0,
         (dQy_seed[i] - dQy_seed.mean()) / dQy_seed.std() if dQy_seed.std() != 0 else 0)
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    dQx_rms = np.mean(np.abs(dQx_seed))
    dQy_rms = np.mean(np.abs(dQy_seed))
    dQx_std = np.std(np.abs(dQx_seed))
    dQy_std = np.std(np.abs(dQy_seed))

    outlier = [outliers, outlier_sigmas]
    return dQx_rms, dQy_rms, dQx_std, dQy_std, outlier ,dQx_devs,dQy_devs

def coupling_vs_s_manyseed(reference_twiss,studied_twisses_dict, outline=None, outlier_sigma=3):

    fig, ax = plt.subplots(figsize=(10, 6))

    c_re_all = []
    c_im_all = []

    seed_keys = list(studied_twisses_dict.keys())

    # compute coupling and plot faint seeds
    for studied_twiss in studied_twisses_dict.values():

        c_re = studied_twiss.c_minus_re
        c_im = studied_twiss.c_minus_im

        c_re_all.append(c_re)
        c_im_all.append(c_im)

        ax.plot(studied_twiss.s, c_re, '-', color='grey', alpha=0.04)
        ax.plot(studied_twiss.s, c_im, '-', color='grey', alpha=0.04)

    c_re_all = np.array(c_re_all)
    c_im_all = np.array(c_im_all)

    c_re_mean = c_re_all.mean(axis=0)
    c_im_mean = c_im_all.mean(axis=0)

    c_re_std = c_re_all.std(axis=0)
    c_im_std = c_im_all.std(axis=0)

    # Plot dominant component first
    if np.max(np.abs([c_re_mean.max(), c_re_mean.min()])) > \
       np.max(np.abs([c_im_mean.max(), c_im_mean.min()])):

        ax.plot(studied_twiss.s, c_re_mean, '-', color='crimson')
        ax.fill_between(studied_twiss.s,
                        c_re_mean - c_re_std,
                        c_re_mean + c_re_std,
                        color='crimson', alpha=0.55,
                        label=r'$\Re\{c_-\}$ mean ±1σ')

        ax.plot(studied_twiss.s, c_im_mean, '-', color='royalblue')
        ax.fill_between(studied_twiss.s,
                        c_im_mean - c_im_std,
                        c_im_mean + c_im_std,
                        color='royalblue', alpha=0.55,
                        label=r'$\Im\{c_-\}$ mean ±1σ')

    else:

        ax.plot(studied_twiss.s, c_im_mean, '-', color='royalblue')
        ax.fill_between(studied_twiss.s,
                        c_im_mean - c_im_std,
                        c_im_mean + c_im_std,
                        color='royalblue', alpha=0.55,
                        label=r'$\Im\{c_-\}$ mean ±1σ')

        ax.plot(studied_twiss.s, c_re_mean, '-', color='crimson')
        ax.fill_between(studied_twiss.s,
                        c_re_mean - c_re_std,
                        c_re_mean + c_re_std,
                        color='crimson', alpha=0.55,
                        label=r'$\Re\{c_-\}$ mean ±1σ')

    # IP markers
    ip_names = studied_twiss.rows['ip.[1-8]$'].name
    for ip in ip_names:
        s_ip = studied_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8*ax.get_ylim()[1], ip,
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )

    # Labels
    ax.set_xlabel('s [m]', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{c_- }$', fontsize=24, fontweight='bold')

    # Scientific notation styling
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.offsetText.set_size(24)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=18, loc='lower left')
    ax.grid()
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s
        ax.axvline(x=s_ip, linestyle='--', color='black', linewidth=0.5)
        ax.text(
            s_ip, 0.8*ax.get_ylim()[1], ip,
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )
    plt.title(f'C minus', fontsize=30, fontweight='bold')

    plt.tight_layout()

    if outline is not None:
        plt.savefig(outline, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # RMS per seed
    rms_re_seeds = np.sqrt(np.mean(c_re_all**2, axis=1))
    rms_im_seeds = np.sqrt(np.mean(c_im_all**2, axis=1))

    # detect outliers
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
                (rms_re_seeds[i] > rms_re_seeds.mean() + outlier_sigma*rms_re_seeds.std() or
                 rms_re_seeds[i] < rms_re_seeds.mean() - outlier_sigma*rms_re_seeds.std() or
                 rms_im_seeds[i] > rms_im_seeds.mean() + outlier_sigma*rms_im_seeds.std() or
                 rms_im_seeds[i] < rms_im_seeds.mean() - outlier_sigma*rms_im_seeds.std())]
    outlier_sigmas = [
        (
            (rms_re_seeds[i] - rms_re_seeds.mean()) / rms_re_seeds.std() if rms_re_seeds.std() != 0 else 0,
            (rms_im_seeds[i] - rms_im_seeds.mean()) / rms_im_seeds.std() if rms_im_seeds.std() != 0 else 0
        )
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]
    rms_re = np.mean(rms_re_seeds)
    rms_im = np.mean(rms_im_seeds)
    outlier= [outliers,outlier_sigmas]
    return rms_re, rms_im, np.std(rms_re_seeds), np.std(rms_im_seeds), outlier
    
def coupling_rdt(reference_twiss, studied_twisses_dict, outline=None, outlier_sigma=3, zoom=None):
    """
    Many-seed coupling RDTs split into f1001 and f1010 plots with mean ±1σ bands and RMS statistics.
    """
    # separate figures
    fig_f1001, ax_f1001 = plt.subplots(figsize=(10, 6))
    fig_f1010, ax_f1010 = plt.subplots(figsize=(10, 6))

    seed_keys = list(studied_twisses_dict.keys())

    f1001_re_all, f1001_im_all = [], []
    f1010_re_all, f1010_im_all = [], []
    s = reference_twiss.s

    for tw in studied_twisses_dict.values():
        f1001_re, f1001_im = np.real(tw.f1001), np.imag(tw.f1001)
        f1010_re, f1010_im = np.real(tw.f1010), np.imag(tw.f1010)

        f1001_re_all.append(f1001_re)
        f1001_im_all.append(f1001_im)
        f1010_re_all.append(f1010_re)
        f1010_im_all.append(f1010_im)

        # faint seeds
        ax_f1001.plot(s, f1001_re, '-', color='grey', zorder=2, alpha=0.5)
        ax_f1001.plot(s, f1001_im, '-', color='grey',  zorder=2,alpha=0.5)
        ax_f1010.plot(s, f1010_re, '-', color='grey',  zorder=2,alpha=0.5)
        ax_f1010.plot(s, f1010_im, '-', color='grey',  zorder=2,alpha=0.5)

    # arrays
    f1001_re_all = np.array(f1001_re_all)
    f1001_im_all = np.array(f1001_im_all)
    f1010_re_all = np.array(f1010_re_all)
    f1010_im_all = np.array(f1010_im_all)

    # mean and std
    stats = {}
    stats['f1001_re'] = f1001_re_all.mean(axis=0), f1001_re_all.std(axis=0)
    stats['f1001_im'] = f1001_im_all.mean(axis=0), f1001_im_all.std(axis=0)
    stats['f1010_re'] = f1010_re_all.mean(axis=0), f1010_re_all.std(axis=0)
    stats['f1010_im'] = f1010_im_all.mean(axis=0), f1010_im_all.std(axis=0)

    bpm_rows = reference_twiss.rows['bpm.*']
    bpm_s = bpm_rows.s
    ip_names = reference_twiss.rows['ip.[1-8]$'].name

    # zoom
    if zoom is not None:
        ax_f1001.set_ylim(-zoom, zoom)
        ax_f1010.set_ylim(-zoom, zoom)

    # IP markers
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s
        bpm_idx = np.argmin(np.abs(bpm_s - s_ip))
        bpm_s_closest = bpm_s[bpm_idx]
        for ax in [ax_f1001, ax_f1010]:
            ax.axvline(x=bpm_s_closest, linestyle='--', color='black',zorder=1, linewidth=0.7)
            ax.text(bpm_s_closest, 0.8*ax.get_ylim()[1], ip,
                    ha='center', va='center', bbox=dict(boxstyle='round',zorder=1, facecolor='wheat', alpha=1))

    # f1001 plot
    ax_f1001.plot(s, stats['f1001_re'][0], '-', zorder=10, color='crimson')
    ax_f1001.fill_between(s, stats['f1001_re'][0]-stats['f1001_re'][1], stats['f1001_re'][0]+stats['f1001_re'][1],
                          color='crimson', alpha=0.6,  zorder=8, label=r'$\Re f_{1001}$ mean ±1σ')
    ax_f1001.plot(s, stats['f1001_im'][0], '-',  zorder=10,color='darkorange')
    ax_f1001.fill_between(s, stats['f1001_im'][0]-stats['f1001_im'][1], stats['f1001_im'][0]+stats['f1001_im'][1],
                          color='darkorange', alpha=0.6, zorder=8, label=r'$\Im f_{1001}$ mean ±1σ')

    ax_f1001.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)
    ax_f1001.set_ylabel('Amplitude', fontsize=24, fontweight='bold')
    ax_f1001.tick_params(axis='both', which='major', labelsize=24)
    ax_f1001.legend(fontsize=18, loc='lower left')
    ax_f1001.grid(True)
    # ax_f1001.set_title('Coupling RDT f1001', fontsize=30, fontweight='bold')

    # f1010 plot
    ax_f1010.plot(s, stats['f1010_re'][0], '-',zorder=10, color='royalblue')
    ax_f1010.fill_between(s, stats['f1010_re'][0]-stats['f1010_re'][1], stats['f1010_re'][0]+stats['f1010_re'][1],
                          color='royalblue', alpha=0.6,zorder=8, label=r'$\Re f_{1010}$ mean ±1σ')
    ax_f1010.plot(s, stats['f1010_im'][0], '-', zorder=10,color='limegreen')
    ax_f1010.fill_between(s, stats['f1010_im'][0]-stats['f1010_im'][1], stats['f1010_im'][0]+stats['f1010_im'][1],
                          color='limegreen', alpha=0.6,zorder=8, label=r'$\Im f_{1010}$ mean ±1σ')

    ax_f1010.set_xlabel(r'$\boldsymbol{s}\ [\boldsymbol{m}]$', fontsize=24)
    ax_f1010.set_ylabel('Amplitude', fontsize=24, fontweight='bold')
    ax_f1010.tick_params(axis='both', which='major', labelsize=24)
    ax_f1010.legend(fontsize=18, loc='lower left')
    ax_f1010.grid(True)
    # ax_f1010.set_title('Coupling RDT f1010', fontsize=30, fontweight='bold')
    ip_names = reference_twiss.rows['ip.[1-8]$'].name

    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s

        ax_f1010.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.5)
        ax_f1010.text(s_ip, 0.8 * ax_f1010.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1))

    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s

        ax_f1001.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.5)
        ax_f1001.text(s_ip, 0.8 * ax_f1001.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1))

    plt.tight_layout()

    if outline is not None:
        fig_f1001.savefig(outline.replace(".png", "_f1001.png"), dpi=300, bbox_inches="tight")
        fig_f1010.savefig(outline.replace(".png", "_f1010.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_f1001)
        plt.close(fig_f1010)
    else:
        plt.show()

    # RMS per seed
    rms_f1001_re_seeds = np.sqrt(np.mean(f1001_re_all**2, axis=1))
    rms_f1001_im_seeds = np.sqrt(np.mean(f1001_im_all**2, axis=1))
    rms_f1010_re_seeds = np.sqrt(np.mean(f1010_re_all**2, axis=1))
    rms_f1010_im_seeds = np.sqrt(np.mean(f1010_im_all**2, axis=1))

    # outlier detection
    outliers = [seed_keys[i] for i in range(len(seed_keys)) if
        (rms_f1001_re_seeds[i] > rms_f1001_re_seeds.mean() + outlier_sigma*rms_f1001_re_seeds.std() or
         rms_f1001_im_seeds[i] > rms_f1001_im_seeds.mean() + outlier_sigma*rms_f1001_im_seeds.std() or
         rms_f1010_re_seeds[i] > rms_f1010_re_seeds.mean() + outlier_sigma*rms_f1010_re_seeds.std() or
         rms_f1010_im_seeds[i] > rms_f1010_im_seeds.mean() + outlier_sigma*rms_f1010_im_seeds.std())]

    outlier_sigmas = [
        (
            (rms_f1001_re_seeds[i] - rms_f1001_re_seeds.mean()) / rms_f1001_re_seeds.std() if rms_f1001_re_seeds.std() != 0 else 0,
            (rms_f1001_im_seeds[i] - rms_f1001_im_seeds.mean()) / rms_f1001_im_seeds.std() if rms_f1001_im_seeds.std() != 0 else 0,
            (rms_f1010_re_seeds[i] - rms_f1010_re_seeds.mean()) / rms_f1010_re_seeds.std() if rms_f1010_re_seeds.std() != 0 else 0,
            (rms_f1010_im_seeds[i] - rms_f1010_im_seeds.mean()) / rms_f1010_im_seeds.std() if rms_f1010_im_seeds.std() != 0 else 0,
        )
        for i in range(len(seed_keys)) if seed_keys[i] in outliers
    ]

    return {
        'f1001_re': (np.mean(rms_f1001_re_seeds), np.std(rms_f1001_re_seeds)),
        'f1001_im': (np.mean(rms_f1001_im_seeds), np.std(rms_f1001_im_seeds)),
        'f1010_re': (np.mean(rms_f1010_re_seeds), np.std(rms_f1010_re_seeds)),
        'f1010_im': (np.mean(rms_f1010_im_seeds), np.std(rms_f1010_im_seeds))
    }, [outliers, outlier_sigmas]

def plot_corrector_mean_manyseed(lines_dict, highlight_seed=None, zoom=None, outline=None):
    """
    Plot mean ± std of sextupole corrector strengths (knl[1] and ksl[1]) across multiple lines.
    Optionally overlay a specific seed.
    """
    seed_keys = list(lines_dict.keys())
    reference_twiss=lines_dict[seed_keys[0]].twiss4d()
    # get s positions and element names
    tt_ref = lines_dict[seed_keys[0]].get_table()
    sext_names = tt_ref.rows[tt_ref.element_type == 'Sextupole'].name
    s_all = np.array([float(tt_ref.rows[name].s) for name in sext_names])

    # collect data across seeds
    knl_all = np.array([[lines_dict[key][name].knl[1] for name in sext_names] for key in seed_keys])
    ksl_all = np.array([[lines_dict[key][name].ksl[1] for name in sext_names] for key in seed_keys])

    # compute mean and std
    knl_mean = knl_all.mean(axis=0)
    knl_std = knl_all.std(axis=0)
    ksl_mean = ksl_all.mean(axis=0)
    ksl_std = ksl_all.std(axis=0)

    colors = ['crimson', 'goldenrod']

    # --- knl[1] plot ---
    plt.figure(figsize=(10,6))

    # seed in background
    if highlight_seed is not None and highlight_seed in seed_keys:
        knl_seed = [lines_dict[highlight_seed][name].knl[1] for name in sext_names]
        plt.scatter(s_all, knl_seed, color='blue', s=35, alpha=0.4,
                    label=f'Seed {highlight_seed}', zorder=1)

    # mean + std on top
    plt.fill_between(s_all, knl_mean-knl_std, knl_mean+knl_std,
                     color=colors[0], alpha=0.2, label='±1σ', zorder=2)
    plt.plot(s_all, knl_mean, '-', color=colors[0], linewidth=1.5,
             label=r'$knl_{1}$ mean', zorder=3)
     # IP markers (both plots)
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    ax = plt.gca()
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s

        plt.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.5)
        plt.text(s_ip, 0.8 * ax.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1))

    plt.xlabel('s [m]', fontsize=24, fontweight='bold')
    plt.ylabel(
    r'$\mathbf{Integrated\ Corrector\ Strength}\ [\boldsymbol{m}^{\boldsymbol{-1}}]$',
    fontsize=24, fontweight='bold'
    )
    plt.title('Normal Quadrupole Corrector Mean', fontsize=30, fontweight='bold')
    plt.grid(True)
    if zoom is not None:
        plt.ylim(-zoom, zoom)

    # scientific formatting
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0,0))
    ax.yaxis.offsetText.set_size(14)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if outline is not None:
        plt.savefig(outline+'_knl_mean.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # --- ksl[1] plot ---
    plt.figure(figsize=(10,6))

    if highlight_seed is not None and highlight_seed in seed_keys:
        ksl_seed = [lines_dict[highlight_seed][name].ksl[1] for name in sext_names]
        plt.scatter(s_all, ksl_seed, color='blue', s=35, alpha=0.6,
                    label=f'Seed {highlight_seed}', zorder=1)

    plt.fill_between(s_all, ksl_mean-ksl_std, ksl_mean+ksl_std,
                     color=colors[1], alpha=0.2, label='±1σ', zorder=2)
    plt.plot(s_all, ksl_mean, '-', color=colors[1], linewidth=1.5,
             label=r'$ksl_{1}$ mean', zorder=3)
    # IP markers (both plots)
    ip_names = reference_twiss.rows['ip.[1-8]$'].name
    ax = plt.gca()
    for ip in ip_names:
        s_ip = reference_twiss.rows[ip].s

        plt.axvline(x=s_ip, linestyle='--', color='black', zorder=1,linewidth=0.5)
        plt.text(s_ip, 0.8 * ax.get_ylim()[1], ip,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', zorder=1,alpha=1))

    plt.xlabel('s [m]', fontsize=24, fontweight='bold')
    plt.ylabel(
    r'$\mathbf{Integrated\ Corrector\ Strength}\ [\boldsymbol{m}^{\boldsymbol{-1}}]$',
    fontsize=24, fontweight='bold'
    )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.title('Skew Quadrupole Corrector Mean', fontsize=30, fontweight='bold')
    plt.grid(True)
    if zoom is not None:
        plt.ylim(-zoom, zoom)

    # scientific formatting

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0,0))
    ax.yaxis.offsetText.set_size(14)

    plt.legend(fontsize=12)
    plt.tight_layout()
    if outline is not None:
        plt.savefig(outline+'_ksl_mean.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return knl_mean, ksl_mean, knl_std, ksl_std

def plot_DA_MA_seeds(results_file, reference_seed_key,highlight_worst_seed_DA=None, highlight_worst_seed_MA=None, outline_DA=None, outline_MA=None):

    seed_keys = list(results_file.keys())

    ref_DA_y = np.array(results_file[reference_seed_key]["DA"]["y"])
    ref_MA_x = np.array(results_file[reference_seed_key]["MA"]["x"])

    DA_deviation = []
    MA_deviation = []

    # ===== compute deviations =====
    for seed in seed_keys:
        if seed == reference_seed_key:
            continue

        y_DA = np.array(results_file[seed]["DA"]["y"])
        x_MA = np.array(results_file[seed]["MA"]["x"])

        mask_DA = y_DA < ref_DA_y
        if np.any(mask_DA):
            DA_deviation.append((seed, np.mean(ref_DA_y[mask_DA] - y_DA[mask_DA])))

        mask_MA = x_MA < ref_MA_x
        if np.any(mask_MA):
            MA_deviation.append((seed, np.mean(ref_MA_x[mask_MA] - x_MA[mask_MA])))

    # top 5
    DA_top5 = [s for s, _ in sorted(DA_deviation, key=lambda x: x[1], reverse=True)[:1]]
    MA_top5 = [s for s, _ in sorted(MA_deviation, key=lambda x: x[1], reverse=True)[:1]]

    # assign colors
    worst_colors = ['gold']
    color_map = {}

    for i, seed in enumerate(DA_top5):
        color_map[seed] = worst_colors[i]

    # ===== DA plot =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in seed_keys:
        x_DA = np.array(results_file[seed]["DA"]["x"])
        y_DA = np.array(results_file[seed]["DA"]["y"])

        if seed == reference_seed_key:
            ax.plot(x_DA, y_DA, '-', color='crimson', linewidth=3, label='Reference')

        elif seed in DA_top5:
            if highlight_worst_seed_DA:
                ax.plot(x_DA, y_DA, '-', color=color_map[seed], linewidth=2,
                    label=f'{seed}')
            else:
                ax.plot(x_DA, y_DA, '-', color='grey', alpha=0.2, linewidth=1)        
        else:
            ax.plot(x_DA, y_DA, '-', color='grey', alpha=0.2, linewidth=1)

    ax.set_xlabel(
    r'$\mathbf{\hat{x}}\,[\mathbf{\sqrt{\varepsilon_x}}]$',
    fontsize=24
    )

    ax.set_ylabel(
    r'$\mathbf{\hat{y}}\,[\mathbf{\sqrt{\varepsilon_x/1000}}]$',
    fontsize=24
    )
    ax.tick_params(axis='both', labelsize=24)
    ax.grid()
    ax.legend(fontsize='large')
    plt.title(f'Dynamic Aperture', fontsize=30, fontweight='bold')
    plt.tight_layout()

    if outline_DA:
        plt.savefig(outline_DA, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # ===== MA plot =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in seed_keys:
        x_MA = np.array(results_file[seed]["MA"]["x"])
        delta_MA = np.array(results_file[seed]["MA"]["delta"])

        if seed == reference_seed_key:
            ax.plot(100*delta_MA, x_MA, '-', color='royalblue', linewidth=3, label='Reference')

        elif seed in MA_top5:
            if highlight_worst_seed_MA:
                ax.plot(100*delta_MA, x_MA, '-', color=color_map.get(seed, 'red'),
                    linewidth=2, label=f'{seed}')
            else:
                ax.plot(100*delta_MA, x_MA, '-', color='grey', alpha=0.2, linewidth=1)
        else:
            ax.plot(100*delta_MA, x_MA, '-', color='grey', alpha=0.2, linewidth=1)

    ax.set_xlabel(
    r'$\mathbf{\delta}\,[\mathbf{\%}]$',
    fontsize=24
    )
    ax.set_ylabel(
    r'$\mathbf{\hat{x}}$ [$\mathbf{\sqrt{\varepsilon_x}}$], $\mathbf{\hat{y}}$ [Tan(%.1f)$\mathbf{\sqrt{\varepsilon_x/1000}}$]' % 45,
    fontsize=24,
    fontweight='bold'
    )

    #ax.set_ylabel(
    #r'$\mathbf{\hat{x}}$',
    #fontsize=24
    #)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid()
    ax.legend(fontsize='large')
    plt.title(f'Momentum Acceptance', fontsize=30, fontweight='bold')
    plt.tight_layout()

    if outline_MA:
        plt.savefig(outline_MA, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {
        "seeds":seed_keys
    }

def plot_emittance_seeds(results_file, eps_x_ideal, eps_y_ideal):
    seed_keys = list(results_file.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    all_eps_x = []
    all_eps_y = []
    above_limit_seeds = []

    for seed in seed_keys:
        eps_x = np.atleast_1d(results_file[seed]["geometric_equilibrium_emittance_x"])
        eps_y = np.atleast_1d(results_file[seed]["geometric_equilibrium_emittance_y"])

        all_eps_x.append(eps_x)
        all_eps_y.append(eps_y)

        # collect seeds above threshold
        if np.any(eps_y > eps_y_ideal):
            above_limit_seeds.append(seed)

        # plot (no highlighting)
        ax.plot(eps_x * 1e9, eps_y * 1e12, 'o',
                color='red', alpha=1, markersize=10)

    ax.set_yscale('log')

    # ideal reference lines
    ax.axvline(eps_x_ideal * 1e9, linestyle='--', color='seagreen',
               linewidth=3, label='Nominal $\epsilon_x$ = 0.7138nm')
    ax.axhline(eps_y_ideal * 1e12, linestyle='--', color='royalblue',
               linewidth=3, label='Nominal $\epsilon_y$ = 0.7138pm')

    ax.set_xlabel(
        r'$\mathbf{\varepsilon_x}\,[\mathbf{nm}\cdot\mathbf{rad}]$',
        fontsize=24
    )
    ax.set_ylabel(
        r'$\mathbf{\varepsilon_y}\,[\mathbf{pm}\cdot\mathbf{rad}]$',
        fontsize=24
    )

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.offsetText.set_size(24)
    ax.xaxis.offsetText.set_size(24)

    ax.grid()
    ax.legend(fontsize='large', loc='best')
    plt.title('Equilibrium Emittance', fontsize=30, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # flatten arrays
    all_eps_x = np.concatenate(all_eps_x)
    all_eps_y = np.concatenate(all_eps_y)

    # statistics
    mean_eps_x = np.mean(all_eps_x)
    std_eps_x = np.std(all_eps_x)
    mean_eps_y = np.mean(all_eps_y)
    std_eps_y = np.std(all_eps_y)

    return (mean_eps_x, std_eps_x), (mean_eps_y, std_eps_y), above_limit_seeds



















#trying the IR biasing
def pseudo_inverse_IR(responce_matrix,weights, Tikhonov_lambda=None):
    W = np.sqrt(weights)
    Wmat = np.diag(W)

    # Weighted system
    Rw = Wmat @ responce_matrix

    U, S, Vt = np.linalg.svd(Rw, full_matrices=False)

    S_reg = np.array([s / (s**2 + Tikhonov_lambda) for s in S])
    S_inv = np.diag(S_reg)
    

    p_inverse = Vt.T @ S_inv @ U.T
    return p_inverse

def optics_corrections_IR_BIASING(line, reference_twiss, observables, 
                       observation_points, correctors, p_inverse, weights=None):
    W = np.sqrt(weights)
    Wmat = np.diag(W)
    twiss = line.twiss4d()

    ideal_values = {o: reference_twiss.rows['bpm.*'][o] for o in observables}
    measured_values = {o: twiss.rows['bpm.*'][o] for o in observables}

    # Compute difference vector
    delta_y = {o: np.array(ideal_values[o]) - np.array(measured_values[o]) for o in observables}
    delta_y_vec = np.concatenate([delta_y[o] for o in observables]) 
   
    if weights is not None:
        W = np.sqrt(weights)
        Wmat = np.diag(W)
        delta_y_vec = Wmat @ delta_y_vec
    #delta_y_vec=get_delta_vec(line.twiss4d(), tw_ref, observation_points,observables, Delta_mu=Delta_mu)
    # Compute corrections
    delta_p = p_inverse @ delta_y_vec   
    dp = delta_p.flatten()             
          

    # Apply corrections
    assert len(correctors) == len(dp), f"Length mismatch: {len(correctors)} knobs, {len(dp)} deltas"
    for name, magnet_shift in zip(correctors, dp):
        line.vars[name] += magnet_shift

    tw_corr = line.twiss4d()
    return tw_corr

def weighted_svd_correction(R, b, weights, lam=0.0):
    """
    Weighted least squares using SVD.
    R: response matrix (m x n)
    b: error vector     (m,)
    weights: weights for each measurement (m,)
    lam: Tikhonov regularization
    """
    # Build weighting matrix W^(1/2)
    W = np.sqrt(weights)
    Wmat = np.diag(W)

    # Weighted system
    Rw = Wmat @ R
    bw = Wmat @ b

#     # SVD
#     U, S, VT = np.linalg.svd(Rw, full_matrices=False)

#     # Regularized inverse
#     S_reg = np.array([s / (s**2 + lam) for s in S])
#     S_inv = np.diag(S_reg)

#     # S_inv = S / (S**2 + lam)

#     p_inverse = VT.T @ S_inv @ U.T

# #    dk = VT.T @ (S_inv * (U.T @ bw))

#     dk = p_inverse @ bw
    return Rw







def plot_misalignments(line):
    arc_quads = ['q[fd].*a.*']
    arc_dipoles = ['dl1a.*']
    arc_sext = ['s[fd][12]a.*']

    quads = extract_misalignments(line, arc_quads,only_nonzero=False)
    sext = extract_misalignments(line, arc_sext,only_nonzero=False)
    dipoles = extract_misalignments(line, arc_dipoles,only_nonzero=False)

    for name, data_dict in zip(['Quadrupole', 'Sextupole', 'Dipole'], [quads, sext, dipoles]):
        names = list(data_dict.keys())
        shift_x = [data_dict[k]["shift_x"] for k in names]
        shift_y = [data_dict[k]["shift_y"] for k in names]
        shift_s = [data_dict[k]["shift_s"] for k in names]
        rot_s   = [data_dict[k]["rot_s_rad_no_frame"] for k in names]

        plt.figure(figsize=(8,5))

        # Plot histograms
        plt.hist(shift_s, bins=100, alpha=0.5, color='teal', label='shift_s')
        plt.hist(shift_x, bins=100, alpha=0.5, color='blue', label='shift_x')
        plt.hist(shift_y, bins=100, alpha=0.5, color='green', label='shift_y')
        plt.hist(rot_s,   bins=100, alpha=0.5, color='red', label='rot_s_rad')

        # Use scientific notation for x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-6,6))

        plt.xlabel("Misalignment value (m)")
        plt.ylabel("Counts")
        plt.title(f"{name} misalignments")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    return 


def try_scale(line, correctors, scale, dp_full,
              reference_twiss, observables, observation_points):
    try:
        # correction
        for name, d in zip(correctors, scale * dp_full):
            line.vars[name] += float(d)

        # evaluate Twiss
        tw = line.twiss4d()
        dvec = get_delta_vec(twiss=tw, reference_twiss=reference_twiss,
                            observables=observables, observation_points=observation_points)
        dist = np.linalg.norm(dvec)

    except Exception as e:
        # mark as failed
        tw = None
        dvec = None
        dist = None
        failed = True
    else:
        failed = False

    finally:
        # revert applied correction
        for name, d in zip(correctors, scale * dp_full):
            try:
                line.vars[name] -= float(d)
            except:
                pass

    return dict(
        scale=scale,
        distance=dist,
        delta_vec=dvec,
        twiss=tw,
        failed=failed
    )
def corr_and_scan_min(
        
    line, response_matrix, reference_twiss, observables,
    observation_points, correctors,
    step=0.01, tol=1e-7, max_scan_steps=20):
    U, S, Vt = np.linalg.svd(response_matrix, full_matrices=False)
    S_inv = np.diag([1/s if s > tol*np.max(S) else 0 for s in S])
    p_inverse = Vt.T @ S_inv @ U.T

    failed_scales = []

    try:
        twiss = line.twiss4d()
        d0 = get_delta_vec(twiss=twiss, reference_twiss=reference_twiss,
                        observables=observables, observation_points=observation_points)
        dp_full = (p_inverse @ d0).flatten()
    except Exception:
        print("Twiss evaluation failed at nominal solution (scale=1.0)")
        failed_scales.append(1.0)
        twiss = None
        d0 = None
        dp_full = None

    trials = []
    trials.append(try_scale(line=line, correctors=correctors, scale=1.0, dp_full=dp_full,
              reference_twiss=reference_twiss, observables=observables, observation_points=observation_points))

    cur_best = trials[-1]["distance"]

    # scan +
    for k in range(1, max_scan_steps + 1):
        sc = 1.0 + k * step
        t=try_scale(line=line, correctors=correctors, scale=sc, dp_full=dp_full,
              reference_twiss=reference_twiss, observables=observables, 
              observation_points=observation_points)
        if t["failed"]:
            failed_scales.append(sc)
        else:
            trials.append(t)

    # scan - 
    for k in range(1, max_scan_steps + 1):
        sc = 1.0 - k * step
        t = try_scale(line=line, correctors=correctors, scale=sc, dp_full=dp_full,
              reference_twiss=reference_twiss, observables=observables, 
              observation_points=observation_points)
        if t["failed"]:
            failed_scales.append(sc)
        else:
            trials.append(t)
    # best
    # filter only successful trials
    successful_trials = [t for t in trials if t["distance"] is not None]

    if successful_trials:
        best = min(successful_trials, key=lambda x: x["distance"])
        best_dp = best["scale"] * dp_full
    else:
        best = None
        best_dp = None
        print("No successful trial found")


    # apply best
    for name, d in zip(correctors, best_dp):
        line.vars[name] += d

    final_tw = line.twiss4d()
    final_tw.plot(" ".join(observables))

    # ✅ top-5 smallest error
    sorted_trials = sorted(successful_trials, key=lambda x: x["distance"])
    top5 = sorted_trials[:5]


    successful_trials_dict = {
        "scale":   [successful_trials[i]["scale"]   for i in range(np.shape(successful_trials)[0] )], 
        "distance":[successful_trials[i]["distance"] for i in range(np.shape(successful_trials)[0]
    )]
    }
    tw_data = {k: np.array(final_tw[k]) for k in ["s","betx","bety","dx","dy","x","y","mux","muy","c_minus_re", "c_minus_im", "dx"]}

    return dict(
        twiss_final=tw_data,
        best_trial=best['scale'],
        top5_trials= [t["scale"] for t in top5],
        successful_trials=successful_trials_dict,
        failed_scales=failed_scales
    )

def jacobian():
    
    # targets = [
    #     xt.TargetSet(observables[0], value=tw_ref, at=bpm, tol=1)
    #     for bpm in obs_points]
    #     + [ xt.TargetSet(observables[1], value=tw_ref, at=bpm, tol=1)
    #     for bpm in obs_points]
    #     + [ xt.TargetSet(observables[2], value=tw_ref, at=bpm, tol=1)
    #     for bpm in obs_points
    # ]

    # opt_p = line2.match(
    #     solve=False,
    #     method='4d',
    #     start=xt.START, end=xt.END,
    #     init=tw_ref, init_at=xt.START,
    #     vary=xt.VaryList(corr_elements, step=dk, limits=[-10, 10]),
    #     targets=targets
    # )

    # jac_p = opt_p._err.get_jacobian(opt_p._err._get_x())

    # opt_m = line2.match(
    #     solve=False,
    #     method='4d',
    #     start=xt.START, end=xt.END,
    #     init=tw_ref, init_at=xt.START,
    #     vary=xt.VaryList(corr_elements, step=-dk, limits=[-10, 10]),
    #     targets=targets
    # )

    # jac_m = opt_m._err.get_jacobian(opt_m._err._get_x())
    # jac=(jac_m+jac_p)/2


    with open("jacobian.pkl", "wb") as f:
        pickle.dump({
            "jac": jac,
        }, f)



#unsure if they are useful
def beta_score(twiss, ref, tau_x, tau_y):

    beta_x      = twiss.betx
    beta_y      = twiss.bety
    beta_x_ref  = ref.betx
    beta_y_ref  = ref.bety

    beat_x = np.abs((beta_x_ref - beta_x) / beta_x_ref)
    beat_y = np.abs((beta_y_ref - beta_y) / beta_y_ref)

    score_x = np.mean(beat_x < tau_x)
    score_y = np.mean(beat_y < tau_y)

    return score_x, score_y

def scan_weights(line, responce_matrix, reference_twiss, observables, observation_points,
                 correctors, weight_list, tau_x=0.05, tau_y=0.1, p_inverse=None, Delta_mu=False):

    results = []

    for w in weight_list:
        # copy line to avoid accumulating corrections
        line_copy = line.copy()
        try:
            tw_corr, p_inv = optics_corrections(
                line_copy, responce_matrix, reference_twiss, observables,
                observation_points, correctors, p_inverse=p_inverse,
                Delta_mu=Delta_mu, weight=w
            )
            score_x, score_y = beta_score(tw_corr, reference_twiss, tau_x, tau_y)

        except Exception as e:
            score_x, score_y = 0, 0

        results.append({
            'weight': w,
            'score_x': score_x,
            'score_y': score_y,
        })
    results_dict = {
    'weight': [r['weight'] for r in results],
    'score_x': [r['score_x'] for r in results],
    'score_y': [r['score_y'] for r in results]
    }
    return results_dict
def plot_scan_weights():
    weights = results['weight']
    score_x = results['score_x']
    score_y = results['score_y']
    plt.figure(figsize=(8,6))

    sc = plt.scatter(score_x, score_y, c=weights, cmap='viridis')
    plt.colorbar(sc, label="weight")

    # Annotate each point with its weight value
    for x, y, w in zip(score_x, score_y, weights):
        plt.text(x, y, str(w), fontsize=10, ha='center', va='bottom')
    plt.title(
        "Impact of solution weight on beta beating\n"
        "for orbit-corrected response matrix with bet(x/y) observables")
    plt.xlabel("score x (threshold 5%)")
    plt.ylabel("score y (threshold 10%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







#trying the coupling
def skew_strength(line, element_type):
    '''
    For the sextupoleit is important to check how far the center of the magnet is from the beam. 
    Now only the center of the magnet shift is checked, the orbit deviation also needs to be implemented for it to be more accurate.
    '''
    tt= line.get_table(attr=True)
    if element_type=='Quadrupole':
        elems = tt.rows[tt.element_type == 'Quadrupole']
        strengths = pd.Series(elems.k1l, index=elems.name)

        mis = extract_misalignments(line, ['Quadrupole'], only_nonzero=False)
        mis_df = pd.DataFrame.from_dict(mis, orient="index")[['rot_s_rad_no_frame']]
        mis_df = mis_df.reindex(strengths.index).fillna(0)

        Jw =- strengths * np.sin(2 * mis_df['rot_s_rad_no_frame'].to_numpy()) 
       
    elif element_type == 'Sextupole':
        elems = tt.rows[tt.element_type == 'Sextupole']
        strengths = pd.Series(elems.k2l, index=elems.name)

        mis = extract_misalignments(line, ['s[fd][12]a.*'], only_nonzero=False)
        mis_df = pd.DataFrame.from_dict(mis, orient="index")[['shift_x', 'shift_y', 'rot_s_rad_no_frame']]
        mis_df = mis_df.reindex(strengths.index).fillna(0)
        ######################################
        tw = line.twiss()

        # get closed orbit at elements
        co_x = pd.Series(tw.x, index=tw.name)
        co_y = pd.Series(tw.y, index=tw.name)

        co_x = co_x.reindex(strengths.index).fillna(0)
        co_y = co_y.reindex(strengths.index).fillna(0)

        # add orbit to misalignment (effective offset seen by magnet)
        mis_df['shift_x'] += co_x
        mis_df['shift_y'] += co_y
        ######################################
        Jw = -strengths * (
            mis_df['shift_x'] * np.sin(2 * mis_df['rot_s_rad_no_frame']) +
            mis_df['shift_y'] * np.cos(2 * mis_df['rot_s_rad_no_frame']))
    return Jw

def compute_f_terms_skew(skew_strength, beta_x, beta_y, dphi_x, dphi_y, Qx, Qy):
    Jw=skew_strength

    h = (Jw * np.sqrt(beta_x * beta_y))/4

    f1001 = np.sum(
        h * np.exp(1j * (dphi_x - dphi_y)),
        axis=0
    ) /(1 - np.exp(2j * np.pi * (Qx - Qy)))

    f1010 = np.sum(
        h * np.exp(1j * (dphi_y + dphi_x)),
        axis=0
    ) / (1 - np.exp(2j * np.pi * (Qy + Qx)))

    return f1001, f1010



def gcv_svd(M, b, lambdas):
    """
    Compute GCV(lambda) for the Tikhonov solution of b = M c,
    using the SVD of M.

    Returns:
        lambda_opt : best lambda (minimizer of GCV)
        Vvals      : GCV values
        lambdas    : lambdas tested
    """
    n = M.shape[0]

    # SVD: M = U S V^T
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Projection of b onto the column space
    z = U.T @ b
    b_perp_norm2 = np.sum(b**2) - np.sum(z**2)

    Vvals = np.empty_like(lambdas, dtype=float)

    for i, lam in enumerate(lambdas):
        nl = n * lam
        denom_terms = s**2 + nl

        factors = nl / denom_terms  # (I - A(λ)) factors along singular vectors

        numerator = (np.sum((factors**2) * (z**2)) + b_perp_norm2) / n
        trace_I_minus_A = n - np.sum(s**2 / denom_terms)
        denom = (trace_I_minus_A / n)**2

        Vvals[i] = numerator / denom if denom > 0 else np.inf

    # Best lambda
    min_idx = np.argmin(Vvals)
    lambda_opt = lambdas[min_idx]

    # Plot
    plt.figure(figsize=(8,5))
    plt.semilogx(lambdas, Vvals, lw=2)
    plt.semilogx(lambda_opt, Vvals[min_idx], 'ro',
                 label=f'best λ = {lambda_opt:.2e}')
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel(r'GCV(\lambda)$', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.title('GCV function vs. λ (SVD)')
    plt.show()

    return lambda_opt, lambdas
# %%



def off_switch_old(element_list, switch_rate):
    ''' 
    Given the list of elements affected it will remove elements randomly. Used for correctors and BPM's to see how stable the solution is.

    element_list: list of elements to be treated. Accepts nested lists eg. [[family10, family11],[family20, family21]]
    switch_rate: the rate at which the element will be removed, maximum value 1, minimum 0.

    returns the lists in the order and format provided with the modifications.
    '''
    new_element_list = []
    removed_element_list = []
    for sublists in element_list:
        cleaned_sublists = []
        trash = []
        for sub in sublists:
            kept = [e for e in sub if random.random() >= switch_rate]
            discarded = [e for e in sub if random.random() < switch_rate]
            cleaned_sublists.append(kept)
            trash.append(discarded)
        new_element_list.append(cleaned_sublists)
        removed_element_list.append(trash)
    return new_element_list, removed_element_list
