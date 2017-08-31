#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:54:52 2017

@author: spolex
"""

def create_reg_flow(name='registration'):
    from nipype.interfaces.ants import Registration
    from nipype.interfaces.ants import ApplyTransforms
    import nipype.pipeline.engine as pe          
    from nipype.interfaces.utility import Merge
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.utility as util
    
    """Register time series to template through subjects' T1 and template 
 
    Example
    -------
 
    >>> wf.inputs.inputspec.func = op.join(resdir,'func.nii.gz')
    >>> wf.inputs.inputspec.struct = op.join(resdir,'t1.nii.gz')
    >>> wf.inputs.inputspec.Template = op.join(resdir,'MNI152_T1_2mm_brain.nii.gz')
    >>> wf.inputs.inputspec.Template_3mm = op.join(resdir,'MNI152_T1_3mm_brain.nii.gz')
    >>> wf.run()
 
    """
    
    def select_volume(filename, which):
        """Return the middle index of a file
        """
        from nibabel import load
        import numpy as np
        if which.lower() == 'first':
            idx = 0
        elif which.lower() == 'middle':
            idx = int(np.ceil(load(filename).get_shape()[3]/2))
        else:
            raise Exception('unknown value for volume selection : %s'%which)
        return idx
 
    extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1),
                      name = 'extractref')
    
    # first 'bet' the anatomy file, including bias reduction step
    bet = pe.Node(interface=fsl.BET(),name='bet')
    bet.inputs.reduce_bias = True
    
    # coregistration step based on affine transformation using ANTs
    coreg = pe.Node(Registration(), name='CoregAnts')
    coreg.inputs.output_transform_prefix = 'func2highres'
    coreg.inputs.output_warped_image = 'func2highres.nii.gz'
    coreg.inputs.output_transform_prefix = "func2highres_"
    coreg.inputs.transforms = ['Rigid', 'Affine']
    coreg.inputs.transform_parameters = [(0.1,), (0.1,)]
    coreg.inputs.number_of_iterations = [[100, 100]]*3 
    coreg.inputs.dimension = 3
    coreg.inputs.write_composite_transform = True
    coreg.inputs.collapse_output_transforms = False
    coreg.inputs.metric = ['Mattes'] * 2 
    coreg.inputs.metric_weight = [1] * 2 
    coreg.inputs.radius_or_number_of_bins = [32] * 2 
    coreg.inputs.sampling_strategy = ['Regular'] * 2 
    coreg.inputs.sampling_percentage = [0.3] * 2 
    coreg.inputs.convergence_threshold = [1.e-8] * 2 
    coreg.inputs.convergence_window_size = [20] * 2
    coreg.inputs.smoothing_sigmas = [[4, 2]] * 2 
    coreg.inputs.sigma_units = ['vox'] * 4
    coreg.inputs.shrink_factors = [[6, 4]] + [[3, 2]]
    coreg.inputs.use_estimate_learning_rate_once = [True] * 2
    coreg.inputs.use_histogram_matching = [False] * 2 
    coreg.inputs.initial_moving_transform_com = True
    
    # registration or normalization step based on symmetric diffeomorphic image registration (SyN) using ANTs 
    reg = pe.Node(Registration(), name='NormalizationAnts')
    reg.inputs.output_transform_prefix = 'highres2template'
    reg.inputs.output_warped_image = 'highres2template.nii.gz'
    reg.inputs.output_transform_prefix = "highres2template_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2 + [[40, 10, 5]])
    #reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2 + [[100, 60, 35]])
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.args = '--float'
    
    # fetch input 
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'struct',
                                                                 'Template',
                                                                 'Template_3mm'
                                                                 ]),
                        name='inputspec')
    
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['registered_func','registered_T1']), 
                        name='outputspec')
    
    # combine transforms
    pickfirst = lambda x: x[0]
    merge = pe.MapNode(Merge(2), iterfield=['in2'], name='mergexfm')
    
    # apply the combined transform 
    applyTransFunc = pe.MapNode(ApplyTransforms(),iterfield=['input_image', 'transforms'],
                         name='applyTransFunc')
    applyTransFunc.inputs.input_image_type = 3
    applyTransFunc.inputs.interpolation = 'BSpline'
    applyTransFunc.inputs.invert_transform_flags = [False, False]
    applyTransFunc.inputs.terminal_output = 'file'
    
    regworkflow = pe.Workflow(name=name)    
    regworkflow.connect(inputnode, 'struct', bet, 'in_file')
    regworkflow.connect(bet,'out_file', coreg, 'fixed_image')
   
    regworkflow.connect(inputnode, 'func', extract_ref, 'in_file')
    regworkflow.connect(inputnode, ('func', select_volume, 'middle'), extract_ref, 't_min')
    regworkflow.connect(extract_ref, 'roi_file', coreg, 'moving_image')
    
    regworkflow.connect(bet, 'out_file', reg, 'moving_image')
    regworkflow.connect(inputnode, 'Template', reg, 'fixed_image')
    
    # get transform of functional image to template and apply it to the functional images to template_3mm (same space as     
    # template)
    regworkflow.connect(inputnode, 'Template_3mm', applyTransFunc, 'reference_image')
    regworkflow.connect(inputnode, 'func', applyTransFunc, 'input_image')
    
    regworkflow.connect(coreg, ('composite_transform', pickfirst), merge, 'in1')
    regworkflow.connect(reg, ('composite_transform', pickfirst), merge, 'in2')  
    regworkflow.connect(merge, 'out', applyTransFunc, 'transforms')
         
    #output
    regworkflow.connect(applyTransFunc, 'output_image', outputnode, 'registered_func')
    regworkflow.connect(reg, 'warped_image',outputnode, 'registered_T1')
 
    regworkflow.write_graph()
    return regworkflow