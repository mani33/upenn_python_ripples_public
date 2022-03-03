# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:34:39 2022
@author: Mani
This module gets trial by trial ripple time points for the light stimulation-mediated
ripple suppression.
"""

# Get database tables
from itertools import compress
import re
import numpy as np
import pandas as pd
import acq as acq
import cont as cont
import ripples as ripples
import general_functions as gfun
import djutils as dju


# Class for default values of parameters
class Args():
    def __init__(self):
        # Data length pre and post light pulse
        self.xmin = -5.0 # Add sign appropriately. If you want 1sec before light pulse, use -1
        self.xmax = 10.0
        
        self.beh_state = 'all' #  beh state to collect ripples from: 'all','rem','nrem','awake'
        self.bin_width = 100.0 # binning width in msec
        self.motion_quantile_to_keep = [1.0]; # below this quantile value, we will retain stimulation trials, list if >1 session
        
        # in sec: time window to detect motion. Can be 4 element [-5 0 2 3] where [-5 0] and [2 3] are two separate windows.
        self.motion_det_win = [self.xmin, self.xmax]
        
        # Remove trials without ripples?
        self.remove_norip_trials = True
        self.remove_noise_spikes = True
        self.pre_and_stim_only = True
        

# main function that calls other functions
def get_processed_rip_data(keys, pulse_per_train, std, minwidth, **kwargs):
    """ This function takes one or more channel keys as input. Possible input key-value pairs
        are indicated in the Args class.
        
        Inputs:
            keys - list of dict. Must be a list even if just one dict. When giving multiple keys
                    user must ensure that all keys had the same light pulse stimulation protocol
            pulse_per_train - how many pulses were given in each train?
            std - list of std values for restricting ripples table
            minwidth - list of minwidth params for restricting ripples table
            kwargs - user's updated params-values listed in the Args class
        Outputs:
            rdata - list of dict. len(rdata) = num of trials. Each dict contains the following keys:
                    'rel_mt' - numpy array of time (sec) relative to stim train onset
                    'mx','my' - numpy array of tracker LED x and y coodinate respectively
                    'head_disp' - numpy array of head displacement per video frame (pix/frame)
                    'rip_evt'- numpy array of ripple event times (sec) relative to stim train onset
                    
            args - Args class object contaning a bunch of experiment related info.
        """
        

    # Ensure list type for keys, std and minwidth
    std = std if type(std) is list else [std]
    minwidth = minwidth if type(minwidth) is list else [minwidth]
    keys = keys if type(keys) is list else [keys]
    # Make sure chan_num exists in keys    
    assert all(['chan_num' in kk for kk in keys]), 'chan_num does not exist in given keys'
    
    # Update optional params
    args = Args()
    for k, v in kwargs.items():
        setattr(args,k,v)

    # Create binning params
    pre = args.xmin * 1e6 # to microsec
    post = args.xmax * 1e6 # to microsec
    bw = args.bin_width * 1e3 # Given originally in msec. Change to microsec
    rbin_edges = np.arange(pre, post, bw)
    assert any(rbin_edges==0), 'Zero is missing in bin edges'
    
    # Go through each channel key and collect ripple and motion data
    cc = 0
    rdata = {}
    for ikey, key in enumerate(keys):
        # Get light pulse train info
        pon_times, poff_times, pulse_width, pulse_freq = get_light_pulse_train_info(key, pulse_per_train)
        # For plotting purpose, extract one pulse train on and off times in sec
        # We assume that all keys had the same stimulus train parameters, so we pick one key
        if ikey==0:
            one_train_on = (pon_times[0:pulse_per_train] - pon_times[0]) * 1e-6
            one_train_off = (poff_times[0:pulse_per_train] - poff_times[0]) * 1e-6
            # Extract pulse width,  in ms
            pulse_width = np.median(poff_times - pon_times)*1e-3
        
        # Pick the first pulse times in the pulse train for time referencing
        tr_on = pon_times[::pulse_per_train] # shape: (rows,), int64
        tr_off = poff_times[::pulse_per_train] # shape: (rows,), int64
        
        # Get motion info from tracker data. tr_on is returned because it may be trimmed
        # inside the get_perievent_motion function for shorter trial length
        # Note - output of get_perievent_motion, mt, will be in seconds.
        same_len = True
        mt, mx, my, good_trials_len = gfun.get_perievent_motion(key, tr_on, args.xmin, 
                                                                args.xmax, same_len)
        
        rel_mt = center_and_scale_time(mt, tr_on)
     
        # Mark (True/False) good trials by amount of head movement 
        good_trials_mov = filter_trials_by_head_disp(rel_mt, mx, my, 
                                                     args.motion_det_win, 
                                                     args.motion_quantile_to_keep[ikey])
        
        # Mark (True/False) trials by their presence inside given behavioral state (e.g NREM)
        good_trials_beh = filter_trials_by_beh_state(tr_on, tr_off, args.beh_state, key)
        
        # Combine all selections and re-filter event times and trial data
        good_trials = good_trials_len & good_trials_mov & good_trials_beh
        tr_on = tr_on[good_trials]
        rel_mt = list(compress(rel_mt, good_trials))
        mx = list(compress(mx, good_trials))
        my = list(compress(my, good_trials))
        
        # Get ripple events
        fstr = f'std = {std[ikey]} and minwidth = {minwidth[ikey]}'
        rpt = np.array((ripples.RipEvents & key & fstr).fetch('peak_t'))
        
        # Check if the RipEvents table was populated
        assert rpt.size > 0, 'No ripples! Check if the RipEvents table is populated'
        
        # Go through each pulse train and collect ripples near by
        
        for t_idx, ct_on in enumerate(tr_on):
            twin = np.array([pre, post]) + ct_on
            # Pick ripples in window
            sel_rt = rpt[(rpt >= twin[0]) & (rpt < twin[1])]
            cmx = mx[t_idx]
            cmy = my[t_idx]
            head_disp = np.sqrt(np.diff(cmx)**2 + np.diff(cmy)**2)
            # Add a filler value to compensate for one sample loss due to diff
            head_disp = np.append(head_disp, np.nan)
          
            # Center ripple event times also as above
            re_rel_t = (sel_rt - ct_on)*1e-6
            rdata[cc] = {'rel_mt': rel_mt[t_idx], 'mx': mx[t_idx], 'my': my[t_idx],\
                        'head_disp': head_disp, 'rip_evt': re_rel_t}
            cc += 1           
    
    # Watch out: here we pick these params from the last key of the key list with
    # the assumption that all keys had same parameters.
    args.one_train_on = one_train_on
    args.one_train_off = one_train_off
    args.pulse_width = pulse_width
    args.pulse_freq = pulse_freq
    args.pulse_per_train = pulse_per_train
    args.mouse_id = key['animal_id']
    args.std = std
    args.minwidth = minwidth
    args.chan_name = (cont.Chan & key & f'chan_num = {key["chan_num"]}').fetch('chan_name')[0]
    
    return rdata, args
        
def center_and_scale_time(mt, evt):
    """
    Center the time vector to event time and convert units from microsec to sec
    Inputs: 
        mt - list of numpy arrays of time in microsec
        evt - numpy array of event times in microsec corresponding to the list
                    of times in mt
    Outputs: 
        rel_mt - list of numpy arrays of time in sec relative to event times
    """
    rel_mt = [(t-et)*1e-6 for t,et in zip(mt, evt)]   
    return rel_mt

def filter_trials_by_beh_state(tr_on, tr_off, beh_state, key):
    """
    Filter photostim trials based on their presence within a given behavioral state
    Inputs:
        tr_on and tr_off  - numpy arrays of photostim train on and off times
        beh_state - string, should be 'awake','nrem','rem' or 'all'
        key - database key, a dict
    Outputs:
        good_trials_beh - numpy array of boolean, with True indicating if a given 
                          photostim trial was within the given behavioral state.
    """
    good_trials_beh = np.full((tr_on.size,),False)
    
    #First get the behavior state segment on and off times
    if beh_state == 'awake':        
        st1, st2 = (cont.AwakeSegManual & key).fetch('seg_begin','seg_end')
        st1 = st1[0][0]
        st2 = st2[0][0]
        ephys_start, ephys_end = (acq.Ephys & key).fetch('ephys_start_time','ephys_stop_time')
       
    elif beh_state == 'nrem':
        t1, t2 = (cont.NremSegManual & key).fetch('seg_begin','seg_end')
        t1 = t1[0][0]
        t2 = t2[0][0]
        
    elif beh_state == 'rem':
        t1, t2 = (cont.RemSegManual & key).fetch('seg_begin','seg_end')
        t1 = t1[0][0]
        t2 = t2[0][0]
        
    elif beh_state == 'all':
        good_trials_beh = np.full((tr_on.size,), True)
        return good_trials_beh
    else:
        raise ValueError(f'the behavior state {beh_state} is unaccepted:' 
                         f'should be one of "awake","rem","nrem" or "all"')
    
    # Mark as True if a given photostim train is within any of the behavioral state segments
    for i_train, cton in enumerate(tr_on):
        ctoff = tr_off[i_train]
        # Take on and off times of each train. If train is inside any one of the NREM
        # segment, it's a good trial.
        inseg = np.full((t1.size,), False)
        for idx, st1 in enumerate(t1):
            st2 = t2[idx]
            if (cton >= st1) & (ctoff < st2):
                inseg[idx] = True
        if any(inseg):
            good_trials_beh[i_train] = True
    
    return good_trials_beh
        
def filter_trials_by_head_disp(mt, mx, my, motion_det_win, motion_quantile_to_keep): 
    """       
    Determine head displacement level in the given window and remove trials above
    threshold obtained based user specified quantile value of head dispalcement
    Inputs:
        mt, mx, and my are outputs of get_perievent_motion(...)
        motion_det_win - list of 2 or 4 relative times (sec) marking boundaries of
                        one or two time windows respecively. e.g: [-5, 10] or [-5,0,2,10] 
        motion_quantile_to_keep - quantile value below which trials will be kept. e.g: 0.9
    Outputs:
        good_trials -  numpy array of boolean, with True indicating trials satisfying 
                       head displacement limit condition.
    """      
    # For the selected time windows, compute averaged head displacement.
    r = np.zeros((len(mx),))
    print('Filtering trials by head displacement ...')
    for idx, x in enumerate(mx):
        tt = mt[idx]
        # Selected time period for computing motion
        sel_t = (tt >= motion_det_win[0]) & (tt < motion_det_win[1])
        if len(motion_det_win)==4:
            sel_t = sel_t | (tt >= motion_det_win[2]) & (tt < motion_det_win[3])
            
        y = my[idx]
        # Get data for time windows
        dx = np.diff(x[sel_t])
        dy = np.diff(y[sel_t])
        r[idx] = np.mean(np.sqrt(dx**2 + dy**2))
        
    q = np.quantile(r, motion_quantile_to_keep)
    good_trials = r <= q
    
    return good_trials

def get_ripple_rate(rdata, bin_width, xmin, xmax):
    """
    Computes ripple rates in rip/s at bins of width bin_width between xmin and xmax
    times points relative to event onset.
    
    Inputs: -----------------------------------
        rdata: list of dict, output of get_processed_rip_data(...) function call
        bin_width: in millisec
        xmin: a negative number in sec
        xmax: a positive number in sec
    
    Outputs:-----------------------------------
        rip_rate: 1D np array of ripple rate (rip/s)
        bin_edges: 1D np array of bin edges
        bin_cen: 1D np array of bin centers in sec, same length as rip_rate.
    
    """

    evt = np.concatenate([v['rip_evt'] for _,v in rdata.items()])
    # Create binedges
    bw =  bin_width/1000
    be = np.arange(xmin, xmax+bw, bw)
    # Plot hist
    counts, bin_edges = np.histogram(evt, be)
    rip_rate = (counts/len(rdata))/bw
    bin_cen = bin_edges[:-1]+bw
    
    return rip_rate, bin_edges, bin_cen

def collect_mouse_group_rip_data(data_sessions, beh_state, xmin, xmax, 
                        bin_width, **kwargs):
    """
    For a given list of sessions, collect ripple data.
    Inputs:
        data_sessions : Pandas data frame containing the following columns:
                        animal_id, session_ts, pulse_per_train, pulse_width, chan, std, minwidth, 
                        laser_color, laser_knob, motion_quantile       
        beh_state: behavior state to collect ripples from: 'all','rem','nrem','awake'       
        xmin: a negative number in sec
        xmax: a positive number in sec
        bin_width: in millisec   
    Outputs:
        group_data - list of dict, containing info of ripples for each chan of each mouse     
    """
    
    group_data = [[]]*data_sessions.shape[0]
    idx = 0
    
    for _, dd in data_sessions.iterrows():
        key = dju.get_key_from_session_ts(dd['session_ts'])[0]
       
        # Go through each channel of the given session
        chans = [int(cc) for cc in dd['chan'].split(',')]
        for ch in chans:
            key.update({'chan_num': ch})
            print(key)
            rdata, args = get_processed_rip_data (key, dd['pulse_per_train'], 
                                                  dd['std'], dd['minwidth'], 
                                                  beh_state = beh_state, 
                                                  motion_quantile_tokeep = dd['motion_quantile'], 
                                                  xmin = xmin, xmax = xmax, 
                                                  bin_width = bin_width)             
            tdic = {'animal_id': key['animal_id'], 'chan_num': ch,'rdata': rdata, 'args': args}
            group_data[idx].append(tdic)
        print(f'Done with mouse {idx}')
        idx += 1
    return group_data
        
def select_one_elec_per_mouse(group_data, elec_sel_meth):
    """
    When multiple electrodes had ripples, pick one based on given selection method.
    Inputs:
        group_data : list (mice) of list(channels) of dict (ripple data), this is an output from
                     collect_mouse_group_rip_data(...) function call.       
        elec_sel_meth: str, should be one of 'random','max_effect', 'highest_baseline_rip_rate'
                      When more than one electrode had ripples, tells you which electrode to pick.
    Outputs: 
        t - 1D numpy array of time in sec.
        mean_rr - 1D numpy array, mean ripple rate(Hz)
        std_rr - 1D numpy array, standard deviation of each time bin
    MS 2022-03-02
        
    """
    
    
def average_rip_rate_across_mice(group_data): 

      
def get_light_pulse_train_info(key, pulse_per_train):
    
    """
    Inputs:
        key : dict, key of one session or one recording channel
        pulse_per_train : int, number of light pulses per stimulus train
    
    Outputs:
        pon: light pulse on times (us) - 1D numpy array of int64 time stamps
        poff: light pulse off times (us) - 1D numpy array of int64 time stamps
        pulse_width: float, pulse width in ms
        pulse_freq: Hz, float (if pulse train) or NaN (if single pulse)
        
    """
    # Get pulse onset and offset times (microsec)
    pon, poff = dju.get_light_pulse_times(key)
    
    
    # Get pulse width in ms
    pulse_width = np.median((poff-pon)*1e-3)
    
    # Compute pulse freq
    if pulse_per_train == 1:
        pulse_freq = np.NaN
    else:
        # Get the first train as an example & get pulse freq in Hz
        train_on = pon[0:pulse_per_train]        
        pulse_freq = 1/(np.median(np.diff(train_on))*1e-6)
            
       
    return pon, poff, pulse_width, pulse_freq    

# def pool_ripple_data_by_session():
    
            

        
        
        
        
        
        
        
        
        
        
        
        
        
            
            