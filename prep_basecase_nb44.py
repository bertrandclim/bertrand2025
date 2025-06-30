import numpy as np
import xarray as xr
from metpy import calc
from metpy.units import units

####################################################
### NB42: 3. Classify sondes by cloud phase (v1) ###
####################################################

#convert from bit-packed to broad categories (mixed, liquid, ice)
def bitlist_to_int(bitlist):
    '''take a list of ints indicating bit numbers to set to true'''
    return sum([2**x for x in bitlist])

def bit_packed_to_broad_categories(phases):
    '''convert bit-packed phase mask to bool for a group of bits'''
    
    #define phase categories via the flag values satisifying each category
    cloud_liquid = [3,5]              #liquid, liquid+drizzle
    cloud_mixed  = [7]                #mixed-phase
    cloud_ice    = [1,2]              #ice, snow
    phase_vars   = {'liquid': cloud_liquid,
                    'mixed': cloud_mixed,
                    'ice': cloud_ice}
    
    #get profiles satisfying broad categories specified in phase_vars
    phase_res = { }
    for name,flags in phase_vars.items():
        #convert list of flag values to int with bits on for flag values
        flag_map = bitlist_to_int(flags)
        #check if at least one of the flag values is present in profile
        flag_res = (phases&flag_map) > 0
        phase_res[name] = flag_res
    
    #include ice and liquid at different levels as mixed-phase
    phase_res['mixed'] = phase_res['mixed'] | (phase_res['liquid'] & phase_res['ice'])

    #exclude liquid-containing profiles as ice
    phase_res['ice'] = phase_res['ice'] & ~(phase_res['mixed'] | phase_res['liquid'])

    #return as dataset
    return xr.Dataset(phase_res)

###########################################
### NB43: 6. Quicklook new (v2) methods ###
###########################################

def exclusion_criteria(top,btm,zrange_max=700,top_height_max=3000):
    #maximum allowable top and base height variation
    zlt_range       = top.max('dt') - top.min('dt')
    zlb_range       = btm.max('dt') - btm.min('dt')
    top_range_test  = zlt_range<=zrange_max
    base_range_test = zlb_range<=zrange_max

    #maximum allowable average cloud top height (min pressure)
    top_thresh_test = top.mean('dt')<=top_height_max

    #minimum number of liquid-containing observations in 31-minute window

    return top_range_test & base_range_test & top_thresh_test

##################################
### CLOUD BOUND HEIGHT METHODS ###
##################################
def get_layer_bounds(iscld):
    '''find layer boundaries in binary cloud mask
    
    given an array of cloud fraction with dimension 'height', return an
    array filled with layer top and bottom heights
    '''
    ht_orig    = iscld['height'].data
    iscld      = iscld.pad({'height':(1,0)},constant_values=0.)     # pad with a blank 0-100 m level
    ht_pad     = np.pad(ht_orig,(1,0),constant_values=50.)          # add 50 m height level
    iscld      = iscld.assign_coords(height=ht_pad)                 # update DataArray coordinate
    clddiff    = iscld.astype(int).diff(dim='height',label='upper') #1 if layer top, 0 if layer base
    tops       = clddiff.height.where(clddiff==-1)                  #height of layer tops
    bottoms    = clddiff.height.where(clddiff==1)                   #height of layer bottoms
    return tops, bottoms

def get_lowest_phase_layer_bounds(micro,flags=[3,5,7]):
    '''get top and bottom heights of lowest layer containing flags'''

    #3,5,7: liquid, liquid+drizzle, mixed-phase
    #1,2,7: snow, ice, mixed-phase
    
    #get liquid layer top and bottom heights
    iscld         = micro.CloudPhaseMask.isin(flags)    #True if liquid droplets in volume
    tops, bottoms = get_layer_bounds(iscld)             #get layer top and bottom heights
    top_lo        = tops.min(dim='height')              #minimum layer top height in profile
    btm_lo        = bottoms.min(dim='height')           #minimum layer bottom height in profile
    ntops         = tops.notnull().sum(dim='height')    #number of layer tops
    nbtms         = bottoms.notnull().sum(dim='height') #number of layer bottoms
    return top_lo, btm_lo, ntops, nbtms

def get_layer_containing_bounds(micro, top_lo, btm_lo):
    '''get top and bottom heights of cloud layer containing top_lo and btm_lo'''
    #get full hydrometeor extent of lowest liquid-containing layer
    iscld         = micro.Avg_CloudFraction>0      #True if any hydrometeor
    tops, bottoms = get_layer_bounds(iscld)        #get layer top and bottom heights
    tops          = tops.where(tops>=top_lo)       #restrict to tops >= liquid top
    bottoms       = bottoms.where(bottoms<=btm_lo) #restrict to bottoms <= liquid bottom
    anytop_lo     = tops.min(dim='height')         #lowest top >= liquid top
    anybtm_hi     = bottoms.max(dim='height')      #highest base <= liquid bottom
    return anytop_lo, anybtm_hi

def get_ice_bounds_in_cloud_layer(micro, anytop_lo, anybtm_hi):
    '''get ice boundaries between anybtm_hi and anytop_lo'''
    #get ice top and bottom heights within liquid-containing layer extent
    iscld         = micro.CloudPhaseMask.isin([1,2,7])      #True if ice crystals in volume
    tops, bottoms = get_layer_bounds(iscld)                 #get layer top and bottom heights
    height        = iscld.height.broadcast_like(iscld)      #height of each measurement
    liqlay_msk    = (anybtm_hi<=height)&(height<=anytop_lo) #mask of liq-containing layer
    tops          = tops.where(liqlay_msk)                  #restrict to liq-containing layer
    bottoms       = bottoms.where(liqlay_msk)               #restrict to liq-containing layer
    icetop_hi     = tops.max(dim='height')                  #highest ice top in liq-containing layer
    icebtm_lo     = bottoms.min(dim='height')               #lowest ice bottom in liq-containing layer
    return icetop_hi, icebtm_lo

##################################
### THERMODYNAMIC CALCULATIONS ###
##################################

def calc_theta_e(sonde):
    '''calculate equivalent potential temperature from P, T, RH using metpy'''
    tunits = sonde['tdry'].attrs['units']
    if tunits=='C':
        sonde['tdry'].attrs['units'] = 'degC'
    p  = sonde['pres']*units(sonde['pres'].attrs['units'])
    T  = sonde['tdry']*units(sonde['tdry'].attrs['units'])
    rh = sonde['rh']*units(sonde['rh'].attrs['units'])
    Td = calc.dewpoint_from_relative_humidity(T, rh)
    return calc.equivalent_potential_temperature(p,T,Td)

def calc_theta(sonde):
    '''calculate potential temperature from P, T using metpy'''
    tunits = sonde['tdry'].attrs['units']
    if tunits=='C':
        sonde['tdry'].attrs['units'] = 'degC'
    p  = sonde['pres']*units(sonde['pres'].attrs['units'])
    T  = sonde['tdry']*units(sonde['tdry'].attrs['units'])
    return calc.potential_temperature(p,T)

def get_tropopause_height_vector(prof,htdim='height'):
    '''get tropopause height in radiosonde from WMO standard thermal defintion.'''
    #"The tropopause is defined as the lowest level at which the lapse rate decreases to 2°C/km or less, 
    # provided that the average lapse-rate, between that level and all other higher levels within 2.0 km 
    # does not exceed 2°C/km." WMO, https://en.wikipedia.org/wiki/Tropopause
    # "htdim" gives the name of the height dimension ('height' or 'alt')
    
    #coarsen from 50 m to 500 m resolution
    tdry = prof.tdry.coarsen(dim={htdim:10},boundary='pad').mean()
    gamma = -1e3*tdry.differentiate(htdim) #K/km
    #satisfactory heights
    heights = gamma[htdim].broadcast_like(gamma).where(gamma<=2)
    #lapse rate from candidate heights to higher levels
    t_at_lvl = tdry.where(heights.notnull())
    gamma_d1 = (t_at_lvl - tdry.shift({htdim:-1}).where(heights.notnull()))/(0.5*1)<=2 #K/km, T(z_i)-T(z_i+500 m)/500 m
    gamma_d2 = (t_at_lvl - tdry.shift({htdim:-2}).where(heights.notnull()))/(0.5*2)<=2
    gamma_d3 = (t_at_lvl - tdry.shift({htdim:-3}).where(heights.notnull()))/(0.5*3)<=2
    gamma_d4 = (t_at_lvl - tdry.shift({htdim:-4}).where(heights.notnull()))/(0.5*4)<=2
    #check where lapse rate to higher levels stays below 2 K/km
    heights = heights.where(gamma_d1 & gamma_d2 & gamma_d3 & gamma_d4)
    #restrict physically plausible height range
    heights = heights.where(lambda da: (5e3<=da)&(da<=18e3))
    #get lowest height
    return heights.min(htdim)

###################################
### NORMALIZED SEGMENT ROUTINES ###
###################################

def make_height_norm_da(iscld,num_levels=50):
    '''form an array like input but with num_levels of height
    
    Parameters:
    -----------
    iscld         xarray object with desired shape and dimensions. must have a dimension named 'height'
    num_levels    size of 'height_norm' dimension which will replace the 'height' dimension

    Returns:
    ---------
    0-1 linspaced xr.DataArray with a height_norm dimension in place of height
    '''
    shape        = list(iscld.shape)
    iaxis        = iscld.get_axis_num('height')         #check dimension order is correct
    shape[iaxis] = num_levels
    znorm_range  = [0,1] #0: layer base, 1: layer top
    zstar_flat   = np.linspace(*znorm_range,num_levels)
    zstar        = zstar_flat.copy()
    while len(zstar.shape)<iaxis+1:    #prepend axes until height is the correct axis
        zstar    = np.expand_dims(zstar,0)
    while len(zstar.shape)<len(shape): #append axes until the total number of dimensions is reached
        zstar    = np.expand_dims(zstar,-1)
    zstar        = np.broadcast_to(zstar,shape)
    dims         = list(iscld.dims)
    dims[iaxis]  = 'height_norm'
    zstar        = xr.DataArray(data=zstar, dims=dims, coords={'height_norm':zstar_flat})
    #put back coordinates
    zstar        = zstar.assign_coords({name: iscld[name] for name in set(iscld.coords).difference(['height'])}) 
    return zstar

def get_profile_normalized_segment(da,top,btm,ref=None,num_levels=50,htdim='height'):
    '''get a profile of da normalized to top, btm
    '''
    if ref == None: 
        ref = da
    zstar    = make_height_norm_da(ref,num_levels=num_levels)
    da_zstar = da.interp({htdim:(zstar*(top-btm)+btm)})
    return da_zstar

def dict_to_da(dd):
    '''concat dict treating keys as layer coordinate'''
    items = [d.assign_coords(layer=k).expand_dims('layer') for k,d in dd.items()]
    return xr.concat(items,dim='layer')

def calculate_normalized_segments(var,bounds):
    '''process var into a dataset summarizing normalized segments as defined by bounds dict
    '''
    #form into dictionaries
    layer_profs  = { }
    base_temps   = { }
    base_heights = { }
    base_depths  = { }
    for k, (btm, top) in bounds.items():
        #compute
        do_avg_dt  = lambda da: da.mean('dt') if 'dt' in da.dims else da
        top, btm   = do_avg_dt(top), do_avg_dt(btm)
        prof_zstar = get_profile_normalized_segment(var,top,btm)#.mean('dt')
        #save
        base_temps[k]   = prof_zstar.sel(height_norm=0,method='nearest')
        base_heights[k] = btm
        base_depths[k]  = top-btm
        layer_profs[k]  = prof_zstar-base_temps[k]
    
    #convert dictionaries to xarray.Dataset
    data_vars = {
        'base_values': base_temps,
        'base_heights': base_heights,
        'base_depths': base_depths,
        'profile': layer_profs
    }
    return xr.Dataset({k:dict_to_da(d) for k,d in data_vars.items()})

def stack_normalized_segments(tesegs):
    '''stack 'layer' segments end-to-end (e.g. layer=1, height_norm=0.5 -> zk=1.5)'''
    shifts  = tesegs.sel(height_norm=1).cumsum('layer').shift(layer=1,fill_value=0)
    tesegs  = tesegs+shifts #ONLY FOR PROFILE.profile.isel(time=250).plot()
    dastack = tesegs.profile.stack(zk=('layer','height_norm'))
    zk      = (tesegs.layer+tesegs.height_norm*(1-1e-6)).data.flatten()
    dastack = dastack.drop_vars(['zk', 'layer', 'height_norm']).assign_coords(zk=zk)
    return dastack

def apply(ds,func,arg,ignore=['base_heights','base_depths']):
    '''apply a function to each variable of a dataset 
    and join the results along new dimension 'var' '''
    res = [ ]
    for name, da in ds.items():
        result = func(da,arg)
        passed = result[ignore]
        result = result.assign_coords(var=name).expand_dims('var')
        result = result.drop_vars(ignore)
        res.append(result)
    joined = xr.concat(res,dim='var')
    joined = joined.assign(passed)
    return joined

#################################################
### NB45: encapsulate full base case pipeline ###
#################################################
    
##########################################
### LAUNCH WINDOW AGGREGATING ROUTINES ###
##########################################

def get_bounds(micro,tps,which='liquid'):
    '''return dictionaries defining atmospheric layer bounds for cloud phase

    Parameters:
    --------------
    micro         dataset containing shupe-turner microphysics
    tps           data array containing tropopause heights at each timestep
    which         which phase to return bounds for. Options are 'liquid', 'mixed', or 'ice'.

    Returns:
    -------------
    bounds_dict            if which == 'ice' or 'liquid'

    liq_dict, ice_dict     if which == 'mixed'
    '''
    type_flags = {
        'mixed':  [3,5,7], #liq, liq+drizzle, mixed-phase
        'liquid': [3,5],   #liq, liq+drizzle
        'ice':    [1,2]    #snow, ice
    }
    ### GET LIQUID AND ICE LAYER BOUNDARIES ###
    #lowest layer of phase top and bottom heights
    top_lo, btm_lo, ntops, nbtms = get_lowest_phase_layer_bounds(micro,flags=type_flags[which])

    if which == 'mixed':
        #hydrometeor extent of lowest liquid-containing layer
        anytop_lo, anybtm_hi = get_layer_containing_bounds(micro, top_lo, btm_lo)
        #ice extent within lowest liquid-containing cloud layer
        icetop_hi, icebtm_lo = get_ice_bounds_in_cloud_layer(micro, anytop_lo, anybtm_hi)
        
        ### MAKE BOUNDS DICT ###
        #define bounds
        liq_bounds = {
            0: [xr.zeros_like(btm_lo), btm_lo], #surface to liquid base
            1: [btm_lo, top_lo],                #liquid base to top
            2: [top_lo, tps]                    #rest of troposphere
        }
        
        ice_bounds = {
            0: [xr.zeros_like(icebtm_lo), icebtm_lo], #surface to ice base
            1: [icebtm_lo, icetop_hi],                #ice base to top
            2: [icetop_hi, tps]                       #rest of troposphere
        }
        return liq_bounds, ice_bounds
    
    else:
        this_bounds = {
            0: [xr.zeros_like(btm_lo), btm_lo], #surface to single-phase base
            1: [btm_lo, top_lo],                #single-phase base to top
            2: [top_lo, tps]                    #rest of troposphere
        }
        return this_bounds  

def get_window_average_micro(micro,bounds,which='mixed',dz=150):
    '''return window-averaged relative profiles and column-integrated microphysics

    Parameters:
    --------------
    micro         dataset containing shupe-turner microphysics
    
    bounds        dictionary of layer bounds if which == 'liquid' or 'ice'.
                  Tuple of (liq_bounds, ice_bounds) if which == 'mixed'. Format of bounds dict is:
                        0: [surface, cloud base],
                        1: [cloud base, cloud top],
                        2: [cloud top, tropopause]
                    
    which         which phase to return bounds for. Options are 'liquid', 'mixed', or 'ice'.

    dz            height (in meters) to consider microphysical properties beyond average cloud edge.

    Returns:
    -------------
    profile_vars    dictionary containing relative profiles of microphysics

    rel_norms       dictionary of layer-integrated or layer-averaged bulk microphysics
    '''

    #unpack cloud bounds
    if which == 'mixed':
        liq_bounds, ice_bounds = bounds        #unpack argument
        btm_lo, top_lo         = liq_bounds[1] #cloudy layer
        icebtm_lo, icetop_hi   = ice_bounds[1] #cloudy layer
    else:
        btm_lo, top_lo         = bounds[1]     #cloudy layer

    #clip microphysics to a single cloudy layer
    keep_vars = ['Avg_Retrieved_LWC','Avg_Retrieved_IWC','Avg_LiqEffectiveRadius','Avg_IceEffectiveRadius']
    micro_avg = micro[keep_vars].where(lambda da: da>=0).mean('dt')
    height    = micro.height
    if which == 'mixed':
        micro_liq    = micro_avg.where((btm_lo.mean('dt')-dz<=height)&(height<=top_lo.mean('dt')+dz))
        micro_ice    = micro_avg.where((icebtm_lo.mean('dt')-dz<=height)&(height<=icetop_hi.mean('dt')+dz))
    else:
        micro_this = micro_avg.where((btm_lo.mean('dt')-dz<=height)&(height<=top_lo.mean('dt')+dz))
        micro_liq, micro_ice = micro_this, micro_this

    #output dicts
    profile_vars = { } #scaled profiles
    rel_norms    = { } #normalization

    #calculate microphysics
    if which != 'ice':
        #relative LWP
        LWC     = micro_liq.Avg_Retrieved_LWC #g/m3
        LWP     = LWC.fillna(0).integrate('height') 
        rel_LWC = LWC/LWP

        #relative liquid effective radius
        reliq       = micro_liq.Avg_LiqEffectiveRadius #um
        reliq_avg   = reliq.where(lambda da: da>0).mean('height') #um
        rel_reliq   = reliq/reliq_avg

        #save
        profile_vars['rel_lwc'] = rel_LWC
        profile_vars['rel_reliq'] = rel_reliq
        rel_norms['rel_lwc'] = LWP
        rel_norms['rel_reliq'] = reliq_avg

    if which != 'liquid':
        #relative IWP
        IWC     = micro_ice.Avg_Retrieved_IWC/1e3 #mg to g/m3
        IWP     = IWC.fillna(0).integrate('height') 
        rel_IWC = IWC/IWP
    
        #relative ice effective radius
        reice       = micro_ice.Avg_IceEffectiveRadius #um
        reice_avg   = reice.where(lambda da: da>0).mean('height') #um
        rel_reice   = reice/reice_avg

        #save
        profile_vars['rel_iwc'] = rel_IWC
        profile_vars['rel_reice'] = rel_reice
        rel_norms['rel_iwc'] = IWP
        rel_norms['rel_reice'] = reice_avg

    #average over launch window
    #rel_norms = {key:da.mean('time').data for key, da in rel_norms.items()}

    return profile_vars, rel_norms

def check_for_liquid_only_layer(micro,this_bounds):
    '''ensure no ice is present in lowest liquid-containing layer'''
    #get lowest liquid-containing layer bounds
    base, top = this_bounds[1]
    #get hydro extent containing lowest liquid layer
    top_outer, base_outer = get_layer_containing_bounds(micro, top, base)
    
    #get frequency of snow within the outer bounds of lowest liquid-containing layer
    phase  = micro.CloudPhaseMask
    phase  = phase.where((base_outer<=phase.height)&(phase.height<=top_outer))
    hasice = phase.isin([1,2,7]).any('height') #snow, ice, mixed-phase
    noice  = hasice.sum('dt')<15 #<15 minutes of snow in 31-minute window
    return noice

def compute_base_case(which, sondes, micro_days_win_by_type, is_phase):
    #subset micro to category
    micro       = micro_days_win_by_type[which]
    #subset sondes to season and category
    keep_months = [12,1,2,3]
    sub_mos     = lambda ds: ds.sel(time=ds['time.month'].isin(keep_months))
    sonde_i     = sub_mos(sondes.sel(time=is_phase[which]))
    
    #get lowest layer bounds satisfying category
    this_bounds = get_bounds(micro,sonde_i['tropopause_height'],which)
    
    #unpack when two sets of bounds are present
    if which == 'mixed':
        liq_bounds, ice_bounds = this_bounds
    
    #aggregate microphysics within layer bounds
    profile_vars, rel_norms = get_window_average_micro(micro,this_bounds,which)
    
    #add radiosonde variables to variables to profile
    sonde_vars   = ['theta_e','theta','tdry','rh','pres']
    sonde_vars   = {name:sonde_i[name] for name in sonde_vars}
    profile_vars = profile_vars | sonde_vars
    #specify no scaling for radiosonde variables... pad the dict
    ref = list(rel_norms.values())[0] #get a value with 'time' dimension
    sonde_norms  = {name: xr.ones_like(ref) for name,da in sonde_vars.items()}
    rel_norms    = rel_norms | sonde_norms
    
    #calculate normalized segments
    if which == 'mixed':
        liq_segs  = apply(profile_vars,calculate_normalized_segments,liq_bounds)
        ice_segs  = apply(profile_vars,calculate_normalized_segments,ice_bounds)
    else:
        this_segs = apply(profile_vars,calculate_normalized_segments,this_bounds)
    
    #apply additional filtering to cases
    if which == 'mixed':
        #exclude "non-stationary" radiosonde launch cases
        base, top = liq_bounds[1]
        mask = exclusion_criteria(top,base,top_height_max=19000,zrange_max=1000)
        #mask = xr.ones_like(liq_segs,dtype=bool)
        liq_segs_masked = liq_segs.where(mask)
        ice_segs_masked = ice_segs.where(mask)
        return profile_vars, rel_norms, liq_segs_masked, ice_segs_masked
    elif which == 'liquid':
        #exclude "non-stationary" radiosonde launch cases
        base, top = this_bounds[1]
        mask  = exclusion_criteria(top,base,top_height_max=19000,zrange_max=1000)
        #mask = xr.ones_like(this_segs,dtype=bool)
        #exclude cases snowing more than half the time
        noice = check_for_liquid_only_layer(micro,this_bounds)
        mask  = mask & noice
        this_segs_masked = this_segs.where(mask)
        return profile_vars, rel_norms, this_segs_masked
    else:
        #don't exclude any ice
        mask = xr.ones_like(this_segs,dtype=bool)
        this_segs_masked = this_segs.where(mask)
        return profile_vars, rel_norms, this_segs_masked

#########################################
### HOMOGENEOUS BASE CASE CALCULATION ###
#########################################

def get_actual_heights(tesegs,htdim='height',reduce_proc=lambda da: da.mean('time')):
    '''find average literal heights corresponding to normalized segments
    '''
    #squish height_norm so it's slightly less than 0 to 1
    lit_ht = reduce_proc(0.999*tesegs.height_norm*tesegs.base_depths+tesegs.base_heights)
    lit_ht = lit_ht.stack({htdim:('layer','height_norm')})
    alt    = lit_ht.data.flatten()
    lit_ht = lit_ht.drop_vars([htdim, 'layer', 'height_norm']).assign_coords({htdim:alt})
    return lit_ht

def make_var_mask(false_vars,data_vars):
    '''make a DataArray for 'var' coordinate which is true when var != false_vars'''
    where_true_vars   = [varname not in false_vars for varname in data_vars]
    where_true_vars   = xr.DataArray(data=where_true_vars,dims=['var'],coords={'var':data_vars})
    return where_true_vars

def min_layer_where_notnull(base_value):
    '''get position of minimum layer where DataArray is not NaN'''
    return base_value.layer.broadcast_like(base_value).where(base_value.notnull()).min('layer').astype(int)

def apply_water_path_scaling(profile,wp):
    '''scale water content profile to water path'''
    profile    = profile-profile.min() #minimum value of 0
    norm       = profile.fillna(0).integrate('height') #find normalization constant
    return wp/norm*profile

