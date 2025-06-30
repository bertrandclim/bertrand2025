import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from metpy.units import units
import statsmodels.api as sm

from pathlib import Path

from load_preproc_nb40 import *
from prep_basecase_nb44 import *

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
    #phase_res['mixed'] = phase_res['mixed'] | (phase_res['liquid'] & phase_res['ice'])

    #exclude liquid-containing profiles as ice
    phase_res['ice'] = phase_res['ice'] & ~(phase_res['mixed'] | phase_res['liquid'])

    #exclude mixed-containing profiles as liquid
    phase_res['liquid'] = phase_res['liquid'] & ~phase_res['mixed']

    #return as dataset
    return xr.Dataset(phase_res)

def get_is_phase(folder=Path('~/Documents/arm_data/').expanduser()):
    #load column-integrated phases and convert to broad categories 
    fps = sorted((folder/'ipynb_processed'/'28-microshupeturn-columnint-minutely-by-year').glob('*.nc'))
    st_prof  = xr.open_mfdataset(fps)
    is_phase = st_prof.phase.compute().astype(int)
    is_phase = bit_packed_to_broad_categories(is_phase)
    residual = (st_prof.tot_cld>0) & ~(is_phase['liquid'] | is_phase['mixed'] | is_phase['ice'])
    clear    = st_prof.tot_cld==0
    is_phase = is_phase.assign({'clear':clear,'other':residual})
    is_phase = is_phase.assign(all=xr.ones_like(is_phase['liquid']))
    is_phase = is_phase.compute()
    for name,total in is_phase.sum('time').items():
        if total == 0:
            print(f'dropping category "{name}" because it contains no observations')
            is_phase = is_phase.drop(name)
    #apply layer-based screening of liquid-only cases
    fps = sorted((folder/'ipynb_processed'/'59-betterliquidonly-columnint-minutely-by-year').glob('*.nc'))
    is_liquid = xr.open_mfdataset(fps).compute()
    is_phase['mixed']  = is_phase['mixed']  | is_liquid['liquid_snowing']
    is_phase['liquid'] = is_phase['liquid'] & is_liquid['liquid_nosnow']
    return is_phase



def get_ds_feat_typed(file='47-monthly_features_by_base_case_median.nc',folder=Path.cwd()):
    ds_feat_typed = xr.open_dataset(folder/file)
    ############################
    ### MINOR POSTPROCESSING ###
    ############################
    # Just some things I forgot to do before saving the netCDF to make life easier down the line
    #units error snuck in
    if (ds_feat_typed['temp'].min() < 0) and (ds_feat_typed['temp'].attrs['units'] != 'degC'):
        print('overwriting temperature units')
        ds_feat_typed['temp'].attrs['units'] = 'degC' #not K
    #multi-sensor cloudiness and single-sensor base disagree, giving "clear-sky cloud bases"
    #it would be better to fix upstream, but just overwrite it to NaN here
    ds_feat_typed['cbh'] = ds_feat_typed['cbh'].where(lambda da: da.type!='clear')
    #ice-over-liquid creates nonzero iwp during liquid cases
    #this inconsistency is visually distracting, so just overwrite it here (no radiative significance)
    ds_feat_typed['iwp'] = ds_feat_typed['iwp'].where(lambda da: da.type!='liquid')
    #align units with perturbation units

    ######################################
    ### ALIGN UNITS WITH PERTURBATIONS ###
    ######################################
    # Convert units for consistency with `rrtm/15-...ipynb`.
    perturb_units = {
        'cbh': 'km',
        'iwp': 'g/m$^2$',
        'column_air_temperature': 'K',
        'temp': 'K',
        'lwp': 'g/m$^2$',
        'tot_cld': '%',
        'pwv': 'mm',
        'co2': 'ppm',
        'lwd': 'W/m$^2$',
        'lwu': 'W/m$^2$',
        'lwn': 'W/m$^2$'
    }
    
    for name in ds_feat_typed:
        da = ds_feat_typed[name]
        #load in units and replace 'ratio' with '' for metpy to work
        units_src  = da.attrs['units']
        units_dest = perturb_units[name]
        filt_ratio = lambda s: '' if s == 'ratio' else s
        units_src  = filt_ratio(units_src)
        units_dest = filt_ratio(units_dest)
        #convert units and replaced desired_range with converted array
        da = da*units(units_src)
        da = da.metpy.convert_units(units_dest).metpy.dequantify()
        da.attrs = ds_feat_typed[name].attrs
        da.attrs['units'] = perturb_units[name] #unnecessary but keeps out long unit names
        ds_feat_typed[name] = da

    return ds_feat_typed

def get_ds_typed_anom(ds_feat_typed):
    #ds_feat_typed = xr.open_dataset('47-monthly_features_by_base_case_median.nc')
    #get anomalies
    declim        = lambda ds: ds.groupby('time.month')-ds.groupby('time.month').mean()
    ds_typed_anom = declim(ds_feat_typed)
    #drop 'other' category because no 'temp' values are valid
    ds_typed_anom = ds_typed_anom.where(lambda da: da.type != 'other', drop=True)

    ###############################
    ### ADD CONSISTENT METADATA ###
    ###############################
    # Adjust metadata for consistency with `rrtm/15-...ipynb`, 
    # rearrange to desired bar chart plot order, and infill decomposed CO$_2$ with all-time values.

    #transfer attributes from before monthly aggregation
    for name in ds_typed_anom:
        ds_typed_anom[name].attrs = ds_feat_typed[name].attrs
    
        #use long/short names from rrtm/15-...ipynb
        #add monthly CO2 assuming daily fluctuations are negligible
        co2            = get_monthly_co2()
        co2_anom       = declim(co2.sel(time=ds_typed_anom.time))
        co2_anom, _    = xr.broadcast(co2_anom, ds_typed_anom)
        co2_anom.attrs = co2.attrs
        ds_typed_anom  = xr.merge([co2_anom,ds_typed_anom])
        
    long_names = {
        'cbh': 'cloud base',      
        'iwp': 'ice water path',        
        'iwphi': 'opaque ice water path',
        'column_air_temperature': 'temperature profile',         
        'lwp': 'liquid water path',                
        'lwphi': 'opaque liquid water path',
        'tot_cld': 'cloud fraction',  
        'pwv': 'water vapor',         
        'co2': 'carbon dioxide',
        'lwu': 'upward longwave flux',
        'lwd': 'downward longwave flux',
        'lwn': 'net longwave flux',
        'temp': 'surface air temperature'
    }
    
    short_names = {
        'tot_cld': 'CF',
        'lwp': 'LWP',
        'lwphi': 'LWP$_\mathrm{opaq}$',
        'pwv': 'PWV',
        'pwv_cc': r'PWV$_\mathrm{CC}$',
        'Temp_Air': r'$T_{2m}$',
        'temp': r'$T_{2m}$',
        'column_air_temperature': r'$T_{air}$',
        'cbh': r'$z_{cld}$',
        'iwp': 'IWP',
        'iwphi': r'IWP$_\mathrm{opaq}$',
        'cloud_liquid': r'$f_{liq}$',
        'cloud_mixed': r'$f_{mix}$',
        'cloud_ice': r'$f_{ice}$',
        'co2': r'CO$_2$',
        'lwu': r'$F_U$',
        'lwd': r'$F_D$',
        'lwn': r'$F_N$'}
    
    for name in ds_typed_anom:
        ds_typed_anom[name].attrs['long_name'] = long_names[name]
        ds_typed_anom[name].attrs['short_name'] = short_names[name]
    
    #reorder for plotting
    desired_order = {'clear': 0, 'ice': 1, 'liquid': 2, 'mixed': 3,'all': 4}
    da_order      = xr.DataArray(data=list(desired_order.values()), dims='type', 
                                 coords={'type': list(desired_order.keys())})
    ds_typed_anom = ds_typed_anom.assign({'desired_order': da_order})
    ds_typed_anom = ds_typed_anom.sortby('desired_order')
    ds_typed_anom = ds_typed_anom.drop('desired_order')

    ######################################
    ### ALIGN UNITS WITH PERTURBATIONS ###
    ######################################
    # Convert units for consistency with `rrtm/15-...ipynb`.
    perturb_units = {
        'cbh': 'km',
        'iwp': 'g/m$^2$',
        'iwphi': 'g/m$^2$',
        'column_air_temperature': 'K',
        'temp': 'K',
        'lwp': 'g/m$^2$',
        'lwphi': 'g/m$^2$',
        'tot_cld': '%',
        'pwv': 'mm',
        'co2': 'ppm',
        'lwd': 'W/m$^2$',
        'lwu': 'W/m$^2$',
        'lwn': 'W/m$^2$'
    }
    
    for name in ds_typed_anom:
        da = ds_typed_anom[name]
        #load in units and replace 'ratio' with '' for metpy to work
        units_src  = da.attrs['units']
        units_dest = perturb_units[name]
        filt_ratio = lambda s: '' if s == 'ratio' else s
        units_src  = filt_ratio(units_src)
        units_dest = filt_ratio(units_dest)
        #convert units and replaced desired_range with converted array
        da = da*units(units_src)
        da = da.metpy.convert_units(units_dest).metpy.dequantify()
        da.attrs = ds_typed_anom[name].attrs
        da.attrs['units'] = perturb_units[name] #unnecessary but keeps out long unit names
        ds_typed_anom[name] = da

    return ds_typed_anom

def parse_result(res):
    '''get slope best-estimate and confidence interval from a statsmodels results object'''
    slope_lo = res.conf_int(0.05)[1,0]
    slope_hi = res.conf_int(0.05)[1,1]
    slope_med = res.params[1]
    return np.array(slope_lo), np.array(slope_hi), np.array(slope_med)
    
def regress(dax,day):
    '''replaced with statsmodels.OLS on 10/6/2024'''
    #drop missing values
    mask = ~np.isnan(dax) & ~np.isnan(day)
    dax,day = dax.where(mask,drop=True), day.where(mask,drop=True)
    if (dax.size == 0) or (day.size == 0):
        print(f'{day.name} empty, let slope = 0')
        slope, intercept, low_slope, high_slope = 0.0, 0.0, 0.0, 0.0
    else:
        Y, X = day.data, dax.data
        X = sm.add_constant(X)
        olsr_djfm_res = sm.OLS(Y,X, missing='drop').fit()
        low_slope, high_slope, slope = parse_result(olsr_djfm_res)
        intercept = olsr_djfm_res.params[0]
    return {'slope':slope,'intercept':intercept,
            'low_slope':low_slope,'high_slope':high_slope}

def regress_theilslopes(dax,day):
    '''before 10/6/2024 this was the method for "regress" '''
    #drop missing values
    mask = ~np.isnan(dax) & ~np.isnan(day)
    dax,day = dax.where(mask,drop=True), day.where(mask,drop=True)
    if (dax.size == 0) or (day.size == 0):
        print(f'{day.name} empty, let slope = 0')
        slope, intercept, low_slope, high_slope = 0.0, 0.0, 0.0, 0.0
    else:
        slope, intercept, low_slope, high_slope = stats.theilslopes(day,dax)
    return {'slope':slope,'intercept':intercept,
            'low_slope':low_slope,'high_slope':high_slope}

def aggregate(ds_typed_anom,key):
    '''apply regression against temperature across type and response variables, saving key'''
    assert key in ['slope','intercept','low_slope','high_slope','fullresult'], 'invalid choice of key'
    reses_by_key = { }
    temp_anom = ds_typed_anom['temp']
    for da_typed_anom in ds_typed_anom.values():
        reses = [ ]
        for this_type in ds_typed_anom.type:
            xi_anom_sub   = da_typed_anom.sel(type=this_type)
            temp_anom_sub = temp_anom.sel(type=this_type)
            res = regress(temp_anom_sub,xi_anom_sub)[key]
            reses.append(res)
        da_res = xr.DataArray(data=reses,dims=['type'],coords={'type':ds_typed_anom.type})
        reses_by_key[da_typed_anom.name] = da_res
    ds_slope_typed = xr.Dataset(reses_by_key)
    return ds_slope_typed

def get_weights(is_phase,subset=True):
    #get monthly frequency of conditions associated with each base case
    num_cases = is_phase.astype(int).resample(time='MS').sum()
    abs_cases = ['clear','all']
    num_cases_abs = num_cases[abs_cases]
    num_cases_rel = num_cases.drop(abs_cases)
    weights_abs = num_cases_abs/num_cases_abs['all']
    sum_kinds   = lambda ds: ds['liquid']+ds['ice']+ds['mixed']+ds['clear']
    weights_rel = num_cases_rel/sum_kinds(num_cases)
    weights = xr.merge([weights_abs,weights_rel])
    
    #reshape to data array
    slices = [ ]
    for name,da in weights.items():
        if name != 'all': 
            slices.append(da.assign_coords({'type':name}))
    da_weights      = xr.concat(slices,dim='type')
    sub_mos         = lambda ds: ds.sel(time=ds['time.month'].isin([12,1,2,3]))
    if subset:
        da_weights_djfm = sub_mos(da_weights)
    else:
        da_weights_djfm = da_weights
    return da_weights_djfm

def join_by_weights(da_var,weights):
    '''get weighted sum of variable and append result to type dimension'''
    #equivalent to dxdt_med.weighted(da_weights.mean('time')).mean('type')
    sum_type = (da_var*weights).sum('type')
    sum_type = sum_type.assign_coords(type='sum')
    return xr.concat([da_var,sum_type],dim='type')

def find_weight_response_to_temp(da_weights,temp_anom,key):
    '''apply regression against temperature across type and response variables, saving key'''
    assert key in ['slope','intercept','low_slope','high_slope'], 'invalid choice of key'
    reses = [ ]
    for this_type in da_weights.type:
        fj_anom_sub   = da_weights.sel(type=this_type)
        temp_anom_sub = temp_anom.sel(type=this_type)
        res = regress(temp_anom_sub,fj_anom_sub)[key]
        reses.append(res)
    da_res = xr.DataArray(data=reses,dims=['type'],coords={'type':da_weights.type})
    return da_res

def weightsum_intensive(dyj_dt,fj):
    '''
    total change in linear combination variable due to changing values
    if y = sum_j f_j*y_j, the change in y implied by changing y_j
    
    dyj_dt: slopes of response 'y' to predictor 't' by case 'j'
    fj:   frequency of cases by case 'j'
    '''
    #equivalent to dxdt_med.weighted(da_weights.mean('time')).mean('type')
    dydt_int = (dyj_dt*fj).sum('type')
    return dydt_int

def weightsum_intensive_err(dyj_dt,fj,dyj_dt_err):
    '''
    error propagation for weightsum_intensive. Assume fj are known exactly.
    '''
    #multiply by a constant and add in quadrature
    dydt_int = np.sqrt(((dyj_dt_err*fj)**2).sum('type'))
    return dydt_int

def weightsum_extensive(dfj_dt,fj,yj):
    '''
    total change in linear combination response variable due to changing weights
    if y = sum_j f_j*y_j, the change in y implied by changing f_j
    
    dfj_dt: slopes of response frequency 'f' to predictor 't' by case 'j'
    fj:     frequencies by cases by case 'j'
    yj:     values of response 'y' by case 'j' 
    '''
    y        = (yj*fj).sum('type')         #value of y (weighted average of yj)
    dy_dfj   = yj - (y-fj*yj)/(1-fj)       #change in y due to change in fj
    dydt_ext = (dy_dfj*dfj_dt).sum('type') #change in y due to t via changing fj
    return dydt_ext

def weightsum_extensive_err(dfj_dt,fj,yj,dfj_dt_err):
    '''
    error propagation for weightsum_extensive. Assume errors in fj and yj are zero.

    y       = sum_j (dy/df_j) * (df_j/dt)
    '''
    y            = (yj*fj).sum('type')       #value of y (weighted average of yj)
    dy_dfj       = yj - (y-fj*yj)/(1-fj)     #change in y due to change in fj
    dydt_fj_err  = np.abs(dy_dfj)*dfj_dt_err #error propagation for product
    dydt_ext_err = np.sqrt((dydt_fj_err**2).sum('type')) #error propagation for sum
    return dydt_ext_err

def nosum_extensive(dfj_dt,fj,yj):
    '''
    total change in linear combination response variable due to changing weights
    if y = sum_j f_j*y_j, the change in y implied by changing f_j
    
    dfj_dt: slopes of response frequency 'f' to predictor 't' by case 'j'
    fj:     frequencies by cases by case 'j'
    yj:     values of response 'y' by case 'j' 
    '''
    y        = (yj*fj).sum('type')         #value of y (weighted average of yj)
    dy_dfj   = yj - (y-fj*yj)/(1-fj)       #change in y due to change in fj
    dydt_ext = (dy_dfj*dfj_dt)             #change in y due to t via changing fj
    return dydt_ext

def nosum_extensive_err(dfj_dt,fj,yj,dfj_dt_err):
    '''
    error propagation for nosum_extensive. Assume errors in fj and yj are zero.
    '''
    y            = (yj*fj).sum('type')       #value of y (weighted average of yj)
    dy_dfj       = yj - (y-fj*yj)/(1-fj)     #change in y due to change in fj
    dydt_fj_err  = np.abs(dy_dfj)*dfj_dt_err #simple fractional error propagation
    return dydt_fj_err

def add_all_sums_to_type(dxdt,dxdt_int,dxdt_ext,dxdt_tot):
    '''get weighted sum of variable and append result to type dimension'''
    obs  = dxdt.sel(type='all')
    subs = dxdt.sel(type=dxdt.type!='all')
    subs = [subs.sel(type=type) for type in subs.type]
    tots = [dxdt_int.assign_coords(type='sum_int'),
            dxdt_ext.assign_coords(type='sum_ext'),
            dxdt_tot.assign_coords(type='sum_tot')]
    return xr.concat([*subs,*tots,obs],dim='type')

def add_one_sum_to_type(dxdt,dxdt_tot):
    '''get weighted sum of variable and append result to type dimension'''
    obs  = dxdt.sel(type='all')
    subs = dxdt.sel(type=dxdt.type!='all')
    subs = [subs.sel(type=type) for type in subs.type]
    tots = dxdt_tot.assign_coords(type='sum_tot')
    return xr.concat([*subs,tots,obs],dim='type')

def transpose(ds):
    '''switch data var name with coordinate name for 1D vars'''
    das = [ ]
    for name,da in ds.items():
        das.append(da.assign_coords({'feature':name}))
    ds2 = xr.concat(das,dim='feature')
    das = [ ]
    for type in ds2.type.data:
        da       = ds2.sel(type=type).drop('type')
        da.name  = type
        da.attrs = {'units':'W/m$^2$',
                    'long_name': 'fraction of subset longwave response due to feature'}
        das.append(da)
    return xr.merge(das)

def get_processed_perturbations(file):
    '''prepare the raw perturbation output for analysis'''
    cases_anom = xr.open_dataset(file)
    #get anomalies for fluxes for easier comparison
    cases_anom[['lwd','lwu']] = cases_anom-cases_anom[['lwd','lwu']].mean('step')
    
    ####################################################
    ### mask LWP and IWP to small perturbations only ###
    ####################################################
    #get the unperturbed step
    step_da   = xr.DataArray(data=np.arange(0,20),dims='step')
    step_da   = step_da.broadcast_like(cases_anom.offset)
    sign_diff = (cases_anom.offset>=0).astype(int).diff('step').pad({'step':(0,1)},constant_values=0)
    zero_xing = step_da.where(sign_diff,other=np.nan).sum('step')
    #make a mask of three steps centered on the unperturbed step (+/- 5 g/m^2)
    mask_orig = step_da.where(sign_diff,other=np.nan).notnull()
    mask      = mask_orig | mask_orig.shift(step=-1,fill_value=False)
    #mask      = mask | mask_orig.shift(step=-2,fill_value=False)
    mask      = mask | mask_orig.shift(step=1,fill_value=False)
    mask      = mask | mask_orig.shift(step=2,fill_value=False)
    #mask all steps not within this window
    cases_anom_windowed = cases_anom.where(mask)
    #put back into initial array
    cases_anom = cases_anom_windowed.where(lambda ds: ds.feature.isin(['lwp','iwp']),other=cases_anom)
    
    #######################################
    ### mask outer perturbations of PWV ###
    #######################################
    #mask the first two and last three steps
    cases_anom_noedges = cases_anom.where(lambda ds: (3<=ds.step)&(ds.step<17))
    #put back into initial array
    cases_anom = cases_anom_noedges.where(lambda ds: ds.feature.isin(['pwv']),other=cases_anom)
    
    ########################
    ### calculate slopes ###
    ########################
    #finite-difference order
    dl     = 1
    diff   = cases_anom.shift(step=-dl)-cases_anom
    slopes = diff.lwd/diff.offset
    #get lapse rate from temperature profile
    slopes = xr.where(slopes.feature=='column_air_temperature',
                      slopes-slopes.sel(feature='planck').drop('feature'),
                      slopes)
    #drop cloud fraction
    slopes = slopes.sel(feature=(slopes.feature!='tot_cld'))
    #take the average slope across valid steps
    slopes_by_case = slopes.mean('step')    #small set
    return slopes_by_case

def group_by_type(da_waterpath, is_phase):
    '''split water path by phase classification'''
    waterpath_by_phase = [ ]
    for name,mask in is_phase.items():
        da_phase          = da_waterpath.sel(time=mask)
        da_phase          = da_phase.assign_coords({'phase':name})
        da_phase['phase'] = da_phase['phase'].astype(str)
        da_phase.attrs    = da_phase.attrs
        waterpath_by_phase.append(da_phase)
    waterpath_by_phase    = xr.concat(waterpath_by_phase,dim='phase')
    return waterpath_by_phase

def waterpath_histogram(da_waterpath, bins=np.geomspace(1e-2,1e4,30)):
    '''monthly histograms of ice or liquid water path'''
    shortnames = {'ice water path':'iwp','liquid water path':'lwp'}
    shortname  = shortnames[da_waterpath.attrs['long_name']]
    #make bins
    bin_mins, bin_maxes = bins[:-1], bins[1:]
    bin_mids = 0.5*(bin_mins+bin_maxes)
    bin_mins  = xr.DataArray(data=bin_mins, dims=['waterpath'], coords={'waterpath':bin_mids})
    bin_maxes = xr.DataArray(data=bin_maxes, dims=['waterpath'], coords={'waterpath':bin_mids})
    #compute monthly histogram
    da_waterpath, bin_mins  = xr.broadcast(da_waterpath, bin_mins)
    da_waterpath, bin_maxes = xr.broadcast(da_waterpath, bin_maxes)
    hist_wp = ((bin_mins<=da_waterpath)&(da_waterpath<bin_maxes)).astype(int)
    hist_wp = hist_wp.resample(time='MS').sum().dropna('time')
    #add back bin edges
    bin_mins, bin_maxes = bins[:-1], bins[1:]
    bin_edges = np.array([bin_mins,bin_maxes])
    bin_edges = xr.DataArray(data=bin_edges, dims=['bound','waterpath'], coords={'waterpath':bin_mids,'bound':['lower','upper']})
    hist_wp = hist_wp.to_dataset(name='hist').assign_coords({'bin_edges':bin_edges})
    return hist_wp

def cre_by_waterpath_bin(cre, da_wp, bins = np.geomspace(1e-2,1e4,30), 
                         stat_funcs={
                             'mean': lambda da: da.mean('time'),
                             'pct_5th':  lambda da: da.quantile(0.05,'time').drop('quantile'),
                             'pct_95th': lambda da: da.quantile(0.95,'time').drop('quantile')
                         }):
    '''group cre by water path bin and apply statistics to each group in bin'''
    #make bins
    bin_mins, bin_maxes = bins[:-1], bins[1:]
    bin_mids = 0.5*(bin_mins+bin_maxes)
    bin_mins  = xr.DataArray(data=bin_mins, dims=['waterpath'], coords={'waterpath':bin_mids})
    bin_maxes = xr.DataArray(data=bin_maxes, dims=['waterpath'], coords={'waterpath':bin_mids})
 
    #make instantaneous water path bin mask
    da_wp, cre       = xr.align(da_wp, cre, join='inner')
    da_wp, bin_mins  = xr.broadcast(da_wp, bin_mins)
    da_wp, bin_maxes = xr.broadcast(da_wp, bin_maxes)
    hist_wp          = (bin_mins<=da_wp) & (da_wp<bin_maxes)
    #apply it to cloud radiative effect
    cre_cond_wp, _   = xr.broadcast(cre, hist_wp)
    cre_cond_wp      = cre_cond_wp.where(hist_wp)
    #apply statistics to cre grouped by waterpath bin
    das = { }
    for name, stat in stat_funcs.items():
        das[name] = stat(cre_cond_wp)
    return xr.Dataset(das)

def regress_hist_against_temp(xwp_hist_anom, ds_typed_anom):
    '''regress each water path bin against temperature anomaly'''
    exploded = xwp_hist_anom.to_dataset('waterpath')
    exploded = exploded.rename({'phase':'type'})
    exploded = exploded.assign(temp=ds_typed_anom['temp'])
    results  = { }
    for key in ['slope','low_slope','high_slope']:
        result   = aggregate(exploded, key)
        result   = result.drop('temp')
        result   = result.to_array('waterpath')
        results[key] = result
    results = xr.Dataset(results)
    return results.rename({'type':'phase'})

def calculate_lwp_iwp_driving_simpleuncert(ds_typed_anom, use_dask=True, SONDEPRODUCT='interpsonde',
                              iwp_bins=np.geomspace(1e-2,1e4,30),
                              lwp_bins=np.geomspace(1e-2,1e4,30),
                              keep_months=[12,1,2,3]):
    '''CRE-based cloud water path attribution'''

    assert SONDEPRODUCT in ['interpsonde','mergesonde']
    sub_mos = lambda da: da.sel(time=da['time.month'].isin(keep_months))
    
    #load cloud props
    folder   = Path('~/Documents/arm_data').expanduser()
    fname    = '28-microshupeturn-columnint-minutely-by-year'
    fps      = sorted(Path(folder/'ipynb_processed'/fname).glob('*.nc'))
    st_prof  = xr.open_mfdataset(fps)
    iwp      = st_prof.iwp.compute()
    lwp      = st_prof.lwp.compute()
    is_phase = get_is_phase(folder)
    is_phase = is_phase.drop('clear')
    is_phase = is_phase[['ice','liquid','mixed','all']]
    #restrict to DJFM
    iwp, lwp = sub_mos(iwp),sub_mos(lwp)
    is_phase = sub_mos(is_phase)
    #group IWP/LWP by phase
    iwp_by_phase = group_by_type(iwp, is_phase)
    lwp_by_phase = group_by_type(lwp, is_phase)
    #compute histograms
    ds_iwp_hist = waterpath_histogram(iwp_by_phase,iwp_bins)
    ds_lwp_hist = waterpath_histogram(lwp_by_phase,lwp_bins)
    iwp_hist, lwp_hist = ds_iwp_hist['hist'], ds_lwp_hist['hist']

    #calculate cloud radiative effects
    if SONDEPRODUCT == 'mergesonde':
        fname_allsky = '19-nsa_rrtm_minutely_v2_vanilla_surf_all.nc'
        fname_clrsky = '19-nsa_rrtm_minutely_v2_nocld_surf_all.nc'
    elif SONDEPRODUCT == 'interpsonde':
        fname_allsky = '19-nsa_rrtm_minutely_v2_interpsonde_surf_all.nc'
        fname_clrsky = '19-nsa_rrtm_minutely_v2_interpsonde_nocld_surf_all.nc'
    #load RRTM output and difference
    folder = Path('~/Documents/arm_data/').expanduser()
    ds_allsky  = xr.open_dataset(folder/'ipynb_processed'/fname_allsky)
    ds_clrsky  = xr.open_dataset(folder/'ipynb_processed'/fname_clrsky)
    ds_allsky, ds_clrsky = xr.align(ds_allsky, ds_clrsky,join='inner')
    ds_allsky, ds_clrsky = ds_allsky.chunk({'time':90000}), ds_clrsky.chunk({'time':90000})
    cre = ds_allsky.downward_flux - ds_clrsky.downward_flux
    cre = cre.compute()
    #screen to cloudy skies only
    is_cloud = st_prof.tot_cld.astype(bool).compute()
    is_cloud, _ = xr.align(is_cloud, cre, join='inner')
    cre = cre.where(is_cloud)
    #group by phase
    is_phase_cre, _ = xr.align(is_phase, cre)
    cre_by_phase = group_by_type(cre, is_phase_cre)
    #group by iwp bin
    cre_iwp_stats = cre_by_waterpath_bin(cre_by_phase, iwp, iwp_bins)
    #group by lwp bin
    cre_lwp_stats = cre_by_waterpath_bin(cre_by_phase, lwp, lwp_bins)

    ## bring this in for near-surface air temperature anomaly ##
    ds_typed_anom = ds_typed_anom.assign(type=ds_typed_anom['type'].astype(str))
    ds_typed_anom = ds_typed_anom.sel(type=ds_typed_anom['type'].isin(iwp_hist.phase))
    ds_typed_anom, iwp_hist_anom = xr.align(ds_typed_anom, iwp_hist)

    #preprocessing for regressions
    #1. drop AERI on/off
    times  = lambda da: da#.sel(time=slice('2008-01-01',None)) 
    #2. normalize by overall phase frequency every month (remove phase-feedback)
    norm   = lambda da: da/da.sum('waterpath')
    #3. remove seasonal cycle in each histogram bin
    declim = lambda da: da.groupby('time.month')-da.groupby('time.month').mean()
    #4. do all three in a row
    proc   = lambda da: declim(norm(times(da)))
    iwp_hist_anom, lwp_hist_anom = proc(iwp_hist), proc(lwp_hist)
    
    #response to temperature in each water path histogram bin
    iwp_hist_resp = regress_hist_against_temp(iwp_hist_anom, ds_typed_anom)
    lwp_hist_resp = regress_hist_against_temp(lwp_hist_anom, ds_typed_anom)

    #NOTE: mask uncertain IWP response during 'liquid-only' clouds (ice over liquid)
    #integrate over bins
    dFdt_IWP_med = iwp_hist_resp['slope']*cre_iwp_stats['mean']
    dFdt_LWP_med = lwp_hist_resp['slope']*cre_lwp_stats['mean']
    dFdt_IWP_med = dFdt_IWP_med.sum('waterpath')
    dFdt_LWP_med = dFdt_LWP_med.sum('waterpath')
    dFdt_IWP_med = dFdt_IWP_med.where(lambda da: da['phase'] != 'liquid', other=0)
    attributed_med_wp = xr.Dataset({'iwp':dFdt_IWP_med,'lwp':dFdt_LWP_med})
    
    #propagate error
    dFdt_IWP_hi  = iwp_hist_resp['high_slope']*cre_iwp_stats['mean']
    dFdt_LWP_hi  = lwp_hist_resp['high_slope']*cre_lwp_stats['mean']
    dFdt_IWP_lo  = iwp_hist_resp['low_slope']*cre_iwp_stats['mean']
    dFdt_LWP_lo  = lwp_hist_resp['low_slope']*cre_lwp_stats['mean']
    dFdt_IWP_err = 0.5*(dFdt_IWP_hi-dFdt_IWP_lo)
    dFdt_LWP_err = 0.5*(dFdt_LWP_hi-dFdt_LWP_lo)
    dFdt_IWP_err = (dFdt_IWP_err**2).sum('waterpath')
    dFdt_LWP_err = (dFdt_LWP_err**2).sum('waterpath')
    dFdt_IWP_err = dFdt_IWP_err.where(lambda da: da['phase'] != 'liquid', other=0)
    attributed_err_wp = xr.Dataset({'iwp':dFdt_IWP_err,'lwp':dFdt_LWP_err})

    return attributed_med_wp.rename({'phase':'type'}), attributed_err_wp.rename({'phase':'type'})

    

def calculate_lwp_iwp_driving(ds_typed_anom, use_dask=True, SONDEPRODUCT='interpsonde',
                              iwp_bins=np.geomspace(1e-2,1e4,30),
                              lwp_bins=np.geomspace(1e-2,1e4,30),
                              keep_months=[12,1,2,3]):
    '''CRE-based cloud water path attribution. Complex error propagation.'''

    assert SONDEPRODUCT in ['interpsonde','mergesonde']
    sub_mos = lambda da: da.sel(time=da['time.month'].isin(keep_months))
    
    #load cloud props
    folder   = Path('~/Documents/arm_data').expanduser()
    fname    = '28-microshupeturn-columnint-minutely-by-year'
    fps      = sorted(Path(folder/'ipynb_processed'/fname).glob('*.nc'))
    st_prof  = xr.open_mfdataset(fps)
    iwp      = st_prof.iwp.compute()
    lwp      = st_prof.lwp.compute()
    is_phase = get_is_phase(folder)
    is_phase = is_phase.drop('clear')
    is_phase = is_phase[['ice','liquid','mixed','all']]
    #restrict to DJFM
    iwp, lwp = sub_mos(iwp),sub_mos(lwp)
    is_phase = sub_mos(is_phase)
    #group IWP/LWP by phase
    iwp_by_phase = group_by_type(iwp, is_phase)
    lwp_by_phase = group_by_type(lwp, is_phase)
    #compute histograms
    ds_iwp_hist = waterpath_histogram(iwp_by_phase,iwp_bins)
    ds_lwp_hist = waterpath_histogram(lwp_by_phase,lwp_bins)
    iwp_hist, lwp_hist = ds_iwp_hist['hist'], ds_lwp_hist['hist']

    #calculate cloud radiative effects
    if SONDEPRODUCT == 'mergesonde':
        fname_allsky = '19-nsa_rrtm_minutely_v2_vanilla_surf_all.nc'
        fname_clrsky = '19-nsa_rrtm_minutely_v2_nocld_surf_all.nc'
    elif SONDEPRODUCT == 'interpsonde':
        fname_allsky = '19-nsa_rrtm_minutely_v2_interpsonde_surf_all.nc'
        fname_clrsky = '19-nsa_rrtm_minutely_v2_interpsonde_nocld_surf_all.nc'
    #load RRTM output and difference
    folder = Path('~/Documents/arm_data/').expanduser()
    ds_allsky  = xr.open_dataset(folder/'ipynb_processed'/fname_allsky)
    ds_clrsky  = xr.open_dataset(folder/'ipynb_processed'/fname_clrsky)
    ds_allsky, ds_clrsky = xr.align(ds_allsky, ds_clrsky,join='inner')
    ds_allsky, ds_clrsky = ds_allsky.chunk({'time':90000}), ds_clrsky.chunk({'time':90000})
    cre = ds_allsky.downward_flux - ds_clrsky.downward_flux
    cre = cre.compute()
    #screen to cloudy skies only
    is_cloud = st_prof.tot_cld.astype(bool).compute()
    is_cloud, _ = xr.align(is_cloud, cre, join='inner')
    cre = cre.where(is_cloud)
    #group by phase
    is_phase_cre, _ = xr.align(is_phase, cre)
    cre_by_phase = group_by_type(cre, is_phase_cre)
    #group by iwp bin
    cre_iwp_stats = cre_by_waterpath_bin(cre_by_phase, iwp, iwp_bins)
    #group by lwp bin
    cre_lwp_stats = cre_by_waterpath_bin(cre_by_phase, lwp, lwp_bins)

    ## bring this in for near-surface air temperature anomaly ##
    ds_typed_anom = ds_typed_anom.assign(type=ds_typed_anom['type'].astype(str))
    ds_typed_anom = ds_typed_anom.sel(type=ds_typed_anom['type'].isin(iwp_hist.phase))
    ds_typed_anom, iwp_hist_anom = xr.align(ds_typed_anom, iwp_hist)

    #preprocessing for regressions
    #1. drop AERI on/off
    times  = lambda da: da#.sel(time=slice('2008-01-01',None)) 
    #2. normalize by overall phase frequency every month (remove phase-feedback)
    norm   = lambda da: da/da.sum('waterpath')
    #3. remove seasonal cycle in each histogram bin
    declim = lambda da: da.groupby('time.month')-da.groupby('time.month').mean()
    #4. do all three in a row
    proc   = lambda da: declim(norm(times(da)))
    iwp_hist_anom, lwp_hist_anom = proc(iwp_hist), proc(lwp_hist)
    
    #response to temperature in each water path histogram bin
    iwp_hist_resp = regress_hist_against_temp(iwp_hist_anom, ds_typed_anom)
    lwp_hist_resp = regress_hist_against_temp(lwp_hist_anom, ds_typed_anom)

    #NOTE: mask uncertain IWP response during 'liquid-only' clouds (ice over liquid)
    #integrate over bins
    dFdt_IWP_med = iwp_hist_resp['slope']*cre_iwp_stats['mean']
    dFdt_LWP_med = lwp_hist_resp['slope']*cre_lwp_stats['mean']

    maskliq = lambda ds: ds.where(lambda da: da['phase'] != 'liquid', other=0)
    dFdt_IWP_med = maskliq(dFdt_IWP_med)
 
    #propagate error
    #calculate error in source terms
    lwp_hist_resp_err = 0.5*(lwp_hist_resp['high_slope']-lwp_hist_resp['low_slope'])
    iwp_hist_resp_err = 0.5*(iwp_hist_resp['high_slope']-iwp_hist_resp['low_slope'])
    cre_lwp_stats_err = 0.5*(cre_lwp_stats['pct_95th']-cre_lwp_stats['pct_5th'])
    cre_iwp_stats_err = 0.5*(cre_iwp_stats['pct_95th']-cre_iwp_stats['pct_5th'])
 
    #propagate error from source terms
    dFdt_IWP_err = product_uncertainty(dFdt_IWP_med,
                                       [iwp_hist_resp['slope'],cre_iwp_stats['mean']],
                                       [iwp_hist_resp_err, cre_iwp_stats_err])
    dFdt_LWP_err = product_uncertainty(dFdt_LWP_med,
                                       [lwp_hist_resp['slope'],cre_lwp_stats['mean']],
                                       [lwp_hist_resp_err, cre_lwp_stats_err])
    dFdt_IWP_err = (dFdt_IWP_err**2).sum('waterpath')
    dFdt_LWP_err = (dFdt_LWP_err**2).sum('waterpath')
    dFdt_IWP_err = maskliq(dFdt_IWP_err)
    dFdt_IWP_med = dFdt_IWP_med.sum('waterpath')
    dFdt_LWP_med = dFdt_LWP_med.sum('waterpath')
    attributed_err_wp = xr.Dataset({'iwp':dFdt_IWP_err,'lwp':dFdt_LWP_err})
    attributed_med_wp = xr.Dataset({'iwp':dFdt_IWP_med,'lwp':dFdt_LWP_med})

    return attributed_med_wp.rename({'phase':'type'}), attributed_err_wp.rename({'phase':'type'})

def product_uncertainty(out_mean,ins_mean,ins_err):
    '''uncertainty propagation for products/quotients'''
    terms   = [(x_err/x_mean)**2 for x_mean,x_err in zip(ins_mean,ins_err)]
    rel_err = np.sqrt(sum(terms))
    return out_mean*rel_err
