import numpy as np
import xarray as xr
import pandas as pd

from metpy.units import units, concatenate #calc_idealized_pwv
from metpy import calc                     #calc_idealized_pwv

from pathlib import Path
from datetime import datetime #getmonthname
import re

######################################
####### DATA LOADING ROUTINES ########
######################################

def load_minutely_flux_hists(folder=Path('.')):
    '''monthly histograms of minutely or 3-minutely surface longwave fluxes'''
    #load data
    fpaths = sorted(folder.glob("15-LWhist*.nc"))
    dses = [ ]
    for fpath in fpaths:
        ds = xr.open_dataset(fpath)
        ds = ds.assign_coords(datastream=ds.attrs['datastream']).expand_dims('datastream')
        dses.append(ds)
    ds = xr.concat(dses,dim='datastream')

    #merge nsaqcrad1longC1.c1 and nsaqcrad1longC1.c2
    #extract datastreams
    c1, c2 = ds.sel(datastream='nsaqcrad1longC1.c1'), ds.sel(datastream='nsaqcrad1longC1.c2')
    #get where they're missing (bool)
    c1_empty = c1.hist.sum(dim='bin')==0
    c2_empty = c2.hist.sum(dim='bin')==0
    #fill in with c1 if c2 is missing and c1 is nonzero
    merged = xr.where(c2_empty & ~c1_empty, c1, c2)
    #add datastream
    merged = merged.assign_coords(datastream='nsaqcrad1longC1.c2c1')
    ds = xr.concat([ds,merged],dim='datastream')

    #sum of NaN times and times with no files = total # of missing data points
    n_nan = ds.n_miss+ds.n_bad
    
    #divide by the number of observations plus missing data to get missing data fraction
    f_miss = n_nan/(n_nan+ds.hist.sum(dim='bin'))
    
    #before 1998, BRW GML fluxes are reported at 3-min frequency, but the code
    #to calculate missing data (15-...) assumes 1-min frequency. This causes f_miss to be
    #overestimated by 2/3. Correct this by subtracting 2/3 for relevant datastreams+months
    datastream_cond = (ds.datastream=='BSRN BAR') | (ds.datastream=='nsanoaaradbrwX1.b1')
    time_cond = ds.time<pd.to_datetime('1998-01-01')
    f_miss_corr = xr.where(time_cond & datastream_cond,f_miss-2./3., f_miss)
    
    #>50% of data must be 'good' or 'intermediate' to consider the statistics
    mask = f_miss_corr<0.5
    ds = ds.where(mask,other=np.nan)

    #calculate frequency of opaque peak
    da = ds.hist.sel(term='LWnet')
    ncool = da.sel(bin=da.bin_min<-25).sum(dim='bin')
    nopaq = da.sel(bin=da.bin_min>=-25).sum(dim='bin')
    birat = nopaq/(nopaq+ncool)
    birat.attrs = {'long_name':'$n_{opaq}$','units':'ratio','description':'frequency of net LW greater than -25 W/m$^2$'}

    return ds.assign(birat=birat)

def load_arm_temp(folder=Path('~/Documents/arm_data/ipynb_processed/').expanduser()):
    '''load temperature time series from ARM NSA C1'''
    #open datastreams saved to disk from above code, join in time to get full NSA C1 record
    arm_t2m_c1 = xr.open_dataset(folder/'19-Temp_Air-minfreq-nsaqcrad1longC1.c1.nc')
    arm_t2m_c2 = xr.open_dataset(folder/'19-Temp_Air-minfreq-nsaqcrad1longC1.c2.nc')
    #split c1 into 'before' and 'after' chunks for correct time ordering
    arm_t2m_c1_before = arm_t2m_c1.sel(time=arm_t2m_c1.time < arm_t2m_c2.time.min())
    arm_t2m_c1_after  = arm_t2m_c1.sel(time=arm_t2m_c1.time > arm_t2m_c2.time.max())
    #join into one dataset
    arm_t2m = xr.concat([arm_t2m_c1_before,arm_t2m_c2,arm_t2m_c1_after],'time')
    #apply qc (probably already applied)
    arm_t2m = arm_t2m.Temp_Air.where((arm_t2m['qc_Temp_Air']==0) & (arm_t2m['aqc_Temp_Air']<3))
    return arm_t2m

def load_abo_temp(folder=Path('~/Documents/noaa_data/met/brw/meteorology/').expanduser()):
    '''load temperature time series from the NOAA BRW Atmospheric Baseline Observatory'''
    #load data
    fpaths = sorted(folder.glob('*.txt'))
    columns=['site_code','year','month','day','hour',
             'wind_dir','wind_spd','wind_steadiness',
             'pressure','T_2m','T_10m','T_top','RH',
             'precip_rate']
    df = pd.concat([pd.read_csv(fp,delim_whitespace=True,names=columns) for fp in fpaths])
    
    #decode times
    datestring = [f'{y:04d}/{m:02d}/{d:02d} {h:02d}' for y,m,d,h in zip(df['year'],df['month'],df['day'],df['hour'])]
    df['Datetime'] = pd.to_datetime(datestring)
    df = df.set_index('Datetime')
    
    #turn missing values to NaN
    missing_values = {'wind_dir':-999,
                      'wind_spd':-99.9,
                      'wind_steadiness':-9,
                      'pressure':-999.9,
                      'T_2m':-999.9,
                      'T_10m':-999.9,
                      'T_top':-999.9,
                      'RH':-99,
                      'precip_rate':-99}#,
                      #'wind_spd':-999.9}
    for var in missing_values.keys():
        df[var] = df[var].where(df[var] != missing_values[var], other=np.nan)

    #convert to xarray
    da = xr.DataArray(df['T_2m']).rename({'Datetime':'time'}) 
    
    #remove initial stretch of missing data
    t0 = np.where(~np.isnan(da))[0][0]
    da = da.isel(time=slice(t0,None)) #remove all values before the first valid point
    return da

def calc_idealized_pwv(folder = Path('~/Documents/arm_data').expanduser()):
    '''get monthly-mean precipitable water vapor using observed T(z) but climatological RH(z)'''
    #load data
    folder = Path('~/Documents/arm_data')
    folder = Path.expanduser(folder)
    atmz = xr.open_dataset(folder/'ipynb_processed'/'24-armbeatm_merged.nc')
    
    #recalculate relative humidity from dewpoint because armbeatm is blank
    temp  = atmz.temperature_h*units('K')
    tdew  = atmz.dewpoint_h*units('K')
    good_rh = calc.relative_humidity_from_dewpoint(temp,tdew)
    good_rh = good_rh.metpy.convert_units('percent')
    da_rh  = good_rh.metpy.dequantify()
    da_rh.name = 'relative_humidity_h'
    atmz   = atmz.assign(relative_humidity_h=da_rh)
    
    #calculate water vapor mass mixing ratio
    press = xr.open_dataset('32-pressuremonthlymeans.nc').pressure
    press = press.mean(dim='time') #use one pressure profile for all times
    wvmr = calc.mixing_ratio_from_relative_humidity(press*units['hPa'],
                                                    temp,
                                                    good_rh)
    wvmr = wvmr.metpy.convert_units('g/kg')
    wvmr = wvmr.metpy.dequantify()
    wvmr.name = 'water vapor mass mixing ratio'
    atmz   = atmz.assign(vapor_mass_mixing_ratio_h=wvmr)
    
    #resample to monthly
    atmz_mon = atmz.resample(time='MS').mean()
    atmz_mon_clim = atmz_mon.groupby('time.month').mean()
    atmz_mon_anom = atmz_mon.groupby('time.month')-atmz_mon_clim
    Tair = atmz.temperature_p.sel(pressure=atmz.pressure>=700).mean(dim='pressure')
    Tair.name = 'temperature_air'
    Tair.attrs = {'long_name': '1000-700 hPa layer-averaged dry bulb temperature',
                  'units': 'K'}

    #recalculate PWV using the month climatological RH(z) but observed T(z)
    pwvs  = [ ]
    times = [ ]
    for time in atmz_mon.time:
        #load
        rh   = atmz_mon_clim.relative_humidity_h.sel(month=time.dt.month)
        temp = atmz_mon.temperature_h.sel(time=time)
        if np.isnan(temp).all(): continue
        #calc
        tdew   = calc.dewpoint_from_relative_humidity(temp*units['K'], rh*units['%'])
        pwv_cc = calc.precipitable_water(press*units['hPa'],tdew)
        #export
        pwvs.append(pwv_cc)
        times.append(time.data)
    pwv_cc = xr.DataArray(data=concatenate(pwvs),dims='time',coords={'time':times})
    pwv_cc = pwv_cc.metpy.convert_units('mm')
    pwv_cc = pwv_cc.metpy.dequantify()  #mm
    pwv_cc.attrs = {
        'long_name':'Clausius-Clayperon PWV using monthly-climo RH',
        'units':'mm'
    }
    return pwv_cc

def get_monthly_co2(folder=Path('~/Documents/noaa_data/brw_chem').expanduser()):
    df = pd.read_table(folder/'co2_brw_surface-flask_1_ccgg_month.txt',
                   comment='#',delim_whitespace=True,names=['site','year','month','conc'])
    df['day'] = np.ones_like(df['month'])
    df['time'] = pd.to_datetime(df[['year','month','day']])
    df = df.drop(columns=['year','month','day','site'])
    da = df.set_index('time').to_xarray().conc
    da.name = 'co2'
    da.attrs = {'long_name':'flask-measured CO2 concentration',
                'source':'NOAA GML BRW ABO co2_brw_surface-flask_1_ccgg_month.txt',
                'units':'ppm'}
    return da

def add_extra_monthly_vars(ds_xi_monthly):
    #idealized PWV with constant RH profile -- armbeatm
    pwv_cc = calc_idealized_pwv()                           
    ds_xi_monthly = ds_xi_monthly.assign(pwv_cc=pwv_cc)    
    ds_xi_monthly['pwv_cc'].attrs = pwv_cc.attrs

    #co2 concentration
    co2 = get_monthly_co2()
    ds_xi_monthly = ds_xi_monthly.assign(co2=co2)
    ds_xi_monthly['co2'].attrs = co2.attrs

    return ds_xi_monthly

def assemble_feature_datasets(folder = Path('~/Documents/arm_data').expanduser()):
    '''load atmospheric state and flux datasets for sensitivity analysis'''
    datasets = { }
    #PWV -- armbecldrad
    cld = xr.open_dataset(folder/'ipynb_processed'/'01-cldrad.nc')
    datasets['pwv'] = cld.pwv

    #lower tropospheric temperature -- armbeatm
    atmz = xr.open_dataset(folder/'ipynb_processed'/'24-armbeatm_merged.nc')
    Tair = atmz.temperature_p.sel(pressure=atmz.pressure>=700).mean(dim='pressure')
    Tair = Tair.dropna(dim='time')
    Tair.name = 'temperature_air'
    Tair.attrs = {'long_name': '1000-700 hPa layer-averaged dry bulb temperature',
                  'units': 'K'}
    datasets['column_air_temperature'] = Tair
    datasets['pressure_sfc'] = atmz.pressure_sfc

    #surface air temperature -- nsaqcrad1longC1.c1 and c2
    arm_t2m = load_arm_temp()
    datasets['Temp_Air'] = arm_t2m

    #downwelling longwave flux -- nsaqcrad1longC1.c1 and c2
    lwx = xr.open_dataset('22-LWfluxandhist_monthly_from_hourly_mergedstreams.nc')
    lwd = lwx.flux.sel(datastream='nsaqcrad1longC1.c2c1',term='LWdn')
    lwd = lwd.groupby('time.month') - lwd.groupby('time.month').mean()
    lwd.attrs = lwx.flux.attrs
    lwd.attrs['units'] = r'W/m$^2$'
    datasets['lwd'] = lwd

    #n_opaq -- nsaqcrad1longC1.c1 and c2
    #load monthly histograms of minutely data
    ds_hists = load_minutely_flux_hists()
    birat = ds_hists.birat
    nopaq = 100*birat.sel(datastream='nsaqcrad1longC1.c2c1')
    nopaq = nopaq.groupby('time.month') - nopaq.groupby('time.month').mean()
    nopaq.attrs = {'units':'%','long_name':'opaque LWN frequency'}
    datasets['nopaq'] = nopaq

    #ice/liq/mixed phase occurrence, IWP -- nsamicrobase2shupeturnC1.c1-variablecoeff
    st = xr.open_dataset('28_microbase2shupeturn_columnint_hourly.nc')
    for name in ('iwp','cloud_liquid','cloud_ice','cloud_mixed'):
        #REMOVE MIXED PHASE FREQ FROM LIQ AND ICE CATEGORIES
        if name not in ['cloud_liquid','cloud_ice']:
            datasets[name] = st[name]
        else:
            datasets[name] = st[name] - st['cloud_mixed'] 
            datasets[name].attrs = st[name].attrs
            datasets[name].attrs['POST-PROC_NOTE'] = 'removed 7 (mixed-phase)'

    #cloud cover -- nsamplcmask1zwangC1.c1 and nsaceilC1.b1
    fpaths = sorted((folder/'ipynb_processed'/'36-minutely_merge_mplcmask_and_ceil').glob('*.nc'))
    cbh = xr.open_mfdataset(fpaths).compute()
    tot_cld = cbh.tot_cld.where(cbh.detection!=1) #mask mpl clear but ceil invalid -- mpl only reports clouds >1 km or something like that
    datasets['tot_cld'] = tot_cld

    #lowest cloud base height -- nsaceilC1.b1
    fpaths = sorted((folder/'ipynb_processed'/'36-1-minutely_ceil').glob('*.nc'))
    ceil = xr.open_mfdataset(fpaths,preprocess=lambda ds: ds.drop_duplicates('time')).compute()
    datasets['cld_base'] = ceil.cbh
    datasets['tot_cld_ceil'] = ceil.tot_cld

    #liquid water path -- mwrret masked by cloud cover
    fpaths = sorted((folder/'ipynb_processed'/'37-minutely_mwrret').glob('*.nc'))
    mwr = xr.open_mfdataset(fpaths,preprocess=lambda ds: ds.drop_duplicates('time')).compute()
    lwp = mwr.lwp.where(cbh.detection>=4) #enforce either MPL or ceil cloudy -- change to isin([8,9,12]) for ceil cloudy
    lwp = lwp.clip(0,None) #remove subzero LWPs
    datasets['lwp'] = lwp

    return datasets

def assemble_feature_datasets_naivelwp(folder = Path('~/Documents/arm_data').expanduser()):
    '''load atmospheric state and flux datasets for sensitivity analysis'''
    
    datasets = { }
    #LWP, PWV -- armbecldrad
    cld = xr.open_dataset(folder/'ipynb_processed'/'01-cldrad.nc')
    datasets['pwv'],datasets['lwp'] = cld.pwv, cld.lwp

    #lower tropospheric temperature -- armbeatm
    atmz = xr.open_dataset(folder/'ipynb_processed'/'24-armbeatm_merged.nc')
    Tair = atmz.temperature_p.sel(pressure=atmz.pressure>=700).mean(dim='pressure')
    Tair = Tair.dropna(dim='time')
    Tair.name = 'temperature_air'
    Tair.attrs = {'long_name': '1000-700 hPa layer-averaged dry bulb temperature',
                  'units': 'K'}
    datasets['column_air_temperature'] = Tair
    datasets['pressure_sfc'] = atmz.pressure_sfc

    #surface air temperature -- nsaqcrad1longC1.c1 and c2
    arm_t2m = load_arm_temp()
    datasets['Temp_Air'] = arm_t2m

    #downwelling longwave flux -- nsaqcrad1longC1.c1 and c2
    lwx = xr.open_dataset('22-LWfluxandhist_monthly_from_hourly_mergedstreams.nc')
    lwd = lwx.flux.sel(datastream='nsaqcrad1longC1.c2c1',term='LWdn')
    lwd = lwd.groupby('time.month') - lwd.groupby('time.month').mean()
    lwd.attrs = lwx.flux.attrs
    lwd.attrs['units'] = r'W/m$^2$'
    datasets['lwd'] = lwd

    #n_opaq -- nsaqcrad1longC1.c1 and c2
    #load monthly histograms of minutely data
    ds_hists = load_minutely_flux_hists()
    birat = ds_hists.birat
    nopaq = 100*birat.sel(datastream='nsaqcrad1longC1.c2c1')
    nopaq = nopaq.groupby('time.month') - nopaq.groupby('time.month').mean()
    nopaq.attrs = {'units':'%','long_name':'opaque LWN frequency'}
    datasets['nopaq'] = nopaq

    #ice/liq/mixed phase occurrence, IWP -- nsamicrobase2shupeturnC1.c1-variablecoeff
    st = xr.open_dataset('28_microbase2shupeturn_columnint_hourly.nc')
    for name in ('iwp','cloud_liquid','cloud_ice','cloud_mixed'):
        #REMOVE MIXED PHASE FREQ FROM LIQ AND ICE CATEGORIES
        if name not in ['cloud_liquid','cloud_ice']:
            datasets[name] = st[name]
        else:
            datasets[name] = st[name] - st['cloud_mixed'] 
            datasets[name].attrs = st[name].attrs
            datasets[name].attrs['POST-PROC_NOTE'] = 'removed 7 (mixed-phase)'


    #cloud cover and cbh -- nsamplcmask1zwangC1.c1 and nsaceilC1.b1
    fpaths = sorted((folder/'ipynb_processed'/'36-minutely_merge_mplcmask_and_ceil').glob('*.nc'))
    cbh = xr.open_mfdataset(fpaths).drop_vars('detection').compute()
    datasets['cld_base'] = cbh.cbh
    datasets['tot_cld']  = cbh.tot_cld

    return datasets

def convert_units_feature_datasets(datasets):
    '''convert cm to mm and ratio to percent'''
    for name,da in datasets.items():
        #cm to mm
        if da.attrs['units'] == 'cm':
            da = 10*da
            da.attrs['units'] = 'mm'
        #ratio to %
        if da.attrs['units'] == 'ratio':
            da = 100*da
            da.attrs['units'] = '%'
        #C to K
        if da.attrs['units'] == 'C':
            da = 273.15+da
            da.attrs['units'] = 'K'
        #format g/m2
        if da.attrs['units'] in ['g/m2','g/m^2']:
            da.attrs['units'] = 'g/m$^2$'
        #save
        datasets[name] = da
    return datasets

def remove_jump_offsets(ds,step_jumps):
    '''remove timeseries jumps in Dataset with dict of arrayName:jump_time pairs'''
    ds = ds.copy()
    for name,jump_time in step_jumps.items():
        da          = ds[name]
        i_jump      = (da.time==pd.to_datetime(jump_time)).data.nonzero()[0].item(0)
        before_jump = da.isel(time=slice(None,i_jump))
        after_jump  = da.isel(time=slice(i_jump,None))
        offset      = after_jump.mean(dim='time')-before_jump.mean(dim='time')
        after_jump  = after_jump-offset
        no_jump_da  = xr.concat([before_jump,after_jump],dim='time')
        ds          = ds.assign({name:no_jump_da})
        print(f'removed {name} offset of {offset.data}')
    return ds

def mask_bad_times(ds,bad_times):
    '''mask a Dataset with a list of arrayName:[start,stop] dicts'''
    ds = ds.copy()
    cast_get = lambda d: list(d)[0]
    for d in bad_times:
        name     = cast_get(d.keys())
        interval = cast_get(d.values())
        da       = ds[name]
        bad_dts  = da.time.sel(time=slice(*interval))
        masked   = da.where(~da.time.isin(bad_dts))
        ds[name] = masked
    return ds

def prepare_feature_dataset():
    '''prepare a dataset of monthly atmospheric feature anomalies'''

    #load, prep, resample datasets
    datasets = assemble_feature_datasets()                   #load data
    datasets = convert_units_feature_datasets(datasets)      #convert units
    datasets_monthly = resample_feature_datasets(datasets)   #resample to monthly
    ds_xi_monthly = xr.Dataset(datasets_mon)                 #merge to dataset
    for name in ds_xi_monthly:
        ds_xi_monthly[name].attrs = datasets[name].attrs     #put back attributes

    #mask some choice suspect time periods
    bad_times = [
        {'tot_cld':['2007-03-01','2010-01-01']},     #unphysically low time period
        {'iwp':['2008-01-01','2008-03-01']},         #unphysically high outlier month
        {'iwp':['2005-01-01','2007-01-01']},         #basically zero for a year (mmcr sensitivity?)
        {'cloud_mixed':['2005-01-01','2007-01-01']}, #probably same issue
        {'cloud_mixed':['2005-01-01','2007-01-01']}, #probably same issue
        {'cloud_ice':['2005-01-01','2007-01-01']}    #probably same issue
    ]
    
    #steps
    step_jumps = {
        'cld_base':'2006-10-01', #bases get ~2x as high
        'iwp':'2008-01-01'       #iwps double probably from mmcr repairs
    #loud phase RFOs just look unusable across all categories
    #in the later period -- rapid trends over DJFM only
    }

    #mask malfunctioning periods, remove jumps from instrument changes
    ds_masked = mask_bad_times(ds_xi_monthly,bad_times)
    ds_nojump = remove_jump_offsets(ds_masked,step_jumps)

    return ds_nojump

######################################
######## RESAMPLING ROUTINES #########
######################################

def floor_times(times, freq):
    '''floor datetime DataArray to non-fixed frequencies, like months'''
    if freq == 'MS':
        times = times - pd.to_timedelta(times.dt.day-1,unit='day')
        return times.dt.floor('D')
    else:
        return times.dt.floor(freq)

def nanmean(da,ifreq='H',ofreq='MS',valid=lambda da: da.notnull(), miss_thresh=0.5):
    '''Resampling of time series with valid data threshold and rigorous missing data detection

    Frequencies must be manually specified using Pandas frequency aliases. Not all aliases are 
    accepted. For a list of valid frequency alias strings, see 
    https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects. 
    Multiples are supported for frequencies < months. For example, a sample every 10 seconds
    would be '10S'.

    -------------
    Arguments:
    da            : DataArray
                    timeseries data to resample
    ifreq         : str
                    input frequency alias. Multiples are supported for frequencies < months.
                    For example, a 10-second input frequency would be '10S'.
    ofreq         : str
                    output frequency alias.
    valid         : function
                    callback method for evaluating if datapoints are valid
    miss_thresh   : float
                    maximum missing data fraction allowable before mean is invalid
    -------------
    Returns:
    da            : DataArray
    smartly-resampled DataArray
    '''
    return da.mean(dim='time').where(qcmap_resample(valid(da),ifreq=ifreq,ofreq=ofreq,fmin=miss_thresh,inner=True))

def freq_divide(ifreq='H',ofreq='MS',times=None):
    '''For resampling of time series, get the number of input timesteps per output timestep.

    Returns number input timesteps at frequency per output timestep. Frequencies are specified
    using Pandas frequency aliases. If the ratio of input to output timesteps is an interval of 
    constant duration (i.e. shorter than a month), a scalar is returned. If the ratio is non-constant
    (e.g. days in a month or days in a year), the input series of timesteps must be provided in 
    "times" in order get the correct duration, and then returned is an array of the maximum number
    of input timesteps per output timestep.
    
    Frequencies are specified using Pandas frequency aliases. Not all aliases are accepted. 
    For a list of valid frequency alias strings, 
    see https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects. 
    Multiples are supported for frequencies < months. For example, a sample every 10 seconds
    would be '10S'.

    -------------
    Arguments:
    ifreq         : str
                    input frequency alias. Multiples are supported for frequencies < months.
                    For example, a 10-second input frequency would be '10S'.
    ofreq         : str
                    output frequency alias.
    times         : DateTime-like xr.DataArray
                    array of input absolute time objects
    -------------
    Returns:
    Nmax (scalar or xarray.DataArray of ints)
    Maximum number of input timesteps per output timestep. If constant, returns a scalar.
    If non-constant (e.g. days in a month), returns a DataArray.
    '''
    #extract leading digit on frequency alias (if any)
    regexp = re.compile('[0-9]*')
    end = regexp.match(ifreq).end()           # end position of leading digits
    iarg  = int(ifreq[:end]) if end>0 else 1  # default to 1 if no leading number
    ifreq = ifreq[end:]                       # strip leading digits
    
    #map time aliases to pd.Timedelta arguments
    accepted_aliases = {'D':   'days',
                        'H':   'hours',
                        'T':   'minutes',
                        'min': 'minutes',
                        'S':   'seconds'}
    
    #sanity check arguments
    assert ifreq in accepted_aliases.keys()
    accepted_out_aliases = ['MS','H','D','T','min']
    assert ofreq in accepted_out_aliases
    
    #get max number input timesteps (at ifreq) per averaging interval (ofreq)
    if ofreq == 'MS':
        ti   = times.min()
        ti   = ti-pd.Timedelta(days=int(ti.dt.day)-1)           # move start day to beginning of month
        ti   = np.datetime64(ti.data)                           # convert to scalar
        tf   = np.datetime64(times.max().data)
        tt   = pd.date_range(ti, tf, freq=ofreq)                # date_range at output frequency
        ttx  = pd.date_range(ti, periods=len(tt)+1, freq=ofreq) # add an extra period past end
        dt   = ttx[1:]-ttx[:-1]                                 # durations of aggregating interval
        Nmax = dt/pd.Timedelta(**{accepted_aliases[ifreq]:1})   # number of ifreq in ofreq
                                                                # (cf Trenton on S/O #22923775 for context on dividing pd.Timedeltas)
    elif (ofreq in ('H','D','T','min')) and (ifreq in ('T','min','H','S')):
        Nmax = pd.Timedelta(**{accepted_aliases[ofreq]:1})/pd.Timedelta(**{accepted_aliases[ifreq]:iarg})
    else:
        raise NotImplementedError(f'resampling freq {ifreq} to freq {ofreq} not supported!')
    return Nmax

def qcmap_resample(da_notnan,ifreq='H',ofreq='MS',fmin=0.5,inner=True):
    '''For resampling, return intervals where missing data fraction is below threshold.

    Provided a boolean array indicating valid data, the input and output sampling frequencies, 
    and a minumum data fraction (0-1), return a boolean array of when enough valid timestamps
    are present to return a valid mean in the resampled data. Frequencies are specified using 
    Pandas frequency aliases. Not all aliases are accepted. For a list of valid frequency strings, 
    see https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects.

    Note that this method is more robust than simply summing the number of valid data points,
    since this method includes both completely missing timesteps as well as invalid timesteps.

    -------------
    Arguments:
    da_isnan      : array-like (bool)
                    bool indicating if data timestep is valid (e.g. results of .notnull()).
    ifreq         : str
                    input frequency alias. Multiples are supported for frequencies < months.
                    For example, a 10-second input frequency would be '10S'.
    ofreq         : str
                    output frequency alias.
    fmin          : float
                    fraction (0-1) of valid timesteps required.
    inner         : bool
                    whether to use inside a Resampler (True) or as stand-alone (False).
    -------------
    Returns:
    array-like (bool)
    True if the resampled timestep has a valid mean, False otherwise.
    '''
    #print(da_notnan.size)
    #floor_times(da_notnan.time,ofreq).isel(time=0)
    #resample and apply
    if inner:
        #if inner, expect to recieve only one group
        Nmax = freq_divide(ifreq,ofreq,times=floor_times(da_notnan.time,ofreq).isel(time=0)).to_numpy()
        Nmax = np.squeeze(Nmax)
        Nobs = da_notnan.sum(dim='time')
    else:
        Nmax = freq_divide(ifreq,ofreq,times=da_notnan.time)
        Nobs = da_notnan.resample(time=ofreq,label='left').sum()
    qc   = (Nobs/Nmax) >= fmin
    return qc

def resample_feature_datasets(datasets):
    '''resample feature data sets to monthly frequency'''
    datasets_mon = { }
    cf_ifreq=pd.infer_freq(pd.to_datetime(datasets['tot_cld_ceil'].isel(time=slice(0,1000)).time))
    for name,da in datasets.items():
        print(name+' ',end='')
        #get input sampling frequency
        if name=='column_air_temperature':
            ifreq='D'
        else:
            ifreq=pd.infer_freq(pd.to_datetime(da.isel(time=slice(0,1000)).time))
        print(ifreq)
    
        #check if resample is necessary
        if ifreq != 'MS':
            #resample
            cloudy_only = ['lwp','iwp','cld_base','cloud_liquid',
                           'cloud_ice','cloud_mixed']
            if name not in cloudy_only:
                da = da.resample(time='MS').map(nanmean,ifreq=ifreq,ofreq='MS',miss_thresh=0.5)
            else:
                Nobs = da.notnull().resample(time='MS',label='left').sum()
                #consider masking or normalizing by tot_cld here
                Nmax = datasets['tot_cld_ceil']>0
                if cf_ifreq != ifreq:
                    Nmax = Nmax.resample(time=ifreq).sum()>0
                Nmax = Nmax.resample(time='MS').sum()
                qc   = (Nobs/Nmax) >= 0.5
                da   = da.resample(time='MS').mean().where(qc)
    
        datasets_mon[name] = da
    return datasets_mon

################################
##### MISC HELPER METHODS ######
################################

def fmt(f,ndigs=2):
    '''print out floats with a constant number of digits'''
    mindigs = max(np.ceil(np.log10(np.abs(f))),0) #num digits past the ones place
    prec    = 0 if mindigs>ndigs else ndigs-mindigs #precision
    fmt     = '{:.'+str(int(prec))+'f}' if prec>0 else '{:d}' #format string
    g       = float(f) if prec>0 else int(round(f)) #cast to int if needed precision is zero
    s       = fmt.format(g) #print out
    s       = str(float(s)) if prec>0 else s #strip trailing zeros
    #check that if we invert the float that it agrees to desired precision
    np.testing.assert_allclose(float(s),f,rtol=0,atol=0.5*10**(-prec))
    return s

def getmonthname(x,fmt='%B'):
    '''Returns month name from an int. 
    fmt specifies abbreviated ('%b') or full ('%B') name.'''
    return datetime.strptime(f'{x:02d}','%m').strftime(fmt)

def get_autocorr(x,lag=1):
    sigma=np.std(x)  ## calculate the standard deviation
    mean=np.mean(x)  ## calculate the mean
    n=len(x)         ## calculate the length of the timeseries

    ##Create two timeseries of the data at t=t1 and t=t2; remove the mean
    t1_m=x[0:-1*lag]-mean
    t2_m=x[lag:]-mean

    #Method #1
    #Calculate the autocorrelation using numpy correlate lagN
    lagNauto_np=np.correlate(t1_m,t2_m,mode='valid')/(n-lag)/(sigma**2)  ## Eq. 67 divided by the variance
    return lagNauto_np