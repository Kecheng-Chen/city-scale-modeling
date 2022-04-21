# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 10:36:31 2021

@author: Kecheng Chen
"""
import pandas as pd
from scipy import stats
import glob
import numpy as np
import fnmatch

def custom_to_datetime(date):
    # If the time is 24, set it to 0 and increment day by 1
    if date[8:10] == '24':
        return pd.to_datetime(date[:-9], format = ' %m/%d ') + pd.Timedelta(days=1)
    else:
        return pd.to_datetime(date, format = ' %m/%d  %H:%M:%S')
def one_standrad(inpt):
    z = (stats.zscore(inpt))
    med=inpt.mean()
    return inpt.mask(z>1.5,med)

def three_standrad(inpt):
    z = (stats.zscore(inpt))
    med=inpt.mean()
    return inpt.mask(z>3,med)
    
def load_generate(name_list,area_list_org,year_list):
    area_list=area_list_org.copy()
    def closest(lst, K):
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()
        return idx
    def which_school(area):
        area_list=[73960,210900]
        name_list=['PrimarySchool','SecondarySchool']
        return name_list[closest(area_list, area)]
    def which_hotel(area):
        area_list=[43200,122132]
        name_list=['SmallHotel','LargeHotel']
        return name_list[closest(area_list, area)]
    def which_office(area):
        area_list=[5500,53600,498600]
        name_list=['SmallOffice','MediumOffice','LargeOffice']
        return name_list[closest(area_list, area)]
    def which_residential(area):
        area_list=[33700,84360]
        name_list=['MidriseApartment','HighriseApartment']
        return name_list[closest(area_list, area)]
    name_dict={'Retail':'RetailStandalone','Industrial':'Warehouse','School':which_school,'Hotel':which_hotel,\
               'Office':which_office,'Residential':which_residential}
    year_dict={'FullServiceRestaurant': [2004, 2007, 2010, 2013, 2004, 1980],
 'HighriseApartment': [2004, 2007, 2010],
 'Hospital': [2004, 2007, 2010, 2013, 2004, 1980],
 'LargeHotel': [2004, 2007, 2010, 2013, 2004, 1980],
 'LargeOffice': [2004, 2007, 2010, 2013, 2004, 1980],
 'MediumOffice': [2004, 2007, 2010, 2013, 2004, 1980],
 'MidriseApartment': [2004, 2007, 2010, 2013, 2004, 1980],
 'PrimarySchool': [2004, 2007, 2010, 2013, 2004, 1980],
 'QuickServiceRestaurant': [2004, 2007, 2010, 2013, 2004, 1980],
 'RetailStandalone': [2004, 2007, 2010, 2013],
 'RetailStripmall': [2004, 2007, 2010, 2013, 2004, 1980],
 'SecondarySchool': [2004, 2007, 2010, 2013, 2004, 1980],
 'SmallHotel': [2004, 1980],
 'SmallOffice': [2004, 2007, 2010, 2013, 2004, 1980],
 'Warehouse': [2004, 2007, 2010, 2013, 2004, 1980]}
    new_name_list=[]
    new_year_list=[]
    index=0
    for name in name_list:
        if name=='Other/Unknown' or name=='Parking':
            new_name_list.append(None)
            area_list[index]=None
        else:
            if type(name_dict[name])==str:
                new_name_list.append(name_dict[name])
            else:
                new_name_list.append(name_dict[name](area_list[index]))
        index+=1
    index=0
    for new_name in new_name_list:
        if new_name==None:
            new_year_list.append(None)
        else:
            new_year_list.append(year_dict[new_name][closest(year_dict[new_name],year_list[index])])
        index+=1
    list1=glob.glob("../base loads/*.csv")
    load_array=np.zeros((1,8760))
    load_array2=np.zeros((1,8760))
    new_name_list.remove(None)
    new_year_list.remove(None)
    area_list.remove(None)
    for file1 in list1:
        current_load=pd.read_csv(file1,sep="\t",index_col=0)['SH(W/m2)'].values
        current_load2=pd.read_csv(file1,sep="\t",index_col=0)['SC(W/m2)'].values
        remove_list1=[]
        remove_list2=[]
        remove_list3=[]
        for i in range(len(new_name_list)):
            if fnmatch.fnmatch(file1,'*'+new_name_list[i]+'*'+str(new_year_list[i])+'*'):
                load_array+=current_load*area_list[i]
                load_array2+=current_load2*area_list[i]
                remove_list1.append(new_name_list[i])
                remove_list2.append(new_year_list[i])
                remove_list3.append(area_list[i])
        for i in range(len(remove_list1)):
            new_name_list.remove(remove_list1[i])
            new_year_list.remove(remove_list2[i])
            area_list.remove(remove_list3[i])
        if new_name_list==[]:
            break
    time_array=pd.read_csv(file1,sep="\t",index_col=0)['time'].values
    return time_array,load_array,load_array2

def length(frame,time0,logic):
    #frame['Date/Time'] = frame['Date/Time'].apply(custom_to_datetime)
    #frame['Date/Time']=pd.to_datetime(frame['Date/Time'])
    #frame.set_index('Date/Time', inplace=True)
    #frame=frame.resample("D").mean()
    frame['hour']=frame.index.values//(time0*3600)
    frame=frame.groupby(by="hour", dropna=False).mean()
    if logic:
        temp_heat=frame['net'][frame['net']>0]
        frame['net'][frame['net']>0]=one_standrad(temp_heat)
        temp_heat=frame['net'][frame['net']<0]
        frame['net'][frame['net']<0]=three_standrad(temp_heat)
    #frame.plot()
    #print(frame.index.values)
    
    frame['month']=frame.index.values//(31*24//time0)
    PLFm = frame.groupby(by="month", dropna=False).mean()
    # print(PLFm.index.values)
    Ql_4 = frame
    PLFm['Heating'] = PLFm['net'] / Ql_4['net'].min()
    PLFm['Cooling'] = PLFm['net'] / Ql_4['net'].max()
    diff=frame['net'].values
    sum_pos = -diff[diff>0].sum()*time0
    sum_neg = -diff[diff<0].sum()*time0
    qa=(sum_pos+sum_neg)/8760
    max_PLFm = dict(
        Cooling=PLFm.loc[PLFm.Cooling==PLFm.Cooling.max(), 'Cooling'].values[0],
        Heating=PLFm.loc[PLFm.Heating==PLFm.Heating.max(), 'Heating'].values[0],
    )
    Lc=(qa*0.157+(-frame['net'].max() *(0.12+max_PLFm['Cooling']*0.14+1.04*0.099)))/(14.4-(25+30)/2)
    Lh=(qa*0.157+(-frame['net'].min() *(0.12+max_PLFm['Heating']*0.14+1.04*0.099)))/(14.4-(8+3)/2)
    #print(max(Lc,Lh))
    return max(Lc,Lh)
    #return 0

def length_generate(name_list,area_list,year_list):
    time_array,load_array,load_array2=load_generate(name_list,area_list,year_list)
    net=load_array[0]*6.5/5.5+load_array2[0]*2.5/3.5
    d = {'time': time_array, 'net': -net/1000}
    df = pd.DataFrame(data=d)
    df=df.set_index('time')
    return length(df,6,False)
