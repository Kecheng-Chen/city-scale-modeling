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
        area_list=[75000.0,166666.66666666666]
        name_list=['PS','SS']
        return name_list[closest(area_list, area)]
    def which_office(area):
        area_list=[350000.0,87500.0,10473.684210526315]
        name_list=['LO','MO','SO']
        return name_list[closest(area_list, area)]
    name_dict={'Retail':'Rstand','Industrial':'WH','School':which_school,'Hotel':'LH',\
               'Office':which_office,'Residential':'Res'}
    new_name_list=[]
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
    
    list1=glob.glob("../result loads/*.csv")
    load_array=np.zeros((1,35040))
    load_array2=np.zeros((1,35040))
    while None in new_name_list:
        new_name_list.remove(None)
        area_list.remove(None)
    for file1 in list1:
        current_load=pd.read_csv(file1,sep="\t",index_col=0)['cooling'].values
        current_load2=pd.read_csv(file1,sep="\t",index_col=0)['heating'].values
        remove_list=[]
        for i in range(len(new_name_list)):
            if file1.split('/')[2]==new_name_list[i]+'.csv':
                load_array+=current_load*area_list[i]*0.092903
                load_array2+=current_load2*area_list[i]*0.092903
                remove_list.append(i)
        for i in range(len(remove_list)):
            new_name_list.pop(remove_list[len(remove_list)-i-1])
            area_list.pop(remove_list[len(remove_list)-i-1])
        if new_name_list==[]:
            break
    if new_name_list==[]:
        print("Correct")
    else:
        print("Error")
    time_array=pd.read_csv(file1,sep="\t",index_col=0)['time'].values
    return time_array,load_array,load_array2

def length(frame,time0,logic):
    #frame['Date/Time'] = frame['Date/Time'].apply(custom_to_datetime)
    #frame['Date/Time']=pd.to_datetime(frame['Date/Time'])
    #frame.set_index('Date/Time', inplace=True)
    #frame=frame.resample("D").mean()
    if max(frame.net.values)==0 and min(frame.net.values)==0:
        return 0
    frame['hour']=frame.index.values//(time0*3600)
    frame=frame.groupby(by="hour", dropna=False).mean()
    #frame.plot()
    #print(frame.index.values)
    
    frame['month']=frame.index.values//(31*24//time0)
    PLFm = frame.groupby(by="month", dropna=False).mean()
    # print(PLFm.index.values)
    Ql_4 = frame
    PLFm['Heating'] = PLFm['net'] / Ql_4['net'].min()
    PLFm['Cooling'] = PLFm['net'] / Ql_4['net'].max()
    if Ql_4['net'].min()>0:
        print('Error: minimum load larger than 0')
    diff=frame['net'].values
    sum_pos = -diff[diff>0].sum()*time0
    sum_neg = -diff[diff<0].sum()*time0
    qa=(sum_pos+sum_neg)/8760
    max_PLFm = dict(
        Cooling=PLFm.loc[PLFm.Cooling==PLFm.Cooling.max(), 'Cooling'].values[0],
        Heating=PLFm.loc[PLFm.Heating==PLFm.Heating.max(), 'Heating'].values[0],
    )
    Lc=(qa*0.157+(-frame['net'].max() *(0.12+max_PLFm['Cooling']*0.14+1.04*0.099)))/(18.4-(25+30)/2)
    Lh=(qa*0.157+(-frame['net'].min() *(0.12+max_PLFm['Heating']*0.14+1.04*0.099)))/(18.4-(8+3)/2)
    j=0
    while Lh>Lc and j<20:
        PLFm['Heating'] = PLFm['Heating'] *(1-(Lh-Lc)/(2*Lh))
        max_PLFm['Heating'] = max_PLFm['Heating'] *(1-(Lh-Lc)/(2*Lh))
        diff2 = PLFm['Heating'] * Ql_4['net'].min()
        sum_neg = -diff2[diff2<0].sum()*31*24
        qa=(sum_pos+sum_neg)/8760
        Lc=(qa*0.157+(-frame['net'].max() *(0.12+max_PLFm['Cooling']*0.14+1.04*0.099)))/(18.4-(25+30)/2)
        Lh=(qa*0.157+(-frame['net'].min() *(0.12+max_PLFm['Heating']*0.14+1.04*0.099)))/(18.4-(8+3)/2)
        j+=1
    # print(Lc,Lh)
    return max(Lc,Lh)
    #return 0

def length_generate(name_list,area_list,year_list):
    time_array,load_array,load_array2=load_generate(name_list,area_list,year_list)
    net=load_array[0]*6.5/5.5+load_array2[0]*2.5/3.5
    d = {'time': time_array, 'net': -net}
    df = pd.DataFrame(data=d)
    df=df.set_index('time')
    return length(df,6,False)
