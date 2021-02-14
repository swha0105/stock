#%%
import numpy as np 
import pandas as pd
import os
import datetime
import time
import matplotlib.pyplot as plt
#%%

data_path = '/home/ha/stock/raw_data/'
for name_list in os.listdir(data_path):
    tmp_data = pd.read_csv(data_path + name_list)

    if len(tmp_data) < 100: #데이터 작은거
        continue

    time_list = []

    for tmp_time in np.array(tmp_data['date']):
        time_list.append( datetime.datetime.strptime(tmp_time,"%Y-%m-%d"))
   
    
    ref_time = datetime.datetime.strptime('2018-01-01', "%Y-%m-%d")
    
    if np.abs( (np.min(time_list) - ref_time).days)  > 800: #800일
        continue
    else:
        index = np.argmin(np.array(time_list) - ref_time)
     
    
    date_data = np.array(tmp_data['date'])[start_index:]
    open_data = np.array(tmp_data['open'])[start_index:]
    close_data = np.array(tmp_data['close'])[start_index:]
    volume_data = np.array(tmp_data['volume'])[start_index:]

    if len(volume_data[volume_data == 0]) >=7 : #거래량 없는거 스킵.
        continue

    df = pd.DataFrame({'date':date_data,'open':open_data,'close':close_data,'volume':volume_data})
    
    df.to_csv('/home/ha/stock/sorted_data/' + str(name_list))

    
    


#%%


plt.plot(date_data,open_data)
plt.show()
#%%
norm_data = (open_data - np.min(open_data)) /(np.max(open_data) - np.min(open_data))
volume_norm = (volume_data - np.min(volume_data)) /(np.max(volume_data) - np.min(volume_data))
N = len(date_data) 
welch_window = 1 - ( ( np.arange(0,N) - N/2)/ (N/2))**2
plt.figure(figsize=[14,7])
#plt.plot(date_data,welch_window*norm_data)
plt.plot(date_data,volume_norm*norm_data,'r')
plt.plot(date_data,norm_data,'b')
plt.show()
# %%
