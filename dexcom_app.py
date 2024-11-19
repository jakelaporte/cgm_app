# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:59:55 2024

@author: grover.laporte
"""
import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime


class Dexcom(object):
    def __init__(self,filename):
        """
        Dexcom data object for visualizing, imputing, and using dexcom 
            continuous glucose monitoring data.
            
        data - dictionary of values with keys as days in the file; and values
            as a list of lists for each record which has a list of
            2 components: 1) # of minutes after midnight according to time
            stamp and 2) glucose reading
        df - dataframe with the same information as data just formatted into
            a table 0:1440:5 as the columns which represent the times, the
            index as dates with the format 'mm/dd/yyy' and the values in the
            table as glucose values.
            
        series - dataframe with one column of all the glucose values with 
            a datetime as the index and 'glucose' values as the one column
            
        days - the days in the format 'mm/dd/yyyy' from the time stamp
        
        times - the 288 times which represent the number of minutes 
            after midnight [0,5,10,...1440]
            
        shape - the number of days (rows) and times (columns) in the data
        
        rows - the days in the data that have missing data.
            Example: [0,3,8] day 0,3,8 are missing at least one instance
                of data
        cols - a list of columns from the data that is missing data. These
            will align with rows to give the columns in each row that are 
            missing.
            Example: [array([0,1,2,...238]), array(230,231), 
                      array(159,160,...287)] day 0 is missing the first part
                    of the day, day 3 is missing only the 230 and 231 columns.
        
        """
        self.filename = filename
        self.data = self.dexcom_data(filename)
        self.df = pd.DataFrame([], columns = list(np.arange(0,1440,5)))
        self.dexcom_dataframe()
        
        self.imputed_data = {}
        self.days = list(self.df.index)
        self.times = np.array(list(np.arange(0,1440,5)))
        self.shape = self.df.shape
        self.dataframe_series_build()
        X = self.df.copy()
        rows = X.isnull().any(axis=1)
        self.impute_bool = True if rows.sum()>0 else False
        rows = np.where(rows==True)[0]
        self.rows = rows
        cols = []
        for row in rows:
            col_ = np.where(X.iloc[row].isnull()==True)[0]
            cols.append(col_)
        self.cols = cols
        
        
        
    def dexcom_data(self,filename):
        working_directory = os.getcwd()
        st.write(working_directory)
        infile = open(working_directory+'/'+filename)
        self.header = infile.readline()
        data = {}
        for line in infile:
            row = line.split(',')
            print(row)
            if row[2]=='EGV':
                temp = row[1].split('T')
                date = temp[0]
                date = date.split('-')
                time = temp[1]
                date = f'{date[1]}/{date[2]}/{date[0]}'
                row.append(date)
                temp = [int(t) for t in time.split(':')]
                temp[1]=temp[1]//5*5
                time = temp[0]*60+temp[1]
                if date not in data.keys():
                    data[date]=[]
                data[date].append([time,int(row[7])])
        infile.close()
        return data
    
    def dexcom_dataframe(self):
        for day in self.data.keys():
            temp = np.array(self.data[day])
            temp = temp[temp[:,0].argsort()]
            glucose = pd.DataFrame(temp[:,1],index = temp[:,0],columns = [day]).T
            self.df = pd.concat([self.df,glucose],axis=0)
            
    def dataframe_series_build(self):
        index_=[]
        values = []
        for day in self.df.index:
            for mm in self.df.columns:
                idx = day + f' {mm//60:0>2}:{mm%60:0>2}'
                index_.append(datetime.strptime(idx,'%m/%d/%Y %H:%M'))
                values.append(self.df.loc[day,mm])
        temp=pd.DataFrame(values,index=index_,columns=['glucose'])
        self.series = temp.loc[temp.first_valid_index():temp.last_valid_index()]
        self.time_series = self.series.index
        indices = np.where(self.series.isnull())[0]
        self.series['glucose'] = self.linear_interpolate(self.series['glucose'].values,indices)
            
    def convert_to_time(self,minutes):
        time = []
        for m in minutes:
            hh = m//60
            mm = m%60
            time.append(f'{hh:0>2}:{mm:0>2}')
        return time
    
    def linear_interpolate(self, glucose_values, indices):
        """
        linear_interpolate - interpolating values between data
        
        Input:
            glucose_values - all of the glucose values for a day (288)
            indices - the indicies of the glucose values that need imputing
        Output:
            interpolated_values - all of the glucose values linear interpolated
            for those that are missing.
        """
        x = np.array(np.arange(len(glucose_values)),dtype=int)
        #x = x.astype(int)
        mask = np.ones(len(glucose_values), dtype=bool)
        mask[indices] = False
        interpolated_values = np.interp(x, 
                                        x[mask], 
                                        glucose_values[mask].astype(int))
        return interpolated_values
    
    def itc(self,day,mm):
        """
        itc - (index_time_conversion) - converts a day / time (min past midnight)  
            to a datetime needed to index the series
        """
        return datetime.strptime(day+f' {mm//60:0>2}:{mm%60:0>2}','%m/%d/%Y %H:%M')
    
    def tic(self,idx):
        """
        tic - (time_index_conversion) - converts a datetime to day and min past
            midnight in a day to access self.df
        """
        day = idx.strftime('%m/%d/%Y')
        time = idx.strftime('%H:%M')
        return day,time,int(time[:2])*60+int(time[3:])
        
    def decide_impute_data(self):
        if st.session_state['impute_decide']:
            chkbox = []
            for idx,day in enumerate(self.rows):
                times = self.convert_to_time(self.times[self.cols[idx]])
                chkbox.append(st.checkbox('Day - '+self.days[day]))
                st.write(f'Number of missing values: {len(times)}')
                st.write(f'First missing value time: {times[0]}')
                st.write(f'Last missing value time: {times[-1]}')
                st.write('===========================================')
    
            decide_btn = st.button("Impute")
            if decide_btn:
                self.impute_days = chkbox
                st.session_state.impute_decide = False
                self.impute_data()
        else:
            st.write("Reload page to start over.")
                
    def impute_data(self):
        for idx,day in enumerate(self.rows):
            if self.impute_days[idx]:
                day_str = self.days[day]
                #times_to_impute = self.times[self.cols[idx]]
                st.write(day_str + " is imputed.")
                interpolated_values = self.linear_interpolate(
                                            self.df.loc[day_str].values,
                                            self.cols[idx].astype(int))
                self.imputed_data[day_str]={}
                self.imputed_data[day_str]['times']=self.times[self.cols[idx]]
                self.imputed_data[day_str]['values']=interpolated_values[self.cols[idx]]
                self.imputed_data[day_str]['original'] = self.df.loc[day_str].values
                self.df.loc[day_str] = interpolated_values
                
            
    def view_data(self):
        days = self.days
        day = st.selectbox("Select the day:",
                           index = 0,
                           options = days,
                           key = 'view_data_day')
        vd = self.df.T[[day]]
        vd['time']=self.convert_to_time(self.times)

        fig = px.line(vd,x='time', y=day, markers=True)
        if day in self.imputed_data.keys():
            imp_times = self.convert_to_time(self.imputed_data[day]['times'])
            imp_vals = self.imputed_data[day]['values']
            fig.add_scatter(x=imp_times,y = imp_vals, 
                            mode = 'lines+markers',
                            marker_color = ['red']*len(imp_vals),
                            showlegend=False)
            fig['data'][1]['line']['color']="#DA5E26"
            fig['data'][1]['marker']['color']= '#953109' #0F8F8C'
        st.plotly_chart(fig)
        
    def time_in_range(self,data,lower,upper):
        """
        time in range - assumes 5 minute intervals for all data and simply
            counts the number between lower and upper / total number of 
            observations.
        Input:  data - needs to be a series object with either datetime or 
            minutes after midnight as the index.
                lower - the lower bound to check
                upper - the upper bound to check
        Output: a single float corresponding to the fraction of time that 
            the glucose values are within the given upper and lower bounds.
            
        """
        i1,i2 = data.first_valid_index(),data.last_valid_index()
        data = data.loc[i1:i2].values
        denom = len(data)
        numer = len(data[(data>=lower) & (data<=upper)])
        
        return numer/denom
        
        
    def conga(self,data,h):
        data_int = np.arange(0,len(data),h*12)
        data_idx = data.index[data_int]
        diff = data.loc[data_idx[1:]].values-data.loc[data_idx[:-1]].values
        #st.write(len(diff),diff.mean(),diff.std())
        return(diff.std())
    
    def mean_absolute_glucose(self,data):
        data = data[~data.isnull()]
        diff = np.abs(data.iloc[1:].values-data.iloc[:-1].values)
        return diff.sum()
        
        
    def daily_summary(self):
        lower,upper = 70,180
        total_time = self.series.index[-1]-self.series.index[0]
        st.markdown("#### Overall")
        res = "|Characteristic|Value|  \n"
        res+= "|--------------|-----|  \n"
        res+= f"|Total time |{total_time}"
        res+=f'|Number of observations|{len(self.series)}|  \n'
        res+=f'|Mean| {self.series.mean().values[0]:0.2f}|  \n'
        res+=f'|Standard Deviation| {self.series.std().values[0]:0.2f}|  \n'
        res+=f'|Start date/time| {self.series.index[0]}|  \n'
        res+=f'|End date/time| {self.series.index[-1]}|  \n'
        res+=f'|Time in Range (70-180)| {self.time_in_range(self.series,70,180):0.4f}|  \n'
        c1 = self.conga(self.series,1)
        c2 = self.conga(self.series,2)
        c4 = self.conga(self.series,4)
        res+=f'|Conga(1)| {c1:0.4f}|  \n'
        res+=f'|Conga(2)| {c2:0.4f}|  \n'
        res+=f'|Conga(4)| {c4:0.4f}|'
        st.markdown(res)
        st.divider()
        self.conga(self.series,1)
        st.markdown("#### Daily")
        res = ""
        res+="|Day | Missing Data | Mean | Std Dev |Time in Range (70-180)|"
        res+="Mean Abs Glucose|  \n"
        res+="|----|--------------|------|---------|-------------|-----|  \n"
        for i,day in enumerate(self.days):
            if i in self.rows:
                idx = list(self.rows).index(i)
                missing_values = len(self.cols[idx])
            else:
                missing_values = 0
            data = self.df.loc[day]
            tir = self.time_in_range(data,lower,upper)
            mag = self.mean_absolute_glucose(data)
            res+=f'|{day}|{missing_values}|{data.mean():0.2f}|{data.std():0.2f}|'
            res+=f'{tir:0.4f}|{mag}|  \n'
            #self.time_in_range(data,70,180)
        st.markdown(res)
        st.divider()
        st.markdown('#### Glucose Distribution by Day')
        days = self.days
        day = st.multiselect("Select the day:",
                           options = days,
                           max_selections=3,
                           key = 'dist_day')
        data = self.df.loc[day].T
        fig = px.histogram(data,x=day,marginal="violin")
        st.plotly_chart(fig)


st.title("Dexcom App")
if 'dex_data' not in st.session_state:
    st.session_state['dex_data'] = None
if 'impute_decide' not in st.session_state:
    st.session_state['impute_decide']=True


## Side bar radio button and options #######################
options = ["Files",
           "Data Tools",
           "Export Data"]
select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')

if select == options[0]:
    tab1, tab2 = st.tabs(["Import Data",
                               "Entire DataFrame"])
    
    with tab1:
        dex_data = st.session_state['dex_data']
        st.subheader("Import csv file")
        if dex_data is None:
            uploaded_file = st.file_uploader("Select .csv survey file.",type='csv')
            if uploaded_file is not None:
                dex_data = Dexcom(uploaded_file.name)
                st.session_state['dex_data'] = dex_data
                st.write(dex_data.df)
        else:
            st.write("Data is loaded")
            reload_button = st.button("Reload new data.")
            if reload_button:
                st.session_state['dex_data']=None
            
    with tab2:
        dex_data = st.session_state['dex_data']
        if dex_data is not None:
            st.write(dex_data.df)

if select == options[1]:
    tab1,tab2,tab3 = st.tabs(["Daily Plot",
                         "Impute Data",
                         "Data Summary"])
    
    with tab1:
        dex_data = st.session_state['dex_data']
        if dex_data is not None:
            st.subheader('Plot of Data by Day')
            dex_data.view_data()
    with tab2:
        dex_data = st.session_state['dex_data']
        if (dex_data is not None):
            st.subheader("Impute Data")
            dex_data.decide_impute_data()
                
    with tab3:
        dex_data = st.session_state['dex_data']
        if dex_data is not None:
            st.subheader("Summary")
            dex_data.daily_summary()
            
