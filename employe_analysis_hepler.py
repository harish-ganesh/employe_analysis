from google.colab import files
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from plotly.subplots import make_subplots
from datetime import date,time
import numpy as np
from math import floor
from datetime import datetime
from pytz import timezone

class EmployeAnalysis:
  def __init__(self):
    self.df = pd.DataFrame()
    self.central = timezone('US/Central')

  def preprocess(self,uploaded):
    for fn in uploaded.keys():
      temp_df = pd.read_csv(fn)
    self.df[['Date1','Date2','Item','Destination','Type','Rate','Duration','Amount','Currency']] = pd.DataFrame(temp_df['Date;Date;Item;Destination;Type;Rate;Duration;Amount;Currency'].str.split(';').tolist())
    # self.df['Date'] = self.df['Date2'].apply(lambda x: x[1:11])
    # self.df['Time'] = self.df['Date2'].apply(lambda x: x[12:20])
    self.df['Date'] = self.df['Date2'].apply(lambda x: self.convert_date_timezone(x))
    self.df['Time'] = self.df['Date2'].apply(lambda x: self.convert_time_timezone(x))
    self.df['Hourly'] = self.df['Date2'].apply(lambda x: str(int(x[12:14])%24)+':00-'+str((int(x[12:14])+1)%24)+':00')
    self.df['Quarterly'] = self.df['Date2'].apply(lambda x: x[12:14]+'Q'+str(floor(int(x[15:17])/15)+1))
    #self.df['Date'] = self.df['Date'].apply(lambda x : date(int(x[:4]), int(x[5:7]), int(x[8:])))
    #self.df = self.df.sort_values(['Date','Time'])
    self.df['Duration_in_sec'] = self.df['Duration'].apply(lambda x: int(x.split(':')[0])*60*60+int(x.split(':')[1])*60+int(x.split(':')[2]) if not x=='' else 0)
  
  def convert_date_timezone(self,t):
    t =t[1:11]+';'+t[12:20]+' GMT'
    published_time = datetime.strptime(t, '%Y-%m-%d;%H:%M:%S %Z')
    published_cst = published_time.astimezone(self.central)
    return date(int(published_cst.strftime('%Y')), int(published_cst.strftime('%m')),int(published_cst.strftime('%d')))

  def convert_time_timezone(self,t):
    t =t[1:11]+';'+t[12:20]+' GMT'
    published_time = datetime.strptime(t, '%Y-%m-%d;%H:%M:%S %Z')
    published_cst = published_time.astimezone(self.central)
    time = published_cst.strftime('%H:%M:%S')
    return time

  def convert(self,seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

  def plot_multiple_days(self,temp,st_date,end_date):
    call_count = pd.DataFrame()
    call_count[['Date','#Calls']] = temp['Date'].value_counts().reset_index()
    call_count['TotalCallDuration (in sec)']=''

    for i in range(0,call_count['Date'].shape[0]):
      call_count.iloc[i,2] = temp[temp['Date']==call_count.iloc[i,0]]['Duration_in_sec'].sum()

    call_count['TotalCallDuration (in minutes)'] = call_count['TotalCallDuration (in sec)'].apply(lambda x : x/60)
    call_count['TotalCallDuration'] = call_count['TotalCallDuration (in sec)'].apply(lambda x : self.convert(x))
    fig = px.bar(call_count, x='Date', y='#Calls',text='TotalCallDuration',color='TotalCallDuration (in minutes)')
    call_count = call_count.sort_values(by=['Date'])
    cols = ['Date','#Calls','TotalCallDuration']
    self.to_excel(call_count,temp,str(st_date)+'_'+str(end_date)+'_analysis.xls',cols)
    fig.update_layout(autosize=False,width=1200,height=500)
    fig.show()

  def get_stats(self,temp):
    sms = temp[temp['Type']=='"SMS"'].shape[0]
    total_calls = temp.shape[0] - sms
    unconnected_calls = temp[temp['Duration_in_sec'] == 0].shape[0] - sms
    temp = temp[temp['Duration_in_sec'] >= 5]
    invalid_calls = total_calls - temp.shape[0] - unconnected_calls  
    return ([unconnected_calls, invalid_calls, temp.shape[0], sms],
            total_calls,self.convert(temp['Duration_in_sec'].sum()))
    
  def plot_pie(self,temp):
    values,total_calls,duration = self.get_stats(temp)
    labels = ['Unconnected calls','Calls <5mins','Valid calls','SMS']  
    pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    colors = ['mediumturquoise', 'darkorange','gold', 'lightgreen']
    pie_fig.update_traces(hoverinfo='label+value', textinfo='value', textfont_size=15,
                          marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    print("Calls: ",total_calls)
    print("Valid Calls Duration: ", duration)
    pie_fig.update_layout(autosize=False,width=1000,height=500)
    pie_fig.show()

  def to_excel(self,call_count,temp,fname,cols):
    values,total_calls,duration = self.get_stats(temp)
    summ = pd.DataFrame([["Total:",values[2],duration]],
                                    columns=cols)
    call_count = call_count.append(summ)
    call_count[cols].to_excel(fname,index=False)

  def plot_one_day(self,temp,ti,Date):
    call_count = pd.DataFrame()
    temp = temp[temp['Duration_in_sec'] >= 5]
    call_count[['TimePeriod','#Calls']] = temp[ti].value_counts().reset_index()
    call_count['TotalCallDuration']=''
    performance = list()
    color = list()
    for i in range(0,call_count['TimePeriod'].shape[0]):
      call_count.iloc[i,2] = temp[temp[ti]==call_count.iloc[i,0]]['Duration_in_sec'].sum()
      if call_count.iloc[i,2] < np.int64(600) and call_count.iloc[i,1]<np.int64(10):
        performance.append(' ಠ╭╮ಠ ')
        color.append("red")
      else:
        performance.append('Okay!')
        color.append("blue")
    
    call_count['Performance'] = performance
    timePeriod = call_count['TimePeriod'].tolist()
    timePeriod.sort()
    call_count['TotalCallDuration'] = call_count['TotalCallDuration'].apply(lambda x : self.convert(x))
    if ti is 'Hourly':
      fig = px.bar(call_count, x='TimePeriod', y='#Calls',text='TotalCallDuration', hover_data=['TotalCallDuration','#Calls'],
                  color='Performance',color_discrete_map={'Okay!':'blue',' ಠ╭╮ಠ ':'red'},
                  category_orders = {'TimePeriod':timePeriod})
    else:
      fig = px.bar(call_count, x='TimePeriod', y='#Calls',text='TotalCallDuration',hover_data=['TotalCallDuration','#Calls'],
                  category_orders = {'TimePeriod':timePeriod}) 
    fig.update_layout(autosize=False,width=1200,height=400)
    call_count = call_count.sort_values(by=['TimePeriod'])
    cols = ['TimePeriod','#Calls','TotalCallDuration']#,'Performance'
    self.to_excel(call_count,temp,str(Date)+'_'+ti+'_analysis.xls',cols)
    fig.show()

  def daily_analysis(self,Date=None,ti='Hourly'): 
    if Date:
      temp = self.df[self.df['Date']==Date]
      self.plot_one_day(temp,ti,Date)

  def monthly_analysis(self,start_date=None,end_date=None): 
    if start_date and end_date:
      temp = self.df[self.df['Date']>=start_date]
      temp = temp[temp['Date']<=end_date]
      self.plot_multiple_days(temp,start_date,end_date)

  def daily_analysis_pie(self,Date=None): 
    if Date:
      temp = self.df[self.df['Date']==Date]
      self.plot_pie(temp)

  def monthly_analysis_pie(self,start_date=None,end_date=None): 
    if start_date and end_date:
      temp = self.df[self.df['Date']>=start_date]
      temp = temp[temp['Date']<=end_date]
      self.plot_pie(temp)


def bar1(x):
  interact(x.daily_analysis,Date=widgets.DatePicker(),
           ti=widgets.SelectionSlider(options=['Hourly', 'Quarterly'],
    value='Hourly',
    description='X-axis Interval: ',
    disabled=False
  ))

def bar2(x):
  interact(x.monthly_analysis,
        start_date=widgets.DatePicker(),
        end_date=widgets.DatePicker())

def dougnut1(x):
  interact(x.daily_analysis_pie,Date=widgets.DatePicker())

def dougnut2(x):
  interact(x.monthly_analysis_pie,start_date=widgets.DatePicker(),
        end_date=widgets.DatePicker())
  
def upload(x):
  file_ = files.upload()
  x.preprocess(file_)
