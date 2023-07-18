import pandas as pd
import numpy as np


follow_up_options = 3
events = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
seq_lengths = [10,11,12,13,14,15,16,17,18]
num_seqs =1000
patid = 'enrolid'
event_column = 'event_real'
length_range = (1,30)
cost_range = (10,1000)
start_date = '01/01/2021'
num_start_points = 2



class SeqGenerator():
    
    '''This class is used to generate event sequences with specified parameters
       * event_options {list} = list of possible events
       * num_seqs {int} = the number of sequences to generate
       * seq_lengths {list} = list of lengths that sequences could be
       * num_start_points {default = 2} = the number of possible starting events
       * follow_up_options {default = 2} = for each event type, the number of possible follow up options for that event
       * length_range {default = tuple (1,30)} = range in number of days that each event can happen after the previous event
       * cost_range {default = tuple (10,1000) = range in number of dollars that each event type can cost}
       * start_date {default = '01/01/2021'} = date in which the first event occurs 
       * event_column {default = 'event_real'} = name of the column that the event column will be called
       * patid {default = 'enrolid'} = name of the column for the unique id of the person
    '''
    def __init__(self,event_options:list, num_seqs:int, seq_lengths:list, num_start_points = 2, follow_up_options = 2, 
                 length_range = (1,30), cost_range = (10,1000), start_date = '01/01/2021', event_column = 'event_real', patid = 'enrolid'):
        
        self.event_options = event_options
        self.num_seqs = num_seqs
        self.seq_lengths = seq_lengths
        self.num_start_points = num_start_points
        self.follow_up_options = follow_up_options
        self.length_range = length_range
        self.cost_range = cost_range
        self.start_date = start_date
        self.event_column = event_column
        self.patid = patid
        self.produced_seqs = []
        self.produced_df = pd.DataFrame()
    
    def __repr__(self) -> pd.DataFrame:
        return self.produced_df
        
    def get_seqs(self) -> list:
        '''return the list of produced seqs'''
        return self.produced_seqs
      
    def get_df(self) -> pd.DataFrame:
        '''return the produced seqs as a dataframe'''
        if self.produced_df.empty:
            self.generate_seqs()
            self.seqs_to_dataframe()
        return self.produced_df
      
    def generate_seqs(self):
        '''returns a list of created sequences with the given criteria'''
        self.produced_seqs = []
        
        seqs_list = []
        start_events = []
        
        sub_events = self._follow_up_events()
        for seqs in range(self.num_seqs):
            seq_building = []
            for i in range(self.seq_lengths[np.random.randint(len(self.seq_lengths))]):
                if len(seq_building)>0 and isinstance(sub_events,dict):
                    seq_building.append(sub_events[seq_building[-1]][0][np.random.randint(len(sub_events[seq_building[-1]][0]))])
                else:
                    if self.num_start_points is not None and len(start_events) == 0:
                        start_events = np.random.choice(self.event_options,self.num_start_points,replace = False)
                        seq_building.append(start_events[np.random.randint(len(start_events))])
                    elif self.num_start_points is not None and len(start_events) > 0:
                        seq_building.append(start_events[np.random.randint(len(start_events))])
                    else:
                        seq_building.append(self.event_options[np.random.randint(len(self.event_options))])

            seqs_list.append(seq_building)
        self.produced_seqs = seqs_list
        return self.get_seqs()
        
    def convert(self) -> np.ndarray:
        '''builds even length event sequences where 0 is the padded number and all other numbers represent event types'''
        converter = self._event_to_number_converter()
        lengths = max([len(seq) for seq in self.produced_seqs])
        all_arr = []
        for i in self.produced_seqs:
            arr = []
            for j in i:
                arr.append(converter[j])
            if len(arr) < lengths:
                 arr += [0 for i in range(lengths - len(arr))]
            all_arr.append(np.array(arr))
        return np.array(all_arr)
      
    def _event_to_number_converter(self) -> dict:
        '''assign a number to each events and return a dictionary'''
        return {j:i+1 for i,j in enumerate(self.event_options)}
      
      
    def _follow_up_events(self) -> dict:
        d = {}
        for e in self.event_options:
            f = np.random.randint(2,self.follow_up_options+1)
            if self.follow_up_options is not None:
                d[e] = [np.random.choice(self.event_options,f,replace = False)]
            else:
                pass
        return d
      
    def generate_random_lengths(self,df:pd.DataFrame,eventtrans_column:str) -> pd.DataFrame:
        return {i: pd.to_timedelta(np.random.randint(self.length_range[0],self.length_range[1]),unit='days') for i in df[eventtrans_column].unique()}

    def generate_random_costs(self,df:pd.DataFrame) -> pd.DataFrame:
        return {i: np.random.randint(self.cost_range[0],self.cost_range[1]) for i in df[self.event_column].unique()}
    
    def event_lag_and_trans(self,df:pd.DataFrame) -> pd.DataFrame:
        df['Eventlag']= df.groupby(self.patid)[self.event_column].shift(-1).fillna('')
        df['Eventtrans'] = df[self.event_column] + ',' + df['Eventlag']
        return df
      
    def costs_and_lengths(self,df:pd.DataFrame) -> pd.DataFrame:
        ''' Add Costs and event lengths to dataframe'''
        df = self.event_lag_and_trans(df)
        lengths = pd.DataFrame.from_dict(self.generate_random_lengths(df,'Eventtrans'),orient = 'index').reset_index().rename({0:'activity_length','index':'Eventtrans'},axis = 1)
        merged = pd.merge(df,lengths,how = 'left',on = 'Eventtrans')
        merged['cum_length'] = merged.groupby(self.patid).activity_length.apply(lambda x: x.fillna(pd.Timedelta(seconds=0)).cumsum())
        merged['svcdate'] = pd.to_datetime(self.start_date) + merged['cum_length']
        costs = pd.DataFrame.from_dict(self.generate_random_costs(merged),orient = 'index').reset_index().rename({0:'cost','index':self.event_column},axis = 1)
        final = pd.merge(merged,costs,how = 'left',on = event_column)
        return final
      
    def seqs_to_dataframe(self,provide_sequences = None, add_context = True) -> pd.DataFrame:
        ''' Turn a list of sequences into a dataframe'''
        if isinstance(provide_sequences,list):
            sequences = {self.patid:[i for i in range(self.num_seqs)], self.event_column:provide_sequences}
        else:
            sequences = {self.patid:[i for i in range(self.num_seqs)], self.event_column:self.get_seqs()}
        df = pd.DataFrame.from_dict(sequences)
        df = df.explode(self.event_column).reset_index().reset_index().drop('index',axis = 1).rename({'level_0':'unique_row'},axis = 1)
        
        if add_context:
            df = self.costs_and_lengths(df)
            
        self.produced_df = df
        
        return df
        




def probability_of_event(df:pd.DataFrame,event_column:str,next_event_column:str,unique_row = 'unique_row', position_col = None) -> pd.DataFrame:
    if position_col:
        group_cols = [event_column,position_col]
    else:
        group_cols = [event_column]
        
    a = df.groupby(group_cols + [next_event_column])[unique_row].count().reset_index()
    b = a.groupby(group_cols)[unique_row].sum().reset_index().rename({unique_row:'sumevent'},axis = 1)
    c = pd.merge(a,b,on = group_cols,how = 'left')
    c['prob'] = c[unique_row]/c['sumevent']
    return c
  
  
sg = SeqGenerator(events,num_seqs,seq_lengths,num_start_points,follow_up_options,length_range,cost_range,start_date,event_column,patid)    
print(sg.get_df())

sg.get_df().to_csv('seqs_example.csv')


