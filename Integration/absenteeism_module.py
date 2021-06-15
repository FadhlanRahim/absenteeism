
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
      
        def __init__(self, model_file):
            # read the 'model' files which were saved
            with open('model','rb') as model_file:
                self.pipe = pickle.load(model_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            # drop the 'ID' column
            df = df.drop(['ID'], axis = 1)
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['Absenteeism Time in Hours'] = 'NaN'

            # create a separate dataframe, containing dummy values for ALL avaiable reasons
            reason_columns = pd.get_dummies(df['Reason for Absence'])
            
            # split reason_columns into 4 types
            Reason1 = reason_columns.loc[:,1:14].max(axis=1)
            Reason2 = reason_columns.loc[:,15:17].max(axis=1)
            Reason3 = reason_columns.loc[:,18:21].max(axis=1)
            Reason4 = reason_columns.loc[:,22:].max(axis=1)
            
            # to avoid multicollinearity, drop the 'Reason for Absence' column from df
            df = df.drop(['Reason for Absence'], axis = 1)
            
            # concatenate df and the 4 types of reason for absence
            df = pd.concat([df, Reason1, Reason2, Reason3, Reason4], axis = 1)
            
            # assign names to the 4 reason type columns
            # note: there is a more universal version of this code, however the following will best suit our current purposes             
            column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                            'Pets', 'Absenteeism Time in Hours', 'Reason1', 'Reason2', 'Reason3', 'Reason4']
            df.columns = column_names

            # re-order the columns in df
            column_names_reordered = ['Reason1', 'Reason2', 'Reason3', 'Reason4', 'Date', 'Transportation Expense', 
                                        'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                        'Children', 'Pets', 'Absenteeism Time in Hours']
            df = df[column_names_reordered]
      
            # convert the 'Date' column into datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

            # create a list with month values retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)

            # insert the values in a new column in df, called 'Month Value'
            df['Month'] = list_months

            # create a new feature called 'Day of the Week'
            df['Day_of_the_week'] = df['Date'].apply(lambda x: x.weekday())


            # drop the 'Date' column from df
            df = df.drop(['Date'], axis = 1)

            # re-order the columns in df
            column_names_upd = ['Month', 'Day_of_the_week', 'Transportation Expense', 'Distance to Work', 'Age', 
                                'Daily Work Load Average', 'Body Mass Index', 'Children',
                                'Pets', 'Absenteeism Time in Hours', 'Reason1', 'Reason2', 'Reason3', 'Reason4', 'Education']
            df = df[column_names_upd]


            # map 'Education' variables; the result is a dummy
            df['Education'] = np.where(df['Education'] == 1, 0, 1)

            # replace the NaN values
            df = df.fillna(value=0)

            # drop the original absenteeism time
            df = df.drop(['Absenteeism Time in Hours'],axis=1)
            
            # drop the variables we decide we don't need
            df = df.drop(['Month','Daily Work Load Average','Distance to Work', 'Reason4'],axis=1)
            
            col_name_upd = ['Reason1', 'Reason2', 'Reason3', 'Day_of_the_week', 'Transportation Expense', 
                            'Age', 'Body Mass Index', 'Education', 'Children', 'Pets']
            
            df = df[col_name_upd]
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = df.copy()

    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.pipe.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.pipe.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.pipe.predict_proba(self.data)[:,1]
                self.preprocessed_data['Prediction'] = self.pipe.predict(self.data)
                return self.preprocessed_data

