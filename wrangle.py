import os
import pandas as pd
from env import get_db_url
from sklearn.model_selection import train_test_split

def new_zillow_data():
    '''
    This reads the zillow data from the Codeup db into a df
    '''
    sql_query = """
                SELECT id, propertylandusetypeid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017
                WHERE propertylandusetypeid = '261';
                """
    
    # Read in DataFrame from Codeup.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    
    return df

def get_zillow_data():
    '''
    This reads in telco data from Codeup database, writes data to
    a csv if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # read in data from csv file if one exists
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache to .csv
        df.to_csv('zillow.csv')
        
    return df

def clean_zillow(df):
    '''
    This gets rid of columns not used in the project and creates dummy variables for use in the model
    '''
    # this drops unnecessary columns
    df = df.drop(columns =['payment_type_id','internet_service_type_id','contract_type_id', 'customer_id', 'gender', 'senior_citizen', 'partner', 'tenure', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'total_charges', 'internet_service_type', 'payment_type'])

    # this creates dummy variables
    dummy_df = pd.get_dummies(df[['dependents', 'contract_type', 'billing_type', 'churn']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df


def split_data(df, test_size=.2, validate_size=.25, col_to_stratify=None, random_state=None):
    '''
    This splits data into test,train and validate data
    '''
    # This takes in a default variable or a variable to determine target variable for stratification
    if col_to_stratify == None:
    # this splits the data
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, validate = train_test_split(train_validate,
                                       test_size=validate_size,
                                       random_state=random_state,)
    else:                                                        
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[col_to_stratify])
        train, validate = train_test_split(train_validate,
                                       test_size=validate_size,
                                       random_state=random_state,
                                       stratify=train_validate[col_to_stratify])
    return train, validate, test

def establish_baseline(y_train):
    '''
    This function establishes a baseline accuracy for comparison to a  model
    '''
    # this finds the mode of the train dataset
    baseline_prediction = y_train.mode()
    # this creates a prediction 
    y_train_pred = pd.Series((baseline_prediction[0]), range(len(y_train)))

    cm = confusion_matrix(y_train, y_train_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp+tn)/(tn+fp+fn+tp)
    return accuracy

def wrangle_zillow():
    '''
    Read zillow data into a pandas DataFrame from mySQL,
    drop property use id column, drop rows with null values,
    return cleaned zillow DataFrame.
    '''

    # Acquire data

    df = get_zillow_data()
    
    # drop property use id column
    
    df = df.drop(columns=['propertylandusetypeid'])

    # Drop all rows with NaN values.
    
    df = df.dropna()

    return df

def minmax_scale(train, validate, test)
    scaler_minmax = sklearn.preprocessing.MinMaxScaler()

    scaler_minmax.fit(x_train)

    x_train_scaled_minmax = scaler_minmax.transform(x_train)
    x_validate_scaled_minmax = scaler_minmax.transform(x_validate)
    x_test_scaled_minmax = scaler_minmax.transform(x_test)
    
    return x_train_scaled_minmax, x_validate_scaled_minmax, x_test_scaled_minmax