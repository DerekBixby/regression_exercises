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

def minmax_scale(train, validate, test):
    scaler_minmax = sklearn.preprocessing.MinMaxScaler()

    scaler_minmax.fit(x_train)

    x_train_scaled_minmax = scaler_minmax.transform(x_train)
    x_validate_scaled_minmax = scaler_minmax.transform(x_validate)
    x_test_scaled_minmax = scaler_minmax.transform(x_test)
    
    return x_train_scaled_minmax, x_validate_scaled_minmax, x_test_scaled_minmax

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    #create subplot structure
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(12,12))

    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()



    # call function with minmax
    visualize_scaler(scaler=MinMaxScaler(), 
                 df=train, 
                 columns_to_scale=to_scale, 
                 bins=50)

# min_max scale function provided 

def scale_data(train, 
               validate, 
               test, 
               to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = test.copy()
    test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled

def X_y_split(df, target):
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    '''  
    train, validate, test = split_data(df)

    X_train = train.drop(columns= target)
    y_train = train[target]

    X_validate = validate.drop(columns= target)
    y_validate = validate[target]

    X_test = test.drop(columns= target)
    y_test = test[target]
        
    # Have function print datasets shape
    print(f'''
    X_train -> {X_train.shape}
    X_validate -> {X_validate.shape}
    X_test -> {X_test.shape}''') 
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test