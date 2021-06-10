import itertools
from matching import Match

def filter_data(df, strata):
    '''
    It samples data from a pandas dataframe using strata.
   
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    '''
    tmp = df[strata]
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    
    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            
            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry)
            first = False
        else:
            tmp_df = df.query(qry)
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df

# helper function 1
def find_types(data, catg_column, strata):
    temp =[]
    for var in catg_column:
        if var in strata:
            kind = list(data[var].value_counts().index)

            if len(strata)==1:
                temp=kind
    
            else: 
                temp.append(kind)
        
    return temp

# helper function 2
def filter_elements(temp_data, s, item):
    for i in strata:
        idx =(temp_data[i]== item[s])
        data = temp_data.loc[idx]
        s=s+1 
        temp_data = data
    
    return temp_data

# original dataframe
def maching_after_stratify(df, cat_column, strata):
    
    new_df = filter_data(df, strata)
    res = find_types(new_df, cat_column, strata)
    output = list()

    if len(strata)==1:
        for x in res:
            dat = new_df.loc[new_df[strata[0]]==x]
            match = Match(dat['treatment'], dat['propensity'])
            result = match.match_method_knn(dat, k=1)
            output.append(result)
            
        return output
            
    else: 
        combination = list(itertools.product(*res))
        for item in combination:
            temp_dat = new_df
            s=0
            stratifed_data = filter_elements(temp_dat, s, item)
            match = Match(stratifed_data['treatment'], stratifed_data['propensity'])
            result = match.match_method_knn(stratifed_data, k=1)
            output.append(result)
    
        return output
    

    
    

# cat_column = ['education', 'housing']
# strata = ['education','housing']

# mat = maching_after_stratify(dfn, cat_column, strata)
# mat
    
    


