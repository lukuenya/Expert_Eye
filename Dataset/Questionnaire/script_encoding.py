import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import datetime


"""
cols_to_drop = [
    'id',
    'submitdate',
    'lastpage',
    'startlanguage',
    'seed',
    'startdate',
    'datestamp',
    'ipaddr',
    'VINCQ05CODEPATICIPAN[SQ001]',
    'VINCQ05CODEPATICIPAN[SQ002]',
    'VINCQ05CODEPATICIPAN[SQ003]',
    'SPPB003',
    'SPPB92',
    'VINCQ06TELPORTABLE',
    'VINCQ07TELFIXE',
    'refurl',
    'VINCQ051[SQ001]',
    'EXAMCLIN011',
    'VINCQ031SEX',
    'MMSE03ORIENTATION1',
    'ACUITEVISUELLE01',
    'ACUITEVISUELLE02',
    'MNAEVALGLOBALEG',
    'MNAEVALGLOBALEH',
    'MNAEVALGLOBALEI',
    'MNAEVALGLOBALEJ',
    'MNAEVALGLOBALEK',
    'MNAEVALGLOBALEL',
    'MNAEVALGLOBALEM',
    'MNAEVALGLOBALEO',
    'MNAEVALGLOBALEP',
    'MNAEVALGLOBALEQ',
    'MNAEVALGLOBALER',
    'MMSE01'
]

# Comment Columns [TO DROP]
comment_cols = [
    'SPPB05',
    'FROPCOM0009[comment]',
    'FROPCOM0010[comment]',
    'FROPCOM0020[comment]',
    'FROPCOM0022[comment]',
    'FROPCOM0026[SQ001comment]',
    'FROPCOM0026[SQ005comment]',
    'FROPCOM0026[SQ003comment]',
    'FROPCOM0026[SQ004comment]',
    'FROPCOM0028[comment]',
    'ABILYCARE2[comment]'
]

# -Time Columns [TO DROP]
time_cols = [
    'groupeTime3',
    'groupeTime12',
    'groupeTime13',
    'groupeTime15',
    'groupeTime23',
    'groupeTime26',
    'interventionTime'
]



"""
# Date de naissance column [TO HANDLE]
# Covert '1947-07-01 00:00:00' to 'Age: 73'
def date_naissance_col(x):
    """
    Convert '1947-07-01 00:00:00' to 'Age: 73'
    """
    if pd.isnull(x):
        return np.nan
    elif isinstance(x, str):
        try:
            # Try to parse the date
            date_of_birth = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            # Calculate and return the age
            return datetime.datetime.now().year - date_of_birth.year
        except ValueError:
            # If the date parsing fails, return the original value
            return x
    return x




# Encoding
def custom_encoding_0(x):
    """
    Encode 'Oui' and 'Non' to 1 and 0
    """
    if x in ['Non', 'NON (2)', 'Non (0)', 'NON ( 0 points)']:
        return 0
    elif x in ['Oui', 'OUI (1)', 'Oui (1) (Spécifier)', 'OUI (4 points)' ]:
        return 1
    elif x == 'PARFOIS ( 2 points)':
        return 0.5
    elif x == 'NaN':
        return np.nan
    elif x == 'Ne sait pas':
        return np.nan
    else:
        return x

# Categories to encode one by one
"""
['Jamais', 'Souvent', 'De temps en temps', 'La plupart du temps', np.nan]
['Pas autant', 'Un peu seulement', 'Presque plus', 'Oui, tout autant', np.nan]
['Pas du tout', 'Un peu, mais cela ne m’inquiète pas', 'Oui, mais ce n’est pas trop grave', 'Oui, très nettement', np.nan]
['Vraiment moins qu’avant', 'Plus du tout', 'Plus autant qu’avant', 'Autant que par le passé', np.nan]
['Très occasionnellement', 'Occasionnellement', 'Assez souvent', 'Très souvent', np.nan]
['Rarement', 'Assez souvent', 'La plupart du temps', np.nan]
['Jamais', 'Rarement', 'Oui, en général', 'Oui, quoi qu’il arrive', np.nan]
['Jamais', 'Parfois', 'Très souvent', 'Presque toujours', np.nan]
['Jamais', 'Parfois', 'Assez souvent', 'Très souvent', np.nan]
['Plus du tout', 'Je n’y accorde pas autant d’attention que je devrais', 'Il se peut que je n’y fasse plus autant attention', 'J’y prête autant d’attention que par le passé', np.nan]
['Pas du tout', 'Pas tellement', 'Un peu', 'Oui, c’est tout à fait le cas', np.nan]
['Presque jamais', 'Bien moins qu’avant', 'Un peu moins qu’avant', 'Autant qu’avant', np.nan]
['Jamais', 'Pas très souvent', 'Assez souvent', 'Vraiment très souvent', np.nan]
['Très rarement', 'Rarement', 'Parfois', 'Souvent', np.nan]
['Pas du tout', 'Un peu', 'Assez', 'Beaucoup', np.nan] -> FRAGIRE02 --

"""

# Encoding 'Jamais' -> `0`, 'Souvent' -> `1`, 'De temps en temps' -> `2`, 'La plupart du temps' -> `3`
def custom_encoding_1(x):
    if x == 'Jamais':
        return 0
    elif x == 'Souvent':
        return 1
    elif x == 'De temps en temps':
        return 2
    elif x == 'La plupart du temps':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS001'] = data['HADS001'].apply(custom_encoding_1)

# Encoding 'Pas autant' -> `0`, 'Un peu seulement' -> `1`, 'Presque plus' -> `2`, 'Oui, tout autant' -> `3`
def custom_encoding_2(x):
    if x == 'Pas autant':
        return 0
    elif x == 'Un peu seulement':
        return 1
    elif x == 'Presque plus':
        return 2
    elif x == 'Oui, tout autant':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS002'] = data['HADS002'].apply(custom_encoding_2)

# Encoding 'Pas du tout' -> `0`, 'Un peu, mais cela ne m’inquiète pas' -> `1`, 'Oui, mais ce n’est pas trop grave' -> `2`, 'Oui, très nettement' -> `3`
def custom_encoding_3(x):
    if x == 'Pas du tout':
        return 0
    elif x == 'Un peu, mais cela ne m’inquiète pas':
        return 1
    elif x == 'Oui, mais ce n’est pas trop grave':
        return 2
    elif x == 'Oui, très nettement':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS003'] = data['HADS003'].apply(custom_encoding_3)


# Encoding 'Vraiment moins qu’avant' -> `0`, 'Plus du tout' -> `1`, 'Plus autant qu’avant' -> `2`, 'Autant que par le passé' -> `3`
def custom_encoding_4(x):
    if x == 'Vraiment moins qu’avant':
        return 0
    elif x == 'Plus du tout':
        return 1
    elif x == 'Plus autant qu’avant':
        return 2
    elif x == 'Autant que par le passé':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS004'] = data['HADS004'].apply(custom_encoding_4)


# Encoding 'Très occasionnellement' -> `0`, 'Occasionnellement' -> `1`, 'Assez souvent' -> `2`, 'Très souvent' -> `3`
def custom_encoding_5(x):
    if x == 'Très occasionnellement':
        return 0
    elif x == 'Occasionnellement':
        return 1
    elif x == 'Assez souvent':
        return 2
    elif x == 'Très souvent':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS005'] = data['HADS005'].apply(custom_encoding_5)


# Encoding 'Rarement' -> `0`, 'Assez souvent' -> `1`, 'La plupart du temps' -> `2`
def custom_encoding_6(x):
    if x == 'Rarement':
        return 0
    elif x == 'Assez souvent':
        return 1
    elif x == 'La plupart du temps':
        return 2
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS006'] = data['HADS006'].apply(custom_encoding_6)


# Encoding 'Jamais' -> `0`, 'Rarement' -> `1`, 'Oui, en général' -> `2`, 'Oui, quoi qu’il arrive' -> `3`
def custom_encoding_7(x):
    if x == 'Jamais':
        return 0
    elif x == 'Rarement':
        return 1
    elif x == 'Oui, en général':
        return 2
    elif x == 'Oui, quoi qu’il arrive':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS007'] = data['HADS007'].apply(custom_encoding_7)


# Encoding 'Jamais' -> `0`, 'Parfois' -> `1`, 'Très souvent' -> `2`, 'Presque toujours' -> `3`
def custom_encoding_8(x):
    if x == 'Jamais':
        return 0
    elif x == 'Parfois':
        return 1
    elif x == 'Très souvent':
        return 2
    elif x == 'Presque toujours':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS008'] = data['HADS008'].apply(custom_encoding_8)


# Encoding 'Jamais' -> `0`, 'Parfois' -> `1`, 'Assez souvent' -> `2`, 'Très souvent' -> `3`
def custom_encoding_9(x):
    if x == 'Jamais':
        return 0
    elif x == 'Parfois':
        return 1
    elif x == 'Assez souvent':
        return 2
    elif x == 'Très souvent':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS009'] = data['HADS009'].apply(custom_encoding_9)


# Encoding 'Plus du tout' -> `0`, 'Je n’y accorde pas autant d’attention que je devrais' -> `1`, 'Il se peut que je n’y fasse plus autant attention' -> `2`, 'J’y prête autant d’attention que par le passé' -> `3`
def custom_encoding_10(x):
    if x == 'Plus du tout':
        return 0
    elif x == 'Je n’y accorde pas autant d’attention que je devrais':
        return 1
    elif x == 'Il se peut que je n’y fasse plus autant attention':
        return 2
    elif x == 'J’y prête autant d’attention que par le passé':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS010'] = data['HADS010'].apply(custom_encoding_10)


# Encoding 'Pas du tout' -> `0`, 'Pas tellement' -> `1`, 'Un peu' -> `2`, 'Oui, c’est tout à fait le cas' -> `3`
def custom_encoding_11(x):
    if x == 'Pas du tout':
        return 0
    elif x == 'Pas tellement':
        return 1
    elif x == 'Un peu':
        return 2
    elif x == 'Oui, c’est tout à fait le cas':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS011'] = data['HADS011'].apply(custom_encoding_11)


# Encoding 'Presque jamais' -> `0`, 'Bien moins qu’avant' -> `1`, 'Un peu moins qu’avant' -> `2`, 'Autant qu’avant' -> `3`
def custom_encoding_12(x):
    if x == 'Presque jamais':
        return 0
    elif x == 'Bien moins qu’avant':
        return 1
    elif x == 'Un peu moins qu’avant':
        return 2
    elif x == 'Autant qu’avant':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS012'] = data['HADS012'].apply(custom_encoding_12)


# Encoding 'Jamais' -> `0`, 'Pas très souvent' -> `1`, 'Assez souvent' -> `2`, 'Vraiment très souvent' -> `3`
def custom_encoding_13(x):
    if x == 'Jamais':
        return 0
    elif x == 'Pas très souvent':
        return 1
    elif x == 'Assez souvent':
        return 2
    elif x == 'Vraiment très souvent':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS013'] = data['HADS013'].apply(custom_encoding_13)


# Encoding 'Très rarement' -> `0`, 'Rarement' -> `1`, 'Parfois' -> `2`, 'Souvent' -> `3`
def custom_encoding_14(x):
    if x == 'Très rarement':
        return 0
    elif x == 'Rarement':
        return 1
    elif x == 'Parfois':
        return 2
    elif x == 'Souvent':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['HADS014'] = data['HADS014'].apply(custom_encoding_14)


# Encoding 'Pas du tout' -> `0`, 'Un peu' -> `1`, 'Assez' -> `2`, 'Beaucoup' -> `3` to all data
def custom_encoding_15(x):
    if x == 'Pas du tout':
        return 0
    elif x == 'Un peu':
        return 1
    elif x == 'Assez':
        return 2
    elif x == 'Beaucoup':
        return 3
    elif pd.isnull(x):
        return np.nan
    else:
        return x

data = data.applymap(custom_encoding_15)

# Encode '0' -> `0` , ['1 à 2 fois', 'Plus de 1'] -> `1` , 'Plus de 2 fois' -> `2` to FRAGIRE02 and FRAGIRE 15 columns
def custom_encoding_16(x):
    if x == '0':
        return 0
    elif x == ['1 à 2 fois', 'Plus de 1']:
        return 1
    elif x == 'Plus de 2 fois':
        return 2
    elif pd.isnull(x):
        return np.nan
    else:
        return x
    
data['FRAGIRE02'] = data['FRAGIRE02'].apply(custom_encoding_16)
data['FRAGIRE15'] = data['FRAGIRE15'].apply(custom_encoding_16)



# Encode by extracting the number in parenthesis from the string
def extract_number(value):
    """
    Return only the numeric part of the string
    """
    if pd.isnull(value):
        return np.nan
    elif isinstance(value, str):
        match = re.search(r'\((\d+)\)', value)
        if match:
            return int(match.group(1))
    return value



# OneHot Encoding of some columns with sklearn [TO HANDLE]
# Travail et Medicaments columns
#def one_hot_encoding(df, cols):
    #"""
    #OneHot Encoding of some columns with sklearn
    #:param df: DataFrame
    #:param cols: list of columns to encode
    #:return: DataFrame
    #"""
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc_df = pd.DataFrame(enc.fit_transform(df[cols]).toarray())
    #enc_df.columns = enc.get_feature_names_out(cols)
    #df = df.drop(cols, axis=1)
    #df = df.join(enc_df)
    #eturn df

def one_hot_encoding(df, cols):
    """
    OneHot Encoding of some columns with sklearn
    :param df: DataFrame
    :param cols: list of columns to encode
    :return: DataFrame
    """
    # convert values to string
    for col in cols:
        df[col] = df[col].astype(str)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(df[cols]).toarray())
    enc_df.columns = enc.get_feature_names_out(cols)
    df = df.drop(cols, axis=1)
    df = df.join(enc_df)
    return df




