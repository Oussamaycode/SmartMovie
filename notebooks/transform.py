import pandas as pd

df=pd.read_json('SmartMovie/data/raw.json')

def analyse():

    print(df.head())
    print(df.tail())
    print(f'les types de donnés:{df.dtypes}')
    print(f'Nombres de columns et des lignes:{df.shape}')
    print(f'info sur le data:{df.info}')

def incoherances():

    positive_values=['runtime','budget','revenue','popularity','vote_average','vote_count']
    for value in positive_values:
        a=df[df[value]<0]
        if len(a)>0:
            print(f'incoherance:{value} <0')
            df.drop(df[df[value]<0],inplace=True)

    b=df[df['vote_average']>10]
    if len(b)>0:
                print('incoherance:vote_average>10')
                df.drop(df['vote_average'>10],inplace=True)
    df.to_json('SmartMovie/data/raw.json')

def missingValues():

    df=pd.read_json('SmartMovie/data/raw.json')

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate movie IDs:")
    print(df["movie_id"].duplicated().sum())

    
    df = df.drop_duplicates(
        subset="movie_id",
        keep="first"
    )

 
    df["overview"] = df["overview"].fillna("Unknown")
    df["genres"] = df["genres"].fillna("Unknown")
    df["keywords"] = df["keywords"].fillna("Unknown")

    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    print(df.isnull().sum())
    df.to_json('SmartMovie/data/raw.json')


def seperate():

    num_cols=df.select_dtypes(include='number')
    text_col=df.select_dtypes(include='string')

    print(num_cols)
    print(text_col)
   
    
analyse()
incoherances()
missingValues()
seperate()



    


