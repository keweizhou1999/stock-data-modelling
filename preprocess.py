
import numpy as np
import pandas as pd

# Read in S&P 500 Data and Sample Date 
sampleDate = pd.read_csv("Standard95-00.csv")
sp500  = pd.read_csv("sp1980.csv") #Excel File

spDate = sp500['Date'].tolist()
spPriceChange = sp500['PriceChange'].tolist()
spVolumeChange = sp500['VolumeChange'].tolist()

standardDate = sampleDate['Date'].tolist()
export = []

for i in range(len(standardDate)):
    new = []
    new.append(standardDate[i])
    if (standardDate[i] in spDate):
        new.append(spPriceChange[spDate.index(standardDate[i])])
        new.append(spVolumeChange[spDate.index(standardDate[i])])
    else:
        # Default price and volume change to 0 if data not found
        new.append(0)
        new.append(0)
    export.append(new)    

# Export to csv, which will be used as input to k-means model in main.py
df = pd.DataFrame(export, columns=['Date', 'PriceChange', 'VolumeChange'])    
df.to_csv(r'Processed95-00.csv', index = False, header = True)
