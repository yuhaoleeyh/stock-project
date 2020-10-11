from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt


ts = TimeSeries(key='OX8RR8VDFUXWLYVD', output_format='pandas')

data, meta_data = ts.get_daily(symbol = "TSE:TD", outputsize='full')

print(data)

# data.to_csv(f'./file_daily.csv') 

data['4. close'].plot()
plt.title('Intraday TimeSeries Google')
plt.show()