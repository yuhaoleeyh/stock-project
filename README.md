## Project title
Analysis of Stock Price Predictions using LSTM

## Motivation
I always wanted to properly delve into machine/deep learning models, and specifically apply it in an area that I am passionate in. Since I was also interested in stocks and wanted to examine the relationships surrounding the various factors (including seasonality, Bollinger bands and price pressures), I wanted to come up with a project which involved these 2 aspects. 

Link to medium article: https://medium.com/analytics-vidhya/analysis-of-stock-price-predictions-using-lstm-models-f993faa524c4

Published in Analytics Vidhya, the 2nd largest data science community in the world 


## Tech/framework used
Python, Pandas, Tensorflow, Keras, Matplotlib, Finta, Seaborn


## Features
In this project, I attempt to improve upon other predictive models of price prediction by altering the number of layers and through feature engineering, come up with accurate models that can as best as possible mimic the price difference. Most importantly, I iteratively tried to improve on the models while avoiding problems mentioned in many other articles, including look-ahead bias or copying yesterday's price. Concepts of seasonality, validation loss, stationarity with proper data visualisation as well as LSTM stacking/boosting were applied to improve the results. 


## How to use?
To use, run: pip install -r requirements.txt to install all required dependencies.

There are 3 models that can be run and tested:

(a) basic_model.py
The basic model with the base LSTM layers with no tuning (only features added in)

(b) magnitude_model.py
This model contains the tuned LSTM layers with feature engineering, that is used to predict the magnitude of the price

(c) difference_model.py
This model contains the tuned and stacked LSTM layers that is used to predict differences that is (a) iteratively added to a base point to generate a predicted graph and (b) added to each actual price at time t to get the predicted price at t + 1.


## License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MIT © Lee Yu Hao