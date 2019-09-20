# Awesome-time-series

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## Contents

1. [Challenges](#Challenges)
2. [Tutorials](#Tutorials)
3. [Books](#Books)
4. [Papers](#Papers)
5. [Scholar](#Scholar)
6. [Tools](#Tools)
7. [Competitions](#Competitions)
8. [Datasets](#Datasets)
9. [Related](#Related)

## Challenges

+ Complex Seasonal Patterns

  > paper: Forecasting time series with complex seasonal patterns using exponential smoothing, Alysha M De Livera, Rob J Hyndman and Ralph D Snyder
  >
  > blog: <http://businessforecastblog.com/analyzing-complex-seasonal-patterns/>

+ Hierarchical

  > http://robjhyndman.com/papers/hierarchical/
  >
  > <http://www.forecastpro.com/Trends/forecasting101January2009.html>

+ Highly Frequency

  > blog: https://www.zhihu.com/question/26464548

  

## Tutorials

1. [Architecture](#Architecture)
2. [Property](#Property)
3. [Feature](#Feature)
4. [Visualization](#Visualization)
5. [Model](#Model)
6. [Strategy](#Strategy)
7. [Application](#Application)
8. [Q&A](#Q&A)

### Architecture

<div align="center">
<img src="http://people.duke.edu/~rnau/411flow.gif" width="500" height="600" alt="flow-chat"></img>
</div>

### Property

+ The Ergodic Theorem [the time series chapter in this book](http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/)
+ 
+ Cross-correlation (Time Delay Analysis)
+ Dynamic time warping

>  https://robjhyndman.com/hyndsight/tscharacteristics/>

### Feature

+ time series feature:

  http://forecastingprinciples.com/index.php/features-of-time-series



+ feature extraction

  Paper: Automatic Identification of Time Series Features for Rule-Based Forecasting



+ feature selection

  Paper: Feature selection for time series prediction – A combined filter and wrapper approach for neural networks

  

### Visualization

+ Recurrence Plots



### Model

+ Kalman filter

  <details>
  https://en.wikipedia.org/wiki/Kalman_filter <br>
  http://www.cs.unc.edu/~welch/kalman/
  http://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
  http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
  http://blog.csdn.net/xiahouzuoxin/article/details/39582483
  </details>

  

+ X13-SEATS-ARIMA

+ TBATS

+ ARIMAX
+ ARX-ARMAX

+ ARDL(Auto Regressive Distributed Lag) [blog](<http://blog.eviews.com/2017/04/autoregressive-distributed-lag-ardl.html>) 

  

+ Dynamic Regression Models

+ ...

  

+ SVM

  > Time series prediction using support vector machines: a survey
  >
  > Support vector machines experts for time series forecasting
  >
  > Predicting time series with support vector machines

+ Boosting

  > A gradient boosting approach to the Kaggle load forecasting competition

+ NARX

  > Long-Term Time Series Prediction with the NARX Network An Empirical Evaluation

+ Bayesian neural network (BNN)

  > paper:
  >
  > Deep and Confident Prediction for Time Series at Uber [arxiv](<https://arxiv.org/abs/1709.01907>)
  >
  > <https://eng.uber.com/neural-networks-uncertainty-estimation/>



+ Neural Network

  > Designing a neural network for forecasting financial time series
  >
  > Neural network forecasting for seasonal and trend time series
  >
  > 
  >
  > RNN:
  >
  > Engineering Extreme Event Forecasting at Uber with Recurrent Neural Networks
  >
  > LSTM:
  >
  > Long Short Term Memory Networks for Anomaly Detection in Time Series
  >
  > 
  >
  > <http://www.neural-forecasting.com/tutorials.htm>

  

  

+ Hybrid

  > Time series forecasting using a hybrid ARIMA and neural network model 
  >
  > An artificial neural network (p,d,q) model for timeseries forecasting
  >
  > 

+ Forecast Combination

  > paper: 
  >
  > Timmermann, Allan, 2006. "**Forecast Combinations**," [Handbook of Economic Forecasting](https://ideas.repec.org/s/eee/ecofch.html), Elsevier. [ppt](<http://www.oxford-man.ox.ac.uk/sites/default/files/events/combination_Sofie.pdf>) 
  >
  > A simple explanation of the forecast combination puzzle
  >
  > Combining time series models for forecasting
  >
  > Optimal combination forecasts for hierarchical time series



### Strategy

+ Machine Learning Strategies for Time Series Prediction [slideshare](<https://www.slideshare.net/gbonte/machine-learning-strategies-for-time-series-prediction>) 

+ Machine learning strategies for multi-step-ahead time series forecasting [PhD thesis](http://souhaib-bentaieb.com/pdf/2014_phd.pdf)

+ Training Strategies for Time Series: Learning for Prediction, Filtering, and Reinforcement Learning [thesis](www.cs.cmu.edu/~arunvenk/papers/thesis.pdf) 

+ 

  

### Application

+ Electricity

  Electricity price forecasting: A review of the state-of-the-art with a look into the future

  Combined modeling for electric load forecasting with adaptive particle swarm optimization

+ Business

  Business Forecasting Practical Problems and Solutions, Edited by Michael Gilliland, Len Tashman, Udo Sglavo

### Q&A

- Is it unusual for the MEAN to outperform ARIMA? [site](<https://stats.stackexchange.com/questions/124955/is-it-unusual-for-the-mean-to-outperform-arima>)
- When to log transform a time series before fitting an ARIMA model [site](https://stats.stackexchange.com/questions/6330/when-to-log-transform-a-time-series-before-fitting-an-arima-model) 
- Don’t Put Lagged Dependent Variables in Mixed Models [site](<https://statisticalhorizons.com/lagged-dependent-variables>) 
- Estimating same model over multiple time series [site](https://stats.stackexchange.com/questions/23036/estimating-same-model-over-multiple-time-series) 
- Is it possible to do time-series clustering based on curve shape? [site](http://stats.stackexchange.com/questions/3331/is-it-possible-to-do-time-series-clustering-based-on-curve-shape?noredirect=1&lq=1)
- Modelling longitudinal data where the effect of time varies in functional form between individuals [site](<https://stats.stackexchange.com/questions/2777/modelling-longitudinal-data-where-the-effect-of-time-varies-in-functional-form-b>) 
- Why can't we use top-down methods in forecasting grouped time series? [site](<https://stats.stackexchange.com/questions/163520/why-cant-we-use-top-down-methods-in-forecasting-grouped-time-series?rq=1>) 
- Proper way of using recurrent neural network for time series analysis [site](<https://stats.stackexchange.com/questions/8000/proper-way-of-using-recurrent-neural-network-for-time-series-analysis>) 
- 
- 

## Papers



### Literature Review

+ 25 Years of Time Series Forecasting, Jan G De Gooijer, Rob J Hyndman

+ A review on time series data mining, Tak Chung Fu

  

+ A Survey on Nonparametric Time Series Analysis, Siegfried Heiler

+ Time-series clustering – A decade review
+ Segmenting Time Series: A Survey and Novel Approach



## Books

+ Time Series Analysis, James D. Hamilton, Princeton University Press, 1994

+ Time Series Analysis Forecasting and Control (5th Edition), George E. P. Box

+ Principles of Forecasting: A Handbook for Researchers and Practitioners, Editors: Armstrong, J.S. (Ed.)

+ Forecasting: Principles and Practice, Rob J Hyndman and George Athanasopoulos [online](<https://otexts.com/fpp2/>) 

  > Notes: the ETS (Error, Trend, Seasonal) framework

+ Forecasting with Exponential Smoothing: The State Space Approach, Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. [online](<http://www.exponentialsmoothing.net/>) 

+ Analysis of Financial Time Series (3ed), Ruey S. Tsay [site](https://faculty.chicagobooth.edu/ruey.tsay/teaching/fts/) 
+ The Elements of Financial Econometrics [site](http://orfe.princeton.edu/~jqfan/fan/FinEcon.html) 
+ Nonlinear Time Series Nonparametric and Parametric Methods [site](http://orfe.princeton.edu/~jqfan/fan/nls.html)
+ Nonlinear Time Series Analysis

## Scholar

+ Makridakis

  > Pioneered Empirical competition on Forecasting called M, M2 and M3, and paved way for evidence based methods in forecasting

+ J. Scott Armstrong [site](<https://marketing.wharton.upenn.edu/profile/jscott/#research>)

  > Provides valuable insights in the form of books/articles on Forecasting Practice
  >
  > Paper: 
  >
  > Simple versus complex forecasting: The evidence
  >
  > Rule-Based Forecasting: Using Judgment in Time-Series Extrapolation
  >
  > Combining Forecasts

+ Kesten C. Green [site](<http://kestencgreen.com/>) [forecasting principles](<http://forecastingprinciples.com/>) [simple forecasting](<http://simple-forecasting.com/>)

  > Unifying theory of forecasting:
  > The Golden Rule of Forecasting provides a unifying theory of forecasting. The Rule is to be conservative when forecasting by relying on cumulative knowledge about the situation and about forecasting. Following the Golden Rule guidelines reduces forecast errors by nearly a third, on average, compared to common practice.
  > Superiority of simple forecasting methods:
  > Sophisticatedly simple forecasting methods, which can be understood by decision makers, reduce forecast errors by nearly a quarter, on average, compared to forecasts from complex statistical methods.
  >
  > ...
  >
  > Paper: 
  >
  > Golden Rule of Forecasting: Be conservative
  >
  > Golden Rule of Forecasting Rearticulated: Forecast Unto Others as You Would Have Them Forecast Unto You
  >
  > ...
  >
  > 

+ Gardner

  > Invented Damped Trend exponential smoothing another simple method which works surprisingly well vs. ARIMA)

+ Eamonn Keogh [site](http://www.cs.ucr.edu/~eamonn/)

  > Dynamic time warping

  

+ Rob J Hyndman [blog](<https://robjhyndman.com/>) [github](https://github.com/robjhyndman) 

  > Some interesting topics on the blog: forecast intervals for aggregates (the aggregate of several time periods), fitting models to short time series, fitting models to long time series, forecasting weekly data, forecasting with daily data, forecasting with long seasonal periods, seasonal periods, rolling forecasts, batch forecasting, facts and fallacies of the AIC...

+ Tao Hong [blog](http://blog.drhongtao.com/)

  > Specialize: energy forecasting, electric load forecasting
  >
  > PhD thesis: Short Term Electric Load Forecasting
  >
  > Course: Electric Load Forecasting I: Fundamentals and Best Practices, Electric Load Forecasting II: Advanced Topics and Case Studies



## Tools

+ R package

  [forecast](<https://github.com/robjhyndman/forecast>), [tsfeatures](<https://github.com/robjhyndman/tsfeatures>), thief(Temporal Hierarchical Forecasting), tsDyn, ForecastCombinations, [forecastHybrid](https://github.com/ellisp/forecastHybrid), opera(Online Prediction by ExpeRt Aggregation) 

+ [prophet](https://github.com/facebookincubator/prophet)

+ [tsfresh](https://github.com/blue-yonder/tsfresh)

+ [palladium](https://github.com/ottogroup/palladium)

  Framework for setting up predictive analytics services

+ [Forecast Pro](https://www.forecastpro.com/) 

+ Autobox

+ [TISEAN](https://www.pks.mpg.de/~tisean/Tisean_3.0.1/index.html) - Nonlinear Time Series Analysis

+ Timeseries analysis for neuroscience data <http://nipy.org/nitime>

  <https://github.com/nipy/nitime>



+ Tensorflow

  <https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series>

  <https://github.com/mouradmourafiq/tensorflow-lstm-regression>

+ Keras

  <https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction>

  <https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/>



## Competitions

+ The M-competitions [dataset](http://robjhyndman.com/software/mcomp/) 
+ The Tourism Forecasting Competition [dataset](https://github.com/ellisp/Tcomp-r-package) 
+ Global Energy Forecasting Competition (GEFCom)
+ <http://www.neural-forecasting-competition.com/index.htm>



## Datasets

+ UCR Time Series Classification Archive
  http://www.cs.ucr.edu/~eamonn/time_series_data/

+ CompEngine

  A self-organizing database of time-series data

  <https://www.comp-engine.org/#!>



## Related

+ Signal Processing

  > time frequency analysis, fourier analysis, wavelets,...







