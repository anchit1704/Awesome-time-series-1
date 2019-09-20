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

  paper: Forecasting time series with complex seasonal patterns using exponential smoothing, Alysha M De Livera, Rob J Hyndman and Ralph D Snyder

  blog: <http://businessforecastblog.com/analyzing-complex-seasonal-patterns/>

+ Highly Frequency

  blog: https://www.zhihu.com/question/26464548

  

## Tutorials

1. [property](#property)
2. [feature](#feature)
3. [visualization](#visualization)
4. [model](#model)
5. [strategy](#strategy)
6. [application](#application)

### property

+ The Ergodic Theorem [the time series chapter in this book](http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/)
+ 



### feature

+ time series feature:

  http://forecastingprinciples.com/index.php/features-of-time-series



+ feature extraction

  Paper: Automatic Identification of Time Series Features for Rule-Based Forecasting



+ feature selection

  Paper: Feature selection for time series prediction – A combined filter and wrapper approach for neural networks

  

### visualization

+ Recurrence Plots



### model

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



### strategy

+ Machine Learning Strategies for Time Series Prediction [slideshare](<https://www.slideshare.net/gbonte/machine-learning-strategies-for-time-series-prediction>) 
+ Machine learning strategies for multi-step-ahead time series forecasting [PhD thesis](http://souhaib-bentaieb.com/pdf/2014_phd.pdf)
+ Training Strategies for Time Series: Learning for Prediction, Filtering, and Reinforcement Learning [thesis](www.cs.cmu.edu/~arunvenk/papers/thesis.pdf) 
+ 

### application

+ Electricity

  Electricity price forecasting: A review of the state-of-the-art with a look into the future

+ Business

  Business Forecasting Practical Problems and Solutions, Edited by Michael Gilliland, Len Tashman, Udo Sglavo



## Papers



### Literature Review

+ 25 Years of Time Series Forecasting, Jan G De Gooijer, Rob J Hyndman

+ A review on time series data mining, Tak Chung Fu

  

+ A Survey on Nonparametric Time Series Analysis, Siegfried Heiler

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

  [forecast](<https://github.com/robjhyndman/forecast>), [tsfeatures](<https://github.com/robjhyndman/tsfeatures>), tsDyn, ForecastCombinations, [forecastHybrid](https://github.com/ellisp/forecastHybrid)

+ [prophet](https://github.com/facebookincubator/prophet)

+ [tsfresh](https://github.com/blue-yonder/tsfresh)

+ [palladium](https://github.com/ottogroup/palladium)

  Framework for setting up predictive analytics services

+ Forecast Pro for Windows

  <http://people.duke.edu/~rnau/autofor.htm>

+ Autobox

+ [TISEAN](https://www.pks.mpg.de/~tisean/Tisean_3.0.1/index.html) - Nonlinear Time Series Analysis

+ Timeseries analysis for neuroscience data <http://nipy.org/nitime>

  <https://github.com/nipy/nitime>





## Competitions

+ The M-competitions [dataset](http://robjhyndman.com/software/mcomp/) 
+ The Tourism Forecasting Competition [dataset](https://github.com/ellisp/Tcomp-r-package) 
+ Global Energy Forecasting Competition (GEFCom)



## Datasets

+ UCR Time Series Classification Archive
  http://www.cs.ucr.edu/~eamonn/time_series_data/

+ CompEngine

  A self-organizing database of time-series data

  <https://www.comp-engine.org/#!>



## Related

+ Signal Processing

  > time frequency analysis, fourier analysis, wavelets,...







