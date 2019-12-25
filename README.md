# Awesome-time-series

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## Contents

1. [Related](#Related)
2. [Challenges](#Challenges)
3. [Tutorials](#Tutorials)
4. [Books](#Books)
5. [Papers](#Papers)
6. [Conference](#Conference)
7. [Scholar](#Scholar)
8. [Competitions](#Competitions)
9. [STOA](#STOA)
10. [Datasets](#Datasets)
11. [Tools](#Tools)

---

## Related

- Information Theory

- Signal Processing/Digital Signal Processing

  > time frequency analysis, fourier analysis, wavelets,...

- Audio Content Analysis

  > fundamentals of sound and time-frequency
  > representations, periodicity detection, novelty detection, sound classification, ...
  >
  > <http://www.nyu.edu/classes/bello/Teaching.html>

- Dynamical Systems Theory

---

## Challenges

+ Complex Seasonal Patterns

  > Forecasting time series with complex seasonal patterns using exponential smoothing, Alysha M De Livera, Rob J Hyndman and Ralph D Snyder
  >
  > <http://businessforecastblog.com/analyzing-complex-seasonal-patterns/>

+ Mixed Frequency

  > Mixed data sampling (MIDAS) models
  >
  > Forecasting with mixed frequencies

+ Irregularly Sampled (Unevenly Spaced, Sparse)

  > https://en.wikipedia.org/wiki/Unevenly_spaced_time_series
  >
  > Comparison of correlation analysis techniques for irregularly sampled time series
  >
  > <http://www.eckner.com/research.html>
  >
  > AWarp: Warping Distance for Sparse Time Series [site](<https://www.cs.unm.edu/~mueen/Projects/AWarp/>)

+ Hierarchical

  > http://robjhyndman.com/papers/hierarchical/
  >
  > <http://www.forecastpro.com/Trends/forecasting101January2009.html>

+ Highly Frequency

  > https://www.zhihu.com/question/26464548

---

## Tutorials

1. [Architecture](#Architecture)
2. [Property](#Property)
3. [Feature](#Feature)
4. [Visualization](#Visualization)
5. [Model](#Model)
6. [Strategy](#Strategy)
7. [Topic](#Topic)
8. [Application](#Application)
9. [Q&A](#Q&A)

---

### Architecture

[timeseries structure design](<https://github.com/bifeng/Awesome-time-series/blob/master/summary/1timeseries-structure-design-for-training-new.xlsx>)

> summary from the competitions experience.

---

### Property

+ Random Walk

  > <https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series/>

+ Stationary

  > mean stationary, variance stationary

+ The Ergodic Theorem 

  > [the time series chapter in this book](http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/)

+ The Takens's theorem

  > https://en.wikipedia.org/wiki/Takens%27_theorem

+ Attractor

  > Attractor reconstruction - Scholarpedia
  >
  > Chaotic Attractor Reconstruction - node99

+ Motif

  > Finding Motifs in Time Series
  >
  > Exact discovery of time series motifs [site](http://alumni.cs.ucr.edu/~mueen/MK/  ) 
  >
  > Detecting time series motifs under uniform scaling [paper](http://www.cs.ucr.edu/~eamonn/motifs_under_scaling.pdf) [site](<http://www.cs.ucr.edu/~eamonn/SIGKDD07/UniformScaling.html>) 

+ Periodicity

  > periodicity detection/estimation:
  >
  > Multi-step approach to find periods of time-series data [site](<https://astromsshin.github.io/science/code/MultiStep_Period/index.html>) 
  >
  > detecting multiple periodicity in time series [site](https://www.analyticbridge.datasciencecentral.com/forum/topics/challenge-of-the-week-detecting-multiple-periodicity-in-time-seri) 

+ Embedding Dimension

  > The false nearest neighbors algorithm: An overview

+ SAX(Symbolic Aggregate approXimation)

  > SAX is the first symbolic representation for time series that allows for dimensionality reduction and indexing with a lower-bounding distance measure. 
  > In classic data mining tasks such as clustering, classification, index, etc., 
  > SAX is as good as well-known representations such as Discrete Wavelet Transform (DWT) and Discrete Fourier Transform (DFT), while requiring less storage space.
  > http://www.cs.ucr.edu/~eamonn/SAX.htm
  >
  > Experiencing SAX: a novel symbolic representation of time series
  >
  > HOT SAX: Efficiently Finding the Most Unusual Time

+ Topological

  > Topological Time Series Analysis:
  >
  > Geometry of sliding window embeddings
  >
  > Persistent Homology of Sliding Window Point Clouds
  >
  > https://www.joperea.com/

---

### Feature

+ data transformation

  > <http://people.duke.edu/~rnau/whatuse.htm>

+ time series features/structure:

  > [characteristics](https://robjhyndman.com/hyndsight/tscharacteristics/) 
  >
  > [features of time series](http://forecastingprinciples.com/index.php/features-of-time-series) 
  >
  > Finding Repeated Structure in Time Series Algorithms and Applications [site](<https://www.cs.unm.edu/~mueen/Tutorial/SDM2015.html>) | [site](http://www.cs.unm.edu/~mueen/Tutorial/ICDMTutorial3.ppt) 
  >
  > Slow feature analysis: Unsupervised learning of invariances

+ feature extraction/transformation

  > [time-based feature construction](<https://github.com/bifeng/Awesome-time-series/blob/master/summary/1time-based-feature-construction.xlsx>) [part1](<https://github.com/bifeng/Awesome-time-series/blob/master/summary/1time-based feature construction-part1.docx>) [part2](<https://github.com/bifeng/Awesome-time-series/blob/master/summary/1time-based feature construction-part2.docx>) [part3](<https://github.com/bifeng/Awesome-time-series/blob/master/summary/1time-based feature construction-part3.doc>)
  >
  > summary from the competitions experience.
  >
  >
  >
  > Automatic Identification of Time Series Features for Rule-Based Forecasting
  >
  > Distributed and parallel time series feature extraction for industrial big data applications

+ feature selection

  > Feature selection for time series prediction – A combined filter and wrapper approach for neural networks

---

### Visualization

+ Recurrence Plots

---

### Model

+ Kalman filter

  <details>
  https://en.wikipedia.org/wiki/Kalman_filter <br>
  http://www.cs.unc.edu/~welch/kalman/
  http://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
  http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
  http://blog.csdn.net/xiahouzuoxin/article/details/39582483
  </details>

+ Fourier transform

  > An Interactive Guide To The Fourier Transform
  >
  > The Fast Fourier Transform - Math ∩ Programming
  >
  > Understanding the FFT Algorithm
  >
  > http://news.mit.edu/2009/explained-fourier
  > http://news.mit.edu/2012/faster-fourier-transforms-0118
  >
  > 我所理解的快速傅里叶变换（FFT）

+ Dynamic time warping

  > A Bibliography of Dynamic Time Warping [site](http://www.cs.unm.edu/~mueen/DTWBib.html) 
  >
  > Extracting Optimal Performance from Dynamic Time Warping [site](http://www.cs.unm.edu/~mueen/DTW.pdf) 
  >
  > Everything you know about Dynamic Time Warping is Wrong
  >
  > Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping
  >
  > <https://gradientflow.com/2012/10/28/mining-time-series-with-trillions-of-points-dynamic-time-warping-at-scale/>
  >
  > <https://blog.acolyer.org/2016/05/11/searching-and-mining-trillions-of-time-series-subsequences-under-dynamic-time-warping/>

+ Simple model (Such as mean, moving averages,...)

  > <http://people.duke.edu/~rnau/whatuse.htm>
  >
  > The mean (constant, intercept-only) model for forecasting
  >
  > Review of basic statistics and the simplest forecasting model: the sample mean
  >
  > <https://people.duke.edu/~rnau/411mean.htm>
  >
  > Simple versus complex forecasting: The evidence
  >
  > Benchmarks for forecasting - Hyndsight
  >
  > Moving averages [paper](https://robjhyndman.com/papers/movingaverage.pdf) 
  >
  > 
  >
  > Theta:
  >
  > The theta model - a decomposition approach to forecasting
  >
  > Unmasking the Theta method
  >
  > 
  >
  > Case:
  >
  > Rossmann sales forecasting(mean and weights)
  > https://github.com/Dyakonov/notebooks
  >
  > 数据竞赛思路分享：机场客流量的时空分布预测 - ZJun Thinking - 博客频道 - CSDN
  >
  > 如何在第一次天池比赛中进入Top 5%（一） - 知乎专栏

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
  >
  > Application of support vector machines in financial time series forecasting

+ Boosting

  > A gradient boosting approach to the Kaggle load forecasting competition

+ NARX

  > Long-Term Time Series Prediction with the NARX Network An Empirical Evaluation

+ Bayesian neural network (BNN)

  > Deep and Confident Prediction for Time Series at Uber [arxiv](<https://arxiv.org/abs/1709.01907>)
  >
  > <https://eng.uber.com/neural-networks-uncertainty-estimation/>

+ Neural Network

  > http://www.neural-forecasting.com/tutorials.htm
  >
  > 
  >
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

+ Hybrid

  > Time series forecasting using a hybrid ARIMA and neural network model 
  >
  > An artificial neural network (p,d,q) model for timeseries forecasting

+ Forecast Combination

  > Timmermann, Allan, 2006. "**Forecast Combinations**," [Handbook of Economic Forecasting](https://ideas.repec.org/s/eee/ecofch.html), Elsevier. [ppt](<http://www.oxford-man.ox.ac.uk/sites/default/files/events/combination_Sofie.pdf>) 
  >
  > A simple explanation of the forecast combination puzzle
  >
  > Combining time series models for forecasting
  >
  > Optimal combination forecasts for hierarchical time series

---

### Strategy

+ Machine Learning Strategies for Time Series Prediction [slideshare](<https://www.slideshare.net/gbonte/machine-learning-strategies-for-time-series-prediction>) 
+ Machine learning strategies for multi-step-ahead time series forecasting [PhD thesis](http://souhaib-bentaieb.com/pdf/2014_phd.pdf)
+ Training Strategies for Time Series: Learning for Prediction, Filtering, and Reinforcement Learning [thesis](www.cs.cmu.edu/~arunvenk/papers/thesis.pdf) 



+ MASE

  > Another look at measures of forecast accuracy

+ Cross-Validation

  > [Time Series Nested Cross-Validation](<https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9>)
  >
  > On the use of cross-validation for time series predictor evaluation
  >
  > A note on the validity of cross-validation for evaluating autoregressive time series prediction
  >
  > Approximate leave-future-out cross-validation for Bayesian time series models

+ Prediction Intervals

  > The difference between prediction intervals and confidence intervals - Hyndsight
  >
  > Why time series forecasts prediction intervals aren't as good as we'd hope
  >
  > Better prediction intervals for time series forecasts
  >
  > Prediction intervals for ensemble time series forecasts

---

### Topic

+ Spectral analysis

  > Spectral analysis is carried out to describe how variation in a time series may be accounted for by cyclic components. This may also be referred to as "Frequency Domain". With this an estimate of the spectrum over a range of frequencies can be obtained and periodic components in a noisy environment can be separated out.

+ Intervention Analysis

  > Time Series Intervention Analysis (or Interrupted Time Series Analysis) can explain if there is a certain event that occurs that changes a time series. This technique is used a lot of the time in planned experimental analysis.
  >
  > The basic question is "Has an event had an impact on a time series?" 

+ Calendar effects

  > <http://calendar-effects.behaviouralfinance.net/>
  >
  > Special days, Holidays,...
  >
  > [Public_holidays_in_China](https://en.wikipedia.org/wiki/Public_holidays_in_China) 

+ Causality 

  > Convergent Cross Mapping
  >
  > <https://github.com/NickC1/skCCM>

+ Similarity

  > https://en.wikipedia.org/wiki/Cross-correlation
  >
  > Detect correlation between multiple time series - Anomaly
  >
  > 
  >
  > Similarity Search on Time Series Data: Past, Present and Future [site](http://www.cs.unm.edu/~mueen/Tutorial/CIKM2016Tutorial.pdf) 
  >
  > Mueen's Algorithm for Similarity Search [site](http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html) 
  >
  > Querying and Mining of Time Series Data Experimental Comparison of Representations and Distance Measures
  >
  > Experimental Comparison of Representation Methods and Distance Measures for Time Series Data

+ Cluster

  > Clustering of time series data—a survey
  >
  > Clustering of Time Series Subsequences is Meaningless Implications for Previous and Future Research
  >
  > Time-series clustering – A decade review
  >
  > Dynamic Time Warping Clustering - Cross Validated
  >
  > k-Shape-Efficient and Accurate Clustering of Time Series, SIGMOD 2015, [site](<http://www.cs.columbia.edu/~jopa/kshape.html>) 

+ Classification

  > Highly comparative feature-based time-series classification

+ Anomaly Detection

  > Outlier Detection for Temporal Data: A Survey
  >
  > Outlier Detection for Temporal Data, Manish Gupta, Microsoft India and IIIT, Jing Gao, SUNY, Buffalo, Charu Aggarwal, IBM TJ Watson, Jiawei Han, UIUC - book
  >
  > https://github.com/rob-med/awesome-TS-anomaly-detection
  >
  > http://stats.stackexchange.com/questions/137094/algorithms-for-time-series-anomaly-detection
  >
  > Time Series Anomaly Detection Algorithms – Stats and Bots
  >
  > <https://github.com/twitter/AnomalyDetection>
  >
  > https://github.com/twitter/BreakoutDetection
  >
  > Anomaly Detection for Airbnb's Payment Platform - Airbnb Engineering
  >
  > <https://anomaly.io/about/index.html>
  >
  > Time-series novelty detection using one-class support vector machines
  
+ time space

  > 大佬用四句五个字来概括了这个领域的整体问题：
  > 空间不变性 
  > 空间可变性 
  > 时间不变性 
  > 时间可变性

---

### Application

+ Electricity

  Electricity price forecasting: A review of the state-of-the-art with a look into the future

  A neural network based several-hour-ahead electric load forecasting using similar days approach

  Modelling and forecasting daily electricity load curves: a hybrid approach 

  Combined modeling for electric load forecasting with adaptive particle swarm optimization

  Triple seasonal methods for short-term electricity demand forecasting

  Short-term forecasting of anomalous load using rule-based triple seasonal methods

  Rule-based autoregressive moving average models for forecasting load on special days: A case study for France

+ Business

  Business Forecasting Practical Problems and Solutions, Edited by Michael Gilliland, Len Tashman, Udo Sglavo

---

### Q&A

- <http://www.forecastingprinciples.com/index.php/faq>

- [Answers to Frequently Asked Questions ](http://qbox.wharton.upenn.edu/documents/mktg/research/FAQ.pdf) 

- Pitfalls in time series analysis - Cross Validated

- Is it unusual for the MEAN to outperform ARIMA? [site](<https://stats.stackexchange.com/questions/124955/is-it-unusual-for-the-mean-to-outperform-arima>) 

- How to know if a time series is stationary or non-stationary - Cross Validated

- When to log transform a time series before fitting an ARIMA model [site](https://stats.stackexchange.com/questions/6330/when-to-log-transform-a-time-series-before-fitting-an-arima-model) 

- Don’t Put Lagged Dependent Variables in Mixed Models [site](<https://statisticalhorizons.com/lagged-dependent-variables>) 

- Best method for short time-series [site](<https://stats.stackexchange.com/questions/135061/best-method-for-short-time-series>) 

- Estimating same model over multiple time series [site](https://stats.stackexchange.com/questions/23036/estimating-same-model-over-multiple-time-series) 

- correlating volume time series [site](http://stats.stackexchange.com/questions/26842/correlating-volume-timeseries)

- correlation between two time series [site](http://stats.stackexchange.com/questions/29096/correlation-between-two-time-series) 

- Is it possible to do time-series clustering based on curve shape? [site](http://stats.stackexchange.com/questions/3331/is-it-possible-to-do-time-series-clustering-based-on-curve-shape?noredirect=1&lq=1)

- features for time series classification [site](http://stats.stackexchange.com/questions/50807/features-for-time-series-classification)

- Modelling longitudinal data where the effect of time varies in functional form between individuals [site](<https://stats.stackexchange.com/questions/2777/modelling-longitudinal-data-where-the-effect-of-time-varies-in-functional-form-b>) 

- Why can't we use top-down methods in forecasting grouped time series? [site](<https://stats.stackexchange.com/questions/163520/why-cant-we-use-top-down-methods-in-forecasting-grouped-time-series?rq=1>) 

- Proper way of using recurrent neural network for time series analysis [site](<https://stats.stackexchange.com/questions/8000/proper-way-of-using-recurrent-neural-network-for-time-series-analysis>) 

- Does the DTW method consider the similarity in scale and time delay of two time series ?

- simple algorithm for online outlier detection of a generic time series [site](http://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series)

- outliers spotting in time series analysis should i pre-process data or not [site](http://stats.stackexchange.com/questions/22955/outliers-spotting-in-time-series-analysis-should-i-pre-process-data-or-not?noredirect=1&lq=1) 

- how to adjusting chinese new year effects [site](https://www.r-bloggers.com/adjusting-chinese-new-year-effects-in-r-is-easy/)

- how to treat holidays when working with time series data [site](http://stats.stackexchange.com/questions/18816/how-to-treat-holidays-when-working-with-time-series-data) 

- Using k-fold cross-validation for time-series model selection - Cross Validated

- Time Series Nested Cross-Validation – Towards Data Science

- Interpretation of mean absolute scaled error (MASE) - Cross Validated

---

## Papers

### Literature Review

+ 25 Years of Time Series Forecasting, Jan G De Gooijer, Rob J Hyndman
+ A review on time series data mining, Tak Chung Fu
+ A Survey on Nonparametric Time Series Analysis, Siegfried Heiler
+ Segmenting Time Series: A Survey and Novel Approach

### Paperlist

+ https://github.com/bighuang624/Time-Series-Papers

---

## Books

+ Introdcution

  https://en.wikipedia.org/wiki/Forecasting

  Statistical forecasting: notes on regression and time series analysis [site](<https://people.duke.edu/~rnau/411home.htm>) :star::star::star::star::star:

  An Introductory Study on Time Series Modeling and Forecasting

  Highly comparative time-series analysis the empirical structure of time series and their methods

+ Time Series Analysis, James D. Hamilton, Princeton University Press, 1994

+ Time Series Analysis Forecasting and Control (5th Edition), George E. P. Box

+ Principles of Forecasting: A Handbook for Researchers and Practitioners, Editors: Armstrong, J.S. (Ed.)

+ Forecasting: Principles and Practice (2ed), Rob J Hyndman and George Athanasopoulos [online](<https://otexts.com/fpp2/>) :star::star::star::star::star:

  > Notes: the ETS (Error, Trend, Seasonal) framework

+ Forecasting with Exponential Smoothing: The State Space Approach, Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. [online](<http://www.exponentialsmoothing.net/>) 

+ Analysis of Financial Time Series (3ed), Ruey S. Tsay [site](https://faculty.chicagobooth.edu/ruey.tsay/teaching/fts/) 
+ The Elements of Financial Econometrics [site](http://orfe.princeton.edu/~jqfan/fan/FinEcon.html) 
+ Nonlinear Time Series Nonparametric and Parametric Methods [site](http://orfe.princeton.edu/~jqfan/fan/nls.html)
+ Nonlinear Time Series Analysis

---

## Conference

+ <https://forecasters.org/>

---

## Scholar

+ Makridakis

  > Pioneered Empirical competition on Forecasting called M, M2 and M3, and paved way for evidence based methods in forecasting

+ J. Scott Armstrong [site](<https://marketing.wharton.upenn.edu/profile/jscott/#research>) :star::star::star::star::star:

  > Provides valuable insights in the form of books/articles on Forecasting Practice
  >
  > Simple versus complex forecasting: The evidence
  >
  > Rule-Based Forecasting: Using Judgment in Time-Series Extrapolation
  >
  > Combining Forecasts
  >
  > Standards and Practices for Forecasting
  > 

+ Kesten C. Green [site](<http://kestencgreen.com/>) [forecasting principles](<http://forecastingprinciples.com/>) [simple forecasting](<http://simple-forecasting.com/>) :star::star::star::star::star:

  > Unifying theory of forecasting:
  > The Golden Rule of Forecasting provides a unifying theory of forecasting. The Rule is to be conservative when forecasting by relying on cumulative knowledge about the situation and about forecasting. Following the Golden Rule guidelines reduces forecast errors by nearly a third, on average, compared to common practice.
  > Superiority of simple forecasting methods:
  > Sophisticatedly simple forecasting methods, which can be understood by decision makers, reduce forecast errors by nearly a quarter, on average, compared to forecasts from complex statistical methods.
  >
  > ... 
  >
  > Golden Rule of Forecasting: Be conservative
  >
  > Golden Rule of Forecasting Rearticulated: Forecast Unto Others as You Would Have Them Forecast Unto You
  >
  > ...

+ Gardner

  > Invented Damped Trend exponential smoothing another simple method which works surprisingly well vs. ARIMA

+ Eamonn Keogh [site](http://www.cs.ucr.edu/~eamonn/) [tutorials](<http://www.cs.ucr.edu/~eamonn/tutorials.html>) :star::star::star::star::star:

  > Dynamic time warping

+ Rob J Hyndman [blog](<https://robjhyndman.com/>) [github](https://github.com/robjhyndman) :star::star::star::star::star:

  > Some interesting topics on the blog: forecast intervals for aggregates (the aggregate of several time periods), fitting models to short time series, fitting models to long time series, forecasting weekly data, forecasting with daily data, forecasting with long seasonal periods, seasonal periods, rolling forecasts, batch forecasting, facts and fallacies of the AIC, cross-validation, ...

+ Tao Hong [blog](http://blog.drhongtao.com/)

  > Specialize: energy forecasting, electric load forecasting
  >
  > PhD thesis: Short Term Electric Load Forecasting
  >
  > Course: Electric Load Forecasting I: Fundamentals and Best Practices, Electric Load Forecasting II: Advanced Topics and Case Studies
  >
  > Some interesting topics on the blog: forecasting and backcasting,...
  
+ 施行建

  香港中文大学。主要研究的方向是时空序列问题，时间维度为主，并且降水预测的应用。

+ 郑宇

  JD的副总裁。主要研究时空序列数据挖掘，空间维度为主，并且用在traffic flow上的应用。主页：http://urban-computing.com/yuzheng

---

## Competitions

+ The M-competitions [dataset](http://robjhyndman.com/software/mcomp/) 
+ The Tourism Forecasting Competition [dataset](https://github.com/ellisp/Tcomp-r-package) 
+ Global Energy Forecasting Competition (GEFCom)
+ <http://www.neural-forecasting-competition.com/index.htm>

---

## STOA

+ https://www.paperswithcode.com/area/time-series

---

## Datasets

+ UCR Time Series Classification Archive
  http://www.cs.ucr.edu/~eamonn/time_series_data/

+ CompEngine

  A self-organizing database of time-series data

  <https://www.comp-engine.org/#!>

---

## Tools

- R package

  [forecast](<https://github.com/robjhyndman/forecast>), [tsfeatures](<https://github.com/robjhyndman/tsfeatures>), thief(Temporal Hierarchical Forecasting), tsDyn, ForecastCombinations, [forecastHybrid](https://github.com/ellisp/forecastHybrid), opera(Online Prediction by ExpeRt Aggregation) 

- [awesome_time_series_in_python](https://github.com/MaxBenChrist/awesome_time_series_in_python) 

- [prophet](https://github.com/facebookincubator/prophet)

- [tsfresh](https://github.com/blue-yonder/tsfresh)

- [palladium](https://github.com/ottogroup/palladium)

  Framework for setting up predictive analytics services

- [Forecast Pro](https://www.forecastpro.com/) 

- Autobox

- [TISEAN](https://www.pks.mpg.de/~tisean/Tisean_3.0.1/index.html) - Nonlinear Time Series Analysis

- Timeseries analysis for neuroscience data <http://nipy.org/nitime>

  <https://github.com/nipy/nitime>

- Tensorflow

  <https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series>

  <https://github.com/mouradmourafiq/tensorflow-lstm-regression>

- Keras

  <https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction>

  <https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline>

  <https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/>