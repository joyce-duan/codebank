"""
analyze credit dataset from kaggle & yhat/kdnugget tutorial

works for data set with reasonable number of variables (not too large) as chartsare used for data exploration.

to-do

8.  make the x-axis for run_binning more meaningful, like hist()
6.  add feature correction: i.e. debt_ratio??? log_scale???
importance of variables
6.  add variable selection for imputation (k=1)
12.  kdnuget winning solution 
11. profiling: one of the method is super slow
test split df into x & y then merge back
8  refactor run_binning make plot a function
9.  explor relationship between 2 variable
                 5.  correlation matrix plot
7.  use feature selection
8.  logistic regression???
10.  explore interactions
11.  add legend for secondary y axis

13.  trellis plot
imputation knn k = 1:
  feature selection:  
  knn =1
??? %d


"""

import random
import sys
import pylab 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsRegressor
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import roc_curve, auc

sys.path.append('../../util')
import myutil

def run_binning(x_num, y_binary, w=2, nbins = 20):
   """ 
   run binning of numerical x vs. binary y

   plot probablity_density (%counts) and mean
   """
   cols = x_num.columns
   yname = y_binary.name
   nplots = len(cols)+1
   nrows = nplots // w
   if nplots % w > 0:
      nrows +=1
   i = 0
   print "\n"
   fig1, ax = plt.subplots(nrows, w, figsize=(15,15))
#   print " x {len_x}, index_x {ind_x}  y {len_y} index_y {ind_y} ".format(len_x = len(x_num), len_y = len(y_binary), ind_x = x_num.index.max(), ind_y = y_binary.index.max())
   #df3 = pd.merge(x_num, y_binary, left_index=True, right_index=True,sort=False) # why does this not work??
   tmp_y = np.array([y_binary.values]).T
   tmp_x = x_num.values

   t2 = np.concatenate((tmp_x,tmp_y), axis = 1)
   cols_new = np.append( xnames.values, yname)
   df3 = pd.DataFrame(t2, columns=cols_new)
   for irow in range(nrows):
      for icol in range(w):
        if i >= len(cols)+1:
           break
        else:
#           print i, cols[i-1], irow, icol, nbins
           if i == 0:
              prob_den = pd.value_counts(y_binary)/len(y_binary)
              ax[irow, icol].bar(prob_den.index, prob_den, label ='prob. density',color='lightgray')
              ax[irow, icol].set_title(yname) 
           else:
              nbins_this =  nbins
              if df3[cols[i-1]].nunique() < nbins:
                  nbins_this = df3[cols[i-1]].nunique()
              df3['bins']=pd.cut(df3[cols[i-1]], nbins_this, labels=False)
#              df3['bins']=pd.cut(df3[cols[i-1]], nbins_this)  
              prob_den = pd.value_counts(df3['bins'])/len(x_num)
              ymean = df3.groupby('bins')[yname].mean()
              ax[irow, icol].bar(prob_den.index, prob_den,label='prob. density',color='lightgray' )
              ax2 = ax[irow,icol].twinx()
#              ax[irow,icol].plot( ymean, label = '%delinquency',color='r' )
              ax2.plot(ymean, label = '%delinquency',color='r' )
              ax[irow, icol].set_title(cols[i-1])
              print df3[cols[i-1]].describe(), "\n","nbins ", nbins_this,"\n"
              if i == 1:
                 ax[irow,icol].legend(loc='upper right')
           i +=1
   plt.tight_layout()
   plt.show()
   fig1.savefig('bining.png', bbox_inches='tight')
   #fig1.show() 


def run_analysis_split_iloc(x,y,clfs,clfnames,run_title, run_des, train_iloc, test_iloc, pct_test =0.25):
   """
   run all the classification methods in clfs, plot roc and return AUC

   traing set, test set specified by train_iloc, test_iloc
   """
   run_name = run_title
   print "-"*60+"\n",run_name+ ": "+run_des
   print count_null(x)
   print "y: ", count_null_s(y)


   print 'training: {len_train}, test: {len_test}, x: {len_x}, y: {len_y}'.format(len_train=len(train_iloc), len_test=len(test_iloc), len_x = len(x), len_y = len(y))

   x_train = x.iloc[train_iloc]
   y_train = y.iloc[train_iloc]
   x_test = x.iloc[test_iloc]
   y_test = y.iloc[test_iloc]
   return run_analysis_split(x_train, y_train, x_test, y_test, clfs, clfnames, run_title, run_des)
   """
   # check  if can remove the block below?????
   auc_s =[]
   f = plt.figure()
   for i, clf in enumerate(clfs):
      print clfnames[i]
      clf.fit(x.iloc[train_iloc,:],y.iloc[train_iloc])
      y_score = clf.predict_proba(x.iloc[test_iloc,:])
      fpr, tpr, t = roc_curve(y.iloc[test_iloc],y_score[:,1])      
      roc_auc = auc(fpr,tpr)
      plt.plot(fpr,tpr, label=clfnames[i]+' auc %0.2f' % roc_auc)
      plt.plot([0,1],[0,1],'k--')
      auc_s.append(roc_auc)
   plt.legend(loc='lower right')
   plt.xlabel('false positive rate')
   plt.ylabel('true positive rate')
   plt.title('ROC'+run_title)
   plt.grid()
   plt.show()
   return auc_s
   """

def run_analysis_split(x_train,y_train, x_test, y_test, clfs,clfnames,run_title, run_des):
   """
   run all the classification methods in clfs, plot roc and return AUC

   input:  traing set, test set dataframe
   """
   run_name = run_title
   print "-"*60+"\n",run_name+ ": "+run_des
   print count_null(x_train)
   print "y: ", count_null_s(y_train)

   print 'training: {len_train}, test: {len_test}'.format(len_train=len(x_train), len_test=len(x_test))
   auc_s =[]
   f = plt.figure()
   for i, clf in enumerate(clfs):
      print clfnames[i]
      clf.fit(x_train,y_train)
      y_score = clf.predict_proba(x_test)
      fpr, tpr, t = roc_curve(y_test,y_score[:,1])      
      roc_auc = auc(fpr,tpr)
      plt.plot(fpr,tpr, label=clfnames[i]+' auc %0.2f' % roc_auc)
      plt.plot([0,1],[0,1],'k--')
      auc_s.append(roc_auc)
   plt.legend(loc='lower right')
   plt.xlabel('false positive rate')
   plt.ylabel('true positive rate')
   plt.title('ROC'+run_title)
   plt.grid()
   plt.show()
   return auc_s




def hists(df, cols, w=2, nbins = 20):
   """
   plot historgram of a list of continuous variables

   """
   nplots = len(cols)
   nrows = nplots // w
   if nplots % w > 0:
      nrows +=1
   i = 0
   print 'hists: ', nplots, nrows, w
   fig1, ax = plt.subplots(nrows, w, figsize=(15,15))
   for irow in range(nrows):
      for icol in range(w):
        if i >= len(cols):
           break
        else:
           print i, cols[i], irow, icol, nbins
           ax[irow,icol].hist(df[cols[i]].dropna(),nbins)
#           ax[irow,icol].hist(df[cols[i]], nbins)
           ax[irow,icol].set_title(cols[i])
        i += 1
#   fig1.show()
   fig1.savefig('hist_capped.png',bbox_inches='tight')
   plt.show()

def count_null_s(s):
    null_var = s.isnull()
    return pd.value_counts(null_var)

def count_null(df):
   df_lng = pd.melt(df)
   null_var = df_lng.value.isnull()
   return pd.crosstab(df_lng.variable, null_var)

def cap_value(x,cap):
    return cap if x > cap else x

def cap_df(df, xname, pct = 0.999, postfix=''):
   """
   cap specified columns xname to pctile
   """

   df2 = pd.DataFrame()
   # postfix = '_capped'
   for c in xname:
     cap_at = df[c].quantile(pct)
     df2[c+postfix]=df[c].apply(lambda x: cap_value(x, cap_at))
   return df2

def impute_knn(x,y,iloc_train, k=1):
   """
   use knn (k=3) to impute missing value y

   need to check x does not have missing value??????
   """



   arr = np.arange(len(y))
   mask = y.iloc[arr].isnull()
   iloc_notmissing = arr[mask.values==False]

   """
   #   segmentation fault???????
   mask_train = y.iloc[iloc_train].isnull()

   print "len y:", len(y), "max iloc:", max(iloc_train)
   print "type arr", type(arr), " iloc_train ", type(iloc_train)
   print "type masktrain ", type(mask_train), " type mask ", type(mask)

   iloc_train_notmissing = np.array(iloc_train)[mask_train.values==False]

   print "max: ", max(iloc_train_notmissing)

   knn_clf = KNeighborsClassifier(n_neighbors=k, warn_on_equidistance=False)
   knn_clf.fit(x.iloc[iloc_train_notmissing], y.iloc[iloc_train_notmissing])
   y_impu = knn.predict(x)
   y_impu.iloc[iloc_notmissing]=y.iloc[iloc_notmissing]

   """
   yname = y.name
   xnames = x.columns
   tmp_y = np.array([y.iloc[iloc_train].values]).T
   t2 = np.concatenate((x.iloc[iloc_train].values, tmp_y), axis=1)
   cols_new = np.append(xnames.values, yname)
   df_new = pd.DataFrame(t2, columns=cols_new)
   print 'cols new: ', cols_new
   print 'df_new shape:',df_new.shape, 'cols_new shape:',cols_new.shape
#   df = pd.merge(x.iloc[iloc_train],y.iloc[iloc_train])
   df2 = df_new.dropna(axis=0)
   print 'df2 shape', df2.shape
   print df2.ix[:5, xnames.values]
   print '---------------y\n'
   print df2.ix[:5, yname]

   """
   clf = RandomForestClassifier(compute_importances=True)
#   clf.fit(df2[xnames.values], df2[yname])
   importances = clf.feature_importances_
   sorted_idx = np.argsort(importances)
   print importances[sorted_idx]
   """
   # ????????????? how to select this cols_imput
   cols_imput = ['number_real_estate_loans_or_lines', 'number_of_open_credit_lines_and_loans']
   knn_imput = KNeighborsRegressor(n_neighbors=k)
   knn_imput.fit(df2[cols_imput], df2[yname])
   y_impu = knn_imput.predict(x[cols_imput])
   y_after_impu = np.where(y.isnull(), y_impu, y)
#   y_impu.iloc[iloc_notmissing]=y.iloc[iloc_notmissing]

#   y_impu = []

   return y_impu

#__main__ 
if 1 == 1:

    datafname = 'data/credit-training.csv'

    flag_testrun = 0  #debug for testrun, use 5% of data; when this flag is on, hist does not work??? bug
    pct_testrun = 0.4
    pct_test = 0.25

    df = pd.read_csv(datafname)
    df.columns = [myutil.camel_to_snake(c) for c in df.columns]

    yname = df.columns[0]
    xnames = df.columns[1:]

    if flag_testrun == 1:
       coin = np.random.rand(len(df))
       df = df[coin<=pct_testrun]
    print "shape: ", df.shape
    print count_null(df)

    #summary of all variables
    df_sum= df.describe()
    for c in df_sum.columns:
      print "----\n", c
      print df_sum.ix[:, c]


    """
    """
    # explore features: historgram before and after capping
    hists(df,df.columns[1:])


    df_x_capped = cap_df(df,xnames, pct = 0.95)
    print "\n", "historgram after capping"
    hists(df_x_capped, df_x_capped.columns)


    df_x_capped = cap_df(df,xnames,pct =0.95)
    run_binning(df_x_capped, df[yname])
#    run_binning(df[xnames], df[yname])

    samp_rows = random.sample(df.index, 5000)
    f=scatter_matrix(df_x_capped.loc[samp_rows], alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()
    plt.savefig('scatter_matrix.png')

    irun = 0
    all_auc = []

    #classification
    clfs = [RandomForestClassifier()
         , GradientBoostingClassifier()
 #          ,  KNeighborsClassifier(n_neighbors=13, warn_on_equidistant=False)
 #        ,  SVC(probability=True)
        ]
    clfnames = ['RF','GBC'
#         , 'KNN 13'
#         , 'SVN' 
    ]





    irun +=1
    run_name = 'run'+str(irun)+ ' drop missing values'
    run_des = 'no transformation'
    df2 = df.dropna(axis=0)
    train_iloc, test_iloc = train_test_split(range(len(df2)),test_size=pct_test)
    all_auc.append([run_name,run_des] + run_analysis_split_iloc(df2[xnames],df2[yname], clfs, clfnames,run_name, run_des, train_iloc, test_iloc))
    del df2
    print all_auc


    irun +=1
    run_name ='run'+str(irun)+ ' fill missing values'
    run_des = '0 or mean'

    df2 = df[df.columns]
    train_iloc, test_iloc = train_test_split(range(len(df2)),test_size=pct_test)
    df2['number_of_dependents']=df2['number_of_dependents'].fillna(0)
    df2['monthly_income']=df2['monthly_income'].fillna(df2.monthly_income.iloc[train_iloc].mean())

    all_auc.append([run_name,run_des] + run_analysis_split_iloc(df2[xnames],df2[yname], clfs, clfnames,run_name, run_des, train_iloc, test_iloc))
    del df2


    irun +=1
    run_name ='run'+str(irun)+ ' fill missing values; capping'
    run_des = '0 or mean'

    df2 = df[df.columns]
    train_iloc, test_iloc = train_test_split(range(len(df2)),test_size=pct_test)

    df2['number_of_dependents']=df2['number_of_dependents'].fillna(0)
    df2['monthly_income']=df2['monthly_income'].fillna(df2.monthly_income.iloc[train_iloc].mean())
    df2_x_capped = cap_df(df2, xnames)
    all_auc.append([run_name,run_des] + run_analysis_split_iloc(df2_x_capped,df2[yname], clfs, clfnames,run_name, run_des, train_iloc, test_iloc))
    del df2


    irun +=1
    run_name ='run'+str(irun)+ ' impute missing value; capping'
    k_knn=3
    run_des = 'knn k='+str(k_knn)

    df2 = df[df.columns]
    train_iloc, test_iloc = train_test_split(range(len(df2)),test_size=pct_test)

    df2['number_of_dependents']=df2['number_of_dependents'].fillna(0)

    x = df[xnames[xnames <>'monthly_income']]
    y = df['monthly_income']
    df2['monthly_income'] = impute_knn(x, y, train_iloc,k_knn)

    df2_x_capped = cap_df(df2, xnames)
    all_auc.append([run_name,run_des] + run_analysis_split_iloc(df2_x_capped,df2[yname], clfs, clfnames,run_name, run_des, train_iloc, test_iloc))
    del df2


    print all_auc

