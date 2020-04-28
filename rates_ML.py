# esessential packages
######################
import pandas as pd
import statsmodels.formula.api as smf
######################

#packages useful for plots
##########################
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
##########################

#Argument parsing
#########################
import argparse
import sys

parser = argparse.ArgumentParser(description='RPC ML rate example.')
parser.add_argument('aRun', metavar='Run', type=int,
                   help='a Run number, e.g. 306138 for training and 306139 for test')
parser.add_argument('do_fit',  metavar='Train', type=str, help='\"train\" for training run and \"predict\" for test run' )

args = parser.parse_args()
#print(args.aRun)
#print(args.do_fit)

aRun=int(args.aRun)
print(aRun)

if args.do_fit == 'train' :
    #print('train')
    fit = True
else :
    if args.do_fit == 'predict' :
        #print('predict')
        fit = False
    else :
        print("Error !!! Not valid argument:")
        print(args.do_fit)
        print("Use  \"train\" or \"predict\" ")
        sys.exit()

print("hi") 
#################################################
#306138 306139
#305814  305902

##fit = True
#fit = False

#if fit is True :
#    aRun = 306138
#else :
#    aRun = 306139

rates_directory = "./rates"

# Reading cvs files 

df_rates = pd.DataFrame()
print(("Loading %s" % aRun))

path = "%s/dt_rates_%s.csv" % (rates_directory, aRun)
df_rates = df_rates.append(pd.read_csv(path, 
                                       names=["run", "time", "board", "RPC1", "RPC2", "RPC3", "RPC4",\
                                              "DT1", "DT2", "DT3", "DT4", "DT5"]), 
                           ignore_index=True)
print("Done.")


df_rates = df_rates[df_rates['board']=="YB0_S7"]

#print(df_rates)

plt.plot(df_rates["RPC1"]/df_rates["RPC2"])
plt.show()
plt.savefig('Rates_ratio_'+str(aRun)+'_'+args.do_fit+'.pdf')
plt.close()


#X=df_rates[['DT1','DT2','DT3','DT4']]
X=df_rates[['RPC2','RPC3','RPC4']]
y=df_rates['RPC1']

if fit is True :
    results=smf.ols('RPC1 ~ RPC2 + RPC3 + RPC4', df_rates).fit()
    results.save("model.pickle")
else :
    from statsmodels.regression.linear_model import OLSResults
    results = OLSResults.load("model.pickle")

print(results.summary())
res=results.predict(X)

#print(res)
#print(y)

xy = np.vstack([y,res])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
sc = ax.scatter(y, res, c=z, s=100, edgecolor='')
plt.title('Predicted vs measured RPC rate')
plt.ylabel('Predicted rate')
plt.xlabel('Measured rate')
plt.colorbar(sc, label="density [a.u]")
plt.show()
plt.savefig('Measured_vs_predicted_scatter_'+str(aRun)+'_'+args.do_fit+'.pdf')
plt.close()

plt.hist(y-res,bins=1000)
plt.title('Measured vs predicted')
plt.ylabel('Entries')
plt.xlabel('Measured - Predicted')
plt.show()
plt.savefig('Measured_vs_predicted_'+str(aRun)+'_'+args.do_fit+'.pdf')
plt.close()

plt.hist(y-res,bins=1000)
plt.title('Measured vs predicted')
plt.ylabel('Entries')
plt.xlabel('Measured - Predicted')
plt.yscale('log', nonposy='clip')
plt.show()
plt.savefig('Measured_vs_predicted_log_'+str(aRun)+'_'+args.do_fit+'.pdf')
plt.close()

