import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pickle
import glob

''' import Emlid Reach M2 .ENU files '''
# create empty dataframe for all .ENU files
df_enu = pd.DataFrame()

# read all .ENU files in folder, parse date and time columns to datetimeindex and add them to the dataframe
for file in glob.iglob('ENU/*.ENU', recursive=True):
    print(file)
    enu = pd.read_csv(file, header=None, delimiter=' ', index_col=['date_time'], usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                skipinitialspace=True, skiprows=0, na_values=["NaN"],
                names=['time', 'date', 'E', 'N', 'U', 'amb_state', 'nr_sat', 'std_e', 'std_n', 'std_u'],
                parse_dates=[['date', 'time']]
                )[['E', 'N', 'U', 'amb_state', 'nr_sat', 'std_e', 'std_n', 'std_u']]
    df_enu = pd.concat([df_enu, enu], axis=0)

# store dataframe as binary pickle format
df_enu.to_pickle('ENU/ENU.pkl')


''' Read binary stored ENU data '''
# read all data from .pkl and combine, if necessary multiple parts
df_enu = pd.read_pickle('ENU/ENU.pkl')

# select data where ambiguities are fixed (amb_state==1)
fil_df = pd.DataFrame(df_enu[(df_enu.amb_state == 1)])
fil_df.index = pd.DatetimeIndex(fil_df.index)
fil = (fil_df.U.dropna() + 2.8) * 1000 + 28  # adapt to reference SWE values in mm

# remove outliers
upper_limit = fil.median() + 3 * fil.std()
lower_limit = fil.median() - 3 * fil.std()
fil_clean = fil[(fil > lower_limit) & (fil < upper_limit)]

# calculate median (per day and 10min) and std (per day)
m = fil_clean.resample('D').median()
s = fil_clean.resample('D').std()
m_10min = fil_clean[1:-1].rolling('D').mean().resample('10min').median()


''' Read reference .csv data '''
# read snow height observations
sh = pd.read_csv('ENU/laret_sh.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0])

# read manual observations
manual = pd.read_csv('ENU/manual_SWE.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True)

# read snow scale observations (10 min resolution)
scale = pd.read_csv('ENU/scale_SWE.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True)

# calculate median (per day and 10min) and relative bias (per day)
scale_res = scale.resample('D').median()
scale_err = scale_res/10     # calculate 10% relative bias
scale_10min = scale.rolling('D').median()   # calculate median per 10min (filtered over one day)

# combine daily manual and resampled scale observations in one dataframe
ref = pd.concat([scale_res, manual], axis=1)

# combine reference and GNSS data
all_daily = pd.concat([ref, m[:-1]], axis=1)
all_10min = pd.concat([scale_10min, m_10min], axis=1)


''' Plot results (SWE, ΔSWE, scatter) '''
low = all_daily.Scale - all_daily.Scale/10
up = all_daily.Scale + all_daily.Scale/10


# plot SWE
plt.figure()
ax = all_daily.Scale.plot(linestyle='--', color='steelblue', fontsize=12, figsize=(6, 5.5), ylim=(0, 600))
ax2 = ax.twinx()
ax2.plot(sh.rolling('D').median()/10, color='darkgrey') # plot snow depth on right axis

all_daily.Manual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2, ax=ax)
ax.errorbar(ref['Manual'].index, ref['Manual'], yerr=ref['Manual']/10, color='k', capsize=4, alpha=0.5)
all_daily.U.plot(color='crimson', ylim=(0, 600), ax=ax)
ax.fill_between(s.index, m - s, m + s, color="crimson", alpha=0.15)
ax.fill_between(all_daily.Scale.index, low, up, color="steelblue", alpha=0.1)

# set left axis limits, labels, params, legends
ax.set_xlim([dt.date(2021, 11, 1), dt.date(2022, 5, 1)])
ax.set_ylim(0, 600) # 1/4 of scale of right y-axes, for sharing 0 and scale
ax.set_xlabel(None)
ax.set_ylabel('SWE (mm w.e.)', color='k', fontsize=14)
ax.grid(True)
ax.tick_params(axis="y", colors='k', labelsize=12)
ax.legend(['Snow scale', 'Manual', 'GNSS'], fontsize=14, loc='upper left')

# set right axis limits, labels, params, no legend
ax2.tick_params(colors='darkgrey', labelsize=12)
ax2.set_xlim([dt.date(2021, 11, 1), dt.date(2022, 5, 1)])
ax2.set_ylim(0,180)
ax2.set_ylabel('Snow depth (cm)', color='darkgrey', fontsize=14)
#plt.show()
plt.savefig('ENU/SWE_SH_Laret.png', bbox_inches='tight')
plt.savefig('ENU/SWE_SH_Laret.pdf', bbox_inches='tight')


# plot SWE difference
plt.close()
plt.figure()
all_daily.Scale.plot(linestyle='--', color='steelblue', fontsize=12, figsize=(6, 5.5), ylim=(0, 600)).grid()
all_daily.Manual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2)
plt.errorbar(ref['Manual'].index, ref['Manual'], yerr=ref['Manual']/10, color='k', capsize=4, alpha=0.5)
all_daily.U.plot(color='crimson', ylim=(0, 600)).grid()
plt.fill_between(s.index, m - s, m + s, color="crimson", alpha=0.1)
plt.fill_between(all_daily.Scale.index, low, up, color="steelblue", alpha=0.1)
plt.xlabel(None)
plt.ylabel('SWE (mm w.e.)', fontsize=14)
plt.legend(['Snow scale', 'Manual', 'GNSS'], fontsize=14, loc='upper left')
plt.xlim(dt.datetime.strptime('2021-11-01', "%Y-%m-%d"), dt.datetime.strptime('2022-05-01', "%Y-%m-%d"))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.savefig('ENU/SWE_Laret.png', bbox_inches='tight')
plt.savefig('ENU/SWE_Laret.pdf', bbox_inches='tight')
plt.figure()
(all_daily.Scale-all_daily.U).plot(linestyle='--', color='steelblue', fontsize=14, figsize=(6, 5.5), ylim=(-200, 200)).grid()
(all_daily.Manual-all_daily.U).plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2).grid()
plt.xlabel(None)
plt.ylabel('ΔSWE (mm w.e.)', fontsize=14)
plt.legend(['Snow scale', 'Manual'], fontsize=14, loc='upper left')
plt.xlim(dt.datetime.strptime('2021-11-01', "%Y-%m-%d"), dt.datetime.strptime('2022-05-01', "%Y-%m-%d"))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.savefig('ENU/diff_SWE_Laret.png', bbox_inches='tight')
plt.savefig('ENU/diff_SWE_Laret.pdf', bbox_inches='tight')

# fit linear regression curve manual vs. GNSS (daily)
all_daily_nonan = all_daily.dropna()
fit = np.polyfit(all_daily_nonan['Manual'], all_daily_nonan['U'], 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=12, m=1.02, b=-8 mm w.e.

# calculate cross correation manual vs. GNSS (daily)
corr = all_daily['Manual'].corr(all_daily['U'])
print('Pearsons correlation: %.2f' % corr)

# plot scatter plot (GNSS vs. manual, daily)
plt.close()
plt.figure()
ax = all_daily.plot.scatter(x='Manual', y='U', figsize=(5, 4.5))
plt.plot(range(10, 450), predict(range(10, 450)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 600)
ax.set_xlim(0, 600)
ax.set_xlabel('Manual SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('ENU/scatter_SWE_manual.png', bbox_inches='tight')
# plt.savefig('ENU/scatter_SWE_manual.pdf', bbox_inches='tight')


# fit linear regression curve scale vs. GNSS (10min)
all_10min_nonan = all_10min.dropna()
fit = np.polyfit(all_10min_nonan['Scale'], all_10min_nonan['U'], 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=11526, m=0.99, b=-36 mm w.e.

# calculate cross correation scale vs. GNSS (10min)
corr = all_10min['Scale'].corr(all_10min['U'])
print('Pearsons correlation: %.2f' % corr)


# plot scatter plot (GNSS vs. scale, 10min data)
plt.close()
plt.figure()
ax = all_10min.plot.scatter(x='Scale', y='U', figsize=(5, 4.5))
plt.plot(range(10, 550), predict(range(10, 550)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 600)
ax.set_xlim(0, 600)
ax.set_xlabel('Snow scale SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('ENU/scatter_SWE_scale_10min.png', bbox_inches='tight')
# plt.savefig('ENU/scatter_SWE_scale_10min.pdf', bbox_inches='tight')


# fit linear regression curve scale vs. GNSS (daily)
all_daily_nonan = all_daily.dropna()
fit = np.polyfit(all_daily_nonan['Scale'], all_daily_nonan['U'], 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=81, m=1, b=-40 mm w.e.

# calculate cross correation scale vs. GNSS (daily)
corr = all_daily['Scale'].corr(all_daily['U'])
print('Pearsons correlation: %.2f' % corr)


# plot scatter plot (GNSS vs. scale, daily data)
plt.close()
plt.figure()
ax = all_daily.plot.scatter(x='Scale', y='U', figsize=(5, 4.5))
plt.plot(range(10, 550), predict(range(10, 550)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 600)
ax.set_xlim(0, 600)
ax.set_xlabel('Snow scale SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('ENU/scatter_SWE_scale_daily.png', bbox_inches='tight')
# plt.savefig('ENU/scatter_SWE_scale_daily.pdf', bbox_inches='tight')

# calculate differences
dscale = (all_10min.Scale - all_10min.U).dropna()
dmanual = (all_daily.Manual-all_daily.U).dropna()
diffs = pd.concat([dmanual, dscale], axis=1)
diffs.columns = ['Manual', 'Snow scale']

# plot boxplot of differences
dscale.describe()
dmanual.describe()
diffs[['Manual', 'Snow scale']].plot.box(ylim=(-50, 200), figsize=(3, 4.5), fontsize=12)
plt.grid()
plt.ylabel('ΔSWE (mm w.e.)', fontsize=12)
plt.show()
plt.savefig('ENU/box_SWE_diff.png', bbox_inches='tight')
plt.savefig('ENU/box_SWE_diff.pdf', bbox_inches='tight')

# plot histogram of differences
diffs[['Snow scale daily', 'Manual']].plot.hist(bins=25, xlim=(-200, 200), figsize=(3, 4.5), fontsize=12, alpha=0.8)
plt.grid()
plt.xlabel('ΔSWE (mm w.e.)', fontsize=12)
plt.legend(loc='upper left')
plt.show()
plt.savefig('ENU/hist_SWE_diff.png', bbox_inches='tight')
plt.savefig('ENU/hist_SWE_diff.pdf', bbox_inches='tight')

''' calculate RMSE and MRB '''
# RMSE
rmse_manual = np.sqrt((np.sum(dmanual**2))/len(dmanual))
rmse_scale = np.sqrt((np.sum(dscale**2))/len(dscale))

# MRB
mrb_manual = (dmanual/all_daily.Manual).mean() * 100
mrb_scale = (dscale/all_daily.Scale).mean() * 100