""" Snow water equivalent estimation based on RTK-GNSS refractometry up-component bias, using Emlid Reach M2 or RTKLIB .ENU baseline solution files

Reference: Steiner et al., (Near) Real-Time Snow Water Equivalent Observation Using GNSS Refractometry and RTKLIB, submitted to Sensors, 2022.

input:  - position (.pos) file; (UTC, E, N, U)
output: - plots (SWE timeseries, DeltaSWE timeseries, scatter plots)

created: L. Steiner (Orchid ID: 0000-0002-4958-0849)
date:    8.8.2022
"""

# IMPORT modules
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


""" 1. Get Emlid Reach M2 or RTKLIB ENU solution files"""
# create empty dataframe for all .ENU solution files
df_enu = pd.DataFrame()

# Q read all .ENU files in folder, parse date and time columns to datetimeindex and add them to the dataframe
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


''' 2. Filter and clean ENU solution data '''
# read all data from .pkl and combine, if necessary multiple parts
# df_enu = pd.read_pickle('ENU/ENU.pkl')

# select only data where ambiguities are fixed (amb_state==1) and sort datetime index
fil_df = pd.DataFrame(df_enu[(df_enu.amb_state == 1)])
fil_df.index = pd.DatetimeIndex(fil_df.index)
fil_df = fil_df.sort_index()

# adapt up values to reference SWE values in mm (here adapt to first manual observation)
fil = (fil_df.U.dropna() - fil_df.U[(fil_df.index < '2021-11-06')].median()) * 1000 + 37

# remove outliers based on a 3*sigma threshold
upper_limit = fil.median() + 3 * fil.std()
lower_limit = fil.median() - 3 * fil.std()
fil_clean = fil[(fil > lower_limit) & (fil < upper_limit)]

# resample data, calculate median and standard deviation (noise) per day to fit manual reference data
m = fil_clean.resample('D').median()
s = fil_clean.resample('D').std()

# filter data with a rolling median and resample resolution to fit reference data (10min)
m_10min = fil_clean.rolling('D').mean().resample('10min').median()


''' 3. Read reference sensors .csv data '''
# read snow depth observations (hourly resolutions)
sh = pd.read_csv('ENU/laret_sh.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0])

# read manual SWE observations (weekly resolution)
manual = pd.read_csv('ENU/manual_SWE.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True)

# read automated SWE observations (10min resolution)
scale = pd.read_csv('ENU/scale_SWE.csv', header=0, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True)
scale_10min = scale.Scale.rolling('D').median()   # calculate median per 10min (filtered over one day)
scale_res = scale.Scale.resample('D').median()
scale_err = scale_res/10     # calculate 10% relative bias


""" 4. Calculate differences, linear regressions, RMSE & MRB between GNSS and reference data """
# Q: calculate differences between GNSS and reference data
dmanual = (manual.Manual - m).dropna()          # daily
dscale_daily = (scale_res - m).dropna()         # daily
dscale = (scale_10min - m_10min).dropna()       # 10min
diffs = pd.concat([dmanual, dscale], axis=1)
diffs.columns = ['Manual', 'Snow scale']

# Q: cross correlation and linear fit (daily & 10min)
# merge manual and gnss data (daily)
all_daily = pd.concat([manual.Manual, m], axis=1)
all_daily_nonan = all_daily.dropna()
# merge scale and gnss data (10min)
all_10min = pd.concat([scale_10min, m_10min], axis=1)
all_10min_nonan = all_10min.dropna()

# cross correation manual vs. GNSS (daily)
corr_daily = all_daily.Manual.corr(all_daily.U)
print('\nPearsons correlation (manual vs. GNSS, daily): %.2f' % corr_daily)
# calculate cross correation scale vs. GNSS (10min)
corr_10min = all_10min.Scale.corr(all_10min.U)
print('Pearsons correlation (scale vs. GNSS, 10min): %.2f' % corr_10min)

# fit linear regression curve manual vs. GNSS (daily)
fit_daily = np.polyfit(all_daily_nonan.Manual, all_daily_nonan.U, 1)
predict_daily = np.poly1d(fit_daily)
print('\nLinear fit (manual vs. GNSS, daily): \nm = ', round(fit_daily[0], 2), '\nb = ', int(fit_daily[1]))
# fit linear regression curve scale vs. GNSS (10min)
fit_10min = np.polyfit(all_10min_nonan.Scale, all_10min_nonan.U, 1)
predict_10min = np.poly1d(fit_10min)
print('Linear fit (scale vs. GNSS, 10min): \nm = ', round(fit_10min[0], 2), '\nb = ', int(fit_10min[1]))     # n=12, m=1.02, b=-8 mm w.e.

# RMSE
rmse_manual = np.sqrt((np.sum(dmanual**2))/len(dmanual))
print('\nRMSE (manual vs. GNSS, daily): %.1f' % rmse_manual)
rmse_scale = np.sqrt((np.sum(dscale**2))/len(dscale))
print('RMSE (scale vs. GNSS, 10min): %.1f' % rmse_scale)

# MRB
mrb_manual = (dmanual/all_daily['Manual']).mean() * 100
print('\nMRB (manual vs. GNSS, daily): %.1f' % mrb_manual)
mrb_scale = (dscale/all_10min['Scale']).mean() * 100
print('MRB (scale vs. GNSS, 10min): %.1f' % mrb_scale)

# Number of samples
n_manual = len(dmanual)
print('\nNumber of samples (manual vs. GNSS, daily): %.0f' % n_manual)
n_scale = len(dscale)
print('Number of samples (scale vs. GNSS, 10min): %.0f' % n_scale)


''' 5. Plot results (SWE, ΔSWE, scatter) '''
# Q: plot SWE
plt.figure()
ax = scale_res.plot(linestyle='--', color='steelblue', fontsize=12, figsize=(6, 5.5), ylim=(0, 600))
ax2 = ax.twinx()
ax2.plot(sh.rolling('D').median()/10, color='darkgrey') # plot snow depth on right axis

manual.Manual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2, ax=ax)
ax.errorbar(manual.index, manual.Manual, yerr=manual.Manual/10, linestyle=' ', color='k', capsize=4, alpha=0.5)
m.plot(color='crimson', ax=ax)
ax.fill_between(m.index, m - s, m + s, color="crimson", alpha=0.15)
ax.fill_between(scale_res.index, scale_res - scale_err, scale_res + scale_err, color="steelblue", alpha=0.1)

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
ax2.set_ylim(0, 180)
ax2.set_ylabel('Snow depth (cm)', color='darkgrey', fontsize=14)
plt.show()
# plt.savefig('ENU/SWE_SH_Laret.png', bbox_inches='tight')
# plt.savefig('ENU/SWE_SH_Laret.pdf', bbox_inches='tight')


# Q. plot SWE difference
plt.close()
plt.figure()
dscale_daily.plot(color='steelblue', linestyle='--', fontsize=14, figsize=(6, 5.5), ylim=(-200, 200)).grid()
dmanual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2).grid()
plt.xlabel(None)
plt.ylabel('ΔSWE (mm w.e.)', fontsize=14)
plt.legend(['Snow scale', 'Manual'], fontsize=14, loc='upper left')
plt.xlim([dt.date(2021, 11, 1), dt.date(2022, 5, 1)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# plt.savefig('plots/diff_SWE_Laret.png', bbox_inches='tight')
# plt.savefig('plots/diff_SWE_Laret.pdf', bbox_inches='tight')


# Q: plot scatter plot (GNSS vs. manual, daily)
plt.close()
plt.figure()
ax = all_daily.plot.scatter(x='Manual', y='U', figsize=(5, 4.5))
plt.plot(range(10, 450), predict_daily(range(10, 450)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 600)
ax.set_xlim(0, 600)
ax.set_xlabel('Manual SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr_daily], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('plots/scatter_SWE_Laret_manual.png', bbox_inches='tight')
# plt.savefig('plots/scatter_SWE_Laret_manual.pdf', bbox_inches='tight')


# Q: plot scatter plot (GNSS vs. scale, 10min)
plt.close()
plt.figure()
ax = all_10min.plot.scatter(x='Scale', y='U', figsize=(5, 4.5))
plt.plot(range(10, 550), predict_10min(range(10, 550)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 600)
ax.set_xlim(0, 600)
ax.set_xlabel('Snow scale SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr_10min], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('plots/scatter_SWE_Laret_scale_10min.png', bbox_inches='tight')
# plt.savefig('plots/scatter_SWE_Laret_scale_10min.pdf', bbox_inches='tight')

# Q: plot boxplot of differences
dscale.describe()
dmanual.describe()
diffs[['Manual', 'Snow scale']].plot.box(ylim=(-100, 200), figsize=(3, 4.5), fontsize=12)
plt.grid()
plt.ylabel('ΔSWE (mm w.e.)', fontsize=12)
plt.show()
# plt.savefig('plots/box_SWE_WFJ_diff.png', bbox_inches='tight')
# plt.savefig('plots/box_SWE_WFJ_diff.pdf', bbox_inches='tight')
