#%%
# Functions
import numpy as np
import matplotlib.pyplot as plt

def availability(MTTR, MTBF):
    return MTBF/(MTTR + MTBF)

def determineProduction(dayCount, demandCurve, nominalDemand):
    if dayCount == 0:
        prevDemand = nominalDemand
        prevprevDemand = nominalDemand
    elif dayCount == 1:
        prevDemand = demandCurve[dayCount-1]
        prevprevDemand = nominalDemand
    else:
        prevDemand = demandCurve[dayCount-1]
        prevprevDemand = demandCurve[dayCount-2]
    
    gradient = prevDemand - prevprevDemand
    prediction = prevDemand + gradient

    return prediction

def determineTTB(batchSize, hourlyRate):
    return batchSize/hourlyRate

def determineStartOrder(IM_hourly_rate, AA_hourly_rate, batchSize):
    IM_ttb = determineTTB(batchSize, IM_hourly_rate)
    AA_ttb = determineTTB(batchSize, AA_hourly_rate)
    startUpDelay = IM_ttb - AA_ttb

    return startUpDelay

#%%
# Parameters
# Global
shiftsPerDay   = 3
shiftLength    = 7.5   # hours
nominalDemand  = 81116 # units/day
surgeMagnitude = 0.46
onTimeTarget   = 0.95
preEObuffer    = 7.1   # hours
postEObuffer   = 12    # hours
surgeDemand    = nominalDemand * (1+surgeMagnitude)

# Injection Moulding
IM_count  = 5
IM_time   = 20.7 # s/part
IM_FPY    = 0.9962
IM_scrap  = 0.0059
IM_barrel_cavities  = [24, 16, 16]
IM_plunger_cavities = [32, 24,  0]

# Automated Assembly
AA_count = 3
AA_time  = 0.54 # s/part
AA_FPY   = 0.9924
AA_rework         = 71 # s
AA_rework_success = 0.915
AA_rework_cost    = 17.13 # £/hour

# Visual Inspection
VI_count = 2
VI_time  = 0.57 # s/part

# EO Batching
EO_count = 1
EO_time  = 246.9       # min/batch
EO_batch = 41_631      # units
EO_quarantine = 12     # hours
EO_batch_cost = 202.73 # £/batch

# Packing
CRP_count = 11
CRP_FPY   = 0.9964

# Costs
scrapCost        = 0.07   # £/unit
packMaterialCost = 0.06   # £/unit
EOCost           = 202.73 # £/batch
labourCost       = 15.7   # £/hour
reworklabourCost = 17.13  # £/hour
qualityEscapes   = 11_187 # £/million_units

# Reliability
IM_MTBF = 634.8 # min
IM_MTTR =  10.4 # min
AA_MTBF = 337.2 # min
AA_MTTR =  12.9 # min
VI_MTBF = 630.8 # min
VI_MTTR =   8.9 # min
EO_MTBF = 867.2 # min
EO_MTTR =  56.4 # min
CRP_MTBF =439.0 # min
CRP_MTTR = 14.8 # min

IM_availability  = availability(IM_MTTR, IM_MTBF)
AA_availability  = availability(AA_MTTR, AA_MTBF)
VI_availability  = availability(VI_MTTR, VI_MTBF)
EO_availability  = availability(EO_MTTR, EO_MTBF)
CRP_availability = availability(CRP_MTTR, CRP_MTBF)

#%%
# Calculated Parameters
# Global

# Injection Moulding
IM_barrel_hourly_rates  = []
IM_plunger_hourly_rates = []
IM_hourly_scrap_rates   = []
for i in range(len(IM_barrel_cavities)):
    barrelCavity  = IM_barrel_cavities[i]
    plungerCavity = IM_plunger_cavities[i]

    barrelRate  = (1 - IM_scrap) * IM_availability * ((3600 * barrelCavity)  / IM_time)
    plungerRate = (1 - IM_scrap) * IM_availability * ((3600 * plungerCavity) / IM_time)

    IM_barrel_hourly_rates.append(barrelRate)
    IM_plunger_hourly_rates.append(plungerRate)

IM_syringe_hourly_rate = min(sum(IM_barrel_hourly_rates), sum(IM_plunger_hourly_rates))
IM_ideal_hourly_rate   = IM_syringe_hourly_rate / ((1-IM_scrap) * IM_availability)
IM_scrap_hourly_rate   = IM_ideal_hourly_rate * IM_scrap
IM_hourly_scrap_cost   = IM_scrap_hourly_rate * scrapCost

print(f'Injection Moulding (x{IM_count})')
print('The chosen (barrel) cavity sizes are: ', IM_barrel_cavities)
print('The chosen (plunger) cavity sizes are:', IM_plunger_cavities)
print('Barrel Hourly Rates:', IM_barrel_hourly_rates, 'units/hour')
print('Plunger Hourly Rates:', IM_plunger_hourly_rates, 'units/hours')
print('Syringe Hourly Rate:', round(IM_syringe_hourly_rate), 'units/hour')
print('Syringe Ideal Hourly Rate:', round(IM_ideal_hourly_rate), 'units/hour')
print('Scrap Hourly Rate:', round(IM_scrap_hourly_rate), 'units/hour')
print('Hourly Scrap Cost:', round(IM_hourly_scrap_cost), '£/hour')
print()

# Automated Assembly
AA_hourly_rates = []
for i in range(AA_count):
    rate = AA_FPY * AA_availability * (3600 / AA_time)
    AA_hourly_rates.append(rate)

AA_syringe_hourly_rate = sum(AA_hourly_rates)
AA_ideal_syringe_hourly_rate = AA_syringe_hourly_rate / (AA_FPY * AA_availability)
AA_defect_hourly_rate = AA_ideal_syringe_hourly_rate * (1 - AA_FPY)
AA_hourly_rework_rate = (3600*AA_count) / AA_rework
AA_hourly_hidden_defects = AA_hourly_rework_rate * (1-AA_rework_success)
AA_hourly_pass_on_rate = AA_syringe_hourly_rate + AA_hourly_rework_rate
AA_hourly_rework_cost  = AA_count * AA_rework_cost

print(f'Automated Assembly (x{AA_count})')
print('Hourly Rates:', AA_hourly_rates, 'units/hour')
print('Syringe Hourly Rate (No Defect):', round(AA_syringe_hourly_rate), 'units/hour')
print('Ideal Syringe Hourly Rate:', round(AA_ideal_syringe_hourly_rate), 'units/hour')
print('Defect Hourly Rate:', round(AA_defect_hourly_rate), 'units/hour')
print('Hourly Rework Rate:', round(AA_hourly_rework_rate), 'units/hour (with 3 rework stations, can rework all defects)')
print('Hourly Hidden Defects:', round(AA_hourly_hidden_defects), 'units/hour')
print('Hourly Pass-On Rate:', round(AA_hourly_pass_on_rate), 'units/hour')
print('Hourly Rework Cost With All Machines:', AA_hourly_rework_cost, '£/hour')
print()

# Visual Inspection
VI_hourly_rates = []
for i in range(VI_count):
    rate = VI_availability * (3600 / VI_time) # CHECK
    VI_hourly_rates.append(rate)

VI_syringe_hourly_rate   = sum(VI_hourly_rates)
m = VI_syringe_hourly_rate * (AA_hourly_hidden_defects/AA_hourly_pass_on_rate)
n = VI_syringe_hourly_rate
VI_hourly_true_defects   = 0.944 * (m)
VI_hourly_false_defects  = 0.0021 * (n-m)
VI_hourly_true_syringes  = 0.9979 * (n-m)
VI_hourly_false_syringes = 0.056 * (m)
VI_hourly_rejects = VI_hourly_true_defects + VI_hourly_false_defects
VI_hourly_pass_on_rate = (VI_hourly_true_syringes + VI_hourly_false_syringes)
VI_hourly_scrap_cost   = VI_hourly_rejects * scrapCost

print(f'Visual Inspection (x{VI_count})')
print('Hourly Rates:', VI_hourly_rates, 'units/hour')
print('Syringe Hourly Rate:', round(VI_syringe_hourly_rate), 'units/hour')
print('Hourly Syringes Incorrectly Rejected:', round(VI_hourly_false_defects), 'units/hour')
print('Hourly Syringes Correctly Rejected:', round(VI_hourly_true_defects), 'units/hour')
print('Hourly Syringes Incorrectly Kept:', round(VI_hourly_false_syringes, 3), 'units/hour')
print('Hourly Syringes Correctly Kept:', round(VI_hourly_true_syringes), 'units/hour')
print('Hourly Reject Rate:', round(VI_hourly_rejects), 'units/hour')
print('Hourly Pass-On Rate;', round(VI_hourly_pass_on_rate), 'units/hour')
print('Hourly Scrap Cost:', round(VI_hourly_scrap_cost), '£/hour')
print()

# EO Sterilisation
EO_time_hours = EO_time / 60

print(f"EO Sterilisation (x{EO_count})")
print('Batch Time:', EO_time_hours, 'hours')

#%%
# Schedule Sequence
# Time to batch
IM_ttb = EO_batch / IM_syringe_hourly_rate # when IM & AA will finish
AA_ttb = EO_batch / AA_hourly_pass_on_rate
VI_ttb = EO_batch / VI_hourly_pass_on_rate

# Queuing Theory
IM_batch_rate = 1 / IM_ttb
AA_batch_rate  = 1 / AA_ttb
startUpDelay = AA_ttb * (AA_batch_rate/IM_batch_rate - 1) # hours
VI_ends = startUpDelay + VI_ttb
EO_finish = VI_ends + EO_time_hours

# Display
print("Time to Complete 1 Batch")
print("Injection Moulding:", round(IM_ttb, 3), 'hours')
print("Automated Assembly:", round(AA_ttb, 3), 'hours')
print("Visual Inspection:",  round(VI_ttb, 3), 'hours')
print()
print("Production Rates [batch/hour]")
print("Injection Moulding:", round(IM_batch_rate, 3))
print("Automated Assembly:", round(AA_batch_rate, 3))
print()
print("Injection Moulding switch on first, and after some time")
print("Automated Assembly and Visual Inspection is switched on:")
print("Start-Up Delay:", round(startUpDelay, 3), "hours")
print("IM and AA finish in:", round(IM_ttb, 3), "hours")
print("VI finishes in:", round(VI_ends, 3), 'hours')
print("EO finishes in:", round(EO_finish,3), 'hours')

#%%

batch_size = EO_batch
t = np.linspace(0, preEObuffer, 500)

IM_output = np.minimum(t * IM_syringe_hourly_rate, batch_size)

AA_output = np.zeros_like(t)
AA_start = startUpDelay
AA_output[t >= AA_start] = np.minimum(
    (t[t >= AA_start] - AA_start) * AA_hourly_pass_on_rate, batch_size
)

VI_output = np.zeros_like(t)
VI_start = AA_start
VI_output[t >= VI_start] = np.minimum(
    (t[t >= VI_start] - VI_start) * VI_hourly_pass_on_rate, batch_size
)

IM_output[t > IM_ttb] = batch_size
AA_output[t > IM_ttb] = batch_size
VI_output[t > VI_ends] = batch_size


plt.rcParams.update({
    'font.size': 14,        
    'axes.titlesize': 16,   
    'axes.labelsize': 16,   
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})
plt.rcParams['legend.loc'] = 'upper left'

plt.figure(figsize=(8, 5))
plt.plot(t, IM_output, label='Injection Moulding', linewidth=2)
plt.plot(t, AA_output, label='Automated Assembly', linewidth=2)
plt.plot(t, VI_output, label='Visual Inspection', linewidth=2)

plt.axvline(AA_start, color='k', linestyle='--', alpha=0.6, label='AA/VI start')
plt.axvline(IM_ttb, color='b', linestyle='--', label='IM/AA stop')
plt.axvline(VI_ends, color='red', linestyle='--', label='VI stop')

plt.xlabel('Time [Hours]')
plt.ylabel('Cumulative Syringes Produced')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%