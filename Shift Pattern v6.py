#%%
# Functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import random

def lognormalCurve(length, s=0.8, loc=0, scale=30, base=1, peak=2):
    t = np.arange(length)
    curve = lognorm.pdf(t, s=s, loc=loc, scale=scale)
    curve = (curve - np.min(curve)) / (np.max(curve) - np.min(curve))  # normalize 0–1
    curve = base + (peak - base) * curve                               # scale to [base, peak]
    curve[-1] = base
    return curve

def whiteNoise(curve, noiseLevel=0.01):
    noise = np.random.normal(0, noiseLevel, len(curve))
    noisyCurve = curve * (1 + noise)
    return noisyCurve

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
IM_scrap_probability = 0.0059
IM_barrel_cavities   = [24, 16, 16]
IM_plunger_cavities  = [32, 24,  0]

# Automated Assembly
AA_count          = 3
AA_time           = 0.54   # s/part
AA_FPY            = 0.9924
AA_rework         = 71     # s
AA_rework_success = 0.915
AA_rework_cost    = 17.13  # £/hour

# Visual Inspection
VI_count = 2
VI_time  = 0.57 # s/part

# EO Batching
EO_count      = 1
EO_time       = 246.9  # min/batch
EO_batch      = 41_631 # units
EO_quarantine = 12     # hours
EO_batch_cost = 202.73 # £/batch

# Packing
CRP_count = 11
CRP_FPY   = 0.9964

# Costs
scrapCost        = 0.0704 # £/unit
packMaterialCost = 0.06   # £/unit
EOCost           = 202.73 # £/batch
labourCost       = 15.7   # £/hour
reworklabourCost = 17.13  # £/hour
qualityEscapes   = 11_187 # £/million_units
qualityEscapesPerUnit = qualityEscapes/1_000_000

# Reliability
IM_MTBF  = 634.8 # min
IM_MTTR  =  10.4 # min
AA_MTBF  = 337.2 # min
AA_MTTR  =  12.9 # min
VI_MTBF  = 630.8 # min
VI_MTTR  =   8.9 # min
EO_MTBF  = 867.2 # min
EO_MTTR  =  56.4 # min
CRP_MTBF = 439.0 # min
CRP_MTTR =  14.8 # min

IM_availability  = availability(IM_MTTR, IM_MTBF)
AA_availability  = availability(AA_MTTR, AA_MTBF)
VI_availability  = availability(VI_MTTR, VI_MTBF)
EO_availability  = availability(EO_MTTR, EO_MTBF)
CRP_availability = availability(CRP_MTTR, CRP_MTBF)

#%%
# Calculated Parameters 

# Injection Moulding
IM_barrel_hourly_rates  = []
IM_plunger_hourly_rates = []
for i in range(len(IM_barrel_cavities)):
    barrelCavity  = IM_barrel_cavities[i]
    plungerCavity = IM_plunger_cavities[i]

    barrelRate  = ((3600 * barrelCavity)  / IM_time)
    plungerRate = ((3600 * plungerCavity) / IM_time)

    IM_barrel_hourly_rates.append(barrelRate)
    IM_plunger_hourly_rates.append(plungerRate)

IM_ideal_hourly_rate         = min(sum(IM_barrel_hourly_rates), sum(IM_plunger_hourly_rates))
IM_scrap_hourly_rate         = IM_ideal_hourly_rate * IM_availability * IM_scrap_probability
IM_scrap_cost_hourly_rate    = IM_scrap_hourly_rate * scrapCost
IM_pass_on_hourly_rate       = IM_ideal_hourly_rate * IM_availability - IM_scrap_hourly_rate
IM_hidden_defect_hourly_rate = IM_pass_on_hourly_rate * (1-IM_FPY)

# Automated Assembly
AA_hourly_rates = []
for i in range(AA_count):
    rate = (3600 / AA_time)
    AA_hourly_rates.append(rate)

AA_ideal_hourly_rate         = sum(AA_hourly_rates)
AA_pass_on_hourly_rate       = AA_ideal_hourly_rate * AA_availability
AA_hidden_defect_hourly_rate = AA_pass_on_hourly_rate * (1-AA_FPY)

# Visual Inspection
VI_hourly_rates = []
for i in range(VI_count):
    rate = (3600 / VI_time)
    VI_hourly_rates.append(rate)

VI_ideal_hourly_rate = sum(VI_hourly_rates)
VI_inspection_hourly_rate = VI_ideal_hourly_rate * VI_availability
m = IM_hidden_defect_hourly_rate + AA_hidden_defect_hourly_rate
n = VI_inspection_hourly_rate
VI_true_defects_hourly_rate = 0.944   * m
VI_false_defects_hourly_rate = 0.0021 * (n-m)
VI_true_parts_hourly_rate = 0.9979    * (n-m)
VI_false_parts_hourly_rate = 0.056    * (m)
VI_total_rejected_hourly_rate = VI_true_defects_hourly_rate + VI_false_defects_hourly_rate
VI_total_kept_hourly_rate     = VI_true_parts_hourly_rate + VI_false_parts_hourly_rate
VI_hidden_defect_hourly_rate = VI_false_parts_hourly_rate
VI_quality_escapes_cost_hourly_rate = VI_hidden_defect_hourly_rate * qualityEscapesPerUnit

# Rework Station
# Rework operators look at all rejected parts from visual inspection.
# They can tell the difference between a true defect and a false defect. 
# A false defect will be put back into the system with negligible time.
reworkRatePerOperator = 3600 / AA_rework
reworkRate = reworkRatePerOperator * 4
rework_success_hourly_rate = VI_true_defects_hourly_rate * AA_rework_success
rework_failure_hourly_rate = VI_true_defects_hourly_rate * (1-AA_rework_success)
rework_scrap_cost_hourly_rate  = rework_failure_hourly_rate * scrapCost
rework_labour_cost_hourly_rate = 4 * reworklabourCost
pass_on_VI_rework_hourly_rate = VI_total_kept_hourly_rate + rework_success_hourly_rate
VI_rework_total_cost = rework_labour_cost_hourly_rate + rework_scrap_cost_hourly_rate + VI_quality_escapes_cost_hourly_rate

# Alternative Method: Get rid of rework station and just scrap everything that fails from EO
VI_total_cost_hourly_rate = VI_total_rejected_hourly_rate * scrapCost + VI_quality_escapes_cost_hourly_rate

# Storage costs
palletCost = 2.50                      # £/week for one pallet, which can fit 60k syringes.
palletCostHourly = palletCost / (7*24) # £/hour for one pallet, for 60k syringes.
storageUnitPrice = palletCostHourly / 60_000

# Outputs
print(f'Injection Moulding (x{IM_count}), Availability {round(IM_availability, 3)}')
print('Chosen (barrel) cavity sizes are: ', IM_barrel_cavities)
print('Chosen (plunger) cavity sizes are:', IM_plunger_cavities)
print('Hourly Ideal Rate:', round(IM_ideal_hourly_rate), 'units/hour')
print('Hourly Scrap Rate:', round(IM_scrap_hourly_rate), 'units/hour')
print('Hourly Scrap Cost:', round(IM_scrap_cost_hourly_rate, 2), '£/hour')
print('Hourly Pass-On Rate:', round(IM_pass_on_hourly_rate), 'units/hour')
print('Hourly Hidden Defects:', round(IM_hidden_defect_hourly_rate), 'units/hour')
print()

print(f'Automated Assembly (x{AA_count}), Availability {round(AA_availability, 3)}')
print('Hourly Ideal Rate:', round(AA_ideal_hourly_rate), 'units/hour')
print('Hourly Pass-On Rate:', round(AA_pass_on_hourly_rate), 'units/hour')
print('Hourly Hidden Defects:', round(AA_hidden_defect_hourly_rate), 'units/hour')
print()

print(f'Visual Inspection (x{VI_count}), Availability {round(VI_availability, 3)}')
print('Hourly Ideal Rate:', round(VI_ideal_hourly_rate), 'units/hour')
print('Hourly Inspection Rate:', round(VI_inspection_hourly_rate), 'units/hour')
print('Hourly Defects Arriving to VI:', round(m), 'units/hour')
print('Hourly Syringes Incorrectly Rejected:', round(VI_false_defects_hourly_rate), 'units/hour')
print('Hourly Syringes Correctly Rejected:  ', round(VI_true_defects_hourly_rate), 'units/hour')
print('Hourly Syringes Incorrectly Kept:    ', round(VI_false_parts_hourly_rate), 'units/hour')
print('Hourly Syringes Correctly Kept:      ', round(VI_true_parts_hourly_rate), 'units/hour')
print('Hourly Rejected Total:', round(VI_total_rejected_hourly_rate), 'units/hour')
print('Hourly Kept Total:', round(VI_total_kept_hourly_rate))
print('Hourly Hidden Defects:', round(VI_hidden_defect_hourly_rate), 'units/hour')
print('Hourly Quality Escapes Cost:', round(VI_quality_escapes_cost_hourly_rate, 3), '£/hour')
print()

print(f'Rework Station (4 operators)')
print('Hourly Rework Rate Per Operator:', round(reworkRatePerOperator), 'units/hour-operator')
print('Hourly Rework Rate:', round(reworkRate), 'units/hour')
print('Hourly Rework Success Rate:', round(rework_success_hourly_rate), 'units/hour')
print('Hourly Rework Failure Rate:', round(rework_failure_hourly_rate), 'units/hour')
print('Hourly Scrap Cost:', round(rework_scrap_cost_hourly_rate,2), '£/hour')
print('Hourly Labour Cost:', round(rework_labour_cost_hourly_rate), '£/hour')
print('Hourly pass-on to EO:', round(pass_on_VI_rework_hourly_rate), 'units/hour')
print('Hourly Total Cost from Rework and VI:', round(VI_rework_total_cost), '£/hour')
print()

'''
Alternative Method: Instead of paying operators for a rework station, we instead scrap everything rejected by
visual inspection. This comes out as much cheaper for the operation. 
'''
print('Alternative Method')
print('Hourly Visual Inspection Scrap Cost', round(VI_total_cost_hourly_rate), '£/hour')
print('Note, Pallet Storage Cost:', round(palletCostHourly,3), '£/hour')
print('Unit Storage Price', round(storageUnitPrice, 7), '£/syringe/hour')

#%%
# Demand Curve
def noisyDemandCurve(totalPeriod, surgePeriod, nominalPeriod, s, loc, scale, noiseLevel):
    demandCurve = lognormalCurve(surgePeriod, s=s, loc=loc, scale=scale, 
                                 base=nominalDemand, peak=surgeDemand)
    nominalLine = np.ones(nominalPeriod) * nominalDemand

    noisyDemandCurve = whiteNoise(demandCurve, noiseLevel=noiseLevel)
    noisyNominalLine = whiteNoise(nominalLine, noiseLevel=noiseLevel)

    numRepeats = round(totalPeriod / (surgePeriod))

    totalCurve = np.array([])
    for i in range(numRepeats):
        totalCurve = np.concatenate((totalCurve, noisyDemandCurve, noisyNominalLine))

    totalCurve = totalCurve[:totalPeriod]

    return totalCurve

x = random.uniform(0, 0.02)
totalCurve = noisyDemandCurve(totalPeriod=365, surgePeriod=90, nominalPeriod=90, s=0.5, loc=-2, scale=25, noiseLevel=0.015)

plt.figure(figsize=(10,5))
plt.plot(totalCurve, c='r', label='Syringe Demand')
plt.xlabel('Time [Days]')
plt.ylabel('Demand [syringes/day]')
plt.axhline(y=nominalDemand, c='g', linestyle='--', label='Nominal Demand')
plt.axhline(y=surgeDemand, c='b', linestyle='--', label='Surge Demand')
plt.legend()
plt.grid()
plt.show()

#%% 
# Simulation Function
def run_simulation(batchThreshold = 0.2, timeFrame = 100, totalCurve=totalCurve):
    """
    Simulation logic where IM, AA and VI start at the same time each production period.
    IM is the slowest process and thus AA and VI only process at the IM pass-on throughput.
    """
    dayCount   = 0
    shiftCount = 0
    iterNum    = int(timeFrame * 24 * 60)  # minutes
    peakAlarm  = False
    EO_quarantine_min = EO_quarantine * 60

    # IM-driven start-up delay is no longer used
    IM_buffer = AA_buffer = VI_buffer = EO_buffer = EO_out = QU_buffer = 0.0
    EO_timer = totCost = 0.0
    EO_cost  = 0
    tot_scrap_cost = 0
    tot_storage_cost = 0
    demand = nominalDemand
    specialBatch = EO_batch
    releases_by_minute = np.zeros(iterNum)
    reserve_stock = 0.0
    peak_num = 0
    next_batch_to_reserve = False

    IM_buf = []
    AA_buf = []
    VI_buf = []
    EO_buf = []
    QU_buf = []
    clockTime = []

    EO_ready = True
    quarantine_timers = []

    clock = 0
    # Precompute bottleneck per-minute throughput
    im_per_min = IM_pass_on_hourly_rate / 60.0             # units produced by IM per minute
    aa_capacity_per_min = AA_pass_on_hourly_rate / 60.0    # AA maximum capacity per minute
    vi_capacity_per_min = VI_total_kept_hourly_rate / 60.0 # VI maximum capacity per minute

    aa_throttle_rate = min(aa_capacity_per_min, im_per_min)
    vi_throttle_rate = min(vi_capacity_per_min, im_per_min)

    for i in range(iterNum):
        cycle_pos = clock % 480
        if cycle_pos == 0:
            shiftCount += 1
            if shiftCount % 3 == 0:
                dayCount += 1

            demand = determineProduction(dayCount, totalCurve, nominalDemand)
            batches_needed = demand / EO_batch
            if batches_needed > 2:
                peakAlarm = True
                remainderFraction = batches_needed - 2
                peak_num += 1
            else:
                peakAlarm = False
                remainderFraction = batches_needed - 1

            remainder_units = remainderFraction * EO_batch
            next_batch_to_reserve = False
            if remainder_units <= 0:
                specialBatch = 0.0
            elif remainder_units <= reserve_stock:
                reserve_stock -= remainder_units
                QU_buffer += remainder_units
                specialBatch = 0.0
            elif remainderFraction < batchThreshold:
                specialBatch = EO_batch
                next_batch_to_reserve = True
            else:
                specialBatch = remainder_units

        # Determine whether the plant should run this shift
        shift_of_day = (shiftCount - 1) % 3
        run_schedule = peakAlarm or (shift_of_day in (0, 1))

        # Determine which batch size we're producing in this running block
        producing_batch = EO_batch if (not peakAlarm and shift_of_day in (0,1)) or (peakAlarm and shift_of_day in (0,1)) else specialBatch

        # Batch time determined entirely by IM throughput
        if producing_batch > 0:
            IM_ttb = determineTTB(producing_batch, IM_pass_on_hourly_rate) * 60.0
        else:
            IM_ttb = 0.0

        if run_schedule and cycle_pos < 450:
            # Injection moulding
            if cycle_pos < IM_ttb:
                IM_buffer += im_per_min
                totCost   += IM_scrap_hourly_rate / 60.0
                tot_scrap_cost += IM_scrap_hourly_rate / 60.0

            # AA is started at the same time as IM
            if cycle_pos < IM_ttb:
                moved_to_AA = min(aa_throttle_rate, IM_buffer)
                IM_buffer -= moved_to_AA
                AA_buffer += moved_to_AA

            # VI pulls from AA_buffer at its throttled rate
            if cycle_pos < IM_ttb:
                moved_to_VI = min(vi_throttle_rate, AA_buffer)
                AA_buffer -= moved_to_VI
                VI_buffer += moved_to_VI
                totCost += VI_total_cost_hourly_rate / 60.0
                tot_scrap_cost += VI_total_cost_hourly_rate / 60.0

            # When IM has finished the batch
            if cycle_pos >= IM_ttb and IM_ttb > 0 and EO_ready:
                EO_buffer += VI_buffer
                VI_buffer = 0.0
                AA_buffer = 0.0
                IM_buffer = 0.0
                EO_ready = False

                # track whether this batch should be used to stock later
                current_batch_to_reserve = next_batch_to_reserve
                next_batch_to_reserve = False

        if cycle_pos < 450:
            if not EO_ready:
                EO_timer += 1

            if EO_timer >= EO_time and not EO_ready:
                sterilised_qty = EO_buffer
                EO_out += sterilised_qty
                EO_buffer = 0.0
                EO_ready = True
                EO_timer = 0.0
                totCost += EO_batch_cost
                EO_cost += EO_batch_cost

                # quarantine handling
                to_reserve = "current_batch_to_reserve" in locals() and current_batch_to_reserve
                quarantine_timers.append([EO_quarantine_min, sterilised_qty, bool(to_reserve)])
                if "current_batch_to_reserve" in locals():
                    del current_batch_to_reserve

            # Progress quarantine timers and release when done
            new_quarantine = []
            for remaining, bsize, to_reserve in quarantine_timers:
                remaining -= 1
                if remaining <= 0:
                    EO_out = max(0.0, EO_out - bsize)
                    QU_buffer += bsize
                    releases_by_minute[clock] += bsize
                    if to_reserve:
                        reserve_stock += bsize
                else:
                    new_quarantine.append([remaining, bsize, to_reserve])
            quarantine_timers = new_quarantine

        # Storage cost accrual
        totCost += reserve_stock * storageUnitPrice / 60.0
        tot_storage_cost += reserve_stock * storageUnitPrice / 60.0

        # advance clock and log buffers
        clock += 1
        IM_buf.append(IM_buffer)
        AA_buf.append(AA_buffer)
        VI_buf.append(VI_buffer)
        EO_buf.append(EO_out)
        QU_buf.append(QU_buffer)
        clockTime.append(clock)

    # Prepare outputs
    data = [IM_buf, AA_buf, VI_buf, EO_buf, QU_buf, clockTime, releases_by_minute]

    minutes_per_day = 24 * 60
    num_days = int(np.ceil(iterNum / minutes_per_day))
    daily_throughput = []
    for d in range(num_days):
        s = d * minutes_per_day
        e = min((d + 1) * minutes_per_day, iterNum)
        daily_throughput.append(releases_by_minute[s:e].sum())

    total_output = sum(daily_throughput)
    total_demand = sum(totalCurve[:len(daily_throughput)])
    on_time_percent = 100.0 * total_output / total_demand if total_demand > 0 else 0.0

    print(tot_scrap_cost)
    print(tot_storage_cost)
    print(EO_cost)

    return totCost, on_time_percent, data

#%% 
# Single-run simulation and plotting using the new IM-bottleneck simulation
timeFrame = 365
iterNum   = int(timeFrame * 24 * 60)
totCost, on_time_percent, data = run_simulation(batchThreshold = 1.0, timeFrame = timeFrame, totalCurve=totalCurve)

IM_buf, AA_buf, VI_buf, EO_buf, QU_buf, clockTime, releases_by_minute = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

clockTime_hours = np.array(clockTime) / 60.0
minutes_per_day = 24 * 60
num_days = int(np.ceil(iterNum / minutes_per_day))

#%%
daily_throughput = []
day_times = []
for d in range(num_days):
    s = d*minutes_per_day
    e = min((d+1)*minutes_per_day, iterNum)
    daily_throughput.append(releases_by_minute[s:e].sum())
    day_times.append(d+1)

plt.rcParams.update({
    'font.size': 14,        
    'axes.titlesize': 16,   
    'axes.labelsize': 16,   
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})   

plt.figure()
plt.plot(clockTime_hours, IM_buf)
plt.title('Injection Moulding Buffer Over Time')
plt.xlabel('Time [Hours]')
plt.ylabel('Units In Buffer')
plt.grid()

plt.figure()
plt.plot(clockTime_hours, AA_buf)
plt.title('Automated Assembly Buffer Over Time')
plt.xlabel('Time [Hours]')
plt.ylabel('Units In Buffer')
plt.grid()

plt.figure()
plt.plot(clockTime_hours, VI_buf)
plt.title('Visual Inspection Buffer Over Time')
plt.xlabel('Time [Hours]')
plt.ylabel('Units In Buffer')
plt.grid()

plt.figure()
plt.plot(clockTime_hours, EO_buf)
plt.title('EO Buffer Over Time')
plt.xlabel('Time [Hours]')
plt.ylabel('Pre-Quarantine Buffer')
plt.grid()

plt.figure()
plt.plot(clockTime_hours, QU_buf)
plt.title('Quarantine Buffer Over Time')
plt.xlabel('Time [Hours]')
plt.ylabel('Syringes Released from Quarantine')
plt.grid()

plt.figure(figsize=(9,4))
plt.bar(day_times, daily_throughput, width=0.6)
plt.title('Daily Throughput')
plt.xlabel('Time [Days]')
plt.ylabel('Syringes')
plt.axhline(y=nominalDemand, color='r', linestyle='--', label='Nominal')
plt.axhline(y=surgeDemand, color='b', linestyle='--', label='Surge')
plt.grid(axis='y')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(day_times, daily_throughput, label='Throughput')
plt.plot(day_times, totalCurve[:len(day_times)], c='r', label='Demand')
plt.xlabel('Time [Days]')
plt.ylabel('Syringes')
plt.axhline(y=nominalDemand, color='g', linestyle='--', label='Nominal')
plt.axhline(y=surgeDemand, color='b', linestyle='--', label='Surge')
plt.grid(axis='y')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.ylim(75000, 128_000)
plt.show()

print('Total Cost: £', round(totCost))
print('On-time Fraction:', round(on_time_percent, 3))

#%%
batchThreshold_values = np.linspace(0, 1, 50)
timeFrame = 100
n_runs = 100
results = []
resultsMean = []

for bt in batchThreshold_values:
    costs = []
    ontimes = []
    print(f"Running {n_runs} sims for batchThreshold={bt:.2f}")
    for _ in range(n_runs):
        noise = random.uniform(0, 0.02)
        totalCurve = noisyDemandCurve(
            totalPeriod=timeFrame,
            surgePeriod=90,
            nominalPeriod=90,
            s=0.5,
            loc=-2,
            scale=25,
            noiseLevel=noise
        )
        cost, ontime, _ = run_simulation(bt, timeFrame=timeFrame, totalCurve=totalCurve)
        costs.append(cost)
        ontimes.append(ontime)

    results.append({
        "bt": bt,
        "cost_median": np.median(costs),
        "cost_q1": np.percentile(costs, 25),
        "cost_q3": np.percentile(costs, 75),
        "ontime_median": np.median(ontimes)
    })

    resultsMean.append({
        "bt": bt,
        "cost_mean": np.mean(costs),
        "cost_q1": np.percentile(costs, 25),
        "cost_q3": np.percentile(costs, 75),
        "ontime_mean": np.mean(ontimes)
    })

#%%
# Convert to arrays for plotting
bt_vals = np.array([r["bt"] for r in results])
cost_median = np.array([r["cost_median"] for r in results])
cost_q1 = np.array([r["cost_q1"] for r in results])
cost_q3 = np.array([r["cost_q3"] for r in results])
ontime_median = np.array([r["ontime_median"] for r in results])

bt_vals = np.array([r["bt"] for r in resultsMean])
cost_mean = np.array([r["cost_mean"] for r in resultsMean])
cost_q1_mean = np.array([r["cost_q1"] for r in resultsMean])
cost_q3_mean = np.array([r["cost_q3"] for r in resultsMean])
ontime_mean = np.array([r["ontime_mean"] for r in resultsMean])


# Identify region meeting ≥95% on-time
valid_mask = ontime_median >= 95
if np.any(valid_mask):
    min_idx = np.argmin(cost_median[valid_mask])
    optimal_bt = bt_vals[valid_mask][min_idx]
    print(f"\nOptimal batchThreshold = {optimal_bt:.2f}")
    print(f"Median Cost = {cost_median[valid_mask][min_idx]:.0f}")
    print(f"Median On-Time = {ontime_median[valid_mask][min_idx]:.2f}%")
else:
    print("No batchThreshold achieved ≥95% on-time performance.")
print()

valid_mask = ontime_mean >= 95
if np.any(valid_mask):
    min_idx = np.argmin(cost_mean[valid_mask])
    optimal_bt_mean = bt_vals[valid_mask][min_idx]
    print(f"\nOptimal batchThreshold = {optimal_bt_mean:.2f}")
    print(f"Mean Cost = {cost_mean[valid_mask][min_idx]:.0f}")
    print(f"Mean On-Time = {ontime_mean[valid_mask][min_idx]:.2f}%")
else:
    print("No batchThreshold achieved ≥95% on-time performance.")

#%%
plt.figure(figsize=(8,5))
plt.plot(bt_vals, cost_median, 'o-', label='Median Cost')
plt.fill_between(bt_vals, cost_q1, cost_q3, alpha=0.2, label='Interquartile Range')
plt.axvline(optimal_bt, color='g', linestyle='--', label=f'Optimal BT={optimal_bt:.2f}')
plt.xlabel('Batch Threshold')
plt.ylabel('Total Cost [£]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(bt_vals, cost_mean, 'o-', label='Mean Cost')
plt.fill_between(bt_vals, cost_q1_mean, cost_q3_mean, alpha=0.2, label='Interquartile Range')
plt.axvline(optimal_bt_mean, color='g', linestyle='--', label=f'Optimal BT={optimal_bt_mean:.2f}')
plt.xlabel('Batch Threshold')
plt.ylabel('Total Cost [£]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot on-time %
plt.figure(figsize=(8,5))
plt.plot(bt_vals, ontime_median, 'o-', label='Median On-Time %')
plt.axhline(95, color='r', linestyle='--', label='95% On-Time Target')
plt.xlabel('Batch Threshold')
plt.ylabel('On-Time Percentage [%]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(bt_vals, ontime_mean, 'o-', label='Mean On-Time %')
plt.axhline(95, color='r', linestyle='--', label='95% On-Time Target')
plt.xlabel('Batch Threshold')
plt.ylabel('On-Time Percentage [%]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

timeFrame = 100
iterNum = int(timeFrame * 24 * 60)
n_simulations = 100
start_date_str = "2025-01-01 00:00:00"
batch_threshold_for_csv = 1.0

start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
time_index_full = np.array([start_datetime + timedelta(minutes=i) for i in range(iterNum)])

sim_data_list = []
max_len = 0

for i in range(n_simulations):
    noise = random.uniform(0, 0.02)
    totalCurve = noisyDemandCurve(
        totalPeriod=timeFrame,
        surgePeriod=90,
        nominalPeriod=90,
        s=0.5,
        loc=-2,
        scale=25,
        noiseLevel=noise
    )

    _, _, data = run_simulation(
        batchThreshold=batch_threshold_for_csv, 
        timeFrame=timeFrame, 
        totalCurve=totalCurve
    )
    releases_by_minute = data[6]
    
    non_zero_mask = releases_by_minute > 0
    current_time_index = time_index_full[:len(releases_by_minute)]
    
    release_times = current_time_index[non_zero_mask]
    release_amounts = releases_by_minute[non_zero_mask]
    
    current_len = len(release_times)
    max_len = max(max_len, current_len)
    
    # 5. Store the filtered data
    sim_data_list.append({
        'Time': release_times,
        'Throughput': release_amounts,
        'len': current_len,
        'sim_num': i + 1
    })

final_columns = {}
NULL_VALUE = ''

for sim_data in sim_data_list:
    sim_num = sim_data['sim_num']
    current_len = sim_data['len']
    
    # Pad Time column
    time_series = sim_data['Time'].tolist()
    time_series.extend([NULL_VALUE] * (max_len - current_len))
    final_columns[f'Time_Sim{sim_num}'] = time_series
    
    # Pad Throughput column
    throughput_series = sim_data['Throughput'].tolist()
    throughput_series.extend([NULL_VALUE] * (max_len - current_len))
    final_columns[f'Throughput_Sim{sim_num}'] = throughput_series

all_simulations_df = pd.DataFrame(final_columns)

ordered_cols = []
for i in range(1, n_simulations + 1):
    ordered_cols.append(f'Time_Sim{i}')
    ordered_cols.append(f'Throughput_Sim{i}')

all_simulations_df = all_simulations_df[ordered_cols]

output_filename = "sparse_minute_throughput_batch1.csv"
all_simulations_df.to_csv(output_filename, index=False, date_format='%Y-%m-%d %H:%M:%S')

print(f"\nSuccessfully created {output_filename} with {max_len} rows and minute resolution event data for {n_simulations} simulations.")