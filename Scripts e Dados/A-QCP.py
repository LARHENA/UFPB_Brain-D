"""
A-QCP Consolidated Script - Complete Pipeline
This script combines all data processing steps into a complete pipeline:
1. Data Cleaning
2. Pre-classification 
3. Outlier Detection
4. Quality Assessment (P, Q1, Q2, Q3)
5. Quality Index Calculation
"""

# ==== IMPORTS ====
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os

# ==== FILE PATHS & PARAMETERS ====
input_path = './1 - Organized data gauge/BRAZIL/BRAZIL_DAILY_1961_2024.h5'
cleaned_path = './1 - Organized data gauge/BRAZIL/DATASETS/BRAZIL_DAILY_1961_2024_CLEANED.h5'
neighboring_data_path = './1 - Organized data gauge/BRAZIL'

# Threshold parameters
threshold_pr_min = 0  # [mm]
threshold_pr_max = 600  # [mm]
rainfall_threshold_outlier = 200.0  # mm
chunk_size = 10000000  # Adjust based on available memory

# ==== DATA CLEANING ====
def clean_data():
    """Clean data by applying precipitation thresholds"""
    df_data = pd.read_hdf(input_path, key='table_data', encoding='utf-8')
    df_info = pd.read_hdf(input_path, key='table_info', encoding='utf-8')
    
    df_data_filtered = df_data[(df_data['rain_mm'] >= threshold_pr_min) & 
                              (df_data['rain_mm'] <= threshold_pr_max)]
    df_info_filtered = df_info[df_info['gauge_code'].isin(df_data_filtered['gauge_code'].unique())]
    
    # Print statistics
    # print(f"{(len(df_data) - len(df_data_filtered))} data points removed due to precipitation threshold.")
    # print(f"{(len(df_info) - len(df_info_filtered))} gauges removed due to precipitation threshold.")
    # print(f"{(len(df_data) - len(df_data_filtered))/len(df_data)*100:.2f}% of data points removed")
    # print(f"{(len(df_info) - len(df_info_filtered))/len(df_info)*100:.2f}% of gauges removed")
    
    # Save cleaned data
    df_data_filtered.to_hdf(cleaned_path, key='table_data', mode='w', complevel=9, 
                           append=False, complib='zlib', encoding='utf-8')
    df_info_filtered.to_hdf(cleaned_path, key='table_info', mode='r+', complevel=9, 
                           append=False, complib='zlib', encoding='utf-8')
    
    return df_data_filtered, df_info_filtered

# ==== PRE-CLASSIFICATION ====
def preclassify_data(df_data, df_info):
    """Pre-classify data based on annual metrics"""
    # Add year column
    df_data['year'] = df_data['datetime'].dt.year
    
    # Group by gauge_code and year
    grouped = df_data.groupby(['gauge_code', 'year'])
    
    def calculate_metrics(group):
        annual_rainfall_mm = group['rain_mm'].sum()
        active_days = (group['rain_mm'] >= 0.0).sum()
        
        # Calculate consecutive dry days
        dry_days = (group['rain_mm'] == 0.0).astype(int)
        consecutive_dry_days = (dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumsum() * dry_days).max()
        
        return pd.Series({
            'annual_rainfall_mm': annual_rainfall_mm,
            'active_days': active_days,
            'consecutive_dry_days': consecutive_dry_days
        })
    
    df_preclassif = grouped.apply(calculate_metrics).reset_index()
    
    # Apply pre-classification rules
    df_preclassif['preclassif'] = df_preclassif.apply(
        lambda row: 'LQ' if (row['annual_rainfall_mm'] < 300 or
                            row['annual_rainfall_mm'] > 6000 or
                            row['consecutive_dry_days'] > 200) else "", axis=1)
    
    # # Print statistics
    # preclassif_counts = df_preclassif['preclassif'].value_counts()
    # if 'LQ' in preclassif_counts:
    #     print(f"LQ percentage: {preclassif_counts['LQ'] / preclassif_counts.sum() * 100:.2f}%")
    
    # Save pre-classification results
    df_preclassif.to_hdf(cleaned_path, key='table_preclassif', mode='r+', 
                        complevel=9, complib='zlib', encoding='utf-8')
    
    return df_preclassif

# ==== OUTLIER DETECTION ====
def detect_outliers(df_data, df_info, df_preclassif):
    """Detect outliers using adjacent day and neighboring analysis"""
    
    # Merge with pre-classification and filter out LQ data
    df_complete_info = pd.merge(df_data, df_info, on='gauge_code', how='inner')
    df_outlier = pd.merge(df_complete_info[['gauge_code', 'rain_mm', 'datetime', 'year']], 
                         df_preclassif, on=['gauge_code', 'year'], how='left')
    df_outlier = df_outlier[df_outlier['preclassif'] != 'LQ']
    df_outlier = df_outlier[['gauge_code', 'datetime', 'rain_mm']]
    
    # Outlier detection using adjacent days
    def mark_outlier_rain(df, threshold_rain_mm=200):
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        df_yesterday = df.copy(deep=True)
        df_yesterday['datetime'] = df_yesterday['datetime'] + pd.Timedelta(days=1)
        df_tomorrow = df.copy(deep=True)
        df_tomorrow['datetime'] = df_tomorrow['datetime'] - pd.Timedelta(days=1)
        
        df = pd.merge(df, df_yesterday[['gauge_code', 'datetime', 'rain_mm']], 
                     on=['gauge_code', 'datetime'], how='left', suffixes=('', '_yesterday'))
        df = pd.merge(df, df_tomorrow[['gauge_code', 'datetime', 'rain_mm']], 
                     on=['gauge_code', 'datetime'], how='left', suffixes=('', '_tomorrow'))
        
        df_sorted = df.sort_values(['gauge_code', 'datetime']).reset_index(drop=True)
        
        # Calculate adjacent days sum
        df_sorted['adjacent_days_mm'] = df_sorted['rain_mm_yesterday'] + df_sorted['rain_mm_tomorrow']
        
        # Outlier condition
        condition = (
            (df_sorted['rain_mm'] > threshold_rain_mm) &
            (df_sorted['adjacent_days_mm'] < 0.025 * df_sorted['rain_mm'])
        )
        
        df_sorted['outlier_status_1'] = np.where(condition, 1, 0)
        return df_sorted
    
    df_outlier_filter_1 = mark_outlier_rain(df_outlier)
    
    # Save adjacent day analysis results
    df_outlier_filter_1_export = df_outlier_filter_1[df_outlier_filter_1['outlier_status_1'] == 1]
    df_outlier_filter_1_export.to_hdf(
        os.path.join(neighboring_data_path, "adjacent_day_analysis_filter_1.h5"),
        key='table_data', mode='w', format='table', complevel=9, encoding='utf-8',
        append=False, min_itemsize={'gauge_code': 20}
    )
    
    # Prepare for neighboring analysis
    df_outlier_filter_1 = df_outlier_filter_1[df_outlier_filter_1['outlier_status_1'] == 0]
    df_outlier_filter_1 = pd.merge(df_outlier_filter_1, df_info[['gauge_code', 'lat', 'long']], 
                                  on='gauge_code', how='left')
    
    # Neighboring analysis
    def idw_interpolation(latitude, longitude, df_temp_without_gauge, kdtree, p=2):
        row = [latitude, longitude]
        distances, indices = kdtree.query(row, k=5)
        weights = 1 / (distances + 1e-6) ** p
        values = df_temp_without_gauge.iloc[indices]['rain_mm'].values
        return (np.sum(weights * values) / np.sum(weights))
    
    # Filter for high rainfall periods
    start_date = '2014-01-01'
    end_date = '2020-12-31'
    
    df_date_filter = df_outlier_filter_1.loc[
        (df_outlier_filter_1['datetime'] >= start_date) & 
        (df_outlier_filter_1['datetime'] <= end_date)
    ].sort_values('datetime')
    
    df_date_filter = df_date_filter[df_date_filter['rain_mm'] > rainfall_threshold_outlier]
    
    # Get sorted unique dates
    analysis_dates = df_date_filter['datetime'].unique().tolist()
    analysis_dates.sort()
    
    output_filename = os.path.join(neighboring_data_path, "neighboring_analysis.h5")
    
    # Process each date
    for current_date in analysis_dates[:10]:  # Process first 10 dates for demonstration
        # Filter data for current date
        daily_data = df_outlier_filter_1[df_outlier_filter_1['datetime'] == current_date]
        df_gauge_filter = daily_data[daily_data['rain_mm'] > rainfall_threshold_outlier]
        
        gauge_codes = df_gauge_filter['gauge_code'].unique()
        date_results = []
        
        for gauge in gauge_codes:
            gauge_data = daily_data[daily_data['gauge_code'] == gauge].iloc[0]
            lat, lon = gauge_data['lat'], gauge_data['long']
            observed_rain = gauge_data['rain_mm']
            
            result_row = {
                'gauge_code': gauge,
                'datetime': current_date,
                'lat': lat,
                'long': lon,
                'observed_rain_mm': observed_rain,
                'interpolated_rain_mm': np.nan
            }
            
            if observed_rain > rainfall_threshold_outlier:
                neighbor_data = daily_data[daily_data['gauge_code'] != gauge]
                
                if len(neighbor_data) > 0:
                    kd_tree = KDTree(neighbor_data[['lat', 'long']].values)
                    result_row['interpolated_rain_mm'] = idw_interpolation(lat, lon, neighbor_data, kd_tree)
            
            date_results.append(pd.DataFrame([result_row]))
        
        # Combine results for current date
        daily_results = pd.concat(date_results, ignore_index=True)
        
        # Save to HDF5
        storage_mode = 'w' if current_date == analysis_dates[0] else 'a'
        append_mode = False if current_date == analysis_dates[0] else True
        
        daily_results.to_hdf(
            output_filename,
            key='table_data',
            mode=storage_mode,
            format='table',
            complevel=9,
            encoding='utf-8',
            append=append_mode,
            min_itemsize={'gauge_code': 20}
        )
        
    # Process neighboring analysis results
    df_outlier_2 = pd.read_hdf(output_filename, key='table_data')
    df_outlier_filter_2_export = df_outlier_2[df_outlier_2['interpolated_rain_mm'] >= 0.0]
    df_outlier_filter_2_export = df_outlier_filter_2_export[
        df_outlier_filter_2_export['interpolated_rain_mm'] >= 0.35 * df_outlier_filter_2_export['observed_rain_mm']
    ]
    df_outlier_filter_2_export['outlier_status_2'] = 1
    
    # Save neighboring analysis results
    df_outlier_filter_2_export.to_hdf(
        os.path.join(neighboring_data_path, "neighboring_analysis_filter_2.h5"),
        key='table_data', mode='w', format='table', complevel=9, encoding='utf-8',
        append=False, min_itemsize={'gauge_code': 20}
    )
    
    return df_outlier_filter_1

# ==== FINAL DATA FILTERING ====
def apply_final_filtering(df_data):
    """Apply final filtering based on outlier detection results"""
    
    # Load outlier detection results
    df_filter_1 = pd.read_hdf(os.path.join(neighboring_data_path, "adjacent_day_analysis_filter_1.h5"), 
                             key='table_data')
    df_filter_1 = df_filter_1[['gauge_code', 'datetime', 'outlier_status_1']]
    
    df_filter_2 = pd.read_hdf(os.path.join(neighboring_data_path, "neighboring_analysis_filter_2.h5"), 
                             key='table_data')
    df_filter_2['outlier_status_2'] = 1
    df_filter_2 = df_filter_2[['gauge_code', 'datetime', 'outlier_status_2']]
    
    # Merge and filter data
    df_data_filtered = pd.merge(df_data, df_filter_1, on=['gauge_code', 'datetime'], how='left')
    df_data_filtered = pd.merge(df_data_filtered, df_filter_2, on=['gauge_code', 'datetime'], how='left')
    df_data_filtered = df_data_filtered[
        (df_data_filtered['outlier_status_1'] != 1) & 
        (df_data_filtered['outlier_status_2'] != 1)
    ]
    df_data_filtered = df_data_filtered[['gauge_code', 'datetime', 'rain_mm']].copy(deep=True)
    
    # Save final filtered data
    key = 'table_data_filtered'
    
    with pd.HDFStore(cleaned_path, mode='r+', complevel=9, complib='blosc:zstd') as store:
        for start in range(0, len(df_data_filtered), chunk_size):
            i = start // chunk_size
            end = start + chunk_size
            chunk = df_data_filtered.iloc[start:end]
            append_mode = False if i == 0 else True
            
            store.append(key, chunk, format='table', data_columns=True, 
                        encoding='utf-8', min_itemsize={'gauge_code': 20}, append=append_mode)
                
    return df_data_filtered

# ==== DATA LOADING FUNCTION ====
def load_data_from_hdf(file_path, key, chunk_size=13000000):
    """Load data from HDF5 file in chunks"""
    df_data = pd.DataFrame()
    
    with pd.HDFStore(file_path, mode='r') as store:    
        for i, chunk in enumerate(store.select(key, chunksize=chunk_size)):
            if df_data.empty:
                df_data = chunk
            else:
                df_data = pd.concat([df_data, chunk], ignore_index=True)
            del chunk
    return df_data

# ==== QUALITY ASSESSMENT FUNCTIONS ====
def calculate_p_availability(df):
    """Calculate data availability percentage by year"""
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    df_p_availability = df.groupby(['gauge_code', 'year']).agg({'rain_mm': 'count'}).reset_index()
    df_p_availability['days_in_year'] = df_p_availability.apply(
        lambda x: 365 if (x['year'] % 4 != 0 or (x['year'] % 100 == 0 and x['year'] % 400 != 0)) else 366, axis=1)
    df_p_availability['p_availability'] = df_p_availability['rain_mm'] / df_p_availability['days_in_year'] * 100
    df_p_availability['p_availability'] = df_p_availability['p_availability'].fillna(0).replace([np.inf, -np.inf], 0)
    return df_p_availability[['gauge_code', 'year', 'p_availability']]

def calculate_q1_gaps(df):
    """Calculate gap-related quality metrics"""
    df = df.sort_values(['gauge_code', 'datetime'])
    df['year'] = df['datetime'].dt.year
    df_grouped = df.groupby(['gauge_code', 'year'])
    
    def calculate_gaps(group):
        time_diffs = group['datetime'].diff().dt.days - 1.0
        active_days = group['datetime'].nunique()
        
        first_day = pd.Timestamp(f"{group['year'].iloc[0]}-01-01")
        start_gap = (group['datetime'].iloc[0] - first_day).days
        
        last_day = pd.Timestamp(f"{group['year'].iloc[0]}-12-31")
        end_gap = (last_day - group['datetime'].iloc[-1]).days
        
        all_gaps = time_diffs.tolist() + [start_gap, end_gap]
        max_gap = np.nanmax(all_gaps) if not all(np.isnan(all_gaps)) else np.nan
        
        return pd.Series({
            'max_gap_days': max_gap,
            'start_gap_days': start_gap,
            'end_gap_days': end_gap,
            'active_days': active_days  
        })
    
    df_q1_gaps = df_grouped.apply(calculate_gaps).reset_index()
    df_q1_gaps['days_in_year'] = df_q1_gaps.apply(
        lambda x: 366 if x['year'] % 4 == 0 and (x['year'] % 100 != 0 or x['year'] % 400 == 0) else 365, axis=1)
    df_q1_gaps['total_gaps'] = df_q1_gaps['days_in_year'] - df_q1_gaps['active_days']
    
    for col in ['max_gap_days', 'start_gap_days', 'end_gap_days', 'active_days', 'total_gaps']:
        df_q1_gaps[col] = df_q1_gaps[col].astype(int)
    
    df_q1_gaps['q1_gaps'] = 100.0 - 100.0 * (((2.0 * df_q1_gaps['total_gaps']) + df_q1_gaps['max_gap_days']) / df_q1_gaps['active_days'])
    df_q1_gaps['q1_gaps'] = df_q1_gaps['q1_gaps'].clip(lower=0, upper=100)
    
    return df_q1_gaps[['gauge_code', 'year', 'q1_gaps']]

def calculate_q2_week(df):
    """Calculate weekly cycle quality metric"""
    df_wet_days = df[df['rain_mm'] >= 1.0].copy()
    df_wet_days['year'] = df_wet_days['datetime'].dt.year
    df_wet_days['day_of_week'] = df_wet_days['datetime'].dt.dayofweek
    
    df_grouped = df_wet_days.groupby(['gauge_code', 'year', 'day_of_week']).size().reset_index(name='count')
    df_pivot = df_grouped.pivot(index=['gauge_code', 'year'], columns='day_of_week', values='count').fillna(0).reset_index()
    
    df_pivot['std'] = df_pivot[[0, 1, 2, 3, 4, 5, 6]].std(axis=1)
    df_pivot['mean'] = df_pivot[[0, 1, 2, 3, 4, 5, 6]].mean(axis=1)
    df_pivot['cv'] = df_pivot.apply(lambda x: x['std'] / x['mean'] if x['mean'] != 0 else 1.0, axis=1)
    df_pivot['q2_week'] = 100 - 100 * df_pivot['cv']
    df_pivot['q2_week'] = df_pivot['q2_week'].clip(lower=0, upper=100)
    
    df_data_year = df.copy()
    df_data_year['year'] = df_data_year['datetime'].dt.year
    df_gauge_code = df_data_year[['gauge_code', 'year']].drop_duplicates().sort_values(by=['gauge_code', 'year'])
    
    df_q2_week = df_pivot[['gauge_code', 'year', 'q2_week']]
    df_q2_week = pd.merge(df_gauge_code, df_q2_week, on=['gauge_code', 'year'], how='left')
    return df_q2_week.sort_values(by=['gauge_code', 'year']).fillna(0).reset_index(drop=True)

def calculate_q3_outliers(df):
    """Calculate outlier-related quality metric"""
    df_monthly_thresholds = df.copy(deep=True)
    df_monthly_thresholds['month'] = df_monthly_thresholds['datetime'].dt.month
    df_monthly_thresholds = df_monthly_thresholds[df_monthly_thresholds['rain_mm'] >= 1.0]
    
    monthly_stats = df_monthly_thresholds.groupby(['gauge_code', 'month'])['rain_mm'].agg(
        Q1=lambda x: x.quantile(0.25) if not x.empty else np.nan,
        Q3=lambda x: x.quantile(0.75) if not x.empty else np.nan
    ).reset_index()
    
    monthly_stats['IQR'] = monthly_stats['Q3'] - monthly_stats['Q1']
    monthly_stats['upper_bound'] = monthly_stats['Q3'] + 1.5 * monthly_stats['IQR']
    
    df_with_bounds = df.copy()
    df_with_bounds['month'] = df_with_bounds['datetime'].dt.month
    df_with_bounds = df_with_bounds.merge(monthly_stats[['gauge_code', 'month', 'upper_bound']], 
                                         on=['gauge_code', 'month'], how='left')
    
    df_with_bounds['upper_bound'] = df_with_bounds['upper_bound'].fillna(0)
    df_with_bounds['outlier'] = (df_with_bounds['rain_mm'] > df_with_bounds['upper_bound']).astype(np.uint8)
    df_with_bounds['year'] = df_with_bounds['datetime'].dt.year
    
    df_q3_outliers = df_with_bounds.groupby(['gauge_code', 'year']).agg(
        count_outliers=('outlier', 'sum'),
        active_days=('outlier', 'count')
    ).reset_index()
    
    df_q3_outliers['outlier_percentage'] = df_q3_outliers['count_outliers'] / df_q3_outliers['active_days'] * 100
    df_q3_outliers['q3_outliers'] = 100 - df_q3_outliers['outlier_percentage']
    
    return df_q3_outliers[['gauge_code', 'year', 'q3_outliers']]

def calculate_quality_index():
    """Calculate final quality index"""
    
    # Load all quality metrics
    df_p_availability = pd.read_hdf(cleaned_path, key='table_p_availability', encoding='utf-8')
    df_q1_gaps = pd.read_hdf(cleaned_path, key='table_q1_gaps', encoding='utf-8')
    df_q2_week = pd.read_hdf(cleaned_path, key='table_q2_week', encoding='utf-8')
    df_q3_outliers = pd.read_hdf(cleaned_path, key='table_q3_outliers', encoding='utf-8')
    df_preclassif = pd.read_hdf(cleaned_path, key='table_preclassif', encoding='utf-8')
    df_info = pd.read_hdf(cleaned_path, key='table_info', encoding='utf-8')
    
    # Merge all quality metrics
    df_qc_info = pd.merge(df_preclassif, df_p_availability, on=['gauge_code', 'year'], how='outer')
    df_qc_info = df_qc_info.merge(df_q1_gaps, on=['gauge_code', 'year'], how='outer')
    df_qc_info = df_qc_info.merge(df_q2_week, on=['gauge_code', 'year'], how='outer')
    df_qc_info = df_qc_info.merge(df_q3_outliers, on=['gauge_code', 'year'], how='outer')
    df_qc_info.fillna(0, inplace=True)
    
    # Calculate quality index
    df_qc_info['quality_index'] = (df_qc_info['p_availability'] + df_qc_info['q1_gaps'] + 
                                  df_qc_info['q2_week'] + df_qc_info['q3_outliers']) / 4
    
    # Calculate quality label
    def calculate_quality_label(row):
        if row['quality_index'] >= 90 and row['p_availability'] >= 99:
            return '1 - Excellent Quality'
        elif row['quality_index'] >= 85 and row['p_availability'] >= 95:
            return '2 - Good Quality'
        elif row['quality_index'] >= 80 and row['p_availability'] >= 90:
            return '3 - Acceptable Quality'
        elif row['quality_index'] >= 50:
            return '4 - Low Quality'
        else:
            return '5 - Very Low Quality'
    
    df_qc_info['quality_label'] = df_qc_info.apply(calculate_quality_label, axis=1)
    df_qc_info = pd.merge(df_info, df_qc_info, on='gauge_code', how='outer')
    
    # Add region information
    state_region_dict = {
        'AC': 'North', 'AL': 'Northeast', 'AP': 'North', 'AM': 'North', 'BA': 'Northeast',
        'CE': 'Northeast', 'DF': 'Central-West', 'ES': 'Southeast', 'GO': 'Central-West',
        'MA': 'Northeast', 'MT': 'Central-West', 'MS': 'Central-West', 'MG': 'Southeast',
        'PA': 'North', 'PB': 'Northeast', 'PR': 'South', 'PE': 'Northeast', 'PI': 'Northeast',
        'RJ': 'Southeast', 'RN': 'Northeast', 'RS': 'South', 'RO': 'North', 'RR': 'North',
        'SC': 'South', 'SP': 'Southeast', 'SE': 'Northeast', 'TO': 'North'
    }
    df_qc_info['region'] = df_qc_info['state_abbreviation'].map(state_region_dict)
    
    # Final classification
    df_qc_info['final_classif'] = df_qc_info.apply(
        lambda row: 'LQ' if row['preclassif'] == 'LQ' or row['quality_label'] in ['4 - Low Quality', '5 - Very Low Quality'] else 'HQ',
        axis=1
    )
    
    # Save final results
    df_qc_info.to_hdf(cleaned_path, key='table_qc_info', mode='r+', encoding='utf-8', 
                      append=False, complevel=9, format='table')
    
    return df_qc_info

# ==== MAIN EXECUTION ====
def main():
    """Main execution function"""
    # Step 1-4: Data Processing Pipeline
    df_data_cleaned, df_info_cleaned = clean_data()
    df_preclassif = preclassify_data(df_data_cleaned, df_info_cleaned)
    df_outlier_filtered = detect_outliers(df_data_cleaned, df_info_cleaned, df_preclassif)
    df_data_final = apply_final_filtering(df_data_cleaned)

    # Load the final filtered data for quality assessment
    df_data = load_data_from_hdf(cleaned_path, 'table_data_filtered')
    
    # Calculate quality metrics
    df_p_availability = calculate_p_availability(df_data)
    df_p_availability.to_hdf(cleaned_path, key='table_p_availability', encoding='utf-8',
                             mode='r+', format='table', complevel=9, append=False)
    
    df_q1_gaps = calculate_q1_gaps(df_data)
    df_q1_gaps.to_hdf(cleaned_path, key='table_q1_gaps', encoding='utf-8',
                      mode='r+', append=False, complevel=9, format='table')
    
    df_q2_week = calculate_q2_week(df_data)
    df_q2_week.to_hdf(cleaned_path, key='table_q2_week', encoding='utf-8',
                      mode='r+', format='table', append=False)
    
    df_q3_outliers = calculate_q3_outliers(df_data)
    df_q3_outliers.to_hdf(cleaned_path, key='table_q3_outliers', encoding='utf-8',
                          mode='r+', append=False, complevel=9, format='table')
    
    # Calculate final quality index
    df_qc_info = calculate_quality_index()

    df_qc_info.to_hdf(cleaned_path, key='table_qc_info', encoding='utf-8',
                      mode='r+', append=False, complevel=9, format='table')
    
if __name__ == "__main__":
    main()