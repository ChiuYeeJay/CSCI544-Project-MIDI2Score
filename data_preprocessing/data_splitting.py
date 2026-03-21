import pandas as pd
import numpy as np
import os

PDMX_PREPROCESSED_ROOT = "../dataset/PDMX_preprocessed/"

def stratified_greedy_split(scores_df, ratio=(16, 4, 5)):
    """
    Performs stratified greedy allocation on the scores DataFrame.
    
    Args:
        scores_df (pd.DataFrame): DataFrame containing 'origin', 'total_tokens', and 'stratum'.
        ratio (tuple): Target ratio for (train, val, test).

    Returns:
        dict: A dictionary with keys 'training', 'validation', 'test', 
              where values are lists of 'origin' identifiers assigned to each set.
    """
    # 1. Calculate targets
    total_tokens = scores_df['total_tokens'].sum()
    total_ratio = sum(ratio)
    
    targets = {
        'training': total_tokens * (ratio[0] / total_ratio),
        'validation': total_tokens * (ratio[1] / total_ratio),
        'test': total_tokens * (ratio[2] / total_ratio),
    }

    # 2. Initialize counters and containers
    current_counts = {'training': 0, 'validation': 0, 'test': 0}
    partitions = {'training': [], 'validation': [], 'test': []}

    # 3. Iterate through strata (from longest average tokens to shortest)
    # Ensure strata order is sorted descending by mean token count
    strata_order = sorted(
        scores_df['stratum'].unique(), 
        key=lambda x: scores_df[scores_df['stratum']==x]['total_tokens'].mean(), 
        reverse=True
    )
    
    print(f"\nProcessing strata in the following order: {strata_order}")

    for stratum in strata_order:
        # Get scores for current stratum and sort by token count descending
        stratum_scores = scores_df[scores_df['stratum'] == stratum].sort_values(
            by='total_tokens', ascending=False
        )
        
        # 4. Perform greedy allocation within the stratum
        for _, score in stratum_scores.iterrows():
            # Calculate the deficit (completion ratio) for each partition
            # Handle division by zero for the first allocation
            deficits = {}
            for part_name in partitions.keys():
                if targets[part_name] == 0:
                    deficits[part_name] = np.inf # If target is 0, never allocate to it
                else:
                    deficits[part_name] = current_counts[part_name] / targets[part_name]

            # Find the partition with the smallest deficit (highest need for tokens)
            min_partition = min(deficits, key=deficits.get)
            
            # Allocate score
            partitions[min_partition].append(score['origin'])
            current_counts[min_partition] += score['total_tokens']
            
    return partitions

# --- Main Execution Flow ---

# Assuming info_csv is your loaded DataFrame
info_csv_path = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info.csv")
info_csv = pd.read_csv(info_csv_path)

print("\n--- Step 1: Data Preprocessing - Grouping by 'Origin' Source ---")
# Group by 'origin' and sum the token lengths for each score
scores_df = info_csv.groupby('origin').agg(
    total_tokens=('n_tokens', 'sum')
    # total_tokens=('lmx_token_length', 'sum')
).reset_index()
print("Data aggregated from 'Part' level to 'Score' level:")
print(scores_df.head())

print("\n--- Step 2: Creating Strata ---")
# Use qcut to stratify total_tokens into quartiles
try:
    scores_df['stratum'] = pd.qcut(
        scores_df['total_tokens'],
        q=4,
        labels=['Short', 'Medium', 'Long', 'Very Long'],
        duplicates='drop' # Merge duplicate bin edges if they exist
    )
except ValueError as e:
    print(f"Stratification failed, possibly due to small dataset or extreme distribution: {e}")
    # Fallback to single stratum strategy if qcut fails
    scores_df['stratum'] = 'all'

print("Strata created. Number of scores per stratum:")
print(scores_df['stratum'].value_counts())

print("\n--- Step 3: Executing Stratified Greedy Allocation ---")
partitions = stratified_greedy_split(scores_df, ratio=(16, 4, 5))

print("\n--- Step 4: Verification and Final Data Splitting ---")
print("Allocation results (Number of Scores):")
for name, origins in partitions.items():
    print(f"- {name.capitalize()} set: {len(origins)} scores")

# Verify total Token distribution ratio
results = []
total_actual_tokens = 0
for name, origins in partitions.items():
    actual_tokens = scores_df[scores_df['origin'].isin(origins)]['total_tokens'].sum()
    total_actual_tokens += actual_tokens
    results.append({'Partition': name, 'Actual Tokens': actual_tokens})

# Calculate percentage and error
summary_df = pd.DataFrame(results)
summary_df['Actual Percentage (%)'] = (summary_df['Actual Tokens'] / total_actual_tokens * 100).round(2)

total_ratio = sum((16, 4, 5))
summary_df['Target Percentage (%)'] = [r / total_ratio * 100 for r in (16, 4, 5)]
summary_df['Error (%)'] = (summary_df['Actual Percentage (%)'] - summary_df['Target Percentage (%)']).round(2)

print("\nToken Distribution Verification:")
print(summary_df)

# Finally, split the original info_csv based on the allocation
print("\nGenerating final training/validation/test DataFrames...")
train_df = info_csv[info_csv['origin'].isin(partitions['training'])]
validation_df = info_csv[info_csv['origin'].isin(partitions['validation'])]
test_df = info_csv[info_csv['origin'].isin(partitions['test'])]

print(f"\nTraining set   : {train_df.shape[0]} parts")
print(f"Validation set : {validation_df.shape[0]} parts")
print(f"Test set    : {test_df.shape[0]} parts")
print(f"Total          : {train_df.shape[0] + validation_df.shape[0] + test_df.shape[0]} parts (Consistent with original {info_csv.shape[0]} parts)")

print("\nStatistical Description per Partition:")
print("Training set:")
print(train_df.describe())
print("\nValidation set:")
print(validation_df.describe())
print("\nTest set:")
print(test_df.describe())

# (Execute after obtaining the partitions dictionary)

print("\n--- Extra Verification: Checking Strata Distribution across Partitions ---")

# Create a DataFrame containing allocation results
assigned_df = scores_df.copy()
origin_to_partition = {origin: part for part, origins in partitions.items() for origin in origins}
assigned_df['partition'] = assigned_df['origin'].map(origin_to_partition)

# Use crosstab for cross-analysis
distribution_check = pd.crosstab(
    assigned_df['stratum'],
    assigned_df['partition']
)

print("Distribution of score counts across strata:")
print(distribution_check)

# View percentage distribution for clarity
distribution_perc = pd.crosstab(
    assigned_df['stratum'],
    assigned_df['partition'],
    normalize='index' # Normalize by row to calculate percentage within each stratum
).apply(lambda x: (x * 100).round(2))

print("\nPercentage distribution across strata (Targets: Training:64%, Val:16%, Test:20%):")
print(distribution_perc)

# --- Final Step: Integrating Results back to the Original DataFrame ---

# We already have the partitions dictionary
# partitions = {'training': ['score_001.mxl', ...], 'validation': [...], 'test': [...]}

# Create a reverse mapping dictionary: {origin -> partition} for easier lookup
print("--- Integrating allocation results back into original DataFrame ---")
origin_to_partition_map = {
    origin: partition_name
    for partition_name, origins in partitions.items()
    for origin in origins
}

# Use .map() to label each row (each part) with its assigned partition
info_csv['partition'] = info_csv['origin'].map(origin_to_partition_map)

print("\nSuccessfully added 'partition' column!")

# --- Final Verification ---
print("\nDisplaying first 5 rows of the updated DataFrame:")
print(info_csv.head())

print("\nDisplaying last 5 rows to confirm mapping completeness:")
print(info_csv.tail())

print("\nFinal count of 'Musical Parts' per partition:")
print(info_csv['partition'].value_counts())

# --- Save Results ---
# Suggestion: Save the full DataFrame with partition labels to a new CSV
# for future use without needing to re-run the split.
try:
    output_path = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info_with_partitions.csv")
    info_csv.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
except Exception as e:
    print(f"\nFailed to save file: {e}")