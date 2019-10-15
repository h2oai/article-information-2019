import pandas as pd
import random

"""
Subset the original data and create a training and test set.
"""


random.seed(42)

# Load the data
DATA = pd.read_csv('./hmda_lar_2018_orig_mtg_sample.csv')



# Keep only certain subsets
DATA = DATA[DATA["derived_dwelling_category"]=="Single Family (1-4 Units):Site-Built"]
# Subset to "action_taken"=="Loan originated"
DATA = DATA[DATA["action_taken"] == 1]
# Drop "open_end_line_of_credit"=="Not an open-end line of credit"
DATA = DATA[DATA["open_end_line_of_credit"] != 2] 
# Subset to "business_or_commercial_purpose"=="Not primarily for a business or commercial purpose"
DATA = DATA[DATA["business_or_commercial_purpose"]==2] 
# Subset to "construction_method"==Site-built
DATA = DATA[DATA["construction_method"] == 1] 
# Subset to "occupancy_type"=="Principal residence"
DATA = DATA[DATA["occupancy_type"]== 1] 



# List of columns to keep
keep_columns = ['Unnamed: 0', "high_priced"]
keep_columns += ['derived_loan_product_type', 'loan_purpose', 'reverse_mortgage']
keep_columns += ['loan_amount', 'loan_to_value_ratio', 'discount_points']
keep_columns += ['lender_credits', 'loan_term', 'prepayment_penalty_term']
keep_columns += ['intro_rate_period', 'negative_amortization', 'interest_only_payment']
keep_columns += ['balloon_payment', 'property_value', 'income', 'debt_to_income_ratio']
keep_columns += ['applicant_credit_score_type', 'co_applicant_credit_score_type']

# Keep only the interesting columns
DATA = DATA[keep_columns]



# Shuffle the dataset then create a training and test set
fraction_train = 0.8
DATA = DATA.sample(frac=1)
TRAIN = DATA.iloc[0:(int(fraction_train*len(DATA))),:].copy()
TEST= DATA.iloc[(int(fraction_train*len(DATA))):,:].copy()


# Create a cv column in the training set
TRAIN['cv_fold'] = random.choices(list(range(0, 5)), k=len(TRAIN))


# Save the training and test set
TRAIN.to_csv('output/TRAIN.csv', index=False)
TEST.to_csv('output/TEST.csv', index=False)












