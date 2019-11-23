
from hmda_variable_description_function import get_hmda_descriptions
import pandas as pd
import numpy as np


if __name__ == '__main__':

    lar = pd.read_csv('/mnt/ldrive/census/hmda lar 2018-static/2018_public_lar_csv2.csv.gz',
                      low_memory=False)
    print(f'Original Data Dimensions: {lar.shape}')

    for vli in ['action_taken', 'loan_purpose', 'derived_dwelling_category', 'open_end_line_of_credit',
                'business_or_commercial_purpose', 'construction_method', 'occupancy_type']:
        print(f'Variable {vli} Frequencies:\n{lar[vli].value_counts(dropna=False)}')

    lar = lar.loc[(lar["action_taken"] == 1) &
                  (lar['loan_purpose'] == 1) &
                  (lar["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built") &
                  (lar["open_end_line_of_credit"] == 2) &
                  (lar["business_or_commercial_purpose"] == 2) &
                  (lar["construction_method"] == 1) &
                  (lar["occupancy_type"] == 1) &
                  (lar["reverse_mortgage"] == 2) &
                  (lar["negative_amortization"] == 2) &
                  (lar["interest_only_payment"] != 1111) &
                  (lar["balloon_payment"] != 1111) &
                  (lar["applicant_credit_score_type"] != 1111)]
    print(f'Subset Data Dimensions: {lar.shape}')

    # 'reverse_mortgage'
    # 'negative_amortization'

    keep_cols = ['derived_loan_product_type', 'loan_amount', 'loan_to_value_ratio',
                 'discount_points', 'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period',
                 'interest_only_payment', 'balloon_payment', 'property_value', 'income',
                 'debt_to_income_ratio', 'applicant_credit_score_type', 'co_applicant_credit_score_type']

    sd = lar[keep_cols]
    print(lar[keep_cols].dtypes)
    for ki in keep_cols:
        var_unique = lar[ki].nunique()
        print(f'\n***********************************************\nUnique Values in {ki}: {var_unique}')
        if var_unique <= 10:
            print(f'\nValue counts for Variable {ki}:\n{lar[ki].value_counts(dropna=False)}')

    lar['high_priced'] = np.where(lar.rate_spread >= 1.5, 1, 0)

    lar = lar[output_columns]
    lar_sample = lar.sample(n=40000)
    lar_sample.to_csv('./data/output/hmda_lar_2018_orig_mtg_sample.csv')


    lar = pd.read_csv('/mnt/ldrive/census/hmda lar 2018-static/2018_public_lar_csv2.csv.gz',
                      low_memory=False)
    print(f'Original Data Dimensions: {lar.shape}')

    for vli in ['action_taken', 'loan_purpose', 'derived_dwelling_category', 'open_end_line_of_credit',
                'business_or_commercial_purpose', 'construction_method', 'occupancy_type']:
        print(f'Variable {vli} Frequencies:\n{lar[vli].value_counts(dropna=False)}')

    lar = lar.loc[(lar["action_taken"] == 1) &
                  (lar['loan_purpose'] == 1) &
                  (lar["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built") &
                  (lar["open_end_line_of_credit"] == 2) &
                  (lar["business_or_commercial_purpose"] == 2) &
                  (lar["construction_method"] == 1) &
                  (lar["occupancy_type"] == 1) &
                  (lar["reverse_mortgage"] == 2) &
                  (lar["negative_amortization"] == 2) &
                  (lar["interest_only_payment"] != 1111) &
                  (lar["balloon_payment"] != 1111) &
                  (lar["applicant_credit_score_type"] != 1111)]
    print(f'Subset Data Dimensions: {lar.shape}')

    # 'reverse_mortgage'
    # 'negative_amortization'

    keep_cols = ['derived_loan_product_type', 'loan_amount', 'loan_to_value_ratio',
                 'discount_points', 'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period',
                 'interest_only_payment', 'balloon_payment', 'property_value', 'income',
                 'debt_to_income_ratio', 'applicant_credit_score_type', 'co_applicant_credit_score_type']

    sd = lar[keep_cols]
    print(lar[keep_cols].dtypes)
    for ki in keep_cols:
        var_unique = lar[ki].nunique()
        print(f'\n***********************************************\nUnique Values in {ki}: {var_unique}')
        if var_unique <= 10:
            print(f'\nValue counts for Variable {ki}:\n{lar[ki].value_counts(dropna=False)}')

    lar['interest_only_payment_desc'] = pd.Series(
        np.select(
            (
                lar.interest_only_payment == 1,
                lar.interest_only_payment == 2,
                lar.interest_only_payment == 1111,
            ),
            (
                'Interest-only payments',
                'No interest-only payments',
                'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['balloon_payment_desc'] = pd.Series(
        np.select(
            (
                lar.balloon_payment == 1,
                lar.balloon_payment == 2,
                lar.balloon_payment == 1111,
            ),
            ('Balloon Payment', 'No balloon payment', 'Exempt'),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_credit_score_type_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_credit_score_type == 1,
                lar.applicant_credit_score_type == 2,
                lar.applicant_credit_score_type == 3,
                lar.applicant_credit_score_type == 4,
                lar.applicant_credit_score_type == 5,
                lar.applicant_credit_score_type == 6,
                lar.applicant_credit_score_type == 7,
                lar.applicant_credit_score_type == 8,
                lar.applicant_credit_score_type == 9,
                lar.applicant_credit_score_type == 1111,
            ),
            (
                'Equifax Beacon 5.0',
                'Experian Fair Isaac',
                'FICO Risk Score Classic 04',
                'FICO Risk Score Classic 98',
                'VantageScore 2.0',
                'VantageScore 3.0',
                'More than one credit scoring model',
                'Other credit scoring model',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_credit_score_type_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_credit_score_type == 1,
                lar.co_applicant_credit_score_type == 2,
                lar.co_applicant_credit_score_type == 3,
                lar.co_applicant_credit_score_type == 4,
                lar.co_applicant_credit_score_type == 5,
                lar.co_applicant_credit_score_type == 6,
                lar.co_applicant_credit_score_type == 7,
                lar.co_applicant_credit_score_type == 8,
                lar.co_applicant_credit_score_type == 9,
                lar.co_applicant_credit_score_type == 10,
                lar.co_applicant_credit_score_type == 1111,
            ),
            (
                'Equifax Beacon 5.0',
                'Experian Fair Isaac',
                'FICO Risk Score Classic 04',
                'FICO Risk Score Classic 98',
                'VantageScore 2.0',
                'VantageScore 3.0',
                'More than one credit scoring model',
                'Other credit scoring model',
                'Not applicable',
                'No co-applicant',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['preapproval_desc'] = pd.Series(
        np.select(
            (lar.preapproval == 1, lar.preapproval == 2),
            ('Preapproval requested', 'Preapproval not requested'),
            default=''
        ),
        dtype='category'
    )
    lar['loan_type_desc'] = pd.Series(
        np.select(
            (
                lar.loan_type == 1,
                lar.loan_type == 2,
                lar.loan_type == 3,
                lar.loan_type == 4
            ),
            (
                'Conventional (not insured or guaranteed by FHA, VA, RHS, or FSA)',
                'Federal Housing Administration insured (FHA)',
                'Veterans Affairs guaranteed (VA)',
                'USDA Rural Housing Service or Farm Service Agency guaranteed (RHS or FSA)',
            ),
            default=''
        ),
        dtype='category'
    )

    keep_cols = keep_cols + ["derived_msa_md", "state_code", "county_code", "conforming_loan_limit", "preapproval",
                             "loan_type", "lien_status", "rate_spread", "total_loan_costs", "loan_term"]
    [y for x in keep_cols  for y in lar.columns.values if x + "_desc" in y]
    lar["lien_status"].value_counts(dropna=False)
    '''
    (lar["loan_term"] >= 180) & (lar["loan_term"] <= 360) &
    (lar["hoepa_status"] == 2)
    '''