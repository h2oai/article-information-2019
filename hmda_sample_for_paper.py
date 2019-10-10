from typing import Union
import pandas as pd
import numpy as np
import sqlalchemy


def optimize_data_types(
        df: Union[pd.Series, pd.DataFrame],
        verbose: int = 0,
        shrink_ints=1,
        shrink_floats=1,
        shrink_strings=1
):
    """
    Reviews all variables in a dataframe or values in a series to determine the
     smallest datatype that can be used while not losing information.
    :param df - Pandas dataframe or series for use in the datatype review
    :param verbose - int that determines whether information is printed
    value = 0 means no printing; higher values lead to more detail
    being printed.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        orig_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        if shrink_ints:
            df_int = df.select_dtypes(include=['int'])
            converted_int = df_int.apply(pd.to_numeric, downcast='integer')
            df[converted_int.columns] = converted_int
            int_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2
        else:
            int_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2

        if shrink_floats:
            df_float = df.select_dtypes(include=['float'])
            converted_float = df_float.apply(pd.to_numeric, downcast='float')
            df[converted_float.columns] = converted_float
            float_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2
        else:
            float_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2

        if shrink_strings:
            df_object = df.select_dtypes(include=['object'])
            for col in df_object:
                unique = len(df_object[col].unique())
                total = len(df_object[col])
                if unique / total < 0.5:
                    df[col] = df_object[col].astype('category')
            category_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2
        else:
            category_reduce = df.memory_usage(deep=True).sum() / 1024 ** 2
        if verbose > 0:
            df.info(memory_usage='deep')
            print("{:03.2f} MB Originally".format(orig_mem))
            print(
                "{0:03.2f} MB {1:5.2f}% After Int Reduction".format(
                    int_reduce, (int_reduce - orig_mem) / orig_mem * 100.0
                )
            )
            print(
                "{0:03.2f} MB {1:5.2f}% After Float Reduction".format(
                    float_reduce,
                    (float_reduce - orig_mem) / orig_mem * 100.0
                )
            )
            print(
                "{0:03.2f} MB {1:5.2f}% After Category Reduction".format(
                    category_reduce,
                    (category_reduce - orig_mem) / orig_mem * 100.0
                )
            )
        return df
    elif isinstance(df, pd.Series):
        orig_mem = df.memory_usage(deep=True) / 1024 ** 2
        orig_dtype = df.dtype.name
        if df.dtype.name.startswith('int') or df.dtype.name.startswith('uint'):
            df = pd.to_numeric(df, downcast='unsigned')
        elif df.dtype.name.startswith('float'):
            df = pd.to_numeric(df, downcast='float')
        elif df.dtype.name == 'object':
            unique = len(df.unique())
            total = len(df)
            if unique / total < 0.5:
                df = df.astype('category')
        if verbose > 0:
            post_mem = df.memory_usage(deep=True) / 1024 ** 2
            print("{0:03.2f} MB dtype {1} Originally".format(orig_mem, orig_dtype))
            print(
                "{0:03.2f} MB dtype {1} {2:5.2f}% After Reduction".format(
                    post_mem, df.dtype.name, (post_mem - orig_mem) / orig_mem * 100.0
                )
            )
        return df


def initial_import():
    connect_string = 'mysql://root:XXXXXXX@linus/census'
    db = sqlalchemy.create_engine(connect_string)
    lar_data = pd.read_sql(
        '''
        # Limit Originated Mortgage Records Only
        SELECT *
        FROM census.hmda_lar_national_2018
        WHERE action_taken = 1
        AND loan_purpose = 1
        ''',
        db
    )
    lar_data.columns = [i.lower() for i in lar.columns]
    lar_data = optimize_data_types(lar, verbose=1, shrink_floats=0)
    lar_data['total_loan_costs'] = pd.to_numeric(lar_data['total_loan_costs'], errors='coerce')
    for var in (
            'applicant_ethnicity_1', 'applicant_ethnicity_2',
            'applicant_ethnicity_3', 'applicant_ethnicity_4',
            'applicant_ethnicity_5', 'co_applicant_ethnicity_1',
            'co_applicant_ethnicity_2', 'co_applicant_ethnicity_3',
            'co_applicant_ethnicity_4', 'co_applicant_ethnicity_5',
            'applicant_ethnicity_observed', 'co_applicant_ethnicity_observed',
            'applicant_race_1', 'applicant_race_2', 'applicant_race_3',
            'applicant_race_4', 'applicant_race_5', 'co_applicant_race_1',
            'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',
            'co_applicant_race_5', 'applicant_race_observed',
            'co_applicant_race_observed', 'applicant_sex', 'co_applicant_sex',
            'applicant_sex_observed', 'co_applicant_sex_observed',
            'submission_of_application', 'initially_payable_to_institution',
            'aus_1', 'aus_2', 'aus_3', 'aus_4', 'aus_5',
            'denial_reason_1', 'denial_reason_2', 'denial_reason_3',
            'denial_reason_4',
            'tract_population', 'ffiec_msa_md_median_family_income',
            'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
            'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units',
    ):
        print('downcasting', var)
        lar_data[var] = pd.to_numeric(lar_data[var], errors='coerce', downcast='integer')
    lar_data.to_parquet('/mnt/ldrive/census/hmda lar 2018-static/nschmidt_sample.parquet')

if __name__ == '__main__':
    first_run = False
    if first_run:
        initial_import()
    lar = pd.read_parquet('/mnt/ldrive/census/hmda lar 2018-static/nschmidt_sample.parquet')
    lar['conforming_loan_limit_desc'] = pd.Series(
        np.select(
            [
                lar.conforming_loan_limit == 'C',
                lar.conforming_loan_limit == 'NC',
                lar.conforming_loan_limit == 'U',
                lar.conforming_loan_limit == 'NA'
            ],
            [
                'Conforming',
                'Nonconforming',
                'Undetermined',
                'Not applicable'
            ],
            ''
        ),
        dtype='category'
    )
    lar['action_taken_desc'] = pd.Series(
        np.select(
            (
                lar.action_taken == 1, lar.action_taken == 2, lar.action_taken == 3,
                lar.action_taken == 4, lar.action_taken == 5, lar.action_taken == 6,
                lar.action_taken == 7, lar.action_taken == 8
            ),
            (
                'Loan originated', 'Application approved but not accepted',
                'Application denied', 'Application withdrawn by applicant',
                'File closed for incompleteness', 'Purchased loan',
                'Preapproval request denied',
                'Preapproval request approved but not accepted'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['purchaser_type_desc'] = pd.Series(
        np.select(
            (
                lar.purchaser_type == 0, lar.purchaser_type == 1, lar.purchaser_type == 2,
                lar.purchaser_type == 3, lar.purchaser_type == 4, lar.purchaser_type == 5,
                lar.purchaser_type == 6, lar.purchaser_type == 71, lar.purchaser_type == 72,
                lar.purchaser_type == 8, lar.purchaser_type == 9
            ),
            (
                'Not applicable', 'Fannie Mae', 'Ginnie Mae', 'Freddie Mac',
                'Farmer Mac', 'Private securitizer',
                'Commercial bank, savings bank, or savings association',
                'Credit union, mortgage company, or finance company',
                'Life insurance company', 'Affiliate institution',
                'Other type of purchaser',
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
    lar['loan_purpose_desc'] = pd.Series(
        np.select(
            (
                lar.loan_purpose == 1, lar.loan_purpose == 2, lar.loan_purpose == 31,
                lar.loan_purpose == 32, lar.loan_purpose == 4, lar.loan_purpose == 5,
            ),
            (
                'Home purchase', 'Home improvement', 'Refinancing',
                'Cash-out refinancing', 'Other purpose', 'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['lien_status_desc'] = pd.Series(
        np.select(
            (lar.lien_status == 1, lar.lien_status == 2),
            ('Secured by a first lien', 'Secured by a subordinate lien'),
            default=''
        ),
        dtype='category'
    )
    lar['reverse_mortgage_desc'] = pd.Series(
        np.select(
            (
                lar.reverse_mortgage == 1, lar.reverse_mortgage == 2,
                lar.reverse_mortgage == 1111
            ),
            ('Reverse mortgage', 'Not a reverse mortgage', 'Exempt'),
            default=''
        ),
        dtype='category'
    )
    lar['open_end_line_of_credit_desc'] = pd.Series(
        np.select(
            (
                lar.open_end_line_of_credit == 1,
                lar.open_end_line_of_credit == 2,
                lar.open_end_line_of_credit == 1111
            ),
            (
                'Open-end line of credit',
                'Not an open-end line of credit',
                'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['business_or_commercial_purpose_desc'] = pd.Series(
        np.select(
            (
                lar.business_or_commercial_purpose == 1,
                lar.business_or_commercial_purpose == 2,
                lar.business_or_commercial_purpose == 1111,
            ),
            (
                'Primarily for a business or commercial purpose',
                'Not primarily for a business or commercial purpose',
                'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['hoepa_status_desc'] = pd.Series(
        np.select(
            (lar.hoepa_status == 1, lar.hoepa_status == 2, lar.hoepa_status == 3),
            ('High-cost mortgage', 'Not a high-cost mortgage', 'Not Applicable'),
            default=''
        ),
        dtype='category'
    )
    lar['negative_amortization_desc'] = pd.Series(
        np.select(
            (
                lar.negative_amortization == 1,
                lar.negative_amortization == 2,
                lar.negative_amortization == 1111
            ),
            (
                'Negative amortization', 'No negative amortization', 'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
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
    lar['other_nonamortizing_features_desc'] = pd.Series(
        np.select(
            (
                lar.other_nonamortizing_features == 1,
                lar.other_nonamortizing_features == 2,
                lar.other_nonamortizing_features == 1111,
            ),
            (
                'Other non-fully amortizing features',
                'No other non-fully amortizing features',
                'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['construction_method_desc'] = pd.Series(
        np.select(
            (
                lar.construction_method == 1,
                lar.construction_method == 2
            ),
            ('Site-built', 'Manufactured home'),
            default=''
        ),
        dtype='category'
    )
    lar['occupancy_type_desc'] = pd.Series(
        np.select(
            (
                lar.occupancy_type == 1,
                lar.occupancy_type == 2,
                lar.occupancy_type == 3,
            ),
            (
                'Principal residence',
                'Second residence',
                'Investment property',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['manufactured_home_secured_property_type_desc'] = pd.Series(
        np.select(
            (
                lar.manufactured_home_secured_property_type == 1,
                lar.manufactured_home_secured_property_type == 2,
                lar.manufactured_home_secured_property_type == 3,
                lar.manufactured_home_secured_property_type == 1111,
            ),
            (
                'Manufactured home and land',
                'Manufactured home and not land',
                'Not Applicable',
                'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['manufactured_home_land_property_interest_desc'] = pd.Series(
        np.select(
            (
                lar.manufactured_home_land_property_interest == 1,
                lar.manufactured_home_land_property_interest == 2,
                lar.manufactured_home_land_property_interest == 3,
                lar.manufactured_home_land_property_interest == 4,
                lar.manufactured_home_land_property_interest == 5,
                lar.manufactured_home_land_property_interest == 1111,
            ),
            (
                'Direct ownership', 'Indirect ownership', 'Paid leasehold',
                'Unpaid leasehold', 'Not Applicable', 'Exempt'
            ),
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
    lar['applicant_ethnicity_1_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_1 == 1,
                lar.applicant_ethnicity_1 == 11,
                lar.applicant_ethnicity_1 == 12,
                lar.applicant_ethnicity_1 == 13,
                lar.applicant_ethnicity_1 == 14,
                lar.applicant_ethnicity_1 == 2,
                lar.applicant_ethnicity_1 == 3,
                lar.applicant_ethnicity_1 == 4,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_ethnicity_2_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_2 == 1,
                lar.applicant_ethnicity_2 == 11,
                lar.applicant_ethnicity_2 == 12,
                lar.applicant_ethnicity_2 == 13,
                lar.applicant_ethnicity_2 == 14,
                lar.applicant_ethnicity_2 == 2,
                lar.applicant_ethnicity_2 == 3,
                lar.applicant_ethnicity_2 == 4,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_ethnicity_3_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_3 == 1,
                lar.applicant_ethnicity_3 == 11,
                lar.applicant_ethnicity_3 == 12,
                lar.applicant_ethnicity_3 == 13,
                lar.applicant_ethnicity_3 == 14,
                lar.applicant_ethnicity_3 == 2,
                lar.applicant_ethnicity_3 == 3,
                lar.applicant_ethnicity_3 == 4,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_ethnicity_4_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_4 == 1,
                lar.applicant_ethnicity_4 == 11,
                lar.applicant_ethnicity_4 == 12,
                lar.applicant_ethnicity_4 == 13,
                lar.applicant_ethnicity_4 == 14,
                lar.applicant_ethnicity_4 == 2,
                lar.applicant_ethnicity_4 == 3,
                lar.applicant_ethnicity_4 == 4,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_ethnicity_5_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_5 == 1,
                lar.applicant_ethnicity_5 == 11,
                lar.applicant_ethnicity_5 == 12,
                lar.applicant_ethnicity_5 == 13,
                lar.applicant_ethnicity_5 == 14,
                lar.applicant_ethnicity_5 == 2,
                lar.applicant_ethnicity_5 == 3,
                lar.applicant_ethnicity_5 == 4,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_1_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_1 == 1,
                lar.co_applicant_ethnicity_1 == 11,
                lar.co_applicant_ethnicity_1 == 12,
                lar.co_applicant_ethnicity_1 == 13,
                lar.co_applicant_ethnicity_1 == 14,
                lar.co_applicant_ethnicity_1 == 2,
                lar.co_applicant_ethnicity_1 == 3,
                lar.co_applicant_ethnicity_1 == 4,
                lar.co_applicant_ethnicity_1 == 5,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_2_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_2 == 1,
                lar.co_applicant_ethnicity_2 == 11,
                lar.co_applicant_ethnicity_2 == 12,
                lar.co_applicant_ethnicity_2 == 13,
                lar.co_applicant_ethnicity_2 == 14,
                lar.co_applicant_ethnicity_2 == 2,
                lar.co_applicant_ethnicity_2 == 3,
                lar.co_applicant_ethnicity_2 == 4,
                lar.co_applicant_ethnicity_2 == 5,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_3_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_3 == 1,
                lar.co_applicant_ethnicity_3 == 11,
                lar.co_applicant_ethnicity_3 == 12,
                lar.co_applicant_ethnicity_3 == 13,
                lar.co_applicant_ethnicity_3 == 14,
                lar.co_applicant_ethnicity_3 == 2,
                lar.co_applicant_ethnicity_3 == 3,
                lar.co_applicant_ethnicity_3 == 4,
                lar.co_applicant_ethnicity_3 == 5,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_4_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_4 == 1,
                lar.co_applicant_ethnicity_4 == 11,
                lar.co_applicant_ethnicity_4 == 12,
                lar.co_applicant_ethnicity_4 == 13,
                lar.co_applicant_ethnicity_4 == 14,
                lar.co_applicant_ethnicity_4 == 2,
                lar.co_applicant_ethnicity_4 == 3,
                lar.co_applicant_ethnicity_4 == 4,
                lar.co_applicant_ethnicity_4 == 5,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_5_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_5 == 1,
                lar.co_applicant_ethnicity_5 == 11,
                lar.co_applicant_ethnicity_5 == 12,
                lar.co_applicant_ethnicity_5 == 13,
                lar.co_applicant_ethnicity_5 == 14,
                lar.co_applicant_ethnicity_5 == 2,
                lar.co_applicant_ethnicity_5 == 3,
                lar.co_applicant_ethnicity_5 == 4,
                lar.co_applicant_ethnicity_5 == 5,
            ),
            (
                'Hispanic or Latino',
                'Mexican',
                'Puerto Rican',
                'Cuban',
                'Other Hispanic or Latino',
                'Not Hispanic or Latino',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_ethnicity_observed_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_ethnicity_observed == 1,
                lar.applicant_ethnicity_observed == 2,
                lar.applicant_ethnicity_observed == 3,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not Applicable'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_ethnicity_observed_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_ethnicity_observed == 1,
                lar.co_applicant_ethnicity_observed == 2,
                lar.co_applicant_ethnicity_observed == 3,
                lar.co_applicant_ethnicity_observed == 4,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not Applicable',
                'No co-applicant'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_1_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_1 == 1,
                lar.applicant_race_1 == 2,
                lar.applicant_race_1 == 21,
                lar.applicant_race_1 == 22,
                lar.applicant_race_1 == 23,
                lar.applicant_race_1 == 24,
                lar.applicant_race_1 == 25,
                lar.applicant_race_1 == 26,
                lar.applicant_race_1 == 27,
                lar.applicant_race_1 == 3,
                lar.applicant_race_1 == 4,
                lar.applicant_race_1 == 41,
                lar.applicant_race_1 == 42,
                lar.applicant_race_1 == 43,
                lar.applicant_race_1 == 44,
                lar.applicant_race_1 == 5,
                lar.applicant_race_1 == 6,
                lar.applicant_race_1 == 7,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_2_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_2 == 1,
                lar.applicant_race_2 == 2,
                lar.applicant_race_2 == 21,
                lar.applicant_race_2 == 22,
                lar.applicant_race_2 == 23,
                lar.applicant_race_2 == 24,
                lar.applicant_race_2 == 25,
                lar.applicant_race_2 == 26,
                lar.applicant_race_2 == 27,
                lar.applicant_race_2 == 3,
                lar.applicant_race_2 == 4,
                lar.applicant_race_2 == 41,
                lar.applicant_race_2 == 42,
                lar.applicant_race_2 == 43,
                lar.applicant_race_2 == 44,
                lar.applicant_race_2 == 5,
                lar.applicant_race_2 == 6,
                lar.applicant_race_2 == 7,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_3_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_3 == 1,
                lar.applicant_race_3 == 2,
                lar.applicant_race_3 == 21,
                lar.applicant_race_3 == 22,
                lar.applicant_race_3 == 23,
                lar.applicant_race_3 == 24,
                lar.applicant_race_3 == 25,
                lar.applicant_race_3 == 26,
                lar.applicant_race_3 == 27,
                lar.applicant_race_3 == 3,
                lar.applicant_race_3 == 4,
                lar.applicant_race_3 == 41,
                lar.applicant_race_3 == 42,
                lar.applicant_race_3 == 43,
                lar.applicant_race_3 == 44,
                lar.applicant_race_3 == 5,
                lar.applicant_race_3 == 6,
                lar.applicant_race_3 == 7,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_4_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_4 == 1,
                lar.applicant_race_4 == 2,
                lar.applicant_race_4 == 21,
                lar.applicant_race_4 == 22,
                lar.applicant_race_4 == 23,
                lar.applicant_race_4 == 24,
                lar.applicant_race_4 == 25,
                lar.applicant_race_4 == 26,
                lar.applicant_race_4 == 27,
                lar.applicant_race_4 == 3,
                lar.applicant_race_4 == 4,
                lar.applicant_race_4 == 41,
                lar.applicant_race_4 == 42,
                lar.applicant_race_4 == 43,
                lar.applicant_race_4 == 44,
                lar.applicant_race_4 == 5,
                lar.applicant_race_4 == 6,
                lar.applicant_race_4 == 7,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_5_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_5 == 1,
                lar.applicant_race_5 == 2,
                lar.applicant_race_5 == 21,
                lar.applicant_race_5 == 22,
                lar.applicant_race_5 == 23,
                lar.applicant_race_5 == 24,
                lar.applicant_race_5 == 25,
                lar.applicant_race_5 == 26,
                lar.applicant_race_5 == 27,
                lar.applicant_race_5 == 3,
                lar.applicant_race_5 == 4,
                lar.applicant_race_5 == 41,
                lar.applicant_race_5 == 42,
                lar.applicant_race_5 == 43,
                lar.applicant_race_5 == 44,
                lar.applicant_race_5 == 5,
                lar.applicant_race_5 == 6,
                lar.applicant_race_5 == 7,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_1_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_1 == 1,
                lar.co_applicant_race_1 == 2,
                lar.co_applicant_race_1 == 21,
                lar.co_applicant_race_1 == 22,
                lar.co_applicant_race_1 == 23,
                lar.co_applicant_race_1 == 24,
                lar.co_applicant_race_1 == 25,
                lar.co_applicant_race_1 == 26,
                lar.co_applicant_race_1 == 27,
                lar.co_applicant_race_1 == 3,
                lar.co_applicant_race_1 == 4,
                lar.co_applicant_race_1 == 41,
                lar.co_applicant_race_1 == 42,
                lar.co_applicant_race_1 == 43,
                lar.co_applicant_race_1 == 44,
                lar.co_applicant_race_1 == 5,
                lar.co_applicant_race_1 == 6,
                lar.co_applicant_race_1 == 7,
                lar.co_applicant_race_1 == 8,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_2_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_2 == 1,
                lar.co_applicant_race_2 == 2,
                lar.co_applicant_race_2 == 21,
                lar.co_applicant_race_2 == 22,
                lar.co_applicant_race_2 == 23,
                lar.co_applicant_race_2 == 24,
                lar.co_applicant_race_2 == 25,
                lar.co_applicant_race_2 == 26,
                lar.co_applicant_race_2 == 27,
                lar.co_applicant_race_2 == 3,
                lar.co_applicant_race_2 == 4,
                lar.co_applicant_race_2 == 41,
                lar.co_applicant_race_2 == 42,
                lar.co_applicant_race_2 == 43,
                lar.co_applicant_race_2 == 44,
                lar.co_applicant_race_2 == 5,
                lar.co_applicant_race_2 == 6,
                lar.co_applicant_race_2 == 7,
                lar.co_applicant_race_2 == 8,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_3_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_3 == 1,
                lar.co_applicant_race_3 == 2,
                lar.co_applicant_race_3 == 21,
                lar.co_applicant_race_3 == 22,
                lar.co_applicant_race_3 == 23,
                lar.co_applicant_race_3 == 24,
                lar.co_applicant_race_3 == 25,
                lar.co_applicant_race_3 == 26,
                lar.co_applicant_race_3 == 27,
                lar.co_applicant_race_3 == 3,
                lar.co_applicant_race_3 == 4,
                lar.co_applicant_race_3 == 41,
                lar.co_applicant_race_3 == 42,
                lar.co_applicant_race_3 == 43,
                lar.co_applicant_race_3 == 44,
                lar.co_applicant_race_3 == 5,
                lar.co_applicant_race_3 == 6,
                lar.co_applicant_race_3 == 7,
                lar.co_applicant_race_3 == 8,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_4_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_4 == 1,
                lar.co_applicant_race_4 == 2,
                lar.co_applicant_race_4 == 21,
                lar.co_applicant_race_4 == 22,
                lar.co_applicant_race_4 == 23,
                lar.co_applicant_race_4 == 24,
                lar.co_applicant_race_4 == 25,
                lar.co_applicant_race_4 == 26,
                lar.co_applicant_race_4 == 27,
                lar.co_applicant_race_4 == 3,
                lar.co_applicant_race_4 == 4,
                lar.co_applicant_race_4 == 41,
                lar.co_applicant_race_4 == 42,
                lar.co_applicant_race_4 == 43,
                lar.co_applicant_race_4 == 44,
                lar.co_applicant_race_4 == 5,
                lar.co_applicant_race_4 == 6,
                lar.co_applicant_race_4 == 7,
                lar.co_applicant_race_4 == 8,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_5_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_5 == 1,
                lar.co_applicant_race_5 == 2,
                lar.co_applicant_race_5 == 21,
                lar.co_applicant_race_5 == 22,
                lar.co_applicant_race_5 == 23,
                lar.co_applicant_race_5 == 24,
                lar.co_applicant_race_5 == 25,
                lar.co_applicant_race_5 == 26,
                lar.co_applicant_race_5 == 27,
                lar.co_applicant_race_5 == 3,
                lar.co_applicant_race_5 == 4,
                lar.co_applicant_race_5 == 41,
                lar.co_applicant_race_5 == 42,
                lar.co_applicant_race_5 == 43,
                lar.co_applicant_race_5 == 44,
                lar.co_applicant_race_5 == 5,
                lar.co_applicant_race_5 == 6,
                lar.co_applicant_race_5 == 7,
                lar.co_applicant_race_5 == 8,
            ),
            (
                'American Indian or Alaska Native',
                'Asian',
                'Asian Indian',
                'Chinese',
                'Filipino',
                'Japanese',
                'Korean',
                'Vietnamese',
                'Other Asian',
                'Black or African American',
                'Native Hawaiian or Other Pacific Islander',
                'Native Hawaiian',
                'Guamanian or Chamorro',
                'Samoan',
                'Other Pacific Islander',
                'White',
                'Information not provided by applicant in mail, internet, or telephone application',
                'Not applicable',
                'No co-applicant',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_race_observed_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_race_observed == 1,
                lar.applicant_race_observed == 2,
                lar.applicant_race_observed == 3,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not Applicable'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_race_observed_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_race_observed == 1,
                lar.co_applicant_race_observed == 2,
                lar.co_applicant_race_observed == 3,
                lar.co_applicant_race_observed == 4,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not Applicable',
                'No co-applicant'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_sex_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_sex == 1,
                lar.applicant_sex == 2,
                lar.applicant_sex == 3,
                lar.applicant_sex == 4,
                lar.applicant_sex == 6,
            ),
            (
                'Male', 'Female',
                'Information not provided by applicant in mail, internet or telephone application',
                'Not Applicable', 'Applicant selected both male and female'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_sex_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_sex == 1,
                lar.co_applicant_sex == 2,
                lar.co_applicant_sex == 3,
                lar.co_applicant_sex == 4,
                lar.co_applicant_sex == 5,
                lar.co_applicant_sex == 6,
            ),
            (
                'Male', 'Female',
                'Information not provided by applicant in mail, internet or telephone application',
                'Not Applicable', 'No co-applicant',
                'Applicant selected both male and female'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['applicant_sex_observed_desc'] = pd.Series(
        np.select(
            (
                lar.applicant_sex_observed == 1,
                lar.applicant_sex_observed == 2,
                lar.applicant_sex_observed == 3,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not applicable'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['co_applicant_sex_observed_desc'] = pd.Series(
        np.select(
            (
                lar.co_applicant_sex_observed == 1,
                lar.co_applicant_sex_observed == 2,
                lar.co_applicant_sex_observed == 3,
                lar.co_applicant_sex_observed == 4,
            ),
            (
                'Collected on the basis of visual observation or surname',
                'Not collected on the basis of visual observation or surname',
                'Not applicable',
                'No co-applicant'
            ),
            default=''
        ),
        dtype='category'
    )
    lar['submission_of_application_desc'] = pd.Series(
        np.select(
            (
                lar.submission_of_application == 1,
                lar.submission_of_application == 2,
                lar.submission_of_application == 3,
                lar.submission_of_application == 1111,
            ),
            (
                'Submitted directly to your institution',
                'Not submitted directly to your institution',
                'Not Applicable',
                'Exempt'
            ),
            default='',
        ),
        dtype='category'
    )
    lar['initially_payable_to_institution_desc'] = pd.Series(
        np.select(
            (
                lar.initially_payable_to_institution == 1,
                lar.initially_payable_to_institution == 2,
                lar.initially_payable_to_institution == 3,
                lar.initially_payable_to_institution == 1111,
            ),
            (
                'Initially payable to your institution',
                'Not initially payable to your institution',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['aus_1_desc'] = pd.Series(
        np.select(
            (
                lar.aus_1 == 1,
                lar.aus_1 == 2,
                lar.aus_1 == 3,
                lar.aus_1 == 4,
                lar.aus_1 == 5,
                lar.aus_1 == 6,
                lar.aus_1 == 1111,
            ),
            (
                'Desktop Underwriter (DU)',
                'Loan Prospector (LP) or Loan Product Advisor',
                'Technology Open to Approved Lenders (TOTAL) Scorecard',
                'Guaranteed Underwriting System (GUS)',
                'Other',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['aus_2_desc'] = pd.Series(
        np.select(
            (
                lar.aus_2 == 1,
                lar.aus_2 == 2,
                lar.aus_2 == 3,
                lar.aus_2 == 4,
                lar.aus_2 == 5,
                lar.aus_2 == 6,
                lar.aus_2 == 1111,
            ),
            (
                'Desktop Underwriter (DU)',
                'Loan Prospector (LP) or Loan Product Advisor',
                'Technology Open to Approved Lenders (TOTAL) Scorecard',
                'Guaranteed Underwriting System (GUS)',
                'Other',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['aus_3_desc'] = pd.Series(
        np.select(
            (
                lar.aus_3 == 1,
                lar.aus_3 == 2,
                lar.aus_3 == 3,
                lar.aus_3 == 4,
                lar.aus_3 == 5,
                lar.aus_3 == 6,
                lar.aus_3 == 1111,
            ),
            (
                'Desktop Underwriter (DU)',
                'Loan Prospector (LP) or Loan Product Advisor',
                'Technology Open to Approved Lenders (TOTAL) Scorecard',
                'Guaranteed Underwriting System (GUS)',
                'Other',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['aus_4_desc'] = pd.Series(
        np.select(
            (
                lar.aus_4 == 1,
                lar.aus_4 == 2,
                lar.aus_4 == 3,
                lar.aus_4 == 4,
                lar.aus_4 == 5,
                lar.aus_4 == 6,
                lar.aus_4 == 1111,
            ),
            (
                'Desktop Underwriter (DU)',
                'Loan Prospector (LP) or Loan Product Advisor',
                'Technology Open to Approved Lenders (TOTAL) Scorecard',
                'Guaranteed Underwriting System (GUS)',
                'Other',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['aus_5_desc'] = pd.Series(
        np.select(
            (
                lar.aus_5 == 1,
                lar.aus_5 == 2,
                lar.aus_5 == 3,
                lar.aus_5 == 4,
                lar.aus_5 == 5,
                lar.aus_5 == 6,
                lar.aus_5 == 1111,
            ),
            (
                'Desktop Underwriter (DU)',
                'Loan Prospector (LP) or Loan Product Advisor',
                'Technology Open to Approved Lenders (TOTAL) Scorecard',
                'Guaranteed Underwriting System (GUS)',
                'Other',
                'Not applicable',
                'Exempt',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['denial_reason_1_desc'] = pd.Series(
        np.select(
            (
                lar.denial_reason_1 == 1,
                lar.denial_reason_1 == 2,
                lar.denial_reason_1 == 3,
                lar.denial_reason_1 == 4,
                lar.denial_reason_1 == 5,
                lar.denial_reason_1 == 6,
                lar.denial_reason_1 == 7,
                lar.denial_reason_1 == 8,
                lar.denial_reason_1 == 9,
                lar.denial_reason_1 == 10,
            ),
            (
                'Debt-to-income ratio',
                'Employment history',
                'Credit history',
                'Collateral',
                'Insufficient cash (downpayment, closing costs)',
                'Unverifiable information',
                'Credit application incomplete',
                'Mortgage insurance denied',
                'Other',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['denial_reason_2_desc'] = pd.Series(
        np.select(
            (
                lar.denial_reason_2 == 1,
                lar.denial_reason_2 == 2,
                lar.denial_reason_2 == 3,
                lar.denial_reason_2 == 4,
                lar.denial_reason_2 == 5,
                lar.denial_reason_2 == 6,
                lar.denial_reason_2 == 7,
                lar.denial_reason_2 == 8,
                lar.denial_reason_2 == 9,
                lar.denial_reason_2 == 10,
            ),
            (
                'Debt-to-income ratio',
                'Employment history',
                'Credit history',
                'Collateral',
                'Insufficient cash (downpayment, closing costs)',
                'Unverifiable information',
                'Credit application incomplete',
                'Mortgage insurance denied',
                'Other',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['denial_reason_3_desc'] = pd.Series(
        np.select(
            (
                lar.denial_reason_3 == 1,
                lar.denial_reason_3 == 2,
                lar.denial_reason_3 == 3,
                lar.denial_reason_3 == 4,
                lar.denial_reason_3 == 5,
                lar.denial_reason_3 == 6,
                lar.denial_reason_3 == 7,
                lar.denial_reason_3 == 8,
                lar.denial_reason_3 == 9,
                lar.denial_reason_3 == 10,
            ),
            (
                'Debt-to-income ratio',
                'Employment history',
                'Credit history',
                'Collateral',
                'Insufficient cash (downpayment, closing costs)',
                'Unverifiable information',
                'Credit application incomplete',
                'Mortgage insurance denied',
                'Other',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['denial_reason_4_desc'] = pd.Series(
        np.select(
            (
                lar.denial_reason_4 == 1,
                lar.denial_reason_4 == 2,
                lar.denial_reason_4 == 3,
                lar.denial_reason_4 == 4,
                lar.denial_reason_4 == 5,
                lar.denial_reason_4 == 6,
                lar.denial_reason_4 == 7,
                lar.denial_reason_4 == 8,
                lar.denial_reason_4 == 9,
                lar.denial_reason_4 == 10,
            ),
            (
                'Debt-to-income ratio',
                'Employment history',
                'Credit history',
                'Collateral',
                'Insufficient cash (downpayment, closing costs)',
                'Unverifiable information',
                'Credit application incomplete',
                'Mortgage insurance denied',
                'Other',
                'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    lar['high_priced'] = np.where(lar.rate_spread >= 1.5, 1, 0)
    output_columns = [
        'activity_year',
        'lei',
        'derived_msa_md',
        'state_code',
        'county_code',
        'census_tract',
        'conforming_loan_limit', 'conforming_loan_limit_desc',
        'derived_loan_product_type',
        'derived_dwelling_category',
        'derived_ethnicity',
        'derived_race',
        'derived_sex',
        'action_taken', 'action_taken_desc',
        'purchaser_type', 'purchaser_type_desc',
        'preapproval', 'preapproval_desc',
        'loan_type', 'loan_type_desc',
        'loan_purpose', 'loan_purpose_desc',
        'lien_status', 'lien_status_desc',
        'reverse_mortgage', 'reverse_mortgage_desc',
        'open_end_line_of_credit', 'open_end_line_of_credit_desc',
        'business_or_commercial_purpose', 'business_or_commercial_purpose_desc',
        'loan_amount',
        'loan_to_value_ratio',
        'interest_rate',
        'rate_spread',
        'high_priced',
        'hoepa_status', 'hoepa_status_desc',
        'total_loan_costs',
        'total_points_and_fees',
        'origination_charges',
        'discount_points',
        'lender_credits',
        'loan_term',
        'prepayment_penalty_term',
        'intro_rate_period',
        'negative_amortization', 'negative_amortization_desc',
        'interest_only_payment', 'interest_only_payment_desc',
        'balloon_payment', 'balloon_payment_desc',
        'other_nonamortizing_features', 'other_nonamortizing_features_desc',
        'property_value',
        'construction_method', 'construction_method_desc',
        'occupancy_type', 'occupancy_type_desc',
        'manufactured_home_secured_property_type', 'manufactured_home_secured_property_type_desc',
        'manufactured_home_land_property_interest', 'manufactured_home_land_property_interest_desc',
        'total_units',
        'multifamily_affordable_units',
        'income',
        'debt_to_income_ratio',
        'applicant_credit_score_type', 'applicant_credit_score_type_desc',
        'co_applicant_credit_score_type', 'co_applicant_credit_score_type_desc',
        'applicant_ethnicity_1', 'applicant_ethnicity_1_desc',
        'applicant_ethnicity_2', 'applicant_ethnicity_2_desc',
        'applicant_ethnicity_3', 'applicant_ethnicity_3_desc',
        'applicant_ethnicity_4', 'applicant_ethnicity_4_desc',
        'applicant_ethnicity_5', 'applicant_ethnicity_5_desc',
        'co_applicant_ethnicity_1', 'co_applicant_ethnicity_1_desc',
        'co_applicant_ethnicity_2', 'co_applicant_ethnicity_2_desc',
        'co_applicant_ethnicity_3', 'co_applicant_ethnicity_3_desc',
        'co_applicant_ethnicity_4', 'co_applicant_ethnicity_4_desc',
        'co_applicant_ethnicity_5', 'co_applicant_ethnicity_5_desc',
        'applicant_ethnicity_observed', 'applicant_ethnicity_observed_desc',
        'co_applicant_ethnicity_observed', 'co_applicant_ethnicity_observed_desc',
        'applicant_race_1', 'applicant_race_1_desc',
        'applicant_race_2', 'applicant_race_2_desc',
        'applicant_race_3', 'applicant_race_3_desc',
        'applicant_race_4', 'applicant_race_4_desc',
        'applicant_race_5', 'applicant_race_5_desc',
        'co_applicant_race_1', 'co_applicant_race_1_desc',
        'co_applicant_race_2', 'co_applicant_race_2_desc',
        'co_applicant_race_3', 'co_applicant_race_3_desc',
        'co_applicant_race_4', 'co_applicant_race_4_desc',
        'co_applicant_race_5', 'co_applicant_race_5_desc',
        'applicant_race_observed', 'applicant_race_observed_desc',
        'co_applicant_race_observed', 'co_applicant_race_observed_desc',
        'applicant_sex', 'applicant_sex_desc',
        'co_applicant_sex', 'co_applicant_sex_desc',
        'applicant_sex_observed', 'applicant_sex_observed_desc',
        'co_applicant_sex_observed', 'co_applicant_sex_observed_desc',
        'applicant_age',
        'co_applicant_age',
        'applicant_age_above_62',
        'co_applicant_age_above_62',
        'submission_of_application', 'submission_of_application_desc',
        'initially_payable_to_institution', 'initially_payable_to_institution_desc',
        'aus_1', 'aus_1_desc',
        'aus_2', 'aus_2_desc',
        'aus_3', 'aus_3_desc',
        'aus_4', 'aus_4_desc',
        'aus_5', 'aus_5_desc',
        'denial_reason_1', 'denial_reason_1_desc',
        'denial_reason_2', 'denial_reason_2_desc',
        'denial_reason_3', 'denial_reason_3_desc',
        'denial_reason_4', 'denial_reason_4_desc',
        'tract_population',
        'tract_minority_population_percent',
        'ffiec_msa_md_median_family_income',
        'tract_to_msa_income_percentage',
        'tract_owner_occupied_units',
        'tract_one_to_four_family_homes',
        'tract_median_age_of_housing_units',
    ]
    lar = lar[output_columns]
    lar_sample = lar.sample(n=40000, random_state=5419)
    lar_sample.to_csv(
        '/mnt/ldrive/census/hmda lar 2018-static/hmda_lar_2018_orig_mtg_sample.csv'
    )
