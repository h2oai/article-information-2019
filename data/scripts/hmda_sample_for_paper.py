
"""
License
Copyright 2019 Navdeep Gill, Patrick Hall, Kim Montgomery, Nick Schmidt

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the License.

DISCLAIMER: This notebook is not legal compliance advice.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_hmda_descriptions(data):
    data['conforming_loan_limit_desc'] = pd.Series(index=data.index, data=
        np.select(
            [
                data.conforming_loan_limit == 'C',
                data.conforming_loan_limit == 'NC',
                data.conforming_loan_limit == 'U',
                data.conforming_loan_limit == 'NA'
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
    data['action_taken_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.action_taken == 1, data.action_taken == 2, data.action_taken == 3,
                data.action_taken == 4, data.action_taken == 5, data.action_taken == 6,
                data.action_taken == 7, data.action_taken == 8
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
    data['purchaser_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.purchaser_type == 0, data.purchaser_type == 1, data.purchaser_type == 2,
                data.purchaser_type == 3, data.purchaser_type == 4, data.purchaser_type == 5,
                data.purchaser_type == 6, data.purchaser_type == 71, data.purchaser_type == 72,
                data.purchaser_type == 8, data.purchaser_type == 9
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
    data['preapproval_desc'] = pd.Series(index=data.index, data=
        np.select(
            (data['preapproval'] == 1, data['preapproval'] == 2),
            ('Preapproval requested', 'Preapproval not requested'),
            default=''
        ),
        dtype='object'
    )
    data['loan_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.loan_type == 1,
                data.loan_type == 2,
                data.loan_type == 3,
                data.loan_type == 4
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
    data['loan_purpose_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.loan_purpose == 1, data.loan_purpose == 2, data.loan_purpose == 31,
                data.loan_purpose == 32, data.loan_purpose == 4, data.loan_purpose == 5,
            ),
            (
                'Home purchase', 'Home improvement', 'Refinancing',
                'Cash-out refinancing', 'Other purpose', 'Not applicable',
            ),
            default=''
        ),
        dtype='category'
    )
    data['lien_status_desc'] = pd.Series(index=data.index, data=
        np.select(
            (data.lien_status == 1, data.lien_status == 2),
            ('Secured by a first lien', 'Secured by a subordinate lien'),
            default=''
        ),
        dtype='category'
    )
    data['reverse_mortgage_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.reverse_mortgage == 1, data.reverse_mortgage == 2,
                data.reverse_mortgage == 1111
            ),
            ('Reverse mortgage', 'Not a reverse mortgage', 'Exempt'),
            default=''
        ),
        dtype='category'
    )
    data['open_end_line_of_credit_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.open_end_line_of_credit == 1,
                data.open_end_line_of_credit == 2,
                data.open_end_line_of_credit == 1111
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
    data['business_or_commercial_purpose_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.business_or_commercial_purpose == 1,
                data.business_or_commercial_purpose == 2,
                data.business_or_commercial_purpose == 1111,
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
    data['hoepa_status_desc'] = pd.Series(index=data.index, data=
        np.select(
            (data.hoepa_status == 1, data.hoepa_status == 2, data.hoepa_status == 3),
            ('High-cost mortgage', 'Not a high-cost mortgage', 'Not Applicable'),
            default=''
        ),
        dtype='category'
    )
    data['negative_amortization_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.negative_amortization == 1,
                data.negative_amortization == 2,
                data.negative_amortization == 1111
            ),
            (
                'Negative amortization', 'No negative amortization', 'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    data['interest_only_payment_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.interest_only_payment == 1,
                data.interest_only_payment == 2,
                data.interest_only_payment == 1111,
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
    data['balloon_payment_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.balloon_payment == 1,
                data.balloon_payment == 2,
                data.balloon_payment == 1111,
            ),
            ('Balloon Payment', 'No balloon payment', 'Exempt'),
            default=''
        ),
        dtype='category'
    )
    data['other_nonamortizing_features_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.other_nonamortizing_features == 1,
                data.other_nonamortizing_features == 2,
                data.other_nonamortizing_features == 1111,
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
    data['construction_method_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.construction_method == 1,
                data.construction_method == 2
            ),
            ('Site-built', 'Manufactured home'),
            default=''
        ),
        dtype='category'
    )
    data['occupancy_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.occupancy_type == 1,
                data.occupancy_type == 2,
                data.occupancy_type == 3,
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
    data['manufactured_home_secured_property_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.manufactured_home_secured_property_type == 1,
                data.manufactured_home_secured_property_type == 2,
                data.manufactured_home_secured_property_type == 3,
                data.manufactured_home_secured_property_type == 1111,
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
    data['manufactured_home_land_property_interest_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.manufactured_home_land_property_interest == 1,
                data.manufactured_home_land_property_interest == 2,
                data.manufactured_home_land_property_interest == 3,
                data.manufactured_home_land_property_interest == 4,
                data.manufactured_home_land_property_interest == 5,
                data.manufactured_home_land_property_interest == 1111,
            ),
            (
                'Direct ownership', 'Indirect ownership', 'Paid leasehold',
                'Unpaid leasehold', 'Not Applicable', 'Exempt'
            ),
            default=''
        ),
        dtype='category'
    )
    data['applicant_credit_score_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_credit_score_type == 1,
                data.applicant_credit_score_type == 2,
                data.applicant_credit_score_type == 3,
                data.applicant_credit_score_type == 4,
                data.applicant_credit_score_type == 5,
                data.applicant_credit_score_type == 6,
                data.applicant_credit_score_type == 7,
                data.applicant_credit_score_type == 8,
                data.applicant_credit_score_type == 9,
                data.applicant_credit_score_type == 1111,
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
    data['co_applicant_credit_score_type_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_credit_score_type == 1,
                data.co_applicant_credit_score_type == 2,
                data.co_applicant_credit_score_type == 3,
                data.co_applicant_credit_score_type == 4,
                data.co_applicant_credit_score_type == 5,
                data.co_applicant_credit_score_type == 6,
                data.co_applicant_credit_score_type == 7,
                data.co_applicant_credit_score_type == 8,
                data.co_applicant_credit_score_type == 9,
                data.co_applicant_credit_score_type == 10,
                data.co_applicant_credit_score_type == 1111,
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
    data['applicant_ethnicity_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_1 == 1,
                data.applicant_ethnicity_1 == 11,
                data.applicant_ethnicity_1 == 12,
                data.applicant_ethnicity_1 == 13,
                data.applicant_ethnicity_1 == 14,
                data.applicant_ethnicity_1 == 2,
                data.applicant_ethnicity_1 == 3,
                data.applicant_ethnicity_1 == 4,
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
    data['applicant_ethnicity_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_2 == 1,
                data.applicant_ethnicity_2 == 11,
                data.applicant_ethnicity_2 == 12,
                data.applicant_ethnicity_2 == 13,
                data.applicant_ethnicity_2 == 14,
                data.applicant_ethnicity_2 == 2,
                data.applicant_ethnicity_2 == 3,
                data.applicant_ethnicity_2 == 4,
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
    data['applicant_ethnicity_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_3 == 1,
                data.applicant_ethnicity_3 == 11,
                data.applicant_ethnicity_3 == 12,
                data.applicant_ethnicity_3 == 13,
                data.applicant_ethnicity_3 == 14,
                data.applicant_ethnicity_3 == 2,
                data.applicant_ethnicity_3 == 3,
                data.applicant_ethnicity_3 == 4,
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
    data['applicant_ethnicity_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_4 == 1,
                data.applicant_ethnicity_4 == 11,
                data.applicant_ethnicity_4 == 12,
                data.applicant_ethnicity_4 == 13,
                data.applicant_ethnicity_4 == 14,
                data.applicant_ethnicity_4 == 2,
                data.applicant_ethnicity_4 == 3,
                data.applicant_ethnicity_4 == 4,
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
    data['applicant_ethnicity_5_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_5 == 1,
                data.applicant_ethnicity_5 == 11,
                data.applicant_ethnicity_5 == 12,
                data.applicant_ethnicity_5 == 13,
                data.applicant_ethnicity_5 == 14,
                data.applicant_ethnicity_5 == 2,
                data.applicant_ethnicity_5 == 3,
                data.applicant_ethnicity_5 == 4,
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
    data['co_applicant_ethnicity_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_1 == 1,
                data.co_applicant_ethnicity_1 == 11,
                data.co_applicant_ethnicity_1 == 12,
                data.co_applicant_ethnicity_1 == 13,
                data.co_applicant_ethnicity_1 == 14,
                data.co_applicant_ethnicity_1 == 2,
                data.co_applicant_ethnicity_1 == 3,
                data.co_applicant_ethnicity_1 == 4,
                data.co_applicant_ethnicity_1 == 5,
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
    data['co_applicant_ethnicity_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_2 == 1,
                data.co_applicant_ethnicity_2 == 11,
                data.co_applicant_ethnicity_2 == 12,
                data.co_applicant_ethnicity_2 == 13,
                data.co_applicant_ethnicity_2 == 14,
                data.co_applicant_ethnicity_2 == 2,
                data.co_applicant_ethnicity_2 == 3,
                data.co_applicant_ethnicity_2 == 4,
                data.co_applicant_ethnicity_2 == 5,
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
    data['co_applicant_ethnicity_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_3 == 1,
                data.co_applicant_ethnicity_3 == 11,
                data.co_applicant_ethnicity_3 == 12,
                data.co_applicant_ethnicity_3 == 13,
                data.co_applicant_ethnicity_3 == 14,
                data.co_applicant_ethnicity_3 == 2,
                data.co_applicant_ethnicity_3 == 3,
                data.co_applicant_ethnicity_3 == 4,
                data.co_applicant_ethnicity_3 == 5,
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
    data['co_applicant_ethnicity_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_4 == 1,
                data.co_applicant_ethnicity_4 == 11,
                data.co_applicant_ethnicity_4 == 12,
                data.co_applicant_ethnicity_4 == 13,
                data.co_applicant_ethnicity_4 == 14,
                data.co_applicant_ethnicity_4 == 2,
                data.co_applicant_ethnicity_4 == 3,
                data.co_applicant_ethnicity_4 == 4,
                data.co_applicant_ethnicity_4 == 5,
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
    data['co_applicant_ethnicity_5_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_5 == 1,
                data.co_applicant_ethnicity_5 == 11,
                data.co_applicant_ethnicity_5 == 12,
                data.co_applicant_ethnicity_5 == 13,
                data.co_applicant_ethnicity_5 == 14,
                data.co_applicant_ethnicity_5 == 2,
                data.co_applicant_ethnicity_5 == 3,
                data.co_applicant_ethnicity_5 == 4,
                data.co_applicant_ethnicity_5 == 5,
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
    data['applicant_ethnicity_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_ethnicity_observed == 1,
                data.applicant_ethnicity_observed == 2,
                data.applicant_ethnicity_observed == 3,
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
    data['co_applicant_ethnicity_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_ethnicity_observed == 1,
                data.co_applicant_ethnicity_observed == 2,
                data.co_applicant_ethnicity_observed == 3,
                data.co_applicant_ethnicity_observed == 4,
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
    data['applicant_race_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_1 == 1,
                data.applicant_race_1 == 2,
                data.applicant_race_1 == 21,
                data.applicant_race_1 == 22,
                data.applicant_race_1 == 23,
                data.applicant_race_1 == 24,
                data.applicant_race_1 == 25,
                data.applicant_race_1 == 26,
                data.applicant_race_1 == 27,
                data.applicant_race_1 == 3,
                data.applicant_race_1 == 4,
                data.applicant_race_1 == 41,
                data.applicant_race_1 == 42,
                data.applicant_race_1 == 43,
                data.applicant_race_1 == 44,
                data.applicant_race_1 == 5,
                data.applicant_race_1 == 6,
                data.applicant_race_1 == 7,
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
    data['applicant_race_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_2 == 1,
                data.applicant_race_2 == 2,
                data.applicant_race_2 == 21,
                data.applicant_race_2 == 22,
                data.applicant_race_2 == 23,
                data.applicant_race_2 == 24,
                data.applicant_race_2 == 25,
                data.applicant_race_2 == 26,
                data.applicant_race_2 == 27,
                data.applicant_race_2 == 3,
                data.applicant_race_2 == 4,
                data.applicant_race_2 == 41,
                data.applicant_race_2 == 42,
                data.applicant_race_2 == 43,
                data.applicant_race_2 == 44,
                data.applicant_race_2 == 5,
                data.applicant_race_2 == 6,
                data.applicant_race_2 == 7,
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
    data['applicant_race_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_3 == 1,
                data.applicant_race_3 == 2,
                data.applicant_race_3 == 21,
                data.applicant_race_3 == 22,
                data.applicant_race_3 == 23,
                data.applicant_race_3 == 24,
                data.applicant_race_3 == 25,
                data.applicant_race_3 == 26,
                data.applicant_race_3 == 27,
                data.applicant_race_3 == 3,
                data.applicant_race_3 == 4,
                data.applicant_race_3 == 41,
                data.applicant_race_3 == 42,
                data.applicant_race_3 == 43,
                data.applicant_race_3 == 44,
                data.applicant_race_3 == 5,
                data.applicant_race_3 == 6,
                data.applicant_race_3 == 7,
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
    data['applicant_race_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_4 == 1,
                data.applicant_race_4 == 2,
                data.applicant_race_4 == 21,
                data.applicant_race_4 == 22,
                data.applicant_race_4 == 23,
                data.applicant_race_4 == 24,
                data.applicant_race_4 == 25,
                data.applicant_race_4 == 26,
                data.applicant_race_4 == 27,
                data.applicant_race_4 == 3,
                data.applicant_race_4 == 4,
                data.applicant_race_4 == 41,
                data.applicant_race_4 == 42,
                data.applicant_race_4 == 43,
                data.applicant_race_4 == 44,
                data.applicant_race_4 == 5,
                data.applicant_race_4 == 6,
                data.applicant_race_4 == 7,
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
    data['applicant_race_5_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_5 == 1,
                data.applicant_race_5 == 2,
                data.applicant_race_5 == 21,
                data.applicant_race_5 == 22,
                data.applicant_race_5 == 23,
                data.applicant_race_5 == 24,
                data.applicant_race_5 == 25,
                data.applicant_race_5 == 26,
                data.applicant_race_5 == 27,
                data.applicant_race_5 == 3,
                data.applicant_race_5 == 4,
                data.applicant_race_5 == 41,
                data.applicant_race_5 == 42,
                data.applicant_race_5 == 43,
                data.applicant_race_5 == 44,
                data.applicant_race_5 == 5,
                data.applicant_race_5 == 6,
                data.applicant_race_5 == 7,
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
    data['co_applicant_race_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_1 == 1,
                data.co_applicant_race_1 == 2,
                data.co_applicant_race_1 == 21,
                data.co_applicant_race_1 == 22,
                data.co_applicant_race_1 == 23,
                data.co_applicant_race_1 == 24,
                data.co_applicant_race_1 == 25,
                data.co_applicant_race_1 == 26,
                data.co_applicant_race_1 == 27,
                data.co_applicant_race_1 == 3,
                data.co_applicant_race_1 == 4,
                data.co_applicant_race_1 == 41,
                data.co_applicant_race_1 == 42,
                data.co_applicant_race_1 == 43,
                data.co_applicant_race_1 == 44,
                data.co_applicant_race_1 == 5,
                data.co_applicant_race_1 == 6,
                data.co_applicant_race_1 == 7,
                data.co_applicant_race_1 == 8,
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
    data['co_applicant_race_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_2 == 1,
                data.co_applicant_race_2 == 2,
                data.co_applicant_race_2 == 21,
                data.co_applicant_race_2 == 22,
                data.co_applicant_race_2 == 23,
                data.co_applicant_race_2 == 24,
                data.co_applicant_race_2 == 25,
                data.co_applicant_race_2 == 26,
                data.co_applicant_race_2 == 27,
                data.co_applicant_race_2 == 3,
                data.co_applicant_race_2 == 4,
                data.co_applicant_race_2 == 41,
                data.co_applicant_race_2 == 42,
                data.co_applicant_race_2 == 43,
                data.co_applicant_race_2 == 44,
                data.co_applicant_race_2 == 5,
                data.co_applicant_race_2 == 6,
                data.co_applicant_race_2 == 7,
                data.co_applicant_race_2 == 8,
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
    data['co_applicant_race_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_3 == 1,
                data.co_applicant_race_3 == 2,
                data.co_applicant_race_3 == 21,
                data.co_applicant_race_3 == 22,
                data.co_applicant_race_3 == 23,
                data.co_applicant_race_3 == 24,
                data.co_applicant_race_3 == 25,
                data.co_applicant_race_3 == 26,
                data.co_applicant_race_3 == 27,
                data.co_applicant_race_3 == 3,
                data.co_applicant_race_3 == 4,
                data.co_applicant_race_3 == 41,
                data.co_applicant_race_3 == 42,
                data.co_applicant_race_3 == 43,
                data.co_applicant_race_3 == 44,
                data.co_applicant_race_3 == 5,
                data.co_applicant_race_3 == 6,
                data.co_applicant_race_3 == 7,
                data.co_applicant_race_3 == 8,
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
    data['co_applicant_race_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_4 == 1,
                data.co_applicant_race_4 == 2,
                data.co_applicant_race_4 == 21,
                data.co_applicant_race_4 == 22,
                data.co_applicant_race_4 == 23,
                data.co_applicant_race_4 == 24,
                data.co_applicant_race_4 == 25,
                data.co_applicant_race_4 == 26,
                data.co_applicant_race_4 == 27,
                data.co_applicant_race_4 == 3,
                data.co_applicant_race_4 == 4,
                data.co_applicant_race_4 == 41,
                data.co_applicant_race_4 == 42,
                data.co_applicant_race_4 == 43,
                data.co_applicant_race_4 == 44,
                data.co_applicant_race_4 == 5,
                data.co_applicant_race_4 == 6,
                data.co_applicant_race_4 == 7,
                data.co_applicant_race_4 == 8,
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
    data['co_applicant_race_5_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_5 == 1,
                data.co_applicant_race_5 == 2,
                data.co_applicant_race_5 == 21,
                data.co_applicant_race_5 == 22,
                data.co_applicant_race_5 == 23,
                data.co_applicant_race_5 == 24,
                data.co_applicant_race_5 == 25,
                data.co_applicant_race_5 == 26,
                data.co_applicant_race_5 == 27,
                data.co_applicant_race_5 == 3,
                data.co_applicant_race_5 == 4,
                data.co_applicant_race_5 == 41,
                data.co_applicant_race_5 == 42,
                data.co_applicant_race_5 == 43,
                data.co_applicant_race_5 == 44,
                data.co_applicant_race_5 == 5,
                data.co_applicant_race_5 == 6,
                data.co_applicant_race_5 == 7,
                data.co_applicant_race_5 == 8,
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
    data['applicant_race_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_race_observed == 1,
                data.applicant_race_observed == 2,
                data.applicant_race_observed == 3,
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
    data['co_applicant_race_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_race_observed == 1,
                data.co_applicant_race_observed == 2,
                data.co_applicant_race_observed == 3,
                data.co_applicant_race_observed == 4,
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
    data['applicant_sex_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_sex == 1,
                data.applicant_sex == 2,
                data.applicant_sex == 3,
                data.applicant_sex == 4,
                data.applicant_sex == 6,
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
    data['co_applicant_sex_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_sex == 1,
                data.co_applicant_sex == 2,
                data.co_applicant_sex == 3,
                data.co_applicant_sex == 4,
                data.co_applicant_sex == 5,
                data.co_applicant_sex == 6,
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
    data['applicant_sex_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.applicant_sex_observed == 1,
                data.applicant_sex_observed == 2,
                data.applicant_sex_observed == 3,
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
    data['co_applicant_sex_observed_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.co_applicant_sex_observed == 1,
                data.co_applicant_sex_observed == 2,
                data.co_applicant_sex_observed == 3,
                data.co_applicant_sex_observed == 4,
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
    data['submission_of_application_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.submission_of_application == 1,
                data.submission_of_application == 2,
                data.submission_of_application == 3,
                data.submission_of_application == 1111,
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
    data['initially_payable_to_institution_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.initially_payable_to_institution == 1,
                data.initially_payable_to_institution == 2,
                data.initially_payable_to_institution == 3,
                data.initially_payable_to_institution == 1111,
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
    data['aus_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.aus_1 == 1,
                data.aus_1 == 2,
                data.aus_1 == 3,
                data.aus_1 == 4,
                data.aus_1 == 5,
                data.aus_1 == 6,
                data.aus_1 == 1111,
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
    data['aus_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.aus_2 == 1,
                data.aus_2 == 2,
                data.aus_2 == 3,
                data.aus_2 == 4,
                data.aus_2 == 5,
                data.aus_2 == 6,
                data.aus_2 == 1111,
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
    data['aus_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.aus_3 == 1,
                data.aus_3 == 2,
                data.aus_3 == 3,
                data.aus_3 == 4,
                data.aus_3 == 5,
                data.aus_3 == 6,
                data.aus_3 == 1111,
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
    data['aus_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.aus_4 == 1,
                data.aus_4 == 2,
                data.aus_4 == 3,
                data.aus_4 == 4,
                data.aus_4 == 5,
                data.aus_4 == 6,
                data.aus_4 == 1111,
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
    data['aus_5_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.aus_5 == 1,
                data.aus_5 == 2,
                data.aus_5 == 3,
                data.aus_5 == 4,
                data.aus_5 == 5,
                data.aus_5 == 6,
                data.aus_5 == 1111,
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
    data['denial_reason_1_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.denial_reason_1 == 1,
                data.denial_reason_1 == 2,
                data.denial_reason_1 == 3,
                data.denial_reason_1 == 4,
                data.denial_reason_1 == 5,
                data.denial_reason_1 == 6,
                data.denial_reason_1 == 7,
                data.denial_reason_1 == 8,
                data.denial_reason_1 == 9,
                data.denial_reason_1 == 10,
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
    data['denial_reason_2_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.denial_reason_2 == 1,
                data.denial_reason_2 == 2,
                data.denial_reason_2 == 3,
                data.denial_reason_2 == 4,
                data.denial_reason_2 == 5,
                data.denial_reason_2 == 6,
                data.denial_reason_2 == 7,
                data.denial_reason_2 == 8,
                data.denial_reason_2 == 9,
                data.denial_reason_2 == 10,
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
    data['denial_reason_3_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.denial_reason_3 == 1,
                data.denial_reason_3 == 2,
                data.denial_reason_3 == 3,
                data.denial_reason_3 == 4,
                data.denial_reason_3 == 5,
                data.denial_reason_3 == 6,
                data.denial_reason_3 == 7,
                data.denial_reason_3 == 8,
                data.denial_reason_3 == 9,
                data.denial_reason_3 == 10,
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
    data['denial_reason_4_desc'] = pd.Series(index=data.index, data=
        np.select(
            (
                data.denial_reason_4 == 1,
                data.denial_reason_4 == 2,
                data.denial_reason_4 == 3,
                data.denial_reason_4 == 4,
                data.denial_reason_4 == 5,
                data.denial_reason_4 == 6,
                data.denial_reason_4 == 7,
                data.denial_reason_4 == 8,
                data.denial_reason_4 == 9,
                data.denial_reason_4 == 10,
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
    return data


if __name__ == '__main__':

    lar = pd.read_csv('/mnt/ldrive/census/hmda lar 2018-static/2018_public_lar_csv2.csv.gz',
                      low_memory=False)
    print(f'Original Data Dimensions: {lar.shape}')
    lar_imported_copy = lar.copy()  # delete this line after code is completed
    # lar = lar_imported_copy.copy()

    # lar = lar.loc[(lar["rate_spread"] != "Exempt") & (lar["loan_term"] != "Exempt"), :]

    make_numeric = ['loan_amount', 'loan_to_value_ratio', 'discount_points', 'lender_credits', 'property_value',
                    'income', "rate_spread", "loan_term"]
    for mni in make_numeric:
        lar = lar.loc[lar[mni] != "Exempt", :]
        lar[mni] = lar[mni].astype(dtype='float64')

    lar['high_priced'] = np.where(lar["rate_spread"].isnull(), np.nan,
                                  np.where(lar["rate_spread"] >= 1.5, 1, 0))

    lar = lar.loc[(lar["action_taken"] == 1) &
                  (lar['loan_purpose'] == 1) &
                  (lar["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built") &
                  (lar["open_end_line_of_credit"] == 2) &
                  (lar["business_or_commercial_purpose"] == 2) &
                  (lar["construction_method"] == 1) &
                  (lar["occupancy_type"] == 1) &
                  (lar["reverse_mortgage"] == 2) &
                  (lar["negative_amortization"] == 2) &
                  (lar["interest_only_payment"] == 2) &
                  (lar["balloon_payment"] != 1111) &
                  (lar["applicant_credit_score_type"] != 1111) &
                  ((lar["loan_term"] == 180) | (lar["loan_term"] == 360)) &
                  (lar["hoepa_status"] == 2) &
                  (lar["lien_status"] == 1) &
                  (lar["conforming_loan_limit"] != "U") &
                  (lar["balloon_payment"] == 2) &
                  lar["prepayment_penalty_term"].isnull() &
                  ~lar["rate_spread"].isnull(), :]

    lar["loan_term"] = lar["loan_term"].astype('object')
    print(f'Subset Data Dimensions: {lar.shape}')

    lar = get_hmda_descriptions(data=lar)
    print(f'Original + Description Variables Data Dimensions: {lar.shape}')
    # lar["preapproval_desc"].value_counts(dropna=False)
    # lar["preapproval"].value_counts(dropna=False)
    # pd.crosstab(lar["preapproval_desc"], lar["preapproval"], dropna=False)
    # pd.crosstab(lar["preapproval"], lar["preapproval_desc"], dropna=False)

    # "total_loan_costs", "purchaser_type", "derived_msa_md", "census_tract", 'discount_points', 'lender_credits',
    keep_cols = ['high_priced', 'derived_loan_product_type', 'loan_amount', 'loan_to_value_ratio',
                 'loan_term', 'intro_rate_period',
                 'interest_only_payment', 'balloon_payment', 'property_value', 'income',
                 'debt_to_income_ratio']
    keep_cols = keep_cols + ["state_code", "county_code", "conforming_loan_limit", "preapproval",
                             "lien_status", "rate_spread", "interest_rate",
                             "applicant_age_above_62", "co_applicant_age_above_62", "derived_ethnicity",
                             "derived_race", "derived_sex", "ffiec_msa_md_median_family_income"] + \
                [x for x in lar.columns if str.startswith(x, "tract")]
    keep_cols = keep_cols + [x + "_desc" for x in keep_cols for y in lar.columns.values if x + "_desc" == y]
    assert(len(np.unique(keep_cols)) == len(keep_cols))
    # pd.Series(keep_cols).value_counts()

    view_dropped_vars = lar.drop(columns=keep_cols)
    view_dropped_vars = view_dropped_vars.loc[:, view_dropped_vars.apply(pd.Series.nunique) != 1]
    view_dropped_vars.drop(inplace=True,
                           columns=[x for x in view_dropped_vars.columns if "race" in x or "ethnicity" in x])
    dropped_columns = pd.Series(view_dropped_vars.columns)
    len(dropped_columns)

    lar_subset = lar[keep_cols].copy()
    print(lar_subset.shape)
    lar_subset = lar_subset.loc[:, lar_subset.apply(pd.Series.nunique) != 1]
    print(lar_subset.shape)

    # lar_subset["census_tract"] = lar_subset["census_tract"].astype('object')
    x = lar.loc[lar["loan_type_desc"].isnull(), ["loan_type", "loan_type_desc"]]
    var_desc = pd.merge(lar_subset.apply(pd.Series.nunique).to_frame('num_unique'),
                        lar_subset.dtypes.to_frame('dtypes'), left_index=True, right_index=True,
                        how='outer' ,)
    var_desc.sort_values(by=["num_unique"], inplace=True)
    for vi in var_desc.loc[var_desc["num_unique"] <= 20].index:
        var_freq = lar_subset[vi].value_counts(dropna=False)
        print(f'******************************************\nVariable {vi} Frequencies:\n{var_freq}')

    lar_subset["co_applicant_age_above_62"].value_counts(dropna=False)
    pd.crosstab(lar_subset["applicant_age_above_62"],  lar_subset["co_applicant_age_above_62"])

    lar_subset["agegte62"] = np.where((lar_subset["applicant_age_above_62"] == "No") &
                                      (lar_subset["co_applicant_age_above_62"] == "No"), 0,
                                      np.where((lar_subset["applicant_age_above_62"] == "Yes") |
                                               (lar_subset["co_applicant_age_above_62"] == "Yes"), 1, np.nan))
    lar_subset["agelt62"] = 1 - lar_subset["agegte62"]

    lar_subset["male"] = np.where(lar_subset["derived_sex"] == "Male", 1,
                                  np.where(lar_subset["derived_sex"] == "Female", 0, np.nan))
    lar_subset['female'] = 1 - lar_subset["male"]

    lar_subset["race"] = np.select([lar_subset["derived_race"] == 'White',
                                    lar_subset["derived_race"] == 'Race Not Available',
                                    lar_subset["derived_race"] == 'Asian',
                                    lar_subset["derived_race"] == 'Black or African American',
                                    lar_subset["derived_race"] == 'Joint',
                                    lar_subset["derived_race"] == 'American Indian or Alaska Native',
                                    lar_subset["derived_race"] == 'Native Hawaiian or Other Pacific Islander',
                                    lar_subset["derived_race"] == '2 or more minority races',
                                    lar_subset["derived_race"] == 'Free Form Text Only'],
                                   ["white", "NA", "asian", "black", "NA", "amind", "hipac", "NA", "NA"])
    print(lar_subset["race"].value_counts(dropna=False))
    for racei in ["black", "asian", "white", "amind", "hipac"]:
        lar_subset[racei] = np.where(lar_subset["race"] == racei, 1,
                                       np.where(lar_subset["race"] != "NA", 0, np.nan))
        print(f'\n{lar_subset[racei].value_counts(dropna=False)}\n\n'
              f'{pd.crosstab(lar_subset[racei], lar_subset["race"], dropna=False)}')

    lar_subset["hispanic"] = np.where(lar_subset["derived_ethnicity"] == 'Hispanic or Latino', 1,
                                      np.where(lar_subset["derived_ethnicity"] == 'Not Hispanic or Latino', 0, np.nan))
    lar_subset["non_hispanic"] = 1 - lar_subset["hispanic"]

    lar_subset.drop(inplace=True, columns=["derived_race", "derived_ethnicity", "applicant_age_above_62",
                                           "co_applicant_age_above_62", "derived_sex",
                                           "tract_population", "tract_minority_population_percent",
                                           "tract_owner_occupied_units", "tract_one_to_four_family_homes",
                                           "county_code", 'ffiec_msa_md_median_family_income',
                                           "tract_to_msa_income_percentage", 'tract_median_age_of_housing_units'])

    lar_subset["no_intro_rate_period"] = (lar_subset["intro_rate_period"].isnull()).astype('int8')
    lar_subset.loc[lar_subset["intro_rate_period"].isnull(), "intro_rate_period"] = 0

    lar_subset.dropna(axis=0, how='any', subset=['loan_to_value_ratio', 'property_value', 'income',
                                                 'rate_spread', 'interest_rate'], inplace=True)
    var_order = list()
    for vvi in [x for x in lar_subset.columns if x + "_desc" in lar_subset.columns]:
        var_order.append(vvi)
        var_order.append(vvi + "_desc")
    var_order = var_order + [x for x in lar_subset.columns if x not in var_order]
    lar_subset = lar_subset[var_order]
    # lar_subset["above_spread"] = np.where(lar_subset["rate_spread"] > lar_subset["rate_spread"].quantile(), 1, 0)
    # print(f'High-Priced Loan Percentages:\n{lar_subset["above_spread"].value_counts(dropna=False, normalize=True)}')

    lar_subset["conforming"] = np.where(lar_subset["conforming_loan_limit"] == "C", 1,
                                        np.where(lar_subset["conforming_loan_limit"] == "NC", 0, np.nan))
    lar_subset["term_360"] = np.where(lar_subset["loan_term"] == 360, 1,
                                      np.where(lar_subset["loan_term"] == 180, 0, np.nan))
    # "rate_spread", "interest_rate",  "state_code",
    final_keep_vars = ['loan_amount', 'loan_to_value_ratio', 'no_intro_rate_period', 'intro_rate_period',
                       'property_value', 'income', "debt_to_income_ratio", "term_360", "conforming",
                       "high_priced",
                       "black", "asian", "white", "amind", "hipac", "hispanic", "non_hispanic",
                       "male", "female",
                       "agegte62", "agelt62"]

    lar_sample = lar_subset.loc[:, final_keep_vars].sample(n=200000, random_state=31415)

    np.random.seed(27182845)
    tts = (np.random.rand(len(lar_sample)) <= 0.8)
    train, test = lar_sample[tts].copy(), lar_sample[~tts].copy()
    print(train.shape, test.shape)
    train['cv_fold'] = pd.Series(np.random.randint(low=1, high=6, size=len(train)), index=train.index)

    review_sample = train.sample(n=100)

    train.to_csv('./data/output/hmda_train.csv')
    test.to_csv('./data/output/hmda_test.csv')
