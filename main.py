import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
tr_set = pd.read_csv('data/training_set_features.csv', index_col='respondent_id')  # import training set of data
tr_set_labels = pd.read_csv('data/training_set_labels.csv', index_col='respondent_id')  #  import labels

np.testing.assert_array_equal(tr_set.index.values, tr_set_labels.index.values)  #  verify rows between features & labels match, nothing returned is good

tr_join = tr_set.join(tr_set_labels)

print(tr_join.shape)  #  26707 Rows  & 37 Columns
print(tr_join.dtypes)  #  data types are ok but some columns should be converted to more specific dtype
print(tr_join.columns)  #  Column names are acceptable, making them more descriptive would increase their name size
tr_cnames = tr_join.columns

print(tr_join.nunique())  #  Unique values fall in expected ranges
print(tr_join.describe())  #  Numerical and categorical features values fall in expected ranges

#  Convert to type category
tr_join['h1n1_concern'] = tr_join['h1n1_concern'].astype('category')
tr_join['h1n1_knowledge'] = tr_join['h1n1_knowledge'].astype('category')
tr_join['opinion_h1n1_vacc_effective'] = tr_join['opinion_h1n1_vacc_effective'].astype('category')
tr_join['opinion_h1n1_risk'] = tr_join['opinion_h1n1_risk'].astype('category')
tr_join['opinion_h1n1_sick_from_vacc'] = tr_join['opinion_h1n1_sick_from_vacc'].astype('category')
tr_join['opinion_seas_vacc_effective'] = tr_join['opinion_seas_vacc_effective'].astype('category')
tr_join['opinion_seas_risk'] = tr_join['opinion_seas_risk'].astype('category')
tr_join['opinion_seas_sick_from_vacc'] = tr_join['opinion_seas_sick_from_vacc'].astype('category')
tr_join['household_adults'] = tr_join['household_adults'].astype('category')
tr_join['household_children'] = tr_join['household_children'].astype('category')

#  Convert to boolean
tr_join['behavioral_antiviral_meds'] = tr_join['behavioral_antiviral_meds'].astype('boolean')
tr_join['behavioral_avoidance'] = tr_join['behavioral_avoidance'].astype('boolean')
tr_join['behavioral_face_mask'] = tr_join['behavioral_face_mask'].astype('boolean')
tr_join['behavioral_wash_hands'] = tr_join['behavioral_wash_hands'].astype('boolean')
tr_join['behavioral_large_gatherings'] = tr_join['behavioral_large_gatherings'].astype('boolean')
tr_join['behavioral_outside_home'] = tr_join['behavioral_outside_home'].astype('boolean')
tr_join['behavioral_touch_face'] = tr_join['behavioral_touch_face'].astype('boolean')
tr_join['doctor_recc_h1n1'] = tr_join['doctor_recc_h1n1'].astype('boolean')
tr_join['doctor_recc_seasonal'] = tr_join['doctor_recc_seasonal'].astype('boolean')
tr_join['chronic_med_condition'] = tr_join['chronic_med_condition'].astype('boolean')
tr_join['child_under_6_months'] = tr_join['child_under_6_months'].astype('boolean')
tr_join['health_worker'] = tr_join['health_worker'].astype('boolean')
tr_join['health_insurance'] = tr_join['health_insurance'].astype('boolean')

#  Convert to string
tr_join['age_group'] = tr_join['age_group'].astype('string')
tr_join['education'] = tr_join['education'].astype('string')
tr_join['race'] = tr_join['race'].astype('string')
tr_join['sex'] = tr_join['sex'].astype('string')
tr_join['income_poverty'] = tr_join['income_poverty'].astype('string')
tr_join['marital_status'] = tr_join['marital_status'].astype('string')
tr_join['rent_or_own'] = tr_join['rent_or_own'].astype('string')
tr_join['employment_status'] = tr_join['employment_status'].astype('string')
tr_join['hhs_geo_region'] = tr_join['hhs_geo_region'].astype('string')
tr_join['census_msa'] = tr_join['census_msa'].astype('string')
tr_join['employment_industry'] = tr_join['employment_industry'].astype('string')
tr_join['employment_occupation'] = tr_join['employment_occupation'].astype('string')

print(tr_join.dtypes)  #  checking dtypes are correct

#  There doesn't appear to be any data that is out of range or entered incorrectly.





