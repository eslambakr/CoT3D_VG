import pandas as pd

RELATIONS = [
    'above',
    'behind',
    'below',
    'beneath',
    'beside',
    'between',
    'close to',
    'closer to',
    'far away from',
    'far from',
    'farthest from',
    'in front of',
    'in the center of',
    'in the middle of',
    'lying on',
    'near',
    'next to',
    'on',
    'on the back of',
    'on the left of',
    'on the left side of',
    'on the right of',
    'on the right side of',
    'on top',
    'on top of',
    'over',
    'supporting',
    'to the left of',
    'to the right of',
    'under',
    'underneath',
    'beside',
    'close to',
    'closer to',
    'near',
    'next to',
    'far away from',
    'far from',
    'farthest from',
    'between',
    'in the center of',
    'in the middle of',
    'behind',
    'in front of',
    'on the back of',
    'on the left of',
    'on the left side of',
    'on the right of',
    'on the right side of',
    'to the left of',
    'to the right of',
    'above',
    'below',
    'beneath',
    'lying on',
    'on',
    'on top',
    'on top of',
    'over',
    'supporting',
    'under',
    'underneath',
    'back',
    'front',
    'facing',
    'with',
    'touching',
    'tucked',
    'draped',
    'facing',
    'with',
    'grouped',
    'touching',
    'along',
    'against',
    'through',
    'with',
    'near',
    'opposite',
    'by',
    'closest',
    'down',
    'face',
    'atop',
    'leftmost',
    'along',
    'overlooks',
    'looking',
    'off',
    'across',
    'nearest',
    'towards',
    'faces',
    'higher',
    'beaneath',
    'inbewteen',
    'furthest',
    'besides',
    'nearer',
    'topped',
    'within',
    'alongside',
    'mounted', 
    'containing', 
    'covering', 
    'contain', 
    'contains', 
    'around', 
    'inside', 
    'sits', 
    'lower',
    'than', 
    'among', 
    'overhangs', 
    'holding', 
    'includes', 
    'blocking', 
    'before', 
    'sitting', 
    'block', 
    'laying', 
    'seats', 
    'onto', 
    'taped', 
    'covers', 
    'hangs', 
    'covered', 
    'after', 
    'cover', 
    'include', 
    'beyond', 
    'setting', 
    'touched',
    'inbetween',
    'surrounds',
    'nearby',

]

def relation_check(target_string, string_list):
    for str in RELATIONS:
        if any(substring.lower() in str.lower() for substring in target_string.lower().split()):
            return False
    return True

# load the csv file into a pandas dataframe
df = pd.read_csv('data/pred_objs_total_opt1.csv')
nr_df = pd.read_csv('data/nr3d.csv')

print(df.columns)
print(df.head())
# get the unique values of the 'relations' column
mask = df['relation'].apply(lambda x: relation_check(x, RELATIONS))

# filter the dataframe using the mask
filtered_df = df[mask]
unique_relation = filtered_df['relation'].unique()

#Add unique relation in a csv file 

unique_relation_df = pd.DataFrame(unique_relation)

unique_relation_df.to_csv('unique_relation.csv', index=False)

filtered_df['id'] = filtered_df['id'].astype(int)

filtered_nr_df = nr_df.iloc[filtered_df['id'].unique()]

sampled_df = filtered_nr_df.sample(n=4000, random_state=42)
sampled_df = sampled_df.rename(columns={'assignmentid': 'id'})
# write the filtered dataframe to a new CSV file
filtered_df.to_csv('spacial_relations.csv', index=False)
filtered_nr_df.to_csv('filtered_nr.csv', index=False)
sampled_df.to_csv('filtered_nr4k.csv', index=False)