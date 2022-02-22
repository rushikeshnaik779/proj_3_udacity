import pandas as pd
# Add the necessary imports for the starter code.
from starter.train_model import inference_main, get_cat_features

def create_data_slice(data_path, col_to_slice, value_to_replace=None):

    # Add code to load in the data.
    if value_to_replace:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: str(value_to_replace)
        )

    else:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: input_df[col_to_slice][0]
        )

    return input_df


if __name__ == '__main__':
    CAT_FEATURES = get_cat_features()

    col_to_slice = 'race'
    value_to_replace = 'Black'  # education: ['Bachelors, 'Masters', 'HS-grad']

    print("performance on sliced column\t", col_to_slice, value_to_replace)
    sliced_data = create_data_slice('/Users/rushikeshnaik/Desktop/Project3_udacity/proj_3_udacity/data/cleaned_data.csv',
                                    col_to_slice,
                                    value_to_replace)

    precision, recall, fbeta = inference_main(sliced_data,
                                               "/Users/rushikeshnaik/Desktop/Project3_udacity/proj_3_udacity/model/RF_with_encoder_lb.pkl",
                                               CAT_FEATURES)

    with open('slice_output.txt', 'a') as f:
        result = f"""\n{'-'*50}\nperformance on sliced column -- {col_to_slice} -- {value_to_replace}\n{'-'*50} \
            \nPrecision:\t{precision}\nRecall:\t{recall}\nF-beta score:\t{fbeta}\n"""
        f.write(result)