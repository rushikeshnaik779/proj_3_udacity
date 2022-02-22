from starter.train_model import train_model_main, get_cat_features, load_data, inference_main, api_output


cat_features = get_cat_features()


if __name__ == "__main__":
    data = "/Users/rushikeshnaik/Desktop/Project3_udacity/proj_3_udacity/data/cleaned_data.csv"
    model = "/Users/rushikeshnaik/Desktop/Project3_udacity/proj_3_udacity/model/RF_with_encoder_lb.pkl"


    train_data, test_data = load_data(data)
    train_model_main(train_data, model, cat_features)

    pre, recall, f_beta = inference_main(test_data, 
    model, cat_features)
