import numpy as np

def mlp_config(input_dimension, output_dimension, hidden_size, num_rep_layer, num_adapt_layer):
    config = []
    config.append({"name": 'linear', "adaptation": False, "meta": True,
                "config": {"out": hidden_size, "in": input_dimension}})
    for _ in range(1, num_rep_layer):
        config.append({"name": 'linear', "adaptation": False, "meta": True,
                "config": {"out": hidden_size, "in": hidden_size}})
    
    for _ in range(1, num_adapt_layer):
        config.append({"name": 'linear', "adaptation": True, "meta": True,
                "config": {"out": hidden_size, "in": hidden_size}})
    
    config.append({"name": 'linear', "adaptation": True, "meta": True,
                "config": {"out": output_dimension, "in": hidden_size}})

    return config

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, hidden_size=300):

        if "Sin" == dataset:

            return [

                {"name": 'linear', "adaptation": False, "meta": True,
                    "config": {"out": hidden_size, "in": input_dimension}},
                {"name": 'relu'},
                {"name": 'linear', "adaptation": False, "meta": True,
                    "config": {"out": hidden_size, "in": hidden_size}},
                {"name": 'relu'},
                {"name": 'linear', "adaptation": False, "meta": True,
                    "config": {"out": hidden_size, "in": hidden_size}},
                {"name": 'relu'},
                {"name": 'linear', "adaptation": False, "meta": True,
                    "config": {"out": hidden_size, "in": hidden_size}},
                {"name": 'relu'},
                {"name": 'linear', "adaptation": False, "meta": True,
                    "config": {"out": hidden_size, "in": hidden_size}},
                {"name": 'relu'},
                {"name": 'linear', "adaptation": True, "meta": True,
                    "config": {"out": output_dimension, "in": hidden_size}}
            ]

        elif dataset == "omniglot":
            channels = 256

            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}}

            ]



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
