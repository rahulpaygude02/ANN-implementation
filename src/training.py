from src.utils.common import read_config
import argparse
from src.utils.data_mgmt import get_data
from utils.model import create_model
from src.utils.model import create_model

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (x_train_small,y_train_small),( x_valid,y_valid), (x_test,y_test) = get_data(validation_datasize)
    LOSS_FUNCTION = config["params"]["loss_fuction"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["matrics"]
    num_classes = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,num_classes)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (x_valid,y_valid)

    data=model.fit(x_train_small,y_train_small,epochs = EPOCHS,validation_data=VALIDATION_SET)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default = "config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)

    