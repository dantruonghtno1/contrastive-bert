import argparse
from email.policy import default
import os 

class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unkown = parser.parse_known_args()
        self.args = all_args
    
    def all_param(self, parser):
        parser.add_argument("--max_length", default = 256, type = int)
        parser.add_argument("--model_path", default = "vinai/phobert-base", type = str)
        parser.add_argument("--epochs", default = 5, type = int)
        parser.add_argument("--data_path", type = str)
        parser.add_argument("--batch_size", default = 8, type = int)
        parser.add_argument("--pooling_strategy_type", default = 1, type = int)
        parser.add_argument("--sampled_for_model_test", default = False, type = bool)
        return parser