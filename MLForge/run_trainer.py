import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.trainer import trainer
from configs import workflows

def get_parser():
    
    parser = argparse.ArgumentParser(prog="ML-trainer", description="DNN/XGBoost training", epilog="Good luck!")
    parser.add_argument("-r", "--runConfig", type=str, required=True, help="Path to the configuration file")
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Initialize the workflow
    workflow = workflows[args.runConfig]
    
    # Create the results directory
    for dir in ["CodeANDConfig", "Plots", "Minitrees"]:
        os.makedirs(f"{workflow.OutputDirName}/{dir}", exist_ok=True)
        
    # Copy the configuration file to the results directory
    os.system(f"cp configs/{args.runConfig}.py src/trainer.py {workflow.OutputDirName}/CodeANDConfig/")
    
    # Run the trainer
    trainer(workflow)