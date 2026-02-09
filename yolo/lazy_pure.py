%%writefile yolo/lazy_pure.py
import sys
from pathlib import Path

#import hydra
#from lightning import Trainer

# turn to function like
from hydra import compose, initialize

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
#from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.tools.solver_pure import TrainSolver, ValidateSolver, InferenceSolver
from yolo.utils.logging_utils import setup



def run_solver(cfg: Config, print_frac: float=0.1):
    callbacks, loggers, save_path = setup(cfg)
    print(f"save_path: {save_path, type(save_path)}")

    if cfg.task.task == "train":
        solver = TrainSolver(cfg, save_path, print_frac=print_frac)
        solver.training()
    if cfg.task.task == "validation":
        solver = ValidateSolver(cfg, print_frac=print_frac)
        mAP, mAP50 = solver.val_epoch()
    if cfg.task.task == "inference":
        solver = InferenceSolver(cfg, save_path, print_frac=print_frac)
        solver.run()

    return solver


def main():
    initialize(config_path="config", version_base=None)
    #cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
    '''
    cfg = compose(config_name="config", overrides=["task=inference",
                                                    "name=DemoResult",
                                                    "device=cpu",
                                                    "model=v9-s",
                                                    "task.nms.min_confidence=0.1",
                                                    #"task.fast_inference=onnx",
                                                    "task.data.source=demo/images/inference",
                                                    "+quite=False"
                                                   ])
    '''

    cfg = compose(config_name="config", overrides=["task=validation",
                                                   "task.data.batch_size=1",
                                                   "dataset=coco",
                                                   "name=ValResult",
                                                   "device=cpu", # or cuda
                                                   "model=v9-t",
                                                   "use_wandb=False",
                                                   "cpu_num=1"

                                                   "+limit_val_batches=4"
                                                  ])
    solver = run_solver(cfg, print_frac=0.01)
    

if __name__ == "__main__":
    main()