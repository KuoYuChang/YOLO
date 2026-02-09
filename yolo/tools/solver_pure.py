from math import ceil
from pathlib import Path

from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler

import time
import torch

class BaseSolver():
    def __init__(self, cfg: Config):
        model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        self.model = model
    
    #def forward(self, x):
    #    return self.model(x)

class ValidateSolver(BaseSolver):
    def __init__(self, cfg: Config, print_frac: float=0.1):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.print_frac = print_frac
        
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)

        print_num = int(self.print_frac * len(self.val_loader))
        self.val_print_num = max(print_num, 1)

        self.device = cfg.device

        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

        self.model = self.model.to(self.device)
        

        


    #def validation_step(self, batch, batch_idx):
    def val_epoch(self):
        start_t = time.time()


        self.model.eval()


        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):            
                batch_size, images, targets, rev_tensor, img_paths = batch
                H, W = images.shape[2:]
                images = images.to(self.device)
                targets = targets.to(self.device)
    
                raw_out = self.model(images)
                predicts = self.post_process(raw_out, image_size=[W, H])
                print("predict: ", predicts)
                print()
                print("targets: ", targets)
                mAP = self.metric.update(
                    [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
                )
    
                # activate when full test
                if batch_idx % self.val_print_num == 0:
                    current_time = time.time() - start_t
                    print(f'time elapsed: {current_time}, finish {batch_idx/len(self.val_loader)}')
    
                # remove when full test
                if batch_idx >= 2:
                    break

        print("-------------- Evaluating -------------------")
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        #self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        #self.log_dict(
        #    {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
        #    sync_dist=True,
        #    rank_zero_only=True,
        #)
        print(f"PyCOCO/AP @ .5:.95: {epoch_metrics["map"]}, PyCOCO/AP @ .5: {epoch_metrics["map_50"]}")
        mAP = epoch_metrics["map"]
        mAP_50 = epoch_metrics["map_50"]
        
        self.metric.reset()

        return mAP, mAP_50

class TrainSolver(ValidateSolver):
    def __init__(self, cfg: Config, model_save_fd: str, print_frac: float=0.1):
        super().__init__(cfg, print_frac)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)
        print_num = int(self.print_frac * len(self.train_loader))
        self.train_print_num = max(print_num, 1)

        self.loss_fn = create_loss_function(self.cfg, self.vec2box)
        self.optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        self.scheduler = create_scheduler(self.optimizer, self.cfg.task.scheduler)

        self.model_save_path = model_save_fd / "weights"
        try:
            self.model_save_path.mkdir()
        except FileExistsError:
            print(f"Directory '{self.model_save_path}' already exists.")
        except FileNotFoundError:
            print(f"Parent directory does not exist.")
        
        self.model_save_path = self.model_save_path / cfg.model.name

    # current float 32, para to half float 16?
    def save_model(self, index):
        model_dict = self.model.state_dict()
        #print(f"model_dict keys: {model_dict.keys()}")
        # edit layer names
        model_dict = {name.removeprefix("model."): key for name, key in model_dict.items()}
        #print(f"model_dict keys: {model_dict.keys()}")
        final_path = Path(str(self.model_save_path) + str(index) +".pt")
        torch.save(model_dict, final_path)
    
    def train_epoch(self, epoch_ith):
        start_t = time.time()

        print(f"------------ start epoch {epoch_ith} ------------")

        running_loss = 0.0

        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_size, images, targets, *_ = batch

            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # predict output
            predicts = self.model(images)
            aux_predicts = self.vec2box(predicts["AUX"])
            main_predicts = self.vec2box(predicts["Main"])

            # loss
            loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
            loss.backward()

            running_loss += loss.item()
            
            # gradient optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if batch_idx % self.train_print_num == 0:
                current_time = time.time() - start_t
                print(f'time elapsed: {current_time}, loss: {running_loss / self.train_print_num:.3f}, finish {batch_idx/len(self.train_loader)}')
                running_loss = 0.0
            if batch_idx >= 0:
                break

        # scheduler, current fix as epoch-wise
        self.scheduler.step()
        
        # evaluate valid set?
        print("-------------- Evaluating -------------------")
        # model be eval mode when calling self.val_epoch
        mAP, mAP50 = self.val_epoch()

        print("================ end ====================\n\n")

        return mAP, mAP50

    def training(self, num_epoch=10):
        best_mAP = 0.0
        best_mAP50 = 0.0
        
        for epoch_ith in range(num_epoch):
            # train epoch start
            #self.optimizer[0].next_epoch(
            #    ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
            #)
            self.vec2box.update(self.cfg.image_size)

            # run epoch
            mAP_ith, mAP50_ith = self.train_epoch(epoch_ith)

            # if better, save model
            if mAP_ith > best_mAP:
                print(f"saving better model at epoch: {epoch_ith}")
                best_mAP = mAP_ith
                self.save_model(epoch_ith)

class InferenceSolver(BaseSolver):
    def __init__(self, cfg: Config, save_path: str, print_frac: float=0.1):
        super().__init__(cfg)
        self.cfg = cfg
        self.save_path = save_path
        self.print_frac = print_frac
        
        self.device = cfg.device
        
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)
        
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

        self.model = self.model.to(self.device)

    def run(self):
        start_t = time.time()

        print_num = int(self.print_frac * len(self.predict_loader))

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.predict_loader):
                images, rev_tensor, origin_frame = batch
                images = images.to(self.device)
                rev_tensor = rev_tensor.to(self.device)
                
                predicts = self.post_process(self.model(images), rev_tensor=rev_tensor)
                img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
                if getattr(self.predict_loader, "is_stream", None):
                    fps = self._display_stream(img)
                else:
                    fps = None
                if getattr(self.cfg.task, "save_predict", None):
                    self._save_image(img, batch_idx)
    

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.save_path) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")