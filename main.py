import os
from tqdm import tqdm
import yaml
import torch
import sys

import importlib
import faulthandler
import numpy as np
import torch.nn as nn

faulthandler.enable()
import utils
from utils.sync_batchnorm import convert_model

from evaluation.slr_eval.wer_calculation import evaluate

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


class ModelController():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model = self.load_model()
        self.optimizer = utils.Optimizer(self.model, self.arg.optimizer_args)
        self.model, self.optimizer = self.model_to_device(self.model, self.optimizer)

    def run(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            best_dev = 100.0
            best_epoch = 0
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                dev_flag = epoch % 1 == 0

                # train end2end model
                train_model(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if dev_flag:
                    dev_wer = eval_model(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                if dev_wer < best_dev:
                    best_dev = dev_wer
                    best_epoch = epoch
                    model_path = "{}_best_model.pt".format(self.arg.work_dir)
                    self.save_model(epoch, model_path)
                    self.recoder.print_log('Save best model')
                self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    self.save_model(epoch, model_path)

        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            dev_wer = eval_model(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = eval_model(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def load_model(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        return model

    def model_to_device(self, model, optimizer):
        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()

        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.clip = nn.DataParallel(
                model.clip,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model, optimizer

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        model.load_state_dict(weights, strict=True)

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)
        print("Loading ckpt start!")
        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print("Loading ckpt completed!")
        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = torch.utils.data.DataLoader(
                self.dataset[mode],
                batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=self.arg.num_worker,
                collate_fn=self.feeder.collate_fn, )
        print("Loading data finished.")


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def train_model(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        ann = data[4]
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt, ann=ann)
            loss = model.losses_calculation(ret_dict, label, label_lgt)
        IteLoss = loss["IteLoss"]
        ItaLoss = loss["ItaLoss"]
        loss = loss["total_loss"]
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                'Epoch: {},Batch({}/{})done.Loss: {:.5f} IteLoss: {:.5f} ItaLoss: {:.5f} lr:{:.7f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), IteLoss.item(), ItaLoss.item(),
                            clr[0]))
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def eval_model(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
    try:
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            triplet=True,
        )
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))


if __name__ == '__main__':
    parser = utils.get_parser()
    prs = parser.parse_args()
    with open(prs.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    args = parser.parse_args()
    with open(f"./configs/dataset_cfg.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = ModelController(args)
    processor.run()
