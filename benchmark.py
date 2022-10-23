# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""
import os
import time
import argparse
import torch
from datasets.vid_multi import build_vidmulti
from src.configs.default import _C as cfg
from src.models.model_builder import build_model
from util.misc import nested_tensor_from_tensor_list

@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    for iter_ in range(num_iters):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    print(ts)
    return sum(ts) / len(ts)


def benchmark(cfg=cfg):
    parser = argparse.ArgumentParser(description="Benchmark inference speed of Deformable DETR.")
    parser.add_argument("--config-file", default="./experiments/PTSEFormer_r50_8gpus.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--num_iters', type=int, default=300, help='total iters to benchmark speed')
    parser.add_argument('--warm_iters', type=int, default=5, help='ignore first several iters that are very slow')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in inference')
    parser.add_argument('--resume', type=str, help='load the pre-trained checkpoint')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    assert args.warm_iters < args.num_iters and args.num_iters > 0 and args.warm_iters >= 0
    assert args.batch_size > 0
    assert args.resume is None or os.path.exists(args.resume)

    device = torch.device(cfg.TRAIN.device)
    dataset = build_vidmulti(image_set='val', cfg=cfg, split_name=cfg.DATASET.val_dataset)
    model, _, _ = build_model(cfg)
    model.cuda()
    model.eval()

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])
        # model.load_state_dict(ckpt['model'], , strict=False) # If has unexpected key error
    # inputs = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0].cuda() for _ in range(args.batch_size)])

    # samples, _ = dataset.__getitem__(0)
    samples = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0] for _ in range(args.batch_size)])
    tmp = {}
    for k, v in samples.items():
        if isinstance(v, list):
            tmp[k] = []
            for n_t in v:
                tmp[k].append(n_t.to(device, non_blocking=True))
        else:
            tmp[k] = v.to(device, non_blocking=True)
    inputs = tmp
    t = measure_average_inference_time(model, inputs, args.num_iters, args.warm_iters)
    return 1.0 / t * args.batch_size


if __name__ == '__main__':
    fps = benchmark(cfg)
    print(f'Inference Speed: {fps:.1f} FPS')
