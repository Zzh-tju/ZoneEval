# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs, outputs0_1, outputs1_2, outputs2_3, outputs3_4, outputs4_5 = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if cfg.model.test_cfg.zone_eval == False:
                print(dataset.evaluate(outputs, **eval_kwargs))
            else:
                num_class = cfg.model.bbox_head.num_classes
                if cfg.evaluation.metric == 'mAP':
                    eval_results, map, mar, aps = dataset.zone_evaluate(outputs, **eval_kwargs, ri=0.0, rj=0.5)
                    print('Overall measures:\n')
                    print(eval_results, map, mar, aps)


                    eval_results0_1, map0_1, mar0_1, aps0_1 = dataset.zone_evaluate(outputs0_1, **eval_kwargs, ri=0.0, rj=0.1)
                    print('Zone 0-1:\n')
                    print(eval_results0_1, map0_1, mar0_1, aps0_1)

                    eval_results1_2, map1_2, mar1_2, aps1_2 = dataset.zone_evaluate(outputs1_2, **eval_kwargs, ri=0.1, rj=0.2)
                    print('Zone 1-2:\n')
                    print(eval_results1_2, map1_2, mar1_2, aps1_2)

                    eval_results2_3, map2_3, mar2_3, aps2_3 = dataset.zone_evaluate(outputs2_3, **eval_kwargs, ri=0.2, rj=0.3)
                    print('Zone 2-3:\n')
                    print(eval_results2_3, map2_3, mar2_3, aps2_3)

                    eval_results3_4, map3_4, mar3_4, aps3_4 = dataset.zone_evaluate(outputs3_4, **eval_kwargs, ri=0.3, rj=0.4)
                    print('Zone 3-4:\n')
                    print(eval_results3_4, map3_4, mar3_4, aps3_4)

                    eval_results4_5, map4_5, mar4_5, aps4_5 = dataset.zone_evaluate(outputs4_5, **eval_kwargs, ri=0.4, rj=0.5)
                    print('Zone 4-5:\n')
                    print(eval_results4_5, map4_5, mar4_5, aps4_5)
                
                    ZP = np.zeros([5,11])
                    ZR = np.zeros([5,11])
                    ZP_class = np.zeros([5,num_class])
                    ZP[0,:] = map0_1
                    ZP[1,:] = map1_2
                    ZP[2,:] = map2_3
                    ZP[3,:] = map3_4
                    ZP[4,:] = map4_5

                    ZR[0,:] = mar0_1
                    ZR[1,:] = mar1_2
                    ZR[2,:] = mar2_3
                    ZR[3,:] = mar3_4
                    ZR[4,:] = mar4_5

                    ZP_class[0,:] = aps0_1
                    ZP_class[1,:] = aps1_2
                    ZP_class[2,:] = aps2_3
                    ZP_class[3,:] = aps3_4
                    ZP_class[4,:] = aps4_5

                    ZP_var = np.var(ZP, axis=0)
                    ZR_var = np.var(ZR, axis=0)
                    ZP_class_var = np.var(ZP_class, axis=0)

                    print('ZP variance: ', ZP_var)
                    print('ZR variance: ', ZR_var)
                    print('ZP class variance: ', ZP_class_var)

                    print('Zone:, ZP50, ZP55, ZP60, ZP65, ZP70, ZP75, ZP80, ZP85, ZP90, ZP95, ZP')
                    print('z05: ', np.around(map, 1))
                    print('z01: ', np.around(ZP[0,:], 1))
                    print('z12: ', np.around(ZP[1,:], 1))
                    print('z23: ', np.around(ZP[2,:], 1))
                    print('z34: ', np.around(ZP[3,:], 1))
                    print('z45: ', np.around(ZP[4,:], 1))
                    print('SP50, SP55, SP60, SP65, SP70, SP75, SP80, SP85, SP90, SP95, SP')
                    SP = 0.36 * ZP[0,:] + 0.28 * ZP[1,:] + 0.2 * ZP[2,:] + 0.12 * ZP[3,:] + 0.04 * ZP[4,:]
                    print(np.around(SP, 1))

                    #print('Zone:, ZR50, ZR55, ZR60, ZR65, ZR70, ZR75, ZR80, ZR85, ZR90, ZR95, ZR')
                    #print('z05: ', np.around(mar, 1))
                    #print('z01: ', np.around(ZR[0,:], 1))
                    #print('z12: ', np.around(ZR[1,:], 1))
                    #print('z23: ', np.around(ZR[2,:], 1))
                    #print('z34: ', np.around(ZR[3,:], 1))
                    #print('z45: ', np.around(ZR[4,:], 1))
                    #print('SR50, SR55, SR60, SR65, SR70, SR75, SR80, SR85, SR90, SR95, SR')
                    #SR = 0.36 * ZR[0,:] + 0.28 * ZR[1,:] + 0.2 * ZR[2,:] + 0.12 * ZR[3,:] + 0.04 * ZR[4,:]
                    #print(np.around(SR, 1))

                    print('============================= ZP per class =============================')
                    print('z05: ', np.around(aps, 1))
                    print('z01: ', np.around(ZP_class[0,:], 1))
                    print('z12: ', np.around(ZP_class[1,:], 1))
                    print('z23: ', np.around(ZP_class[2,:], 1))
                    print('z34: ', np.around(ZP_class[3,:], 1))
                    print('z45: ', np.around(ZP_class[4,:], 1))
                    SP_class = 0.36 * ZP_class[0,:] + 0.28 * ZP_class[1,:] + 0.2 * ZP_class[2,:] + 0.12 * ZP_class[3,:] + 0.04 * ZP_class[4,:]
                    print('Per Class SP: ', np.around(SP_class, 1))

                if cfg.evaluation.metric == 'bbox':
                    ZP = np.zeros([5,7])
                    ZP_class = np.zeros([5,num_class])

                    eval_results, metric, ap_class, _= dataset.zone_evaluate(outputs, **eval_kwargs, ri=0.0, rj=0.5)

                    eval_results0_1, metric0_1, ap_class0_1, _ = dataset.zone_evaluate(outputs0_1, **eval_kwargs, ri=0.0, rj=0.1)
                    eval_results1_2, metric1_2, ap_class1_2, _ = dataset.zone_evaluate(outputs1_2, **eval_kwargs, ri=0.1, rj=0.2)
                    eval_results2_3, metric2_3, ap_class2_3, _ = dataset.zone_evaluate(outputs2_3, **eval_kwargs, ri=0.2, rj=0.3)
                    eval_results3_4, metric3_4, ap_class3_4, _ = dataset.zone_evaluate(outputs3_4, **eval_kwargs, ri=0.3, rj=0.4)
                    eval_results4_5, metric4_5, ap_class4_5, _ = dataset.zone_evaluate(outputs4_5, **eval_kwargs, ri=0.4, rj=0.5)
                    ZP[0,:] = metric0_1
                    ZP[1,:] = metric1_2
                    ZP[2,:] = metric2_3
                    ZP[3,:] = metric3_4
                    ZP[4,:] = metric4_5
                    ZP_class[0,:] = ap_class0_1
                    ZP_class[1,:] = ap_class1_2
                    ZP_class[2,:] = ap_class2_3
                    ZP_class[3,:] = ap_class3_4
                    ZP_class[4,:] = ap_class4_5

                    ZP_var = np.var(ZP, axis=0)
                    print('ZP variance: ', ZP_var)

                    ZP_class_var = np.var(ZP_class, axis=0)

                    print('Zone: ZP, ZP50, ZP75, ZPs, ZPm, ZPl, ZR100')
                    print('z05: ', np.around(metric, 1))
                    print('z01: ', np.around(ZP[0,:], 1))
                    print('z12: ', np.around(ZP[1,:], 1))
                    print('z23: ', np.around(ZP[2,:], 1))
                    print('z34: ', np.around(ZP[3,:], 1))
                    print('z45: ', np.around(ZP[4,:], 1))
                    SP = 0.36 * ZP[0,:] + 0.28 * ZP[1,:] + 0.2 * ZP[2,:] + 0.12 * ZP[3,:] + 0.04 * ZP[4,:]
                    SP_class = 0.36 * ZP_class[0,:] + 0.28 * ZP_class[1,:] + 0.2 * ZP_class[2,:] + 0.12 * ZP_class[3,:] + 0.04 * ZP_class[4,:]
                    print('SP, SP50, SP75, SPs, SPm, SPl, SR100')
                    print(np.around(SP, 1))

                    print('============================= ZP per class =============================')
                    print('z05: ', np.around(aps, 1))
                    print('z01: ', np.around(ZP_class[0,:], 1))
                    print('z12: ', np.around(ZP_class[1,:], 1))
                    print('z23: ', np.around(ZP_class[2,:], 1))
                    print('z34: ', np.around(ZP_class[3,:], 1))
                    print('z45: ', np.around(ZP_class[4,:], 1))
                    print('ZP_class_var: ', ZP_class_var)
                    print('Per Class SP: ', np.around(SP_class, 1))
            metric_dict = dict(config=args.config, metric=eval_results)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
