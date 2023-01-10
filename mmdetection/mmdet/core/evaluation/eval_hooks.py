# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import os.path as osp

import mmcv
import numpy as np
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, zone_eval=(True, 'mAP', 80), dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None
        self.zone_eval, self.metric, self.num_class = zone_eval[0], zone_eval[1], zone_eval[2]

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # avoid error when zone eval is True
        if self.zone_eval == True:
            tmpdir = None

        from mmdet.apis import multi_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        outputs, outputs0_1, outputs1_2, outputs2_3, outputs3_4, outputs4_5 = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = outputs
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            print('-----------------------------------')
            eval_results, map, mar, aps = self.dataloader.dataset.zone_evaluate(outputs, metric=self.metric, ri=0.0, rj=0.5)
            if self.zone_eval == True:
                eval_results0_1, map0_1, mar0_1, aps0_1 = self.dataloader.dataset.zone_evaluate(outputs0_1, metric=self.metric, ri=0.0, rj=0.1)
                eval_results1_2, map1_2, mar1_2, aps1_2 = self.dataloader.dataset.zone_evaluate(outputs1_2, metric=self.metric, ri=0.1, rj=0.2)
                eval_results2_3, map2_3, mar2_3, aps2_3 = self.dataloader.dataset.zone_evaluate(outputs2_3, metric=self.metric, ri=0.2, rj=0.3)
                eval_results3_4, map3_4, mar3_4, aps3_4 = self.dataloader.dataset.zone_evaluate(outputs3_4, metric=self.metric, ri=0.3, rj=0.4)
                eval_results4_5, map4_5, mar4_5, aps4_5 = self.dataloader.dataset.zone_evaluate(outputs4_5, metric=self.metric, ri=0.4, rj=0.5)

                if self.metric=='mAP':
                    ZP = np.zeros([5,11])
                    ZR = np.zeros([5,11])
                    ZP_class = np.zeros([5,self.num_class])
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

                else:
                    ZP = np.zeros([5,7])
                    ZP_class = np.zeros([5,self.num_class])
                    ZP[0,:] = metric0_1
                    ZP[1,:] = metric1_2
                    ZP[2,:] = metric2_3
                    ZP[3,:] = metric3_4
                    ZP[4,:] = metric4_5
                    #ZP_class[0,:] = ap_class0_1
                    #ZP_class[1,:] = ap_class1_2
                    #ZP_class[2,:] = ap_class2_3
                    #ZP_class[3,:] = ap_class3_4
                    #ZP_class[4,:] = ap_class4_5

                    ZP_var = np.var(ZP*100, axis=0)
                    print('ZP variance: ', ZP_var)

                    #ZP_class_var = np.var(ZP_class*100, axis=0)
                    #print('mAP_class_var: ', mAP_class_var)

                    print('Zone: ZP, ZP50, ZP75, ZPs, ZPm, ZPl, ZR100')
                    print('z05: ', np.around(metric, 1))
                    print('z01: ', np.around(ZP[0,:], 1))
                    print('z12: ', np.around(ZP[1,:], 1))
                    print('z23: ', np.around(ZP[2,:], 1))
                    print('z34: ', np.around(ZP[3,:], 1))
                    print('z45: ', np.around(ZP[4,:], 1))
                    SP = 0.36 * ZP[0,:] + 0.28 * ZP[1,:] + 0.2 * ZP[2,:] + 0.12 * ZP[3,:] + 0.04 * ZP[4,:]
                    print('SP, SP50, SP75, SPs, SPm, SPl, SR100')
                    print(np.around(SP, 1))

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and eval_results:
                self._save_ckpt(runner, eval_results)
