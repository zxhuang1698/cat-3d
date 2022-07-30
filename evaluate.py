import os,sys
import torch
import model.base
import utils.util_opt as util_opt

from utils.util_run import log


log.process(os.getpid())
log.title("[{}] (evaluating)".format(sys.argv[0]))

opt_cmd = util_opt.parse_arguments(sys.argv[1:])
opt = util_opt.set(opt_cmd=opt_cmd)

with torch.cuda.device(opt.device):

    evaluator = model.base.Runner(opt)

    evaluator.load_dataset(opt,eval_split="test")
    evaluator.build_networks(opt)
    evaluator.restore_checkpoint(opt,best=True,evaluate=True)
    evaluator.setup_visualizer(opt,test=True)

    evaluator.evaluate(opt,ep=0)
