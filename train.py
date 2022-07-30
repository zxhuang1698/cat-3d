import os,sys
import torch
import model.base
import model.pretrain
import utils.util_opt as util_opt

from utils.util_run import log


log.process(os.getpid())
log.title("[{}] (training)".format(sys.argv[0]))

opt_cmd = util_opt.parse_arguments(sys.argv[1:])
opt = util_opt.set(opt_cmd=opt_cmd)
util_opt.save_options_file(opt)

with torch.cuda.device(opt.device):

    if opt.pretrain: trainer = model.pretrain.Runner(opt)
    else: trainer = model.base.Runner(opt)

    trainer.load_dataset(opt)
    trainer.build_networks(opt)
    trainer.setup_optimizer(opt)
    trainer.restore_checkpoint(opt)
    trainer.setup_visualizer(opt)
    trainer.train(opt)
