import os,time,warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import torch.utils.tensorboard
import importlib
import tqdm
import shutil
import utils.util_run as util_run
import utils.util_vis as util_vis
import utils.util_eval as util_eval

from copy import deepcopy
from utils.util_run import log,AverageMeter,toggle_grad,GAN_loss,R1_reg
from utils.util_run import EasyDict as edict
from . import discriminator


class Runner():

    def __init__(self,opt):
        super().__init__()
        if os.path.isdir(opt.output_path) and opt.resume == False:
            for filename in os.listdir(opt.output_path):
                if "tfevents" in filename: os.remove(os.path.join(opt.output_path, filename))
                if "vis" in filename: shutil.rmtree(os.path.join(opt.output_path,filename))
                if "gradient" in filename: shutil.rmtree(os.path.join(opt.output_path,filename))
        os.makedirs(opt.output_path,exist_ok=True)
        self.optimizer = getattr(torch.optim,opt.optim.algo)
        if opt.optim.sched:
            self.scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)

    # concatenate samples along the batch dimension
    # the input is a list, with each element a dictionary
    # the element of this dictionary is either tensor or dictionary containing tensors
    def concat_samples(self,sample_list):
        stacked = sample_list[0]
        for key, value in stacked.items():
            if isinstance(value, torch.Tensor):
                tensor_list = [value]
                for sample in sample_list[1:]:
                    tensor_list.append(sample[key])
                stacked[key] = torch.cat(tensor_list, dim=0)
            elif isinstance(value, dict):
                for key_sub, value_sub in value.items():
                    assert isinstance(value_sub, torch.Tensor)
                    tensor_list = [value_sub]
                    for sample in sample_list[1:]:
                        tensor_list.append(sample[key][key_sub])
                    stacked[key][key_sub] = torch.cat(tensor_list, dim=0)
            else:
                raise NotImplementedError
        return stacked
    
    def append_viz_data(self,opt):
        # append data for visualization
        cat_samples = [0] * opt.data.num_classes
        viz_data = {}
        viz_data_list = []
        while sum(cat_samples) < opt.data.num_classes:
            # load the current batch, a dictionary
            current_batch = next(iter(self.viz_loader))
            # fetch the category of each sample
            for i, cat_tensor in enumerate(current_batch['category_label']):
                cat_idx = cat_tensor.item()
                # check whether we have already got enough samples for this category
                if cat_samples[cat_idx] >= 1: continue
                # if not include the current data
                cat_samples[cat_idx] += 1
                viz_data = {}            
                for key, value in current_batch.items():
                    if isinstance(value, torch.Tensor):
                        viz_data[key] = value[i].unsqueeze(0)
                    elif isinstance(value, dict):
                        viz_data[key] = {}
                        for key_sub, value_sub in value.items():
                            assert isinstance(value_sub, torch.Tensor)
                            viz_data[key][key_sub] = value_sub[i].unsqueeze(0)
                    else:
                        raise NotImplementedError
                viz_data_list.append(viz_data)
        self.viz_data.append(self.concat_samples(viz_data_list))
    
    def load_dataset(self,opt,eval_split="val"):
        data = importlib.import_module("dataset.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.batch_order = []
        self.train_data = data.Dataset(opt,split="train")
        self.train_loader = self.train_data.setup_loader(opt,shuffle=True)
        self.num_batches = len(self.train_loader)
        log.info("loading test data...")
        self.test_data = data.Dataset(opt,split=eval_split)
        self.test_loader = self.test_data.setup_loader(opt,shuffle=False,drop_last=False)
        log.info("creating data for visualization...")
        self.viz_loader = self.test_data.setup_loader(opt,shuffle=True,drop_last=False)
        self.viz_data = []
        num_save = 2
        for _ in range(num_save): self.append_viz_data(opt)

    def build_networks(self,opt):
        log.info("building networks...")
        graph_name = 'pretrain' if opt.pretrain else 'graph'
        module = importlib.import_module("model.{}".format(graph_name))
        self.graph = module.Graph(opt).to(opt.device)
        self.discriminator = discriminator.Discriminator(opt).to(opt.device)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        kwargs = {}
        for k,v in opt.optim.params.items():
            if k == 'betas': kwargs[k] = tuple(v)
            else: kwargs[k] = v
        optim_list_G = []
        optim_list_D = []
        for k,v in self.graph.named_parameters():
            optim_list_G.append(v)
        for k,v in self.discriminator.named_parameters():
            optim_list_D.append(v)
        optim_dict_G = [dict(params=optim_list_G,lr=opt.optim.lr_gen),]
        optim_dict_D = [dict(params=optim_list_D,lr=opt.optim.lr_dis),]
        self.optim_G = self.optimizer(optim_dict_G, **kwargs)
        self.optim_D = self.optimizer(optim_dict_D, **kwargs)
        if opt.optim.sched: self.setup_optimizer_scheduler(opt)

    def setup_optimizer_scheduler(self,opt):
        kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
        self.sched_G = self.scheduler(self.optim_G,**kwargs)
        self.sched_D = self.scheduler(self.optim_D,**kwargs)

    def restore_checkpoint(self,opt,best=False,evaluate=False):
        epoch_start,iter_start = None,None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            epoch_start,iter_start,best_val = util_run.restore_checkpoint(opt,self,resume=opt.resume,best=best,evaluate=evaluate)
            self.best_val = best_val
        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            epoch_start,iter_start,best_val = util_run.restore_checkpoint(opt,self,load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self,opt,test=False):
        if opt.tb:
            log.info("setting up visualizers...")
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path,flush_secs=10)
    
    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.iter_skip = self.iter_start%len(self.train_loader)
        self.it = self.iter_start
        if not opt.resume: 
            self.best_val = np.inf
            self.best_ep = 1
        # training
        if self.iter_start==0: self.evaluate(opt,ep=0,training=True)
        for self.ep in range(self.epoch_start,opt.max_epoch):
            self.train_epoch(opt)
        # after training
        self.save_checkpoint(opt,ep=self.ep+1,it=self.it,best_val=self.best_val)
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        log.title("TRAINING DONE")
        log.info("Best CD: %.4f @ epoch %d" % (self.best_val, self.best_ep))
    
    def train_epoch(self,opt):
        # before train epoch
        loss_D = AverageMeter()
        loss_G = AverageMeter()
        acc_D = AverageMeter()
        batch_progress = tqdm.tqdm(range(self.num_batches),desc="training epoch {}".format(self.ep+1),leave=False)
        self.graph.train()
        # train epoch
        loader = iter(self.train_loader)
           
        for _ in batch_progress:
            # if resuming from previous checkpoint, skip until the last iteration number is reached
            if self.iter_skip>0:
                batch_progress.set_description("(fast-forwarding...)")
                self.iter_skip -= 1
                if self.iter_skip==0: batch_progress.set_description("training epoch {}".format(self.ep+1))
                continue
            batch = next(loader)
            # train iteration
            var = edict(batch)
            var = util_run.move_to_device(var,opt.device)
            loss,acc_D_tensor = self.train_iteration(opt,var,batch_progress)
            loss_D.update(loss.discriminator.item(), len(var.idx))
            loss_G.update(loss.generator.item(), len(var.idx))
            acc_D.update(acc_D_tensor.item(), len(var.idx)) 
            
        # after train epoch
        log.info("Loss_D: %.4f, Loss_G: %.4f, Acc: %.2f%%" % (loss_D.avg,loss_G.avg,acc_D.avg*100))
        lr = self.sched_G.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        if opt.optim.sched: 
            self.sched_G.step()
            self.sched_D.step()
        log.loss_train(opt,self.ep+1,lr,loss,self.timer)
        if (self.ep+1)%opt.freq.eval==0: 
            current_val = self.evaluate(opt,ep=self.ep+1,training=True)
            if current_val < self.best_val:
                self.best_val = current_val
                self.best_ep = self.ep + 1
                self.save_checkpoint(opt,ep=self.ep+1,it=self.it,best_val=self.best_val,best=True,latest=True)

    def train_iteration(self,opt,var,loader):
        # before train iteration
        self.timer.it_start = time.time()

        # train iteration
        loss,var = self.generator_trainstep(opt,var)
        dloss_real, dloss_fake, accuracy = self.discriminator_trainstep(opt, var)
        loss.real = dloss_real
        loss.fake = dloss_fake
        loss.discriminator = loss.real + loss.fake

        # after train iteration
        if (self.it)%opt.freq.vis==0: self.visualize(opt,var,step=self.it,split="train")
        if (self.it+1)%opt.freq.ckpt_latest==0: self.save_checkpoint(opt,ep=self.ep,it=self.it+1,best_val=self.best_val,latest=True)
        if (self.it)%opt.freq.scalar==0: self.log_scalars(opt,var,loss,step=self.it,split="train")
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util_run.update_timer(opt,self.timer,self.ep,len(loader))
        return loss, accuracy

    def generator_trainstep(self,opt,var):
        toggle_grad(self.graph, True)
        toggle_grad(self.discriminator, False)
        self.graph.train()
        self.optim_G.zero_grad()

        # forward for recon loss and rnd view
        var,loss = self.graph(opt,var,training=True,get_loss=True,render_rnd=True)
        loss = self.summarize_loss(opt,var,loss)
        
        # G loss
        x_fake = torch.cat([var.fake_rnd_view,var.fake_recon], dim=0)
        category_fake = torch.cat([var.category_label,var.category_label], dim=0)
        d_fake = self.discriminator(x_fake,category_fake)
        gloss = GAN_loss(d_fake, 1)

        # aggregate and update
        loss_recon_gen = opt.loss_weight.gen * gloss + loss.all
        loss_recon_gen.backward()
        if self.it > opt.optim.clip_iter:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), opt.optim.clip_norm)
        loss.generator = gloss
        self.optim_G.step()
        return loss,var

    def discriminator_trainstep(self, opt, var):
        toggle_grad(self.graph, False)
        toggle_grad(self.discriminator, True)
        self.graph.train()

        # get real and fake images
        x_real = var.real_input.detach().clone()
        x_real.requires_grad_()
        with torch.no_grad():
            var = self.graph(opt,var,training=True,get_loss=False,render_rnd=True)
            x_fake = torch.cat([var.fake_rnd_view.clone(),var.fake_recon.clone()], dim=0)
            category_fake = torch.cat([var.category_label,var.category_label], dim=0)

        # get the real logits
        self.optim_D.zero_grad()
        d_real = self.discriminator(x_real,var.category_label)
        dloss_real = GAN_loss(d_real, 1)
        reg = opt.gan.reg_weight * R1_reg(d_real, x_real).mean()

        # get the fake logits
        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake,category_fake)
        dloss_fake = GAN_loss(d_fake, 0)

        # backward D loss and R1 regularization
        dloss_real.backward(retain_graph=True)
        reg.backward()
        dloss_fake.backward()

        # calculate the accuracy and update
        accuracy = ((d_real>0).float().sum() + (d_fake<0).float().sum()) / (3*len(var.idx))
        self.optim_D.step()
        toggle_grad(self.discriminator, False)

        return dloss_real, dloss_fake, accuracy

    def summarize_loss(self,opt,var,loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key].mean()),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key].mean()),"loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key].mean()
        loss.update(all=loss_all)
        return loss

    @torch.no_grad()
    def evaluate(self,opt,ep,training=False):
        self.graph.eval()

        # params for metrics
        f_scores = []
        loss_eval = edict()
        metric_eval = dict(dist_acc=0.,dist_cov=0.)
        
        # dataloader on the test set
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)

        for it,batch in enumerate(loader):

            # inference the model
            var = edict(batch)
            var,loss = self.evaluate_batch(opt,var)

            # record loss and CD for evaluation
            for key in loss:
                loss_eval.setdefault(key,0.)
                loss_eval[key] += loss[key]*len(var.idx)

            dist_acc,dist_cov = util_eval.eval_metrics(opt,var)

            # accumulate f-score
            f_scores.append(var.f_score)

            metric_eval["dist_acc"] += dist_acc*len(var.idx)
            metric_eval["dist_cov"] += dist_cov*len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))

            # save the predicted mesh for vis data if in train mode
            if it==0 and training: 
                for i in range(len(self.viz_data)):
                    var_viz = edict(deepcopy(self.viz_data[i]))
                    var_viz,_ = self.evaluate_batch(opt,var_viz)
                    util_eval.eval_metrics(opt,var_viz,vis_only=True)
                    self.visualize(opt,var_viz,step=ep,split="eval")
                    self.dump_results(opt,var_viz,ep,train=True)
            
            # dump the result if in eval mode
            if not training: 
                var = self.graph.render_random_view(opt,var)
                self.dump_results(opt,var,ep,write_new=(it==0))
                
        # print f_scores
        f_scores = torch.cat(f_scores, dim=0).mean(dim=0)
        print('##############################')
        for i in range(len(opt.eval.f_thresholds)):
            print('F-score @ %.2f: %.4f' % (opt.eval.f_thresholds[i]*100, f_scores[i].item()))
        print('##############################')

        # write to file
        f_score_file = os.path.join(opt.output_path, 'f_score.txt')
        with open(f_score_file, "w") as outfile:
            for i in range(len(opt.eval.f_thresholds)):
                outfile.write('F-score @ %.2f: %.4f\n' % (opt.eval.f_thresholds[i]*100, f_scores[i].item()))

        # average the metric            
        for key in loss_eval: loss_eval[key] /= len(self.test_data)
        for key in metric_eval: metric_eval[key] /= len(self.test_data)
        
        # print eval info
        log.loss_eval(opt,loss_eval,chamfer=(metric_eval["dist_acc"],
                                             metric_eval["dist_cov"]))
        
        val_metric = (metric_eval["dist_acc"] + metric_eval["dist_cov"]) / 2
        return val_metric.item()

    def evaluate_batch(self,opt,var):
        var = util_run.move_to_device(var,opt.device)
        var,loss = self.graph(opt,var,training=False)
        loss = self.summarize_loss(opt,var,loss)
        return var,loss

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        if split=="train":
            dist_acc,dist_cov = util_eval.eval_metrics(opt,var)
            metric = dict(dist_acc=dist_acc,dist_cov=dist_cov)
        for key,value in loss.items():
            if key=="all": continue
            self.tb.add_scalar("{0}/loss_{1}".format(split,key),value.mean(),step)
        if metric is not None:
            for key,value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split,key),value,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        util_vis.tb_image(opt,self.tb,step,split,"image_input",var.rgb_input_map,masks=None,from_range=(0,1),poses=var.pose)
        util_vis.tb_image(opt,self.tb,step,split,"image_recon",var.rgb_recon_map,masks=None,from_range=(0,1),poses=var.pose)
        util_vis.tb_image(opt,self.tb,step,split,"mask_input",var.mask_input_map)
        util_vis.tb_image(opt,self.tb,step,split,"mask_recon",var.soft_mask_map)

    @torch.no_grad()
    def dump_results(self,opt,var,ep,write_new=False,train=False):
        if train == False:
            current_folder = "dump"
            os.makedirs("{}/dump/".format(opt.output_path),exist_ok=True)
        else:
            current_folder = "vis_{}".format(ep)
            os.makedirs("{}/{}/".format(opt.output_path,current_folder),exist_ok=True)
        
        util_vis.dump_images(opt,var.idx,"image_input",var.rgb_input_map,masks=None,from_range=(0,1),folder=current_folder)
        util_vis.dump_images(opt,var.idx,"image_recon",var.rgb_recon_map,masks=var.mask_map,from_range=(0,1),poses=var.pose,folder=current_folder)
        util_vis.dump_images(opt,var.idx,"mask_recon",var.soft_mask_map,folder=current_folder)
        util_vis.dump_images(opt,var.idx,"mask_input",var.mask_input_map,folder=current_folder)
        util_vis.dump_meshes(opt,var.idx,"mesh",var.mesh_pred,folder=current_folder)
        util_vis.dump_pointclouds_compare(opt,var.idx,"pointclouds_comp",var.dpc_pred,var.dpc.points,folder=current_folder)
        if not train:
            # write chamfer distance results
            chamfer_fname = "{}/chamfer.txt".format(opt.output_path)
            with open(chamfer_fname,"w" if write_new else "a") as file:
                for i,acc,comp in zip(var.idx,var.cd_acc,var.cd_comp):
                    file.write("{} {:.8f} {:.8f}\n".format(i,acc,comp))

    def save_checkpoint(self,opt,ep=0,it=0,best_val=np.inf,latest=False,best=False):
        util_run.save_checkpoint(opt,self,ep=ep,it=it,best_val=best_val,latest=latest,best=best)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group,opt.name,ep,it))
        if best:
            log.info("Saving the current model as the best...")
