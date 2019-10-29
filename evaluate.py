"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import model.data_loader as data_loader
import model.net as net
import utils
from sklearn import linear_model

def evaluate(model, loss_fn, dataloader, metrics, params, setting, epoch, writer=None):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        covar_mode: (bool) include covariate data in dataloader
    """

    # set model to evaluation mode
    model.eval()
    model.to(params.device)

    # summary for current eval loop
    summ  = []
    preds = [] # for saving last predictions
    bn_activations = []

    # create storate for tensors for OLS after minibatches
    Xhats = []
    Zhats = []


    # for counterfactuals
    if setting.counterfactuals:
        y0_hats = []
        y1_hats = []

    # compute metrics over the dataset
    for batch in dataloader:
        summary_batch = {}
        batch = {k: v.to(params.device) for k, v in batch.items()}
        img_batch    = batch["image"].to(params.device, non_blocking=True)
        labels_batch = batch["label"].to(params.device, non_blocking=True)
        if setting.covar_mode and epoch > params.suppress_t_epochs:
            data_batch = batch["t"].to(params.device, non_blocking=True).view(-1,1)
        else:
            data_batch = torch.zeros((params.batch_size, 1), requires_grad=False).to(params.device, non_blocking=True)

        if params.multi_task:
            # x_target_batch = Variable(batch["x"].to(params.device)).type(torch.cuda.LongTensor)
            x_target_batch = batch["x"].to(params.device)
            y_target_batch = batch["y"].to(params.device)
            labels_batch = {'x': x_target_batch, 'y': y_target_batch}
        
        # compute model output
        # output_batch, bn_batch = model(img_batch, data_batch)
        output_batch = model(img_batch, data_batch, epoch)

        # calculate loss
        if setting.fase == "feature":
            # calculate loss for z directly, to get clear how well this can be measured
            loss_fn_z = torch.nn.MSELoss()
            loss_z = loss_fn_z(output_batch["y"].squeeze(), batch["z"])
            loss   = loss_z
            summary_batch["loss_z"] = loss_z.item()
        else:
            loss_fn_y = torch.nn.MSELoss()
            loss_y = loss_fn_y(output_batch["y"].squeeze(), batch["y"])
            loss   = loss_y
            summary_batch["loss_y"] = loss_y.item()

        # calculate loss for colllider x
        loss_fn_x = torch.nn.MSELoss()
        loss_x = loss_fn_x(output_batch["bnx"].squeeze(), batch["x"])
        summary_batch["loss_x"] = loss_x.item()
        if not params.alpha == 1:
            # possibly weigh down contribution of estimating x
            loss_x *= params.alpha
            summary_batch["loss_x_weighted"] = loss_x.item()

        # add x loss to total loss
        loss += loss_x

        # add least squares regression on final layer
        if params.do_least_squares:
            X    = batch["x"].view(-1,1)
            t    = batch["t"].view(-1,1)
            Z    = output_batch["bnz"]
            if Z.ndimension() == 1:
                Z.unsqueeze_(1)
            Xhat = output_batch["bnx"]
            # add intercept
            Zi = torch.cat([torch.ones_like(t), Z], 1)
            # add treatment info
            Zt = torch.cat([Zi, t], 1)
            Y  = batch["y"].view(-1,1)

            # regress y on final layer, without x
            betas_y = net.cholesky_least_squares(Zt, Y, intercept=False)
            y_hat   = Zt.matmul(betas_y).view(-1,1)
            mse_y  = ((Y - y_hat)**2).mean()

            summary_batch["regr_b_t"] = betas_y[-1].item()
            summary_batch["regr_loss_y"] = mse_y.item()

            # regress x on final layer without x
            betas_x = net.cholesky_least_squares(Zi, Xhat, intercept=False)
            x_hat   = Zi.matmul(betas_x).view(-1,1)
            mse_x  = ((Xhat - x_hat)**2).mean()

            # store all tensors for single pass after epoch
            Xhats.append(Xhat.detach().cpu())
            Zhats.append(Z.detach().cpu())

            summary_batch["regr_loss_x"] = mse_x.item()


        # add loss_bn only after n epochs
        if params.bottleneck_loss and epoch > params.bn_loss_lag_epochs:
            # only add to loss when bigger than margin
            if params.bn_loss_margin_type == "dynamic-mean":
                # for each batch, calculate loss of just using mean for predicting x
                mse_x_mean = ((X - X.mean())**2).mean()
                loss_bn = torch.max(torch.zeros_like(mse_x), mse_x_mean - mse_x)
            elif params.bn_loss_margin_type == "fixed":
                mse_diff = params.bn_loss_margin - mse_x
                loss_bn = torch.max(torch.zeros_like(mse_x), mse_diff)
            else:
                raise NotImplementedError(f'bottleneck loss margin type not implemented: {params.bn_loss_margin_type}')

            # possibly reweigh bottleneck loss and add to total loss
            summary_batch["loss_bn"] = loss_bn.item()
            # note is this double?
            if loss_bn > params.bn_loss_margin:
                loss_bn *= params.bottleneck_loss_wt
                loss    += loss_bn

       # generate counterfactual predictions
        if setting.counterfactuals:
            batch_t0 = Variable(torch.zeros_like(data_batch).to(torch.float32), requires_grad=False).to(params.device)
            batch_t1 = Variable(torch.ones_like(data_batch).to(torch.float32), requires_grad=False).to(params.device)
            y0_batch = model(img_batch, batch_t0)
            y1_batch = model(img_batch, batch_t1)
            y0_hats.append(y0_batch["y"].detach().cpu().numpy())
            y1_hats.append(y1_batch["y"].detach().cpu().numpy())


        # write out activations of bottleneck layer
        if params.multi_task:
            bn_activations.append(output_batch["bnz"])
        else:
            bn_activations.append(output_batch["bn"])

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        if (len(setting.outcome) > 1) or params.multi_task:
            for var, batch in labels_batch.items():
                labels_batch[var] = batch.data.cpu().numpy()
        else:
            labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        data_batch = data_batch.data.cpu().numpy()
        for var, batch in output_batch.items():
            output_batch[var] = batch.detach().cpu().numpy()
        if params.multi_task:
            metrics_xy = {m: net.all_metrics[m] for m in setting.metrics_xy}
            for var, batch in labels_batch.items():
                for metric, metric_fn in metrics_xy.items():
                    summary_batch[metric+"_"+var] = metric_fn(setting, model, output_batch[var], labels_batch[var], data_batch)
            if "b_t" in setting.metrics:
                summary_batch["b_t"] = net.all_metrics["b_t"](setting, model, None, None)

        else:
            NotImplementedError
            # summary_batch = {metric: metrics[metric](setting, model, output_batch[setting.outcome[0]], labels_batch, data_batch)
            #                 for metric in metrics}

        summary_batch["loss"]   = loss.item()
        summ.append(summary_batch)
        #if "y" in setting.outcome:
        preds.append(output_batch["y"])
        #else:
        #    preds.append(output_batch[setting.outcome[0]])



    # compute mean of all metrics in summary
    metrics_mean = {metric:np.nanmean([x[metric] for x in summ]) for metric in summ[0]} 

#    if "ate" in setting.metrics:
 #       metrics_mean["ate"] = all_metrics["ate"](setting, model, preds, )
    
    if params.save_bn_activations:
        # write out batch activations
        bn_activations = torch.cat(bn_activations, 0).detach().cpu().numpy()
        writer.add_histogram("bn_activations", bn_activations, epoch+1)


    # get means and covariances
    if "bottleneck_loss" in setting.metrics:
        bn_means    = bn_activations.mean(dim=0)
        bn_sds      = bn_activations.std(dim=0)
        bn_cov      = net.cov(bn_activations)
        bn_offdiags = net.get_of_diag(bn_cov.detach().cpu().numpy())
        writer.add_histogram("bn_covariances", bn_offdiags, epoch+1)



    # export predictions

    preds  = np.vstack([x.reshape(-1,1) for x in preds])
    writer.add_histogram('predictions', preds, epoch+1)
    labels = dataloader.dataset.df[setting.outcome[0]].values.astype(np.float32)

    # predict individual treatment effects (only worth-while when there is an interaction with t)
    if setting.counterfactuals:
        y0_hats = np.vstack(y0_hats)
        y1_hats = np.vstack(y1_hats)
        ite_hats = y1_hats - y0_hats
        metrics_mean["ite_mean"] = ite_hats.mean()

        y0s = dataloader.dataset.df["y0"].values.astype(np.float32)
        y1s = dataloader.dataset.df["y1"].values.astype(np.float32)
        ites = y1s - y0s
        metrics_mean["pehe"] = np.sqrt(np.mean(np.power((ite_hats - ites), 2)))

        metrics_mean["loss_y1"] = ((y1s - y1_hats)**2).mean()
        metrics_mean["loss_y0"] = ((y0s - y0_hats)**2).mean()

    # in case of single last layer where x is part of, do regression on this layer
    if params.bn_place == "single-regressor" and params.do_least_squares:
        Xhat  = torch.cat(Xhats, 0).view(-1,1).float()
        Zhat  = torch.cat(Zhats, 0).float()
        t     = torch.tensor(dataloader.dataset.df["t"].values).view(-1,1).float()
        Y     = torch.tensor(dataloader.dataset.df["y"].values).view(-1,1).float()

        betas_bias   = model.betas_bias.cpu()
        betas_causal = model.betas_causal.cpu()

        y_hat_bias   = torch.cat([torch.ones_like(t), Xhat, Zhat, t], 1).matmul(betas_bias).view(-1,1)
        y_hat_causal = torch.cat([torch.ones_like(t), Zhat, t], 1).matmul(betas_causal).view(-1,1)

        reg_mse_bias   = ((y_hat_bias - Y)**2).mean()
        reg_mse_causal = ((y_hat_causal - Y)**2).mean()

        metrics_mean["regr_bias_loss_y"] = reg_mse_bias
        metrics_mean["regr_causal_loss_y"] = reg_mse_causal

        if setting.counterfactuals:
            y0_hat_bias   = torch.cat([torch.ones_like(t), Xhat, Zhat, torch.zeros_like(t)], 1).matmul(betas_bias).view(-1,1)
            y1_hat_bias   = torch.cat([torch.ones_like(t), Xhat, Zhat, torch.ones_like(t)], 1).matmul(betas_bias).view(-1,1)
            y0_hat_causal = torch.cat([torch.ones_like(t), Zhat, torch.zeros_like(t)], 1).matmul(betas_causal).view(-1,1)
            y1_hat_causal = torch.cat([torch.ones_like(t), Zhat, torch.ones_like(t)], 1).matmul(betas_causal).view(-1,1)
       
            ite_hats_bias = y1_hat_bias - y0_hat_bias
            ite_hats_causal = y1_hat_causal - y0_hat_causal

            writer.add_scalars("pehe", {"regr_bias": np.sqrt(((ite_hat_bias - ites)**2).mean())}, epoch+1)
            writer.add_scalars("pehe", {"regr_causal": np.sqrt(((ite_hat_causal - ites)**2).mean())}, epoch+1)
            writer.add_scalars("loss_y1", {"regr_bias": ((y1s - y1_hat_bias)**2).mean()}, epoch+1)
            writer.add_scalars("loss_y0", {"regr_bias": ((y0s - y0_hat_bias)**2).mean()}, epoch+1)
            writer.add_scalars("loss_y1", {"regr_causal": ((y1s - y1_hat_causal)**2).mean()}, epoch+1)
            writer.add_scalars("loss_y0", {"regr_causal": ((y0s - y0_hat_causal)**2).mean()}, epoch+1)


    outtensors = {
        'bn_activations': bn_activations,
        'predictions': preds,
        'xhat': np.vstack(Xhats)
    }

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean, outtensors 