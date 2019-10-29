# generate bootstrapped samples for simulated datatask

import os
from pathlib import Path
import argparse
import logging
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
import pickle
import shutil
import utils
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--setting-dir', default='settings', help="Directory with different settings")
parser.add_argument('--setting', default='collider-prognosticfactor', help="Directory contain setting.json, experimental setting, data-generation, regression model etc")
parser.add_argument('--N', default = '3000', help = "number of units in simulation")
parser.add_argument('--Nvalid', default = '1000', help = "number of units in simulation for validation")
parser.add_argument('--splits', default = 'train.valid', help = "which splits to do, should be separated by .")
parser.add_argument('--counterfactuals', dest='counterfactuals', action='store_true', help="Also generate outcomes for counterfactuals")
parser.add_argument('--no-counterfactuals', dest='counterfactuals', action='store_false', help="Don't generate outcomes for counterfactuals")
parser.add_argument('--sample-imgs', dest='sample_imgs', action = "store_true", help="sample images along with covariate data")
parser.add_argument('--no-imgs', dest='sample_imgs', action="store_false", help="don't get images, matching with units")
parser.add_argument('--seed', default='1234567', help="seed for simluations")
parser.add_argument('--close-range', default=5, type=int, help="when sampling on continuous variables, pick an image from the closest x observations")
parser.add_argument('--replace', action='store_true', help="sample with replacement from images")
parser.add_argument('--debug', action='store_true')
parser.set_defaults(sample_imgs=True, debug=False, counterfactuals=True, replace=False)


class LinearRegressionModel(nn.Module):
    def __init__(self, p, weights = None, bias = None):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        if weights is not None:
            self.linear.weight = Parameter(torch.Tensor([weights]))
        if bias is not None:
            self.linear.bias = Parameter(torch.Tensor([bias]))

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, p, weights = None, bias = None):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        if weights is not None:
            self.linear.weight = Parameter(torch.Tensor([weights]))
        if bias is not None:
            self.linear.bias = Parameter(torch.Tensor([bias]))

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class ProductModel(nn.Module):
    def __init__(self, p, weights = None, bias = None):
        super(ProductModel, self).__init__()
        # assert len(list(set([0,1]) - set(list(np.unique(weights, return_counts = False))))) == 0, "only weigths 0 and 1 are implemented for ProductModel"

        if weights is not None:
            self.weights = torch.Tensor([weights]).view(1,-1)
        else:
            self.weights = torch.Tensor([1])
    
    def forward(self, x):
        # apply weights (1, m)-tensor, broadcast to (n, m) and multiply elementwise
        x = (x*self.weights).clone() # add copy to prevent changing in place

        # select only those with nonzero weights
        x = x[:,self.weights.squeeze().nonzero().squeeze()]

        # multiply everything in column dimension
        return x.prod(1)


distributiondict = {"Bernoulli": dist.Bernoulli,
                    "Normal": dist.Normal}
model_modules = {
    "Linear": LinearRegressionModel,
    "Logistic": LogisticRegressionModel,
    "Product": ProductModel
    }

# create helper for strechting a list to a given size, repeating elements when necessary
def repeat_list(x, N):
    x_len = len(x)
    n_repeats = int(np.ceil(N / x_len))
    x = x * n_repeats
    x = x[:N]
    return x


def repeat_array(x, N):
    """
    Extend the number of rows in an array by repeating elements, to a specified size
    x: np.ndarray
    N: int, out length
    """
    assert isinstance(x, np.ndarray)

    n_repeats = int(np.ceil(N / x.shape[0]))

    # make sure only the first axis gets repeated
    tile_reps = np.ones((x.ndim,), dtype=np.int32)
    tile_reps[0] = n_repeats

    x = np.tile(x, tile_reps)
    
    return np.take(x, range(N), axis=0)


def grab_closest(x, d, close_range=int(0), replace=False):
    """
    Given a numeric value x, grab an item from dict d that is closest to x.
    For multidimensional x, assumes standard euclidian distance metric
    x: vector
    d: dict with {names: ["name1", "name2", ...], values: np.array([v1, v2, ...])}; values.shape[1] should be x.shape[0]
    close_range: pick an item from the closest n values to x
    replace: don't remove the picked item from d and return updated d

    returns: (name_of_closest_elem, distance_to_x (vector when x is a vector), dict (possibly updated))
    """
    names = d["name"]
    values = d["value"]
    assert type(values) is np.ndarray
    if not isinstance(x, np.ndarray):
        assert x.size==1
    else:
        assert x.shape[0] == values.shape[1]

    dist = (values - x)
    if dist.ndim == 1:
        dist = dist.reshape(-1,1) # reshape to make this work for 1d x, so that dist.shape == (n,1) always
    diff = np.linalg.norm(dist, ord=2, axis=1)

    if close_range > 0:
        closest_idx = np.random.choice(np.argsort(np.abs(diff))[:close_range])
    else:
        closest_idx = np.argmin(np.abs(diff))
    if not replace:
        # print(closest_idx)
        keep_idx = np.array(list(set(np.arange(values.shape[0]).astype(np.int64)) - set([closest_idx])))
        # print(keep_idx[:5])
        # print(keep_idx.shape)
        # print(values.shape)
        assert keep_idx.shape[0] == values.shape[0] - 1
        d = {"name": names[keep_idx],
             "value": np.take(values, keep_idx, axis=0)}
    return names[closest_idx], np.take(dist, closest_idx, axis=0), d


#%% import model specification
def prepare_model(model):
    """
    Prepare a model as defined in a pandas dataframe for sampling
    model: a pandas.DataFrame, see examples
    """
    # TODO add checks on model csv file
    # assert variable has variable_model iff variable_type == dependent
    # assert ordering of structural assignments
    assert isinstance(model, pd.DataFrame)

    model.set_index("variable", drop = False, inplace = True)
    param_cols = [x for x in model.columns if "param" in x]
    model["param_tuple"] = model[param_cols].apply(lambda x: (*x.dropna(),), axis = 1)
    var2label = dict(zip(model.variable.values, model.label.values))
    label2var = dict(zip(model.label.values, model.variable.values))
    return  model, var2label, label2var

def prepare_image_sets(model, img_path = "data", split = "train", N = 1000):
    """
    Prepare a dict of img names which are matched on variables present in 
    the generative model. Presently only works for binary variables
    Generate vectors of length N-samples, of which items can be picked one 
    by one to reduce reduncancy
    """

    gen_labels = model.label.tolist()
    # create list of variable roots, since there are labels with different names, e.g.:
    # malignancy_binary, malignancy_isborderline, malignancy_mean etc
    gen_variable_roots = [x.split("_")[0] for x in gen_labels]

    img_df = pd.read_csv(os.path.join(img_path, "labels.csv"))
    img_df = img_df[img_df.split == split]
    if split in img_df.name.values[0]: # some imgs can contain the split in the name: train/img_01.png
        # img_df["name"] = img_df.name.apply(lambda x: x.split("/")[1])
        img_df["name"] = img_df.name.apply(lambda x: os.path.basename(x))
    
    # print(img_df["name"].values[:10])

    # some generative variables should correspond to image features as recorded in data/labels.csv
    # keep only the columns that appear in the generative model, and name
    img_df = img_df[[x for x in img_df.columns if x.split("_")[0] in gen_variable_roots] + ["name"]]

    # define image vars that are in gen model and image labels
    img_vars = [x for x in img_df.columns if x in gen_labels]
    img_gen_model = model[model.label.isin(img_vars)]

    # distinguish continous generative variables
    img_cont_vars = img_gen_model[img_gen_model.distribution=="Normal"].label.tolist()
    img_disc_vars = [x for x in img_vars if x not in img_cont_vars]

    # assert len(img_cont_vars) < 3, "Currently only implemented for max 2 continuous variables"

    print("img_vars: {}".format(img_vars))
    print("img_cont_vars: {}".format(img_cont_vars))
    print("img_disc_vars: {}".format(img_disc_vars))

    # for the discrete image variables, ensure that for every group, there
    # are enough rows to accomodate the required simulation size

    img_disc_dict = {}
    if len(img_disc_vars) > 0:
        # remove possible 'borderline' images for removing noise in labels
        img_df = img_df[img_df[[x.split("_")[0] + "_isborderline" for x in img_disc_vars]].max(axis=1)==0]

        df_grp = img_df.groupby(img_disc_vars, sort=False)

        img_disc_dict = {}
        img_cont_dict = {}

        for name, group in df_grp:
            print("{} original items for key {}".format(group.shape[0], name))
            # names.append(name)
            img_disc_dict[name]  = repeat_list(group["name"].tolist(), 2*N)
            img_cont_dict[name] = {
                "name":  np.array(repeat_list(group["name"].tolist(), 2*N)),
                "value": np.array(repeat_array(group[img_cont_vars].values, 2*N))
            }
        
        # TODO remake pretty df with proper variable names
    else:
        img_cont_dict = {
            "name": np.array(repeat_list(img_df["name"].tolist(), 2*N)),
            "value":np.array(repeat_array(img_df[img_cont_vars].values, 2*N))
        }

    # print(img_cont_dict)
    
    return img_df, img_cont_vars, img_disc_vars, img_disc_dict, img_cont_dict    

def build_dataset(model, args, setting, N = 100):
    model_vars = model.variable.tolist()

    dep_vars = model[model.type == "dependent"].variable.tolist()

    # create dicts for going from variable name to column index and back
    if args.counterfactuals:
        model_vars = model_vars + ["y0", "y1"]
        dep_vars = dep_vars + ["y0", "y1"]

    n_vars = len(model_vars)
    var2idx = dict(zip(model_vars, range(n_vars)))
    idx2var = dict(zip(range(n_vars), model_vars))

    # TODO inject noise, base on how well the cnn-model can predict a feature 
    # which we are sampling on, to model the expected loss

    # initialize tensor
    X = torch.zeros([N, n_vars], requires_grad = False)

    for var, row in model.iterrows():
        column_idx = var2idx[var]

        # for noise variables, sample from distribution
        if row["type"] == "noise":
            distribution = distributiondict[row["distribution"]]
            params = row["param_tuple"]
            fn = distribution(*params)
            X[:, column_idx] = fn.sample(torch.Size([N])).requires_grad_(False)

        # for dependent variables, sample according to distribution parameterized via noise variables
        else:
            betas = model["b_"+var].values
            if args.counterfactuals:
                betas = np.append(betas, [0.,0.])
            bias = row["param_1"]
            model_type = row["variable_model"]
            variable_model = model_modules[model_type](len(betas), betas, bias)
            distribution = row["distribution"]
            MU = variable_model.forward(X.detach()).squeeze()
            if distribution == "Normal":
                X[:, column_idx] = MU
            #  NB possibility to use Bernoulli(logits = ...) here
            elif distribution == "Bernoulli":
                fn = distributiondict[distribution](MU)
                X[:, column_idx] = fn.sample().squeeze().requires_grad_(False)

        # df = pd.DataFrame(X.detach().numpy(), columns = model_vars)
        # print(df)


    # TODO update counterfactuals to include interactions

    if args.counterfactuals:
        # fill column with 0s and 1s
        X_0 = X.scatter(1, var2idx["t"]*torch.ones((N, 1)).long(), 0.)
        X_1 = X.scatter(1, var2idx["t"]*torch.ones((N, 1)).long(), 1.)

        if "interaction" in model.label.tolist():
            X_0[:,var2idx["zt"]] = X_0[:,var2idx["t"]] # all zeros
            X_1[:,var2idx["zt"]] = X_1[:,var2idx["z"]] # all equal to z 
            
        # get outcome model
        betas          = np.append(model["b_y"].values, [0.,0.])
        bias           = model.loc["y", "param_1"]
        model_type     = model.loc["y", "variable_model"]
        variable_model = model_modules[model_type](len(betas), betas, bias)
        distribution   = model.loc["y", "distribution"]

        MU_0 = variable_model.forward(X_0).squeeze()
        MU_1 = variable_model.forward(X_1).squeeze()

        if distribution == "Normal":
            X[:, var2idx["y0"]] = MU_0
            X[:, var2idx["y1"]] = MU_1
        #  NB possibility to use Bernoulli(logits = ...) here
        elif distribution == "Bernoulli":
            fn_0 = distributiondict[distribution](MU_0)
            fn_1 = distributiondict[distribution](MU_1)
            X[:, var2idx["y0"]] = fn_0.sample().squeeze()
            X[:, var2idx["y1"]] = fn_1.sample().squeeze()
    
    for var in dep_vars:
        print("mean (sd) {}: {:.3f} ({:.3f})".format(var, X[:, var2idx[var]].mean(), X[:, var2idx[var]].std()))

    return X, var2idx, idx2var


if __name__ == '__main__':
    # Load the parameters from json file
    args         = parser.parse_args()

    # Load information from last setting if none provided:
    if args.setting == "" and Path('last-defaults.json').exists():
        print("using last default setting")
        last_defaults = utils.Params("last-defaults.json")
        args.setting = last_defaults.dict["setting"]
        for param, value in last_defaults.dict.items():
            print("{}: {}".format(param, value))
    else:
        with open("last-defaults.json", "r+") as jsonFile:
            defaults = json.load(jsonFile)
            tmp = defaults["setting"]
            defaults["setting"] = args.setting
            jsonFile.seek(0)  # rewind
            json.dump(defaults, jsonFile)
            jsonFile.truncate()

    setting_home = os.path.join(args.setting_dir, args.setting)
    setting      = utils.Params(os.path.join(setting_home, "setting.json"))
    data_dir     = os.path.join(setting_home, "data")
    mode3d       = setting.mode3d
    GEN_MODEL    = setting.gen_model
    N_SAMPLES    = {"train": int(args.N), "valid": int(args.Nvalid), "test": int(args.Nvalid)}
    SPLITS       = str(args.splits).split(".")
    SAMPLE_IMGS  = args.sample_imgs
    MANUAL_SEED  = int(args.seed)
    if mode3d: 
        IMG_DIR = "data" # source location of all images
    else:
        IMG_DIR = Path("data","slices")



    # load and prepare generative model dataframe
    # model_df = pd.read_csv(os.path.join(HOME_PATH, "experiments", "sims", GEN_MODEL + ".csv"))
    model_df = pd.read_csv(os.path.join("experiments", "sims", GEN_MODEL + ".csv"))
    model_df, var2label, label2var = prepare_model(model_df)

    shutil.copy(os.path.join("experiments", "sims", GEN_MODEL + ".csv"),
                os.path.join(setting_home, "generating_model.csv"))

    dfs = {}
    dfs_oracle = {}

    # associate an image with each unit
    for i, split in enumerate(SPLITS):
        # remove earlier possible images
        if os.path.isdir(os.path.join(data_dir, split)) and SAMPLE_IMGS:
            shutil.rmtree(os.path.join(data_dir, split))

        # simulate data
        # logging.info("generating data for %s split" % (split))
        print("generating data for %s split" % (split))
        torch.manual_seed(MANUAL_SEED + i)
        X, var2idx, idx2var = build_dataset(model_df, args, setting, N_SAMPLES[split])
        df_oracle = pd.DataFrame(X.detach().numpy(), columns = list(var2idx.keys()))

        # extract Y and treatment
        y = X[:, var2idx["y"]]
        y = y.detach().numpy()
        t = X[:, var2idx["t"]]
        t = t.detach().numpy()
        if args.counterfactuals:
            y0 = X[:, var2idx["y0"]]
            y0 = y0.detach().numpy()
            y1 = X[:, var2idx["y1"]]
            y1 = y1.detach().numpy()

        # export
        if not os.path.isdir(os.path.join(data_dir, split)):
            logging.info("making dirs")
            os.makedirs(os.path.join(data_dir, split))
        torch.save(X, os.path.join(data_dir, split, "X.pt"))
        np.save(os.path.join(data_dir, split, "X.npy"), X.detach().numpy())

        if SAMPLE_IMGS:
            img_df, img_cont_vars, img_disc_vars, img_disc_dict, img_cont_dict = prepare_image_sets(model_df, IMG_DIR, split, N_SAMPLES[split])

            # when no discrete generative image variables provided,
            # no grouping is necessary
            if len(img_disc_vars) == 0:
                # extract columns from simulated data, corresponding to image vars
                img_cont_var_col_ids = [var2idx[label2var[x]] for x in img_cont_vars]
                x_cont = X[:, img_cont_var_col_ids]
                x_cont = x_cont.detach().squeeze().numpy()

                print(f"number of continuous variables: {len(img_cont_vars)}")

                diffs = np.zeros_like(x_cont)
                if diffs.ndim == 1:
                    diffs = diffs.reshape(-1,1)
                    x_cont = x_cont.reshape(-1,1)

                # sample images for each simulated unit
                img_names_out = []
                for i in tqdm(range(x_cont.shape[0])):
                    img_name, diff, img_cont_dict = grab_closest(x_cont[i,:], img_cont_dict, args.close_range, args.replace)
                    diffs[i,:] = diff
                    # print("image name: {}, x_value: {:.3f}, difference: {:.3f}".format(img_name, x[i], diff))
                    img_name_out = os.path.join(str(i) + "_" + img_name)
                    if "imgs/" in img_name:
                        img_name_out = os.path.basename(img_name_out)
                    # print(img_name_out)
                    # print(img_name)
                    img_names_out.append(img_name_out)
                    shutil.copy(os.path.join(IMG_DIR, split, img_name), 
                                os.path.join(data_dir, split, img_name_out))

            else:
                # sample based on discrete variables
                print(f"number of continuous variables: {len(x_img_cont_vars)}")
                n_img_disc_vars = len(img_disc_vars)
                img_disc_var_col_ids = [var2idx[label2var[x]] for x in img_disc_vars]
                x_disc = X[:, img_disc_var_col_ids].reshape(-1, n_img_disc_vars)
                x_disc = x_disc.detach().numpy().astype(int)

                if len(img_cont_vars) > 0:
                    img_cont_var_col_ids = [var2idx[label2var[x]] for x in img_cont_vars]
                    x_cont = X[:, img_cont_var_col_ids]
                    x_cont = x_cont.detach().squeeze().numpy()
                    diffs = np.zeros_like(x_cont)
                    if diffs.ndim == 1:
                        diffs = diffs.reshape(-1,1)
                        x_cont = x_cont.reshape(-1,1)

                img_names_out = []
                for i in tqdm(range(N_SAMPLES[split])):
                    key = tuple(x_disc[i, :])
                    if n_img_disc_vars == 1:
                        key = key[0]

                    if len(img_cont_vars) == 0:
                        img_names = img_disc_dict[key]
                        # pick first in list, then split this one off
                        img_name = img_names[0]
                        img_dict[key] = img_names[1:]
                    else:
                        # grab the continuous variable dict corresponding to discrete setting
                        cont_var_dict = img_cont_dict[key]
                        img_name, diff, cont_var_dict = grab_closest(x_cont[i,:], cont_var_dict, args.close_range, args.replace)
                        diffs[i,:] = diff
                        img_cont_dict[key] = cont_var_dict
                    
                    img_name_out = os.path.join(str(i) + "_" + img_name)
                    img_names_out.append(img_name_out)
                    shutil.copy(os.path.join(IMG_DIR, split, img_name), 
                                os.path.join(data_dir, split, img_name_out))
            df_oracle["name"] = img_names_out
            if len(img_cont_vars) > 0:
                for i, cont_var in enumerate(img_cont_vars):
                    df_oracle["diff_"+label2var[cont_var]] = diffs[:,i]
                    df_oracle[label2var[cont_var]+"_actual"] = df_oracle[label2var[cont_var]].values + diffs[:,i]

            dict_out = {
                'name': img_names_out,
                't': t,
                'y': y}
            if args.counterfactuals:
                dict_out["y0"] = y0
                dict_out["y1"] = y1
            if "x" in var2idx.keys():
                dict_out["x"] = X[:, var2idx["x"]].detach().numpy()
            if "z" in var2idx.keys():
                dict_out["z"] = X[:, var2idx["z"]].detach().numpy()

            print("unique number of images sampled for split {}: {}".format(split, len(set([x.split("_")[-1] for x in img_names_out]))))
            print("sampling difference sd: {:.3f}".format(diffs.std()))
            df_out = pd.DataFrame(dict_out)
            dfs[split] = df_out
            df_out.to_csv(os.path.join(data_dir, split, "labels.csv"), index = False)
            df_oracle.to_csv(os.path.join(data_dir, split, "oracle.csv"), index = False)
        
        # add oracle data frame to dict
        dfs_oracle[split] = df_oracle

    # save data frame with all splits, and vardicts
    with open(os.path.join(data_dir, "vardicts.pt"), 'wb') as f:
        pickle.dump((var2idx, idx2var, var2label, label2var), f)

    oracle = pd.concat(dfs_oracle, axis = 0)
    oracle.reset_index(inplace=True)
    oracle.rename(index = str, columns = {"level_0": "split"}, inplace=True)
    if SAMPLE_IMGS:
        oracle["name"] = oracle[["split", "name"]].apply(lambda x: os.path.join(x[0], x[1]), axis = 1)
    oracle.to_csv(os.path.join(data_dir, "oracle.csv"), index = False)

    if SAMPLE_IMGS:
        df = pd.concat(dfs, axis = 0)
        df = df.reset_index()
        df["split"] = df.level_0
        df["name"] = df[["split", "name"]].apply(lambda x: os.path.join(x[0], x[1]), axis = 1)
        df = df.drop(["level_0", "level_1"], axis=1)
        df.to_csv(os.path.join(data_dir, "labels.csv"), index = False)

    if args.debug:
        x_train = torch.load(os.path.join(data_dir, "train", "X.pt"))
        x_train = x_train.detach().numpy()
        np.savetxt("scratch/X.csv", x_train, delimiter=',')


    logging.info("- done.")



#%%
