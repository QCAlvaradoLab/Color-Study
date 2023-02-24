from multipledispatch import dispatch

import numpy as np

import cv2

import os
import glob

import re

import traceback

from torch.utils.data import Subset

from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from folder_images_dataset import FolderImages, folder_datasets

class GMMColors(object):
    
    def __init__(self, datasets, gmm_components=5, init_params="random_from_data", 
                bayesian=False, means_init=None, precisions_init=None, batch_size=1,
                warm_start=False, verbose=False, models_dir = "../models",
                save_every=10, start_from=0, num_iters=5, foreground_gaussians=3):
        
        self.datasets = datasets
        self.gmm_components = gmm_components
        
        self.models_dir = models_dir
        
        self.init_params = init_params
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.warm_start = warm_start
        self.verbose = verbose
        
        self.foreground_gaussians = foreground_gaussians
        self.foreground_constraint = lambda foldername: "_valid" in foldername and "_empty" not in foldername
        
        self.gmm_changed = False
        self.start_from, start_iter, self.gaussian_folder = self.load_model(init_params=self.init_params,
                                                                  means_init=self.means_init, precisions_init=self.precisions_init,
                                                                  warm_start=self.warm_start, verbose=self.verbose)
        
        if bayesian:
            raise NotImplementedError
#           self.gmm_model = BayesianGaussianMixture(n_components=gmm_components, init_params=init_params,
#                                                    means_init=means_init, precisions_init=precisions_init,
#                                                    warm_start=warm_start, verbose=verbose)
        
        COLORS = np.random.randint(0, 255, size=(gmm_components + self.foreground_gaussians - 1, 3),
                                        dtype="uint8")
        
        self.COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        

        self.iters_range = [start_iter, num_iters]
        self.save_every = save_every
        
        self.images_len = [len(x) for x in self.datasets] # glob.glob(os.path.join(images_dir, '*'))
        self.images_cumsum = [sum(self.images_len[:idx]) for idx in range(len(self.images_len))] # np.cumsum
        self.images_len = sum(self.images_len)

        self.batch_size = batch_size
    
    def infer_result(self, data, dataset_gaussian_folder, iters, index, img_str=""):

        colors = self.predict_colors(data)

        if colors is None:
            return

        display_img = np.concatenate((data, self.COLORS[colors.astype(np.uint8)]), axis=1)
                
        img_path = os.path.join(self.models_dir, "images", str(self.gmm_components), dataset_gaussian_folder, 
                                 "sample_%d_%d_%s.png" % (iters, index + self.start_from, img_str))
        if not os.path.isdir(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))

        cv2.imwrite(img_path, display_img)    

    def train(self, per_folder_gaussians=True):

        for iters in range(*self.iters_range):
            
            print ("Starting iter: %d ; %d/%d!" % (iters, self.start_from, self.images_len))
                
            dataset_id, start_ptr = 0, 0
            for idx in range(len(self.images_cumsum)-1):
                if self.start_from > self.images_cumsum[idx+1]:
                    dataset_id += 1
                else:
                    start_ptr = self.images_cumsum[idx]
                    break

            for d, dataset in enumerate(self.datasets):
                    
                num_images = len(dataset) - (self.start_from - start_ptr)
                if d < dataset_id:
                    continue
                print ("USING DATASET: %d/%d (%d images)" % (d+1, len(self.datasets), num_images))
                
                dataset_gaussian_folder = dataset.folder.replace(dataset.data_dir, "").replace("/", "_")

                if self.foreground_constraint(dataset_gaussian_folder) and not self.gmm_changed:
                    self.gmm_components += self.foreground_gaussians
                    self.gmm_changed = True
                else:
                    self.gmm_components = self.gmm_components - int(self.gmm_changed) * self.foreground_gaussians
                    self.gmm_changed = False

                index = 0
                   
                try:
                    
                    data_subset_indices = list(range(len(dataset) - num_images, len(dataset)))
                    
                    load_prev_iter_model = True
                    for index, data in enumerate(Subset(dataset, data_subset_indices)):    
                                    
                        if iters > 0 and num_images==len(dataset) and load_prev_iter_model:
                            nearest_save_every = lambda x: x - (x % self.save_every)
                            
                            last_iter_model_dir = self.load_model_from_epoch(iters = iters - 1, 
                                                                                epoch = nearest_save_every(len(data_subset_indices) + self.start_from))
                            
                            print ("."*50, "\n", "Using GMM from last epoch folder %s!" % last_iter_model_dir, "\n", "."*50, "\n")
                            per_folder_gaussians = False
                            load_prev_iter_model = False

                        if per_folder_gaussians and self.gaussian_folder != dataset_gaussian_folder and self.gaussian_folder != "":
                            print ("."*50, "\n", "Using Re-Initialized GMM for new folder %s!" % dataset_gaussian_folder, "\n", "."*50, "\n")
                            self.gmm_model = GaussianMixture(n_components=self.gmm_components, init_params=self.init_params,
                                                                means_init=self.means_init, precisions_init=self.precisions_init,
                                                                warm_start=self.warm_start, verbose=self.verbose)
                            
                            self.gaussian_folder = dataset_gaussian_folder
                        
                        img_data = data 
                        img, _ = self.get_image_vector(img_data)
                        
                        if (index + self.start_from) % self.save_every == 0 or ( # Serialize wrt subset of dataset
                            self.save_every > len(dataset) - data_subset_indices[0] and \
                            index + data_subset_indices[0] == len(dataset) - 1): # Edge case for index < self.save_every
                            
                            self.infer_result(data, dataset_gaussian_folder, iters, index, img_str="before")    
                            self.save_model(iters, index + self.start_from, dataset=dataset, before=True)

                        try:
                            self.gmm_model.fit(img)
        
                        except ConvergenceWarning as e:
                            #TODO: Does this need a solution or dataset construction based on selecting folders of images right?
                            print (e)
                            pass

                        except Exception:
                            print ("Image error: ", dataset.images[index])
                            pass
                        
                        if (index + self.start_from) % self.save_every == 0 or ( # Serialize wrt subset of dataset
                            self.save_every > len(dataset) - data_subset_indices[0] and \
                            index + data_subset_indices[0] == len(dataset) - 1): # Edge case for index < self.save_every
                            
                            print ("Saving model at iter: %d ; Epoch : %d/%d" % (
                                        iters, index + self.start_from, self.images_len), end=' ; ')
                                
                            if not os.path.isdir(self.models_dir):
                                os.mkdir(self.models_dir)
                                
                            self.save_model(iters, index + self.start_from, dataset=dataset)
                            self.infer_result(img_data, dataset_gaussian_folder, iters, index, img_str="")
                               
                except Exception:
                    traceback.print_exc()
                
                # Infer using final GMM model after iteration!
                for idx, data in enumerate(Subset(dataset, data_subset_indices)):
                    if idx % self.save_every == 0:
                        self.infer_result(data, dataset_gaussian_folder, iters, idx, img_str="final")
                            
                self.start_from += index+1
                start_ptr = self.start_from
            
            self.gmm_changed = False
            self.start_from = 0

    def get_image_vector(self, img_data):
        
        img = img_data 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_shape = img.shape
        img = img.reshape((-1, img.shape[-1])) / 255.0
        return img, img_shape

    def load_file(self, file_fn):

        try:
            w, mu, sigma = \
                np.load(file_fn("w")), \
                np.load(file_fn("means")), \
                np.load(file_fn("covariances"))
            
            self.gmm_model.fit(np.zeros((self.gmm_components+1, 3)))
            
            self.gmm_model.weights_ = w
            self.gmm_model.means_ = mu
            self.gmm_model.covariances_ = sigma
        
        except Exception:
            pass
    
    def load_model_from_epoch(self, iters, epoch):
        
        model_fn =  lambda x: os.path.join(self.models_dir, str(self.gmm_components), 
                                            str(epoch).zfill(5), str(iters), "gmm_%s.npy"%x)

        self.load_file(model_fn)
        
        return model_fn("w")

    def load_model(self, init_params, means_init, precisions_init,
                         warm_start, verbose):
        
        models = glob.glob(os.path.join(self.models_dir, str(self.gmm_components), '*', 
                                         '*', "gmm_w*.npy"))
        models.extend(glob.glob(os.path.join(self.models_dir, str(self.gmm_components + self.foreground_gaussians), '*', 
                                         '*', "gmm_w*.npy")))

        models = [x for x in models if "b/gmm_" not in x]
         
        get_num_list = lambda s: re.split(r'/', s)[-4:-1] # gmm_components, epoch, iters=[0,1,...]

        sorted_models = sorted(models, reverse=True, key = lambda x: \
                                int(get_num_list(x)[2]) * 10**6 + int(get_num_list(x)[1]))
        
        folder = self.models_dir
        if len(sorted_models) > 0:
            spl = get_num_list(sorted_models[0])
            
            if "_w.npy" not in sorted_models[0]:
                folder = "_".join("/".join(sorted_models[0][:-4].split("/")[-2:]).replace("/", "_").split("_")[-2:])
            else:
                folder = ""

            iters, epoch = int(spl[2]), int(spl[1])
            if int(spl[0]) == self.gmm_components + self.foreground_gaussians:
                self.gmm_changed = True            
            self.gmm_components = int(spl[0])
            print ("Loading model from %s with #gaussians: %d" % (sorted_models[0], self.gmm_components))
            model_fn = lambda x: os.path.join(self.models_dir, spl[0], 
                                                     spl[1].zfill(5), spl[2], "gmm_%s.npy"%x if folder=="" else "gmm_%s_%s.npy"%(x, folder))
            
        else:
            model_fn = lambda x: os.path.join(self.models_dir, str(self.gmm_components), 
                                                     str(0).zfill(5), str(0), "gmm_%s.npy"%x if folder=="" else "gmm_%s_%s.npy"%(x, folder))
            
            print ("Cannot infer without training first!")
        
        self.gmm_model = GaussianMixture(n_components=self.gmm_components, init_params=init_params,
                                                means_init=means_init, precisions_init=precisions_init,
                                                warm_start=warm_start, verbose=verbose)
        
        self.load_file(model_fn)

        if len(sorted_models) > 0:
            return epoch+1, iters, folder
        else:
            return 0, 0, folder
    
    def save_model(self, iters, epoch, dataset=None, before=False): 
        
        data_folder = "" if dataset is None else dataset.folder.replace(dataset.data_dir, "").replace("/", "_")
        model_fn = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(epoch).zfill(5),
                                                 str(iters) + ("" if not before else "b"), "gmm_%s_%s.npy" % (x, data_folder))
        
        self.gaussian_folder = data_folder 

        if not os.path.isdir(os.path.dirname(model_fn(""))):
            os.makedirs(os.path.dirname(model_fn("")))     
        
        try:
            np.save(model_fn("w"), self.gmm_model.weights_)
            np.save(model_fn("means"), self.gmm_model.means_)
            np.save(model_fn("covariances"), self.gmm_model.covariances_)

            print ("Saved model to : %s" % (model_fn("files")), end='\r')
        
        except Exception:
            pass

    def predict_colors(self, img_data):

        img_vec, shp = self.get_image_vector(img_data)
        
        try:
            preds = self.gmm_model.predict(img_vec)
            output = preds.reshape(shp[:-1]) 
            #np.repeat(preds.reshape(shp[:-1] + (1,)), 3, axis=-1)
        except Exception:
            print ("GMM not fit yet!")
            return None

        return output

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("config_path", help="Path to cfg file with dataset information", nargs="?", default="./resources/gaussians.cfg")
args = ap.parse_args()

cfg_path = args.config_path

import configparser
parser = configparser.RawConfigParser()

with open(cfg_path, 'r') as f:
    st = f.read()
parser.read_string(st)

kwargs = dict(parser["data"])
kwargs.update(dict(parser["gmm"]))
kwargs.update(dict(parser["training"]))

kwargs.update({x: int(kwargs[x]) for x in kwargs.keys() if kwargs[x].isdigit()})
kwargs.update({x: bool(kwargs[x].replace("False", "")) for x in kwargs.keys() if kwargs[x] in ["True", "False"]})

if __name__ == '__main__':
    
    #print ("Using folders:")
    #print ([str(x) for x in folder_datasets])

    color_model = GMMColors(folder_datasets, **kwargs) 
    color_model.train()
