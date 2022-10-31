import traceback
import numpy as np

import cv2

import os
import glob

import re

from torch.utils.data import Subset

from sklearn.mixture import GaussianMixture

from folder_images_dataset import FolderImages, folder_datasets

class GMMColors(object):
    
    def __init__(self, datasets, gmm_components=5, init_params="random_from_data", 
                bayesian=False, means_init=None, precisions_init=None, batch_size=1,
                warm_start=False, verbose=False, models_dir = "../models",
                save_every=10, start_from=0, num_iters=5):
        
        self.datasets = datasets
        self.gmm_components = gmm_components
        
        self.models_dir = models_dir

        self.start_from, start_iter = self.load_model(init_params=init_params,
                                                      means_init=means_init, precisions_init=precisions_init,
                                                      warm_start=warm_start, verbose=verbose)
           
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
        
        if bayesian:
            raise NotImplementedError
            # self.gmm_model = BayesianGaussianMixture(n_components=gmm_components, init_params=init_params,
            #                                    means_init=means_init, precisions_init=precisions_init,
            #                                    warm_start=warm_start, verbose=verbose)
        
        COLORS = np.random.randint(0, 255, size=(gmm_components - 1, 3),
                                        dtype="uint8")
        
        self.COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        

        self.iters_range = [start_iter, num_iters]
        self.save_every = save_every
        
        self.images_len = [len(x) for x in self.datasets] # glob.glob(os.path.join(images_dir, '*'))
        self.images_cumsum = [sum(self.images_len[:idx]) for idx in range(len(self.images_len))] # np.cumsum
        self.images_len = sum(self.images_len)

        self.batch_size = batch_size
        
    def train(self):
        
        if not os.path.isdir(os.path.join(self.models_dir, "images", str(self.gmm_components))):
            os.makedirs(os.path.join(self.models_dir, "images", str(self.gmm_components)))
        
        for iters in range(*self.iters_range):
            
            print ("Starting iter: %d ; %d/%d!" % (iters, self.start_from, self.images_len))
                
            cur_index = 0
            
            dataset_id = 0
            
            for idx in range(len(self.images_cumsum)-1):    
                
                if self.start_from > self.images_cumsum[idx+1]:
                    dataset_id += 1
                else:
                    break

            for d, dataset in enumerate(self.datasets):
                
                if d < dataset_id:
                    continue
                print ("USING DATASET: %d/%d" % (d, len(self.datasets)))

                index = 0
                
                try:
                    for index, data in enumerate(Subset(dataset, 
                                                    list(range(self.start_from - self.images_cumsum[idx], len(dataset))))):    
                            
                        img_data = data # self.images[idx]
                        img, shp = self.get_image_vector(img_data)
                        
                        saving_condition = (cur_index + index + self.start_from + 1) % self.save_every == 0
                        if saving_condition:
                            before_display_img = self.predict_colors(img_data)

                        self.gmm_model.fit(img)
                        
                        if saving_condition:
                            
                            print ("Saving model at iter: %d ; Epoch : %d/%d" % (
                                        iters, cur_index + index + self.start_from, self.images_len), end=' ; ')
                                
                            self.save_model(iters, cur_index + index + self.start_from)
                                
                            colors = self.predict_colors(img_data)
                            
                            display_img = np.concatenate((self.COLORS[img_data.astype(np.uint8)], img_data, self.COLORS[colors.astype(np.uint8)]), axis=1)
                            #cv2.imwrite(os.path.join(self.models_dir, "images", "sample_%d_%d.png" % (iters, index + cur_index + self.start_from)), 
                            #                self.COLORS[colors.astype(np.uint8)])    
                            cv2.imwrite(os.path.join(self.models_dir, "images", str(self.gmm_components), "sample_%d_%d.png" % (iters, index + cur_index + self.start_from)), 
                                            display_img)    
                                
                            # cv2.imshow('f', (colors * 255 / colors.max()).astype(np.uint8))
                            # cv2.imshow('f', (COLORS[colors.astype(np.uint8)]))
                            # cv2.imshow('g', img.reshape(shp))
                            # cv2.waitKey()
                
                except Exception:
                    traceback.print_exc()
                    print ("Skipping images %d-%d in dataset %d" % (self.start_from - self.images_cumsum[idx] + index, len(dataset), d))
                    self.start_from = self.images_cumsum[idx] + len(dataset) 
                    
                cur_index += index
            
            self.start_from = 0

    def get_image_vector(self, img_data):
        
        img = img_data #cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_shape = img.shape
        img = img.reshape((-1, img.shape[-1])) / 255.0
        return img, img_shape
    
    def load_model(self, init_params, means_init, precisions_init,
                         warm_start, verbose):
        
        models = glob.glob(os.path.join(self.models_dir, str(self.gmm_components), '*', 
                                        '*', "gmm_w.npy"))
        
        get_num_list = lambda s: re.split(r'/', s)[6:9]
        
        sorted_models = sorted(models, key = lambda x: \
                                int(get_num_list(x)[0]) * 10**6 + int(get_num_list(x)[1]))
           
        if len(sorted_models) > 0:
            spl = get_num_list(sorted_models[-1])
            iters, epoch = int(spl[1]), int(spl[2])
            self.gmm_components = int(spl[0])
            self.model_path = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(iters).zfill(5), 
                                        str(epoch), "gmm_%s.npy" % (x))
            
        else:
            self.model_path = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(0).zfill(5), 
                                        str(0), "gmm_%s.npy" % (x))
            
            print ("Cannot infer without training first!")
        
        self.gmm_model = GaussianMixture(n_components=self.gmm_components, init_params=init_params,
                                                means_init=means_init, precisions_init=precisions_init,
                                                warm_start=warm_start, verbose=verbose)
        
        try:
            w, mu, sigma = \
                np.load(self.model_path("w")), \
                np.load(self.model_path("means")), \
                np.load(self.model_path("covariances"))
            
            self.gmm_model.fit(np.zeros((self.gmm_components + 1, 3)))
            
            self.gmm_model.weights_ = w
            self.gmm_model.means_ = mu
            self.gmm_model.covariances_ = sigma
        
        except Exception:
            traceback.print_exc()
            pass
 
        if len(sorted_models) > 0:
            return iters, epoch  # iters, epoch+1
        else:
            return 0, 0
    
    def save_model(self, iters, epoch):
        
        self.model_path = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(epoch).zfill(5),
                                                str(iters), "gmm_%s.npy" % (x))

        if not os.path.isdir(os.path.dirname(self.model_path(""))):
            os.makedirs(os.path.dirname(self.model_path("")))        

        np.save(self.model_path("w"), self.gmm_model.weights_)
        np.save(self.model_path("means"), self.gmm_model.means_)
        np.save(self.model_path("covariances"), self.gmm_model.covariances_)

        print ("Saved model to : %s" % (self.model_path("files")), end='\r')

    def predict_colors(self, img_data):

        img_vec, shp = self.get_image_vector(img_data)
        
        preds = self.gmm_model.predict(img_vec)
        
        output = preds.reshape(shp[:-1]) 
        #np.repeat(preds.reshape(shp[:-1] + (1,)), 3, axis=-1)
        
        return output

import configparser
cfg_path = "./resources/gaussians.cfg"
parser = configparser.RawConfigParser()

with open(cfg_path, 'r') as f:
    st = f.read()
parser.read_string(st)

kwargs = dict(parser["data"])
kwargs.update(dict(parser["gmm"]))
kwargs.update(dict(parser["training"]))

kwargs.update({x: int(kwargs[x]) for x in kwargs.keys() if kwargs[x].isdigit()})
kwargs.update({x: bool(kwargs[x].replace("False", "")) for x in kwargs.keys() if kwargs[x] in ["True", "False"]})

#kwargs.update({"save_every": 1})

if __name__ == '__main__':
    
    # images_dir = "../images/"
    # datasets = [FolderImages(images_dir)]
    
    color_model = GMMColors(folder_datasets, **kwargs) 
    color_model.train()
