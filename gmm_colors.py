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

        for iters in range(*self.iters_range):
            
            print ("Starting iter: %d ; %d/%d!" % (iters, self.start_from, self.images_len))
                
            dataset_id, start_ptr = 0, 0
            for idx in range(len(self.images_cumsum)-1):
                if self.start_from > self.images_cumsum[idx+1]:
                    dataset_id += 1
                else:
                    start_ptr = self.images_cumsum[idx]
                    break

#            if (len(self.datasets[dataset_id]) - self.start_from + start_ptr) < self.save_every:
#                start_ptr = 0
#                self.start_from = 0
#                dataset_id += 1


            for d, dataset in enumerate(self.datasets):
                
                num_images = len(dataset) - (self.start_from - start_ptr)
                if d < dataset_id:
                    continue
                print ("USING DATASET: %d/%d (%d images)" % (d, len(self.datasets), num_images))
                   
                index = 0
                   
                try:
                    
                    data_subset_indices = list(range(len(dataset) - num_images, len(dataset)))
                    for index, data in enumerate(Subset(dataset, data_subset_indices)):    
                            
                        img_data = data # self.images[idx]
                        img, shp = self.get_image_vector(img_data)
                        
                        try:
                            self.gmm_model.fit(img)
                            #print (self.start_from - start_ptr, index, len(dataset))
                        except Exception:
                            print ("Image error: ", dataset.images[index])
                            pass
                        
                        if (index + self.start_from) % self.save_every == 0 or ( # Serialize wrt subset of dataset
                            self.save_every > len(dataset) - data_subset_indices[0] and index + data_subset_indices[0] == len(dataset) - 1): # Edge case for index < self.save_every
                            
                            print ("Saving model at iter: %d ; Epoch : %d/%d" % (
                                        iters, index + self.start_from, self.images_len), end=' ; ')
                                
                            if not os.path.isdir(self.models_dir):
                                os.mkdir(self.models_dir)
                                
                            self.save_model(iters, index + self.start_from)
                                
                            colors = self.predict_colors(img_data)
                            
                            display_img = np.concatenate((img_data, self.COLORS[colors.astype(np.uint8)]), axis=1)
                            #cv2.imwrite(os.path.join(self.models_dir, "sample_%d_%d.png" % (iters, index + self.start_from)), 
                            #                self.COLORS[colors.astype(np.uint8)])    
                            cv2.imwrite(os.path.join(self.models_dir, "sample_%d_%d.png" % (iters, index + self.start_from)), 
                                            display_img)    
                                
                            # cv2.imshow('f', (colors * 255 / colors.max()).astype(np.uint8))
                            # cv2.imshow('f', (COLORS[colors.astype(np.uint8)]))
                            # cv2.imshow('g', img.reshape(shp))
                            # cv2.waitKey()
                
                except Exception:
                    print (self.start_from - start_ptr, index, len(dataset))
                    traceback.print_exc()
                
                self.start_from += index+1
                start_ptr = self.start_from
                 
            self.start_from = 0

    def get_image_vector(self, img_data):
        
        img = img_data #cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_shape = img.shape
        img = img.reshape((-1, img.shape[-1])) / 255.0
        return img, img_shape
    
    def load_model(self, init_params, means_init, precisions_init,
                         warm_start, verbose):
        
        #models = glob.glob(os.path.join(self.models_dir, "gmm_*.npy"))
        models = glob.glob(os.path.join(self.models_dir, str(self.gmm_components), '*', 
                                         '*', "gmm_w.npy"))
        models = [x for x in models if "b/gmm_" not in x]
         
        get_num_list = lambda s: re.split(r'/', s)[-4:-1] # gmm_components, epoch, iters=[0,1,...]

        #get_num_list = lambda s: re.split(r'[_:]', s)

        sorted_models = sorted(models, reverse=True, key = lambda x: \
                                int(get_num_list(x)[2]) * 10**6 + int(get_num_list(x)[1]))

        if len(sorted_models) > 0:
            spl = get_num_list(sorted_models[0])
            iters, epoch = int(spl[2]), int(spl[1])
            self.gmm_components = int(spl[0])
            print ("Loading model from %s with #gaussians: %d" % (sorted_models[0], self.gmm_components))
            self.model_path = lambda x: os.path.join(self.models_dir, spl[0], spl[1].zfill(5), spl[2], 
                                        "gmm_%s.npy" % (x))
            
        else:
            self.model_path = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(0).zfill(5), str(0),
                                        "gmm_%s.npy" % (x))
            
            print ("Cannot infer without training first!")
        
        self.gmm_model = GaussianMixture(n_components=self.gmm_components, init_params=init_params,
                                                means_init=means_init, precisions_init=precisions_init,
                                                warm_start=warm_start, verbose=verbose)
        
        try:
            w, mu, sigma = \
                np.load(self.model_path("w")), \
                np.load(self.model_path("means")), \
                np.load(self.model_path("covariances"))
            
            self.gmm_model.fit(np.zeros((6, 3)))
            
            self.gmm_model.weights_ = w
            self.gmm_model.means_ = mu
            self.gmm_model.covariances_ = sigma
        except Exception:
            pass

        if len(sorted_models) > 0:
            return epoch+1, iters
        else:
            return 0, 0
    
    def save_model(self, iters, epoch, before=False): #TODO before

        self.model_path = lambda x: os.path.join(self.models_dir, str(self.gmm_components), str(epoch).zfill(5),
                                                 str(iters) + ("" if not before else "b"), "gmm_%s.npy" % (x))
 
        if not os.path.isdir(os.path.dirname(self.model_path(""))):
            os.makedirs(os.path.dirname(self.model_path("")))     
        
        self.model_path = lambda x: os.path.join(self.models_dir, 
                                    "gmm_%d_%s_%d_%s.npy" % (self.gmm_components, x, iters, str(epoch).zfill(5)))
        
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

if __name__ == '__main__':
    
    # images_dir = "../images/"
    # datasets = [FolderImages(images_dir)]
    
    color_model = GMMColors(folder_datasets, **kwargs) 
    color_model.train()
