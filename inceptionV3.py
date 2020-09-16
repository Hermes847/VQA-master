import tensorflow as tf
import pickle
import os.path 
from tqdm import tqdm
import os

#class FeatureExtractor:
#    def __init__(self,cache_path,num_files=None,img_paths=None):
#        self.cache_path = cache_path
#        self.cache = {}
#        if not num_files or not img_paths:
#            assert os.path.exists(cache_path),'cache not exists'
#            with open(os.path.join(cache_path,'info'),'rb') as f:
#                self.info = pickle.load(f)
#            self.chunk_size = self.info['size'] // self.info['chunks']
#        else:
#            self.info = {}
#            self.info['size'] = len(img_paths)
#            self.info['chunks'] = num_files
#            self.info['remapping'] = {}
#            self.chunk_size = self.info['size'] // self.info['chunks']
#            self.image_model =
#            tf.keras.applications.InceptionV3(weights='imagenet')
#            chunk_index = 0
#            for image_path in img_paths:
#                img = tf.io.read_file(image_path)
#                img = tf.image.decode_jpeg(img, channels=3)
#                img = tf.image.resize(img, (299, 299))
#                feature = self.image_model(img).numpy()
#                index = self.get_index_form_path(img_paths)
#                self.info['remapping'][index] = len(self.remapping)
#                if len(self.cache) > self.chunk_size:
#                    with
#                    open(os.path.join(cache_path,'chunk_{0}'.format(chunk_index)),'wb')
#                    as f:
#                        pickle.dump(cache,f)
#                    self.cache = {}
#                    chunk_index+=1
#                else:
#                    self.cache[index] = feature
#            with open(os.path.join(cache_path,'info'),'wb') as f:
#                pickle.dump(cache,info)



#    def get_index_form_path(self,img_paths):
#        pass


#    def __getitem__(self,index):
#        if index not in self.cache:
#            reindex = self.info['remapping'][index]
#            chunk_index = reindex // self.chunk_size
#            with
#            open(os.path.join(cache_path,'chunk_{0}'.format(chunk_index)),'rb')
#            as f:
#                self.cache = pickle.load(f)
#        return self.cache[index]
class FeatureExtractor:
    def __init__(self,cache_path,img_paths=None,bath_size = 128):
        self.cache_path = cache_path      
        if os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                self.cache = pickle.load(f)    
        else:          
            self.cache = {}
            self.image_model = tf.keras.applications.InceptionV3(weights='imagenet')
            for i in tqdm(range(0,len(img_paths),bath_size)):
                batch = []
                indices = []
                for j in range(bath_size):
                    if i+j>=len(img_paths):break
                    image_path = img_paths[i+j]
                    img = tf.io.read_file(image_path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = tf.image.resize(img, (299, 299))
                    img = tf.expand_dims(img,0)                
                    batch.append(img)
                    indices.append(self.get_index_form_path(image_path))
                batch = tf.concat(batch,axis = 0)
                features = self.image_model(tf.keras.applications.inception_v3.preprocess_input(batch)).numpy()
                for k,index in enumerate(indices):
                    self.cache[index] = features[k]

               
            with open(cache_path,'wb') as f:
                pickle.dump(self.cache,f)

    def get_index_form_path(self,img_path):
        return int(img_path.split('\\')[-1].split('_')[-1].split('.')[0])

    def __getitem__(self,index):
        return self.cache[index]   



if __name__ == "__main__":
    paths = []
    folders = [r'D:\documents\coding\Data\coco\train2014',r'D:\documents\coding\Data\coco\val2014']
    for folder in folders:
        paths += [os.path.join(folder,x) for x in list(os.walk(folder))[0][2]]
    FeatureExtractor('features.bin',paths)


