import json
import os
import math
import librosa 
import numpy as np
import soundfile as sf

DATASET_PATH = "/home/abdul/Documents/dl20/archive/archive/Data/genres_original"
JSON_PATH = "/home/abdul/Documents/dl20/archive/archive/Data/json/data_101.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 9 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):



    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "raw": []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    #num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)            
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                print("len-signal",len(signal))
                   
                
                for d in range(num_segments):

                    
                    
                    v=(int(len(signal)/sample_rate))
                    
                    if(v>=9):
                      start = samples_per_segment * d
                      finish = start + samples_per_segment
                      signal1 = signal[start:finish]
                      
                      signal1 = np.reshape(signal1,(3000,-1))
                      data["raw"].append(signal1.tolist())
                      data["labels"].append(i-1)
                      print("{}, segment:{}".format(file_path, d+1))
                      print("signal",signal1.shape)
                    else:
                        print("loop fail")
                        break
                     
                    
                        
                        

    # save MFCCs to json file
    print("labels",data["labels"])
    print("mappings",data["mapping"])
    #print("raw",data["raw"])
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=3)
