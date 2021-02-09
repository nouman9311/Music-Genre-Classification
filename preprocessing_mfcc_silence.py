import json
import os
import math
import librosa 
import numpy as np
import soundfile as sf

DATASET_PATH = "/home/abdul/Documents/dl20/archive/archive/Data/genres_original"

JSON_PATH = "/home/abdul/Documents/dl20/archive/archive/Data/json/data_10.json"

SAMPLE_RATE = 22050
TRACK_DURATION = 27 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
       
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

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
                #print(file_path)
                try:
                   signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                except:
                    
                    print("load error")
                    break
               
                num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    #print("start",start)
                    #print("finish",finish)
                    # extract mfcc
                    if(int(len(signal)/sample_rate)>=27):
                       mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                       mfcc = mfcc.T
                       print("mfcc shape",mfcc.shape)
                    else:
                        break

                                        
                    
                    

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
                    
                        
                        

    # save MFCCs to json file
    print("labels",data["labels"])
    print("mappings",data["mapping"])
    #print("mfcc",data["mfcc"])
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
