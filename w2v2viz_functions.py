import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch


# Convert annotated timestamps to the model timestamps
def to_transformer_frames(timestamps, spf, round_up=False):
    # spf = samples per frame = len(audio array) / len(latent representations)
    if round_up:
        return np.ceil(np.array(timestamps) / spf).astype(int)
    else:
        return (np.array(timestamps) / spf).round().astype(int)


def phonetic_class_annotator(phone_details, spf_, debug=False,
                             round_up=False, p61=False):
    if p61:
        map_table = pd.read_csv("Tresources/phonemapsat_60_49_40.csv")
        targets = pd.DataFrame(
            columns=['phone61', 'phone49', 'phone40', 'cat', 'poa61', 'moa61', 'voicing61', 'poa49', 'moa49',
                     'voicing49', 'poa40', 'moa40', 'voicing40', 'back61', 'height61', 'rounding61', 'back49',
                     'height49', 'rounding49', 'back40', 'height40', 'rounding40'])
        phone_c = "phone61"
    else:
        map_table = pd.read_csv("resources/phonemapsat.csv")
        targets = pd.DataFrame(columns=['phone', 'cat', 'poa', 'moa', 'voicing', 'height', 'back', 'rounding'])
        phone_c = "phone"

    # print(map_table.loc[map_table["phone"] == "b"].values[0])
    start_ = to_transformer_frames(phone_details["start"], spf_, round_up)
    stop_ = to_transformer_frames(phone_details["stop"], spf_, round_up)
    frames_ = []

    overlap = False
    for indx, p in enumerate(phone_details["utterance"]):
        # if the previous frame was overlapping skip the first frame of the next timestamp
        if overlap:
            start_i = start_[indx] + 1
        else:
            start_i = start_[indx]

        if start_i == stop_[indx]:
            targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]
            overlap = True
        elif start_i > stop_[indx]:
            if debug:
                print(f"Overlapping: {phone_details['utterance'][indx-1]}-{p}")
            overlap = True
        else:
            for t in range(start_i, stop_[indx]):
                targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]  # [1:]
            overlap = False

    if start_[-1] != stop_[-1]:
        # add the last frame
        targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == phone_details["utterance"][-1]].values[0]

    return targets


# Frame finder:
def phonfinder(phondict, pos):
    tempdict = phondict.copy()
    del tempdict["total"]
    last = tempdict[list(tempdict.keys())[0]]
    for i in tempdict.keys():
        # print("Checking pos {} at {}".format(pos,i))
        if pos > int(i):
            pass
        elif pos <= int(i):
            return tempdict[i]


def phondict_gen(timitdata):
    utdic = {}
    testout = timitdata["phonetic_detail"]
    total = testout["stop"][-1]
    utdic["total"] = total
    for i in range(len(testout["utterance"])):
        phoneme = testout["utterance"][i]
        cutoff = testout["stop"][i]
        utdic[str(cutoff)] = phoneme
    return utdic


# %%
# find start and stop timestamps of a given word
def find_word_st(timitdata, word, index_=None):
    if index_ is None:  # in case the word index is not given it will find the index of the first occurrence of the word
        index_ = timitdata["word_detail"]["utterance"].index(word.lower())
    return timitdata["word_detail"]["start"][index_], timitdata["word_detail"]["stop"][index_]


def wav2vec_phondict(timitdata, w2v_rep):
    # w2v_rep is the wav2vec hidden representation of a certain layer
    # Find the phone at a given index in the wav2vec outputs
    frames = {}
    tenlen = len(w2v_rep)
    # print("Frame len:", tenlen)
    phon_dic = phondict_gen(timitdata)
    # print(phon_dic)
    audlen = phon_dic["total"]
    factor = audlen / tenlen
    # print("factor:", factor)
    for i in range(tenlen):
        phone = phonfinder(phon_dic, i * factor)
        # print(phone, i)
        frames[i] = phone
    # print(frames)
    return frames


def timitdata_reader(wavfilepath):
    # print(wavfilepath[:-4])
    speech_, _ = sf.read(wavfilepath)

    if os.path.exists(wavfilepath[:-4] + '.TXT'):
        txt = pd.read_fwf(wavfilepath[:-4] + '.TXT', header=None)
        text = ' '.join(list(txt.loc[0][2:]))
    else:
        text = ""

    if os.path.exists(wavfilepath[:-4] + '.PHN'):
        phn_detail = pd.read_csv(wavfilepath[:-4] + '.PHN', header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
    else:
        print("Phonetic details not found.")
        return

    if os.path.exists(wavfilepath[:-4] + '.WRD'):
        wrd_detail = pd.read_csv(wavfilepath[:-4] + '.WRD', header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
    else:
        print("Word details not found.")
        return

    sep_c = os.sep

    # a naive way to see if the TIMIT file is included in the original folder structure
    if wavfilepath.split(sep_c) in ["train", "test"]:
        sampl_ = {'file': wavfilepath,
                  'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
                  'text': text,
                  'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                      'utterance': list(phn_detail["utterance"])},
                  'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                                  'utterance': list(wrd_detail["utterance"])},
                  'dialect_region': wavfilepath.split(sep_c)[-3],
                  'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
                  'speaker_id': wavfilepath.split(sep_c)[-2],
                  'id': wavfilepath.split(sep_c)[-1].split(".")[0]}
    else:
        sampl_ = {'file': wavfilepath,
                  'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
                  'text': text,
                  'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                      'utterance': list(phn_detail["utterance"])},
                  'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                                  'utterance': list(wrd_detail["utterance"])},
                  'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
                  'id': wavfilepath.split(sep_c)[-1].split(".")[0]}
    return sampl_


# convert a given integer or string to boolean equivalent
def convert_to_bool(val):
    if type(val) is bool:
        return val
    elif type(val) is str:
        if val.lower() in ["true", "t", "1"]:
            return True
        else:
            return False
    elif type(val) in [int, float]:
        if val > 0:
            return True
        else:
            return False
    else:
        return False
