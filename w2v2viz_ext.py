# Updates: v2, 26.01.2024
# Fixed backness and height
# Imports
import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from datasets import load_dataset  # if you want to load TIMIT dataset
from matplotlib import ticker
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import warnings
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math
import matplotlib.mlab as mlab
import sounddevice as sd
import librosa

if "soundfile" not in sys.modules:
    import soundfile as sf
from w2v2viz_functions import phonetic_class_annotator, timitdata_reader


def w2v2viz_ext(filename,
                sample_rate=16000,
                timit_file=False,
                show_spectrogram_window=False,
                plot_window=1.0,
                pred_sil=True):
    warnings.filterwarnings('ignore')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # if a timit sample is specified in code it will be used for audio processing and target definition.
    timit_sample = {}

    if timit_sample == {}:
        # speech, _ = sf.read(filename)
        speech, _ = librosa.load(filename, sr=sample_rate)
    else:
        speech = timit_sample["audio"]["array"]
        sample_rate = timit_sample["audio"]["sampling_rate"]
        filename = timit_sample["file"]
        print(timit_sample["text"])

    print("file:", filename)
    print("sampling rate:", sample_rate)
    print("audio len:", len(speech) / sample_rate)

    if show_spectrogram_window:
        fig_sp, ax_sp = plt.subplots()
        fig_sp.canvas.manager.set_window_title(f'Spectrogram ({filename.split("TIMIT")[-1]})')
        fig_sp.set_figheight(2.5)
        fig_sp.set_figwidth(14)
        plt.subplots_adjust(left=0.041, right=0.995, top=0.959, bottom=0.1)
        ax_sp.specgram(speech, Fs=sample_rate)

    speech_len = len(speech)

    # Load Wav2Vec Models
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

    input_values = processor(speech, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values

    with torch.no_grad():
        mod_output = model(input_values.to(DEVICE), output_hidden_states=True, output_attentions=True)

    num_layers = len(mod_output.hidden_states)

    # Create dictionary of Wav2Vec representations
    rep_dict = {}
    ix = 0
    timest = 0
    for layer in mod_output.hidden_states:
        timest = 0
        rep_dict[str(ix)] = {}

        for timestep in layer[0]:
            rep_dict[str(ix)][str(timest)] = [np.array(timestep.cpu())]
            timest += 1
        ix += 1

    print("wav2vec2 frames:", timest)
    sample_per_timest = speech_len / timest
    print("audio sample per frame:", sample_per_timest)
    ms_per_frame = sample_per_timest / sample_rate
    print("ms per frame:", ms_per_frame)

    # create the list of the phone target according to the frame number
    targets = ["-"] * timest  # use this if no targets are available

    # This part is not needed if loading non-TIMIT audio =================================
    if timit_file:
        sample = timitdata_reader(filename)  # use if load_dataset is not available.
        if sample is not None:
            targets = phonetic_class_annotator(sample['phonetic_detail'], spf_=sample_per_timest,
                                               debug=False, round_up=False)
            targets = targets['phone']
    # =============================================================================

    # print(targets)

    # Convert representations to probability distributions
    prob_dict = {}
    for layer in rep_dict:
        prob_dict[layer] = {}

        # Load models
        # NB - process is: Detect Category, then pick appropriate detectors!
        # Phonmodel is placeholder - trained on averaged phone representations, not definitive - refer to TIMIT timings
        # for authoritative!
        phonmodel = load("probe_models/mlp-LAYER{}.joblib".format(layer))
        catmodel = load("probe_models/CAT_layer{}.joblib".format(layer))
        poamod = load("probe_models/POA_layer{}.joblib".format(layer))
        moamod = load("probe_models/MOA_layer{}.joblib".format(layer))
        voimod = load("probe_models/Voicing_layer{}.joblib".format(layer))
        heimod = load("probe_models/Height_layer{}.joblib".format(layer))
        backmod = load("probe_models/Back_layer{}.joblib".format(layer))
        roumod = load("probe_models/Rounding_layer{}.joblib".format(layer))

        for frame in rep_dict[layer]:
            prob_dict[layer][frame] = {}
            data = rep_dict[layer][frame]
            category = catmodel.predict(data)
            prob_dict[layer][frame]["category"] = category
            # category probabilities
            prob_dict[layer][frame]["cat_prob"] = np.round(catmodel.predict_proba(data) * 100, 2)[0]
            prob_dict[layer][frame]["phone"] = phonmodel.predict(data)

            if category == "con":
                prob_dict[layer][frame]["poa"] = poamod.predict_proba(data)[0]
                prob_dict[layer][frame]["moa"] = moamod.predict_proba(data)[0]
                prob_dict[layer][frame]["voiced"] = voimod.predict(data)
            elif category == "vow":
                prob_dict[layer][frame]["height"] = heimod.predict_proba(data)[0]
                prob_dict[layer][frame]["back"] = backmod.predict_proba(data)[0]
                prob_dict[layer][frame]["rounding"] = roumod.predict(data)
            elif category == "sil":  # predict consonant features for silences (closures) if pred_sil is True
                if pred_sil:
                    prob_dict[layer][frame]["poa"] = poamod.predict_proba(data)[0]
                    prob_dict[layer][frame]["moa"] = moamod.predict_proba(data)[0]
                    prob_dict[layer][frame]["voiced"] = voimod.predict(data)
                else:
                    pass

    # Fill in gaps (Messy - just standardising the shapes of the various arrays - some of the feature values were not
    # present in our dataset, so not in MLP confidences!)
    filledver = {}
    print("short, moa", moamod.classes_)
    print("long, poa", poamod.classes_)
    print("short, back", backmod.classes_)
    print("long, height", heimod.classes_)
    for layer in prob_dict:
        # Conversion to terrain
        filledver[layer] = {}
        for frame in prob_dict[layer]:
            # To fix height
            poa_classes = list(poamod.classes_)
            moa_classes = list(moamod.classes_)
            category = prob_dict[layer][frame]["category"]
            if category == "con":
                longax = prob_dict[layer][frame]["poa"]
                shortax = prob_dict[layer][frame]["moa"]
            elif category == "vow":
                # move class 10 to the end to prevent it from
                # mixing up with class 2 in vowels
                poa_classes.remove('10')
                poa_classes.append('10')
                moa_classes = backmod.classes_
                longax = prob_dict[layer][frame]["height"]
                shortax = prob_dict[layer][frame]["back"]
            elif category == "sil":
                if pred_sil:
                    longax = prob_dict[layer][frame]["poa"]
                    shortax = prob_dict[layer][frame]["moa"]
                else:
                    longax = np.zeros(11)
                    shortax = np.zeros(8)

            longcls = [int(x) for x in poa_classes]  # poa classes
            shortcls = [int(x) for x in moa_classes]  # moa classes
            longlist = dict(zip(longcls, longax))
            shortlist = dict(zip(shortcls, shortax))
            # sort
            longlist = dict(sorted(longlist.items()))
            shortlist = dict(sorted(shortlist.items()))

            for t in range(11):
                if t not in longlist:
                    longlist[t] = 0
                # if longlist[t] == 0:
                #     longlist[t] = 0.000000000000000000000000000001
            longlist = dict(sorted(longlist.items()))

            for f in range(8):
                if f not in shortlist:
                    shortlist[f] = 0
                # if shortlist[f] == 0 and f != 7:
                #     shortlist[f] = 0.000000000000000000000000000001
            shortlist = dict(sorted(shortlist.items()))

            poaprob = list(longlist.values())
            moaprob = list(shortlist.values())
            distframe = []

            for pv in poaprob:
                row = []
                for mv in moaprob:
                    row.append(pv * mv)
                distframe.append(row)
            filledver[layer][frame] = distframe

    # Convert filled tables into X,Y,Z co-ordinates for matplotlib
    xyzver = {}
    for layer in filledver:
        xyzver[layer] = {}
        for frame in filledver[layer]:
            # print(frame)
            distframe = filledver[layer][frame]
            renderframe = []
            for poa in range(0, 11):
                for moa in range(0, 8):
                    output = [poa, moa, distframe[poa][moa]]
                    renderframe.append(output)
            renderframe = pd.DataFrame(renderframe, columns=["X", "Y", "Z"])
            renderframe = renderframe.mask(renderframe < 0, 0)
            xyzver[layer][frame] = renderframe

    # get the axis label based on the category
    def get_ax_label(cat):
        if cat == 'con':
            return ['PoA', 'MoA']
        elif cat == 'vow':
            return ['Height', 'Backness']
        else:
            if pred_sil:
                return ['PoA', 'MoA']
            else:
                return ['', '']

    cat_dict = {"con": "Consonant", "sil": "Silence", "vow": "Vowel"}
    poa_dict = {0: "Bilabial", 1: "Labiodental", 2: "Dental", 3: "Alveolar", 4: "Postalveolar", 5: "Retroflex",
                6: "Palatal", 7: "Velar", 8: "Uvular", 9: "Pharyngeal", 10: "Glottal"}
    moa_dict = {0: "Plosive", 1: "Nasal", 2: "Trill", 3: "Tap/Flap", 4: "Fricative", 5: "Lateral Fricative",
                6: "Approximant", 7: "Lateral Approximant"}

    poa_abbr = ["Bi", "La", "De", "Al", "Po", "Re", "Pa", "Ve", "Uv", "Ph", "Gl"]
    moa_abbr = ["Pl", "Na", "Tr", "TF", "Fr", "LF", "Ap", "LA"]

    # Get the consonant labels
    def get_con_label(cat, xyzver, layer_, frame_):
        max_z = np.argmax(xyzver[layer_][frame_].values, axis=0)[-1]
        # print(max_z, xyzver[layer_][frame_].shape)
        if cat == 'con':
            poa_pr = xyzver[layer_][frame_]['X'][max_z]
            poa_pr = poa_dict[poa_pr]
            moa_pr = xyzver[layer_][frame_]['Y'][max_z]
            moa_pr = moa_dict[moa_pr]
            voi_round = f'Voiced: {bool(int(prob_dict[layer_][frame_]["voiced"][0]))}'
        elif cat == 'vow':
            poa_pr = xyzver[layer_][frame_]['X'][max_z]
            moa_pr = xyzver[layer_][frame_]['Y'][max_z]
            voi_round = f'Rounding: {bool(int(prob_dict[layer_][frame_]["rounding"][0]))}'
        else:
            if pred_sil:
                poa_pr = xyzver[layer_][frame_]['X'][max_z]
                poa_pr = poa_dict[poa_pr]
                moa_pr = xyzver[layer_][frame_]['Y'][max_z]
                moa_pr = moa_dict[moa_pr]
                voi_round = f'Voiced: {bool(int(prob_dict[layer_][frame_]["voiced"][0]))}'
            else:
                poa_pr = '-'
                moa_pr = '-'
                voi_round = ''

        return poa_pr, moa_pr, voi_round

    # Generate visualisation
    # %matplotlib
    # notebook

    class Updater:

        def __init__(self):
            self.subplots = 0  # number of subplots
            self.spec_strt = 0  # spectrogram start time (sec) for current segment
            self.spec_stp = 0  # spectrogram stop time (sec) for current segment
            self.spec_tpf = 0  # spectrogram time per w2v2 frame
            self.spec_tsc = 0  # spectrogram samples per time(sec) to slice spectrogram for plotting
            self.ln_end = 0  # position of the last slicing line on spectrogram
            self.spec_ = None  # spectrogram matrix

        def update(self, val):
            colourmap = {"vow": "gist_ncar", "con": "gist_earth", "sil": "gist_gray"}

            fnt_size = fig.get_figwidth() / (5 * self.subplots) * 9
            for ax_ in fig.get_axes():
                ax_.tick_params(axis='both', labelsize=fnt_size)

            frame_ind = sl_frame.val - 1
            layer_ind = str(sl_layer.val)

            s = 0
            for s in range(self.subplots):
                category = prob_dict[layer_ind][str(frame_ind + s)]["category"][0]
                cat_prob = prob_dict[layer_ind][str(frame_ind + s)]["cat_prob"]
                phone = prob_dict[layer_ind][str(frame_ind + s)]["phone"][0]

                ax[s].clear()

                ax_labels = get_ax_label(category)
                ax[s].set_xlabel(ax_labels[0], fontsize=12)
                ax[s].set_ylabel(ax_labels[1], fontsize=12)
                ax[s].set_zlim([0.0, 1.0])

                line_start = spec_time_per_frame * (frame_ind + s)

                # if line_start < self.spec_strt and s == 0:
                #     self.update_spec_segment(-1, line_start)

                if line_start < self.spec_strt + (plot_window / 2) - (spec_time_per_frame * subplots_ / 2) and s == 0:
                    self.update_spec_segment(-1, max(0, line_start - (plot_window / 2) + (
                                spec_time_per_frame * subplots_ / 2)))

                speclines[s].set_xdata(line_start)

                if category in ["con", "sil" if pred_sil else ""]:
                    ax[s].xaxis.set_major_locator(ticker.FixedLocator(list(range(11))))
                    ax[s].xaxis.set_major_formatter(ticker.FixedFormatter([v for v in poa_abbr]))
                    ax[s].yaxis.set_major_locator(ticker.FixedLocator(list(range(8))))
                    ax[s].yaxis.set_major_formatter(ticker.FixedFormatter([v for v in moa_abbr]))

                # actually, current frame
                nextframe = xyzver[layer_ind][str(frame_ind + s)]

                Z = nextframe["Z"]
                zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')

                newzi = []
                for i in range(len(zi)):
                    temp = []
                    for p in range(len(zi[i])):
                        number = zi[i][p]
                        if number > 0:
                            temp.append(number)
                        else:
                            temp.append(0)
                    newzi.append(temp)
                newzi = np.array(newzi)
                surf = ax[s].plot_surface(xig, yig, newzi, cmap=colourmap[category])
                # Add important info

                cat_prob = " ".join(['{0}: {1:.2f}'.format(c, p) for c, p in zip(list(cat_dict.keys()), cat_prob)])
                poa_pred, moa_pred, voicing_round = get_con_label(category, xyzver, layer_ind, str(frame_ind + s))
                cattext[s].set_text(f'Frame: {frame_ind + s} | {cat_dict[category]} ({cat_prob}) \n'
                                    f'Phone - Predicted: {phone}, Target: {targets[int(frame_ind + s)]}\n'
                                    f'{("MOA:" if category != "vow" else "Backness:")} {moa_pred} | '
                                    f'{("POA:" if category != "vow" else "Height: ")} {poa_pred} | '
                                    f'{voicing_round}')

            line_end = spec_time_per_frame * (frame_ind + s + 1)
            # line_end = (t[0]) + spec_time_per_frame * (frame_ind + s + 1)
            # print(line_end)

            speclines[s + 1].set_xdata(line_end)

            # if line_end > self.spec_stp:
            #     self.update_spec_segment(1, line_end)

            if line_end > self.spec_stp - (plot_window / 2) + (spec_time_per_frame * subplots_ / 2):
                self.update_spec_segment(1, min(timest * ms_per_frame,
                                                line_end + (plot_window / 2) - (spec_time_per_frame * subplots_ / 2)))

            fig.canvas.draw_idle()

        def update_spec_segment(self, plus_minus, line_):
            if plus_minus > 0:
                self.spec_stp = line_
                self.spec_strt = line_ - plot_window
            else:
                self.spec_strt = line_
                self.spec_stp = line_ + plot_window

            spec_slice = self.spec_[:, int(self.spec_strt * self.spec_tsc):int(self.spec_stp * self.spec_tsc)]
            spec_im.set_data(spec_slice)
            extent = self.spec_strt, self.spec_stp, freqs[0], freqs[-1]
            spec_im.set_extent(extent)

            x_spec = np.linspace(self.spec_strt, self.spec_stp, x_tick_num)
            x_spec_sec = ["{:4.2f}".format(i) for i in np.linspace(self.spec_strt, self.spec_stp, x_tick_num)]
            spec_axes.set_xticks(x_spec, minor=False)
            spec_axes.set_xticklabels(x_spec_sec, fontdict=None, minor=False)

    # only works for 3 subplots for now
    subplots_ = 3
    max_cols = 3

    rows = math.ceil(subplots_ / max_cols)
    fig = plt.figure(figsize=(5 * subplots_, 8))
    fig.canvas.manager.set_window_title(f'W2V2VIZ + Spectrogram ({filename.split("TIMIT")[-1]})')

    ax = [''] * subplots_  # axes for each frame
    cattext = [''] * subplots_  # info text for each frame
    speclines = [''] * (subplots_ + 1)  # spectrogram seek lines
    # %%
    # Generate the spectrogram ========================================================================================
    # NFFT = 256  # default: 256
    # noverlap = int(NFFT * 0.75)  # default: 128
    NFFT = int(0.025 * sample_rate)  # 25 ms
    noverlap = int(0.02 * sample_rate)  # 20 ms
    Fs = sample_rate  # default: 2

    spec_axes = plt.axes([0.04, 0.11, 0.94, 0.30])
    from scipy.signal.windows import gaussian
    spec, freqs, t = mlab.specgram(x=speech, NFFT=NFFT, noverlap=noverlap, Fs=Fs, pad_to=512,
                                   window=gaussian(M=NFFT, std=int(NFFT / 6)))
    spec = 10. * np.log10(spec)
    pad_xextent = (NFFT - noverlap) / Fs / 2
    xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    xmin, xmax = xextent

    if plot_window > t[-1]:
        plot_window = t[-1]

    spec_t_scale = spec.shape[-1] / xmax  # spectrogram samples per time(sec) to slice spectrogram for plotting
    spec_time_per_frame = ms_per_frame

    print("Spectrogram min-max freqs (original):", freqs[0], freqs[-1])
    print("Spectrogram start-end:", t[0], t[-1])
    print("Spectrogram time (sec) per w2v2 frame:", spec_time_per_frame)
    print("spectrogram samples per time (sec):", spec_t_scale)

    frame_per_window = plot_window / spec_time_per_frame
    print("w2v2 frames per plotted section:", frame_per_window)
    start_ = 0  # t[0]
    stop_ = start_ + plot_window

    extent = start_, stop_, freqs[0], freqs[-1]
    spec_start = int(start_ * spec_t_scale)
    spec_stop = int(stop_ * spec_t_scale)
    spec_slice = spec[:, spec_start:spec_stop]

    spec_im = spec_axes.imshow(spec_slice, aspect='auto', origin='lower', extent=extent, interpolation=None)

    # set the ylim to the index of 5khz band
    freq_5k_indx = list(freqs).index(5000)
    spec_axes.set_ylim((0, freq_5k_indx))

    # Draw slicers on spectrogram
    spec_alpha = 1.0
    spec_width = 0.7
    line_start = 0  # t[0]

    x_tick_num = 10
    x_spec = np.linspace(start_, stop_, x_tick_num)
    x_spec_sec = ["{:4.2f}".format(i) for i in np.linspace(start_, stop_, x_tick_num)]
    spec_axes.set_xticks(x_spec, minor=False)
    spec_axes.set_xticklabels(x_spec_sec, fontdict=None, minor=False)

    y_tick_num = 6
    y_spec = np.linspace(freqs[0], freqs[freq_5k_indx], y_tick_num)
    y_spec_freq = [f"{i / 1000} k" for i in np.linspace(freqs[0], freqs[freq_5k_indx], y_tick_num)]
    spec_axes.set_yticks(y_spec, minor=False)
    spec_axes.set_yticklabels(y_spec_freq, fontdict=None, minor=False)
    y_spec_m = np.linspace(freqs[0], freqs[freq_5k_indx], (y_tick_num * 2) - 1)
    spec_axes.set_yticks(y_spec_m, minor=True)

    # %%
    updater = Updater()
    updater.subplots = subplots_
    updater.spec_strt = start_
    updater.spec_stp = stop_
    updater.spec_tpf = spec_time_per_frame
    updater.spec_tsc = spec_t_scale
    updater.ln_end = line_start
    updater.spec_ = spec

    sb = 0

    for sb in range(subplots_):
        frame = str(sb)
        layer = "0"

        curr_frame = xyzver[layer][frame]
        X, Y, Z = xyzver[layer][frame]["X"], xyzver[layer][frame]["Y"], xyzver[layer][frame]["Z"]

        category = prob_dict[layer][frame]["category"][0]
        cat_prob = prob_dict[layer][frame]["cat_prob"]
        phone = prob_dict[layer][frame]["phone"][0]
        poa_pred, moa_pred, voicing_round = get_con_label(category, xyzver, layer, str(sb))

        speclines[sb] = spec_axes.axvline(x=line_start + (sb * spec_time_per_frame), ymin=0, ymax=1, color="red",
                                          linestyle="--", linewidth=spec_width, alpha=spec_alpha)
        # spec_axes.vlines(int(frame_) * 25 * 16, 0, 1, "red", "--")

        ax[sb] = fig.add_subplot(rows, max_cols, sb + 1, projection='3d')
        plt.subplots_adjust(left=0.015, right=0.95, top=1.0, bottom=0.370)

        ax_labels = get_ax_label(category)
        ax[sb].set_xlabel(ax_labels[0], fontsize=12)
        ax[sb].set_ylabel(ax_labels[1], fontsize=12)
        ax[sb].set_zlim([0.0, 1.0])

        xi = np.linspace(X.min(), X.max(), int((len(Z) / 3)))
        yi = np.linspace(Y.min(), Y.max(), int((len(Z) / 3)))
        zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')

        newzi = []
        for i in range(len(zi)):
            temp = []
            for p in range(len(zi[i])):
                number = zi[i][p]
                if number > 0:
                    temp.append(number)
                else:
                    temp.append(0)
            newzi.append(temp)
        newzi = np.array(newzi)

        ##
        xig, yig = np.meshgrid(xi, yi)

        surf = ax[sb].plot_surface(xig, yig, newzi, cmap='gist_earth')

        if category == "con":
            ax[sb].xaxis.set_major_locator(ticker.FixedLocator(list(range(11))))
            ax[sb].xaxis.set_major_formatter(ticker.FixedFormatter([v for v in poa_abbr]))
            ax[sb].yaxis.set_major_locator(ticker.FixedLocator(list(range(8))))
            ax[sb].yaxis.set_major_formatter(ticker.FixedFormatter([v for v in moa_abbr]))

        cat_prob = " ".join(['{0}: {1:.2f}'.format(c, p) for c, p in zip(list(cat_dict.keys()), cat_prob)])
        cattext[sb] = fig.text(0.04 + (0.33 * sb),
                               0.92, f'Frame: {frame} | {cat_dict[category]} ({cat_prob}) \n'
                                     f'Phone - Predicted: {phone}, Target: {targets[0]}\n'
                                     f'{("MOA:" if category != "vow" else "Backness:")} {moa_pred} | '
                                     f'{("POA:" if category != "vow" else "Height: ")} {poa_pred} | '
                                     f'{voicing_round}', fontsize=10, linespacing=1.75)

    update = updater.update

    speclines[sb + 1] = spec_axes.axvline(x=line_start + ((sb + 1) * spec_time_per_frame), ymin=0, ymax=1, color="red",
                                          linestyle="--", linewidth=spec_width, alpha=spec_alpha)

    sldr_y1 = 0.012
    sldr_y2 = 0.045
    sldr_x = 0.065
    sldr_w = 0.83
    allowed_layers = np.array([x for x in range(0, num_layers)])
    ax_layer = plt.axes([sldr_x, sldr_y1, sldr_w, 0.03])
    sl_layer = Slider(ax_layer, 'Layer', 0, num_layers - 1, valinit=0, valstep=allowed_layers)
    sl_layer.on_changed(update)

    allowed_frames = np.array([x for x in range(1, timest - 1)])
    ax_frame = plt.axes([sldr_x, sldr_y2, sldr_w, 0.03])
    sl_frame = Slider(ax_frame, 'Frame', 1, timest - 2, valinit=0, valstep=allowed_frames)
    sl_frame.on_changed(update)

    btn_color = 'white'

    def btn_prev_l(event):
        sl_layer.set_val(max(0, int(sl_layer.val - 1)))

    def btn_next_l(event):
        sl_layer.set_val(min(num_layers - 1, int(sl_layer.val + 1)))

    axnext_l = fig.add_axes([sldr_x + sldr_w + 0.055, sldr_y1, 0.01, 0.03])
    bnext_l = Button(axnext_l, '>', color=btn_color)
    bnext_l.on_clicked(btn_next_l)

    axprev_l = fig.add_axes([sldr_x + sldr_w + 0.04, sldr_y1, 0.01, 0.03])
    bprev_l = Button(axprev_l, '<', color=btn_color)
    bprev_l.on_clicked(btn_prev_l)

    def btn_prev_f(event):
        sl_frame.set_val(max(1, int(sl_frame.val - 1)))

    def btn_next_f(event):
        sl_frame.set_val(min(timest - 2, int(sl_frame.val + 1)))

    axnext_f = fig.add_axes([sldr_x + sldr_w + 0.055, sldr_y2, 0.01, 0.03])
    bnext_f = Button(axnext_f, '>', color=btn_color)
    bnext_f.on_clicked(btn_next_f)

    axprev_f = fig.add_axes([sldr_x + sldr_w + 0.04, sldr_y2, 0.01, 0.03])
    bprev_f = Button(axprev_f, '<', color=btn_color)
    bprev_f.on_clicked(btn_prev_f)

    def on_press(event):
        # print('press', event.key)
        # no match statement for python 3.9, so not using it.
        if event.key == 'right':
            sl_frame.set_val(min(timest - 2, int(sl_frame.val + 1)))
        elif event.key == 'left':
            sl_frame.set_val(max(1, int(sl_frame.val - 1)))
        elif event.key == 'shift+right':
            sl_layer.set_val(min(num_layers - 1, int(sl_layer.val + 1)))
        elif event.key == 'shift+left':
            sl_layer.set_val(max(0, int(sl_layer.val - 1)))
        elif event.key == 'ctrl+right':
            sl_frame.set_val(min(timest - 2, int(sl_frame.val + 3)))
        elif event.key == 'ctrl+left':
            sl_frame.set_val(max(1, int(sl_frame.val - 3)))
        elif event.key == 'ctrl+alt+right':
            sl_frame.set_val(min(timest - 2, int(sl_frame.val + 30)))
        elif event.key == 'ctrl+alt+left':
            sl_frame.set_val(max(1, int(sl_frame.val - 30)))
        elif event.key == 'alt+shift+right':
            sl_layer.set_val(num_layers - 1)
        elif event.key == 'alt+shift+left':
            sl_layer.set_val(0)

    fig.canvas.mpl_connect('key_press_event', on_press)

    def play_sound(event):
        sd.play(speech, sample_rate)

    def play_frame(event):
        sp_ = (int(int(sl_frame.val - 1) * sample_per_timest), int(int(sl_frame.val + 2) * sample_per_timest))
        # print("frames playback:", int(int(sl_frame.val - 1) * sample_per_timest), "-",
        #       int(int(sl_frame.val) * sample_per_timest), ",",
        #       int(int(sl_frame.val) * sample_per_timest), "-", int(int(sl_frame.val + 1) * sample_per_timest), ",",
        #       int(int(sl_frame.val + 1) * sample_per_timest), "-", int(int(sl_frame.val + 2) * sample_per_timest))
        sp_ = speech[sp_[0]:sp_[-1]]
        sd.play(sp_, sample_rate)

    axplay = fig.add_axes([sldr_x + sldr_w + 0.07, sldr_y1, 0.015, 0.03])
    bplay = Button(axplay, '►', color=btn_color)
    bplay.on_clicked(play_sound)

    axplay_frame = fig.add_axes([sldr_x + sldr_w + 0.07, sldr_y2, 0.015, 0.03])
    bplay_frame = Button(axplay_frame, '|►', color=btn_color)
    bplay_frame.on_clicked(play_frame)

    fnt_size = fig.get_figwidth() / (5 * subplots_) * 9
    for ax_ in fig.get_axes():
        ax_.tick_params(axis='both', labelsize=fnt_size)

    # remove right and left from the default shortcuts to prevent the spectrogram bug when using arrow keys.
    plt.rcParams['keymap.forward'].remove('right')
    plt.rcParams['keymap.back'].remove('left')

    plt.show()
