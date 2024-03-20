from w2v2viz_ext import w2v2viz_ext

file = r"TIMIT_sample/LDC93S1.wav"

w2v2viz_ext(filename=file,
            sample_rate=16000,
            timit_file=True,  # Default: False. Display the targets using any annotation formatted similar to TIMIT
            show_spectrogram_window=False,  # display a separate window with the spectrogram of the entire utterance
            plot_window=1,  # (sec) plot window for the integrated spectrogram
            pred_sil=False  # predict POA/MOA/Voicing for frames categorised as silence
            )
