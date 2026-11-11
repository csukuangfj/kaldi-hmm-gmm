MFA alignment — English
=======================

A complete, copy-paste runnable English example.  See :doc:`mfa` for how
the pipeline works and :doc:`setup` for installing the dependencies.

Try it in Colab: |mfa english colab|

.. |mfa english colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/csukuangfj/kaldi-hmm-gmm/blob/master/mfa_english.ipynb

Test audio
----------

.. raw:: html

   <audio controls src="../_static/en.wav">
     Your browser does not support the audio element.
   </audio>

- file: ``en.wav`` (mono, 16-bit PCM, **24 kHz**, 3.85 s)
- transcript: *Ask not what your country can do for you, ask what you can
  do for your country.*

Run it — copy and paste
-----------------------

Everything from zero (Python 3.8+):

.. code-block:: bash

   # 1. environment (once)
   pip install -U kaldi-hmm-gmm kaldi-native-fbank kaldifst numpy soundfile librosa
   git clone https://github.com/csukuangfj/kaldi-hmm-gmm
   cd kaldi-hmm-gmm

   # 2. download the acoustic model, the dictionary and the test audio
   mkdir -p english && cd english
   wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip
   unzip english_mfa.zip
   wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav
   cd ..

   # 3. align
   python3 tools/align_mfa.py --wav english/en.wav \
     --text "Ask not what your country can do for you, ask what you can do for your country." \
     --model-dir english/english_mfa --dict english/english_mfa.dict \
     --phones-txt english/english_mfa/phones.txt --out english/en.TextGrid

What the steps do:

* ``english_mfa`` is the MFA v3.1.0 acoustic model; ``english_mfa.dict`` is
  the matching pronunciation dictionary (both from
  `mfa-models <https://github.com/MontrealCorpusTools/mfa-models>`_);
  the model directory also contains ``final.alimdl``, ``tree``, ``lda.mat``,
  ``meta.json`` and ``phones.txt``.
* the wav is 24 kHz while the model expects 16 kHz; the tool resamples
  automatically (librosa, MFA's ``allow_downsample`` semantics);
* ``--out english/en.TextGrid`` writes a Praat TextGrid with phone and word
  tiers (open it next to ``en.wav`` in `Praat <https://praat.org/>`_).

Expected output
---------------

Measured on an Apple M-series laptop (the timing line shows the audio
duration, the wall-clock time of the pipeline — features + graph +
alignment + TextGrid, **excluding one-time initialization** such as
model/dictionary loading — and the real-time factor, lower is faster):

.. code-block:: text

   OK: english/en.TextGrid  frames=385 dur=3.85s phones=54 words=17
   timing: audio duration 3.85s, elapsed 0.60s, RTF 0.156

The resulting TextGrid
----------------------

.. only:: html

   .. raw:: html

      <details>
      <summary>Show english/en.TextGrid (304 lines)</summary>

   .. code-block:: text

      File type = "ooTextFile"
      Object class = "TextGrid"
      
      xmin = 0
      xmax = 3.8500
      tiers? <exists>
      size = 2
      item []:
          item [1]:
              class = "IntervalTier"
              name = "phones"
              xmin = 0
              xmax = 3.8500
              intervals: size = 54
              intervals [1]:
                  xmin = 0.0000
                  xmax = 0.1100
                  text = "sil"
              intervals [2]:
                  xmin = 0.1100
                  xmax = 0.2600
                  text = "a"
              intervals [3]:
                  xmin = 0.2600
                  xmax = 0.2900
                  text = "k"
              intervals [4]:
                  xmin = 0.2900
                  xmax = 0.4500
                  text = "s"
              intervals [5]:
                  xmin = 0.4500
                  xmax = 0.5100
                  text = "n"
              intervals [6]:
                  xmin = 0.5100
                  xmax = 0.5900
                  text = "a"
              intervals [7]:
                  xmin = 0.5900
                  xmax = 0.6300
                  text = "t"
              intervals [8]:
                  xmin = 0.6300
                  xmax = 0.6600
                  text = "ʋ"
              intervals [9]:
                  xmin = 0.6600
                  xmax = 0.7000
                  text = "ɒ"
              intervals [10]:
                  xmin = 0.7000
                  xmax = 0.7300
                  text = "ʈ"
              intervals [11]:
                  xmin = 0.7300
                  xmax = 0.7900
                  text = "j"
              intervals [12]:
                  xmin = 0.7900
                  xmax = 0.8200
                  text = "ɒ"
              intervals [13]:
                  xmin = 0.8200
                  xmax = 0.8500
                  text = "ɹ"
              intervals [14]:
                  xmin = 0.8500
                  xmax = 0.9600
                  text = "k"
              intervals [15]:
                  xmin = 0.9600
                  xmax = 1.0100
                  text = "ə"
              intervals [16]:
                  xmin = 1.0100
                  xmax = 1.0500
                  text = "n"
              intervals [17]:
                  xmin = 1.0500
                  xmax = 1.1300
                  text = "ʈ"
              intervals [18]:
                  xmin = 1.1300
                  xmax = 1.1700
                  text = "ɹ"
              intervals [19]:
                  xmin = 1.1700
                  xmax = 1.2200
                  text = "i"
              intervals [20]:
                  xmin = 1.2200
                  xmax = 1.3100
                  text = "cʰ"
              intervals [21]:
                  xmin = 1.3100
                  xmax = 1.3500
                  text = "æ"
              intervals [22]:
                  xmin = 1.3500
                  xmax = 1.4100
                  text = "n"
              intervals [23]:
                  xmin = 1.4100
                  xmax = 1.4800
                  text = "d"
              intervals [24]:
                  xmin = 1.4800
                  xmax = 1.5600
                  text = "əw"
              intervals [25]:
                  xmin = 1.5600
                  xmax = 1.6600
                  text = "f"
              intervals [26]:
                  xmin = 1.6600
                  xmax = 1.6900
                  text = "a"
              intervals [27]:
                  xmin = 1.6900
                  xmax = 1.8100
                  text = "j"
              intervals [28]:
                  xmin = 1.8100
                  xmax = 1.8700
                  text = "ə"
              intervals [29]:
                  xmin = 1.8700
                  xmax = 2.0500
                  text = "sil"
              intervals [30]:
                  xmin = 2.0500
                  xmax = 2.2500
                  text = "a"
              intervals [31]:
                  xmin = 2.2500
                  xmax = 2.2800
                  text = "k"
              intervals [32]:
                  xmin = 2.2800
                  xmax = 2.3700
                  text = "s"
              intervals [33]:
                  xmin = 2.3700
                  xmax = 2.4600
                  text = "ʋ"
              intervals [34]:
                  xmin = 2.4600
                  xmax = 2.4900
                  text = "ɒ"
              intervals [35]:
                  xmin = 2.4900
                  xmax = 2.5300
                  text = "ʈ"
              intervals [36]:
                  xmin = 2.5300
                  xmax = 2.6300
                  text = "j"
              intervals [37]:
                  xmin = 2.6300
                  xmax = 2.6400
                  text = "ə"
              intervals [38]:
                  xmin = 2.6400
                  xmax = 2.7200
                  text = "cʰ"
              intervals [39]:
                  xmin = 2.7200
                  xmax = 2.7600
                  text = "æ"
              intervals [40]:
                  xmin = 2.7600
                  xmax = 2.8300
                  text = "n"
              intervals [41]:
                  xmin = 2.8300
                  xmax = 2.8900
                  text = "d"
              intervals [42]:
                  xmin = 2.8900
                  xmax = 2.9700
                  text = "əw"
              intervals [43]:
                  xmin = 2.9700
                  xmax = 3.0700
                  text = "f"
              intervals [44]:
                  xmin = 3.0700
                  xmax = 3.1000
                  text = "a"
              intervals [45]:
                  xmin = 3.1000
                  xmax = 3.1400
                  text = "j"
              intervals [46]:
                  xmin = 3.1400
                  xmax = 3.1700
                  text = "ɒ"
              intervals [47]:
                  xmin = 3.1700
                  xmax = 3.2100
                  text = "ɹ"
              intervals [48]:
                  xmin = 3.2100
                  xmax = 3.3200
                  text = "k"
              intervals [49]:
                  xmin = 3.3200
                  xmax = 3.3800
                  text = "ə"
              intervals [50]:
                  xmin = 3.3800
                  xmax = 3.4500
                  text = "n"
              intervals [51]:
                  xmin = 3.4500
                  xmax = 3.5500
                  text = "ʈ"
              intervals [52]:
                  xmin = 3.5500
                  xmax = 3.6000
                  text = "ɹ"
              intervals [53]:
                  xmin = 3.6000
                  xmax = 3.7900
                  text = "i"
              intervals [54]:
                  xmin = 3.7900
                  xmax = 3.8500
                  text = "sil"
          item [2]:
              class = "IntervalTier"
              name = "words"
              xmin = 0
              xmax = 3.8500
              intervals: size = 17
              intervals [1]:
                  xmin = 0.1100
                  xmax = 0.4500
                  text = "ask"
              intervals [2]:
                  xmin = 0.4500
                  xmax = 0.6300
                  text = "not"
              intervals [3]:
                  xmin = 0.6300
                  xmax = 0.7300
                  text = "what"
              intervals [4]:
                  xmin = 0.7300
                  xmax = 0.8500
                  text = "your"
              intervals [5]:
                  xmin = 0.8500
                  xmax = 1.2200
                  text = "country"
              intervals [6]:
                  xmin = 1.2200
                  xmax = 1.4100
                  text = "can"
              intervals [7]:
                  xmin = 1.4100
                  xmax = 1.5600
                  text = "do"
              intervals [8]:
                  xmin = 1.5600
                  xmax = 1.6900
                  text = "for"
              intervals [9]:
                  xmin = 1.6900
                  xmax = 1.8700
                  text = "you"
              intervals [10]:
                  xmin = 2.0500
                  xmax = 2.3700
                  text = "ask"
              intervals [11]:
                  xmin = 2.3700
                  xmax = 2.5300
                  text = "what"
              intervals [12]:
                  xmin = 2.5300
                  xmax = 2.6400
                  text = "you"
              intervals [13]:
                  xmin = 2.6400
                  xmax = 2.8300
                  text = "can"
              intervals [14]:
                  xmin = 2.8300
                  xmax = 2.9700
                  text = "do"
              intervals [15]:
                  xmin = 2.9700
                  xmax = 3.1000
                  text = "for"
              intervals [16]:
                  xmin = 3.1000
                  xmax = 3.2100
                  text = "your"
              intervals [17]:
                  xmin = 3.2100
                  xmax = 3.7900
                  text = "country"

   .. raw:: html

      </details>

.. only:: html

   Listen to each word
   --------------------
   
   Transcript: *Ask not what your country can do for you, ask what you can
   do for your country.*

   Every word of the transcript, cut from ``en.wav`` by the word intervals of
   the TextGrid.  Reproduce the clips with:

   .. code-block:: bash

      python3 tools/textgrid_to_word_wavs.py --wav english/en.wav \
        --textgrid english/en.TextGrid --out-dir english/en.words

   .. raw:: html

      <table>
        <tr><th>Word</th><th>Time (s)</th><th>Audio</th></tr>
          <tr><td>ask</td><td>0.11 – 0.45</td><td><audio controls src="../_static/words/en/w01.wav"></audio></td></tr>
          <tr><td>not</td><td>0.45 – 0.63</td><td><audio controls src="../_static/words/en/w02.wav"></audio></td></tr>
          <tr><td>what</td><td>0.63 – 0.73</td><td><audio controls src="../_static/words/en/w03.wav"></audio></td></tr>
          <tr><td>your</td><td>0.73 – 0.85</td><td><audio controls src="../_static/words/en/w04.wav"></audio></td></tr>
          <tr><td>country</td><td>0.85 – 1.22</td><td><audio controls src="../_static/words/en/w05.wav"></audio></td></tr>
          <tr><td>can</td><td>1.22 – 1.41</td><td><audio controls src="../_static/words/en/w06.wav"></audio></td></tr>
          <tr><td>do</td><td>1.41 – 1.56</td><td><audio controls src="../_static/words/en/w07.wav"></audio></td></tr>
          <tr><td>for</td><td>1.56 – 1.69</td><td><audio controls src="../_static/words/en/w08.wav"></audio></td></tr>
          <tr><td>you</td><td>1.69 – 1.87</td><td><audio controls src="../_static/words/en/w09.wav"></audio></td></tr>
          <tr><td>ask</td><td>2.05 – 2.37</td><td><audio controls src="../_static/words/en/w10.wav"></audio></td></tr>
          <tr><td>what</td><td>2.37 – 2.53</td><td><audio controls src="../_static/words/en/w11.wav"></audio></td></tr>
          <tr><td>you</td><td>2.53 – 2.64</td><td><audio controls src="../_static/words/en/w12.wav"></audio></td></tr>
          <tr><td>can</td><td>2.64 – 2.83</td><td><audio controls src="../_static/words/en/w13.wav"></audio></td></tr>
          <tr><td>do</td><td>2.83 – 2.97</td><td><audio controls src="../_static/words/en/w14.wav"></audio></td></tr>
          <tr><td>for</td><td>2.97 – 3.10</td><td><audio controls src="../_static/words/en/w15.wav"></audio></td></tr>
          <tr><td>your</td><td>3.10 – 3.21</td><td><audio controls src="../_static/words/en/w16.wav"></audio></td></tr>
          <tr><td>country</td><td>3.21 – 3.79</td><td><audio controls src="../_static/words/en/w17.wav"></audio></td></tr>
      </table>
