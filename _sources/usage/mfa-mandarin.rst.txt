MFA alignment — Mandarin
========================

Complete, copy-paste runnable Mandarin examples (two pinyin models).
See :doc:`mfa` for how the pipeline works and :doc:`setup` for installing
the dependencies (``pypinyin`` is required for ``--tokenize pinyin``).

Test audio
----------

.. raw:: html

   <audio controls src="../_static/zh.wav">
     Your browser does not support the audio element.
   </audio>

- file: ``zh.wav`` (mono, 16-bit PCM, **24 kHz**, 4.76 s)
- transcript: 不要问你的国家能为你做什么，而要问你能为你的国家做什么。

The transcript is Chinese characters; ``--tokenize pinyin`` converts each
character to pinyin for dictionary lookup and keeps the characters for the
TextGrid word tier.

AISHELL-3 pinyin model
----------------------

Run it — copy and paste:

.. code-block:: bash

   # 1. environment (once)
   pip install -U kaldi-hmm-gmm kaldi-native-fbank kaldifst numpy soundfile librosa pypinyin
   git clone https://github.com/csukuangfj/kaldi-hmm-gmm
   cd kaldi-hmm-gmm

   # 2. download the model (the dictionary ships inside the zip) and the audio
   mkdir -p aishell3_pinyin mandarin && cd aishell3_pinyin
   wget https://github.com/lars76/forced-alignment-chinese/releases/download/model_files/aishell3_pinyin.zip
   unzip aishell3_pinyin.zip          # -> aishell3_pinyin_acoustic.zip + dictionaries
   unzip aishell3_pinyin_acoustic.zip # -> aishell3_pinyin_acoustic/
   cd ..
   wget -O mandarin/zh.wav \
     https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav

   # 3. align
   python3 tools/align_mfa.py --wav mandarin/zh.wav \
     --text "不要问你的国家能为你做什么，而要问你能为你的国家做什么。" \
     --model-dir aishell3_pinyin/aishell3_pinyin_acoustic \
     --dict aishell3_pinyin/aishell3_pinyin_dictionary.dict \
     --phones-txt aishell3_pinyin/aishell3_pinyin_acoustic/phones.txt \
     --tokenize pinyin --out mandarin/zh-aishell3.TextGrid

Expected output:

.. code-block:: text

   OK: mandarin/zh-aishell3.TextGrid  frames=476 dur=4.76s phones=71 words=26
   timing: audio duration 4.76s, elapsed 0.67s, RTF 0.140

The resulting TextGrid
----------------------

.. only:: html

   .. raw:: html

      <details>
      <summary>Show mandarin/zh-aishell3.TextGrid (408 lines)</summary>

   .. code-block:: text

      File type = "ooTextFile"
      Object class = "TextGrid"
      
      xmin = 0
      xmax = 4.7600
      tiers? <exists>
      size = 2
      item []:
          item [1]:
              class = "IntervalTier"
              name = "phones"
              xmin = 0
              xmax = 4.7600
              intervals: size = 71
              intervals [1]:
                  xmin = 0.0000
                  xmax = 0.0600
                  text = "sil"
              intervals [2]:
                  xmin = 0.0600
                  xmax = 0.1200
                  text = "p"
              intervals [3]:
                  xmin = 0.1200
                  xmax = 0.1800
                  text = "u"
              intervals [4]:
                  xmin = 0.1800
                  xmax = 0.2700
                  text = "j"
              intervals [5]:
                  xmin = 0.2700
                  xmax = 0.3200
                  text = "au̯"
              intervals [6]:
                  xmin = 0.3200
                  xmax = 0.3900
                  text = "w"
              intervals [7]:
                  xmin = 0.3900
                  xmax = 0.4200
                  text = "ə"
              intervals [8]:
                  xmin = 0.4200
                  xmax = 0.4800
                  text = "n"
              intervals [9]:
                  xmin = 0.4800
                  xmax = 0.5500
                  text = "n"
              intervals [10]:
                  xmin = 0.5500
                  xmax = 0.6400
                  text = "i"
              intervals [11]:
                  xmin = 0.6400
                  xmax = 0.7100
                  text = "t"
              intervals [12]:
                  xmin = 0.7100
                  xmax = 0.7600
                  text = "ɤ"
              intervals [13]:
                  xmin = 0.7600
                  xmax = 0.8500
                  text = "k"
              intervals [14]:
                  xmin = 0.8500
                  xmax = 0.9000
                  text = "w"
              intervals [15]:
                  xmin = 0.9000
                  xmax = 0.9500
                  text = "o"
              intervals [16]:
                  xmin = 0.9500
                  xmax = 0.9900
                  text = "sil"
              intervals [17]:
                  xmin = 0.9900
                  xmax = 1.0500
                  text = "tɕ"
              intervals [18]:
                  xmin = 1.0500
                  xmax = 1.1000
                  text = "j"
              intervals [19]:
                  xmin = 1.1000
                  xmax = 1.1400
                  text = "a"
              intervals [20]:
                  xmin = 1.1400
                  xmax = 1.2300
                  text = "n"
              intervals [21]:
                  xmin = 1.2300
                  xmax = 1.2900
                  text = "ə"
              intervals [22]:
                  xmin = 1.2900
                  xmax = 1.3200
                  text = "ŋ"
              intervals [23]:
                  xmin = 1.3200
                  xmax = 1.3800
                  text = "w"
              intervals [24]:
                  xmin = 1.3800
                  xmax = 1.4400
                  text = "ei̯"
              intervals [25]:
                  xmin = 1.4400
                  xmax = 1.5300
                  text = "n"
              intervals [26]:
                  xmin = 1.5300
                  xmax = 1.6000
                  text = "i"
              intervals [27]:
                  xmin = 1.6000
                  xmax = 1.6300
                  text = "sil"
              intervals [28]:
                  xmin = 1.6300
                  xmax = 1.6900
                  text = "ts"
              intervals [29]:
                  xmin = 1.6900
                  xmax = 1.7200
                  text = "w"
              intervals [30]:
                  xmin = 1.7200
                  xmax = 1.7300
                  text = "o"
              intervals [31]:
                  xmin = 1.7300
                  xmax = 1.7600
                  text = "sil"
              intervals [32]:
                  xmin = 1.7600
                  xmax = 1.8600
                  text = "ʂ"
              intervals [33]:
                  xmin = 1.8600
                  xmax = 1.9100
                  text = "ə"
              intervals [34]:
                  xmin = 1.9100
                  xmax = 1.9400
                  text = "n"
              intervals [35]:
                  xmin = 1.9400
                  xmax = 2.0200
                  text = "m"
              intervals [36]:
                  xmin = 2.0200
                  xmax = 2.1900
                  text = "ɤ"
              intervals [37]:
                  xmin = 2.1900
                  xmax = 2.4800
                  text = "sil"
              intervals [38]:
                  xmin = 2.4800
                  xmax = 2.5500
                  text = "aɚ̯"
              intervals [39]:
                  xmin = 2.5500
                  xmax = 2.6800
                  text = "j"
              intervals [40]:
                  xmin = 2.6800
                  xmax = 2.7400
                  text = "au̯"
              intervals [41]:
                  xmin = 2.7400
                  xmax = 2.8000
                  text = "w"
              intervals [42]:
                  xmin = 2.8000
                  xmax = 2.8400
                  text = "ə"
              intervals [43]:
                  xmin = 2.8400
                  xmax = 2.8900
                  text = "n"
              intervals [44]:
                  xmin = 2.8900
                  xmax = 2.9600
                  text = "n"
              intervals [45]:
                  xmin = 2.9600
                  xmax = 3.0400
                  text = "i"
              intervals [46]:
                  xmin = 3.0400
                  xmax = 3.1000
                  text = "n"
              intervals [47]:
                  xmin = 3.1000
                  xmax = 3.1700
                  text = "ə"
              intervals [48]:
                  xmin = 3.1700
                  xmax = 3.2000
                  text = "ŋ"
              intervals [49]:
                  xmin = 3.2000
                  xmax = 3.2600
                  text = "w"
              intervals [50]:
                  xmin = 3.2600
                  xmax = 3.3300
                  text = "ei̯"
              intervals [51]:
                  xmin = 3.3300
                  xmax = 3.4300
                  text = "n"
              intervals [52]:
                  xmin = 3.4300
                  xmax = 3.5000
                  text = "i"
              intervals [53]:
                  xmin = 3.5000
                  xmax = 3.5700
                  text = "t"
              intervals [54]:
                  xmin = 3.5700
                  xmax = 3.6100
                  text = "ɤ"
              intervals [55]:
                  xmin = 3.6100
                  xmax = 3.6900
                  text = "k"
              intervals [56]:
                  xmin = 3.6900
                  xmax = 3.7200
                  text = "w"
              intervals [57]:
                  xmin = 3.7200
                  xmax = 3.7700
                  text = "o"
              intervals [58]:
                  xmin = 3.7700
                  xmax = 3.8400
                  text = "tɕ"
              intervals [59]:
                  xmin = 3.8400
                  xmax = 3.8800
                  text = "j"
              intervals [60]:
                  xmin = 3.8800
                  xmax = 3.9500
                  text = "a"
              intervals [61]:
                  xmin = 3.9500
                  xmax = 3.9900
                  text = "sil"
              intervals [62]:
                  xmin = 3.9900
                  xmax = 4.0500
                  text = "ts"
              intervals [63]:
                  xmin = 4.0500
                  xmax = 4.0800
                  text = "w"
              intervals [64]:
                  xmin = 4.0800
                  xmax = 4.1100
                  text = "o"
              intervals [65]:
                  xmin = 4.1100
                  xmax = 4.1400
                  text = "sil"
              intervals [66]:
                  xmin = 4.1400
                  xmax = 4.2500
                  text = "ʂ"
              intervals [67]:
                  xmin = 4.2500
                  xmax = 4.3000
                  text = "ə"
              intervals [68]:
                  xmin = 4.3000
                  xmax = 4.3400
                  text = "n"
              intervals [69]:
                  xmin = 4.3400
                  xmax = 4.4400
                  text = "m"
              intervals [70]:
                  xmin = 4.4400
                  xmax = 4.5500
                  text = "ɤ"
              intervals [71]:
                  xmin = 4.5500
                  xmax = 4.7600
                  text = "sil"
          item [2]:
              class = "IntervalTier"
              name = "words"
              xmin = 0
              xmax = 4.7600
              intervals: size = 26
              intervals [1]:
                  xmin = 0.0600
                  xmax = 0.1800
                  text = "不"
              intervals [2]:
                  xmin = 0.1800
                  xmax = 0.3200
                  text = "要"
              intervals [3]:
                  xmin = 0.3200
                  xmax = 0.4800
                  text = "问"
              intervals [4]:
                  xmin = 0.4800
                  xmax = 0.6400
                  text = "你"
              intervals [5]:
                  xmin = 0.6400
                  xmax = 0.7600
                  text = "的"
              intervals [6]:
                  xmin = 0.7600
                  xmax = 0.9500
                  text = "国"
              intervals [7]:
                  xmin = 0.9900
                  xmax = 1.1400
                  text = "家"
              intervals [8]:
                  xmin = 1.1400
                  xmax = 1.3200
                  text = "能"
              intervals [9]:
                  xmin = 1.3200
                  xmax = 1.4400
                  text = "为"
              intervals [10]:
                  xmin = 1.4400
                  xmax = 1.6000
                  text = "你"
              intervals [11]:
                  xmin = 1.6300
                  xmax = 1.7300
                  text = "做"
              intervals [12]:
                  xmin = 1.7600
                  xmax = 1.9400
                  text = "什"
              intervals [13]:
                  xmin = 1.9400
                  xmax = 2.1900
                  text = "么"
              intervals [14]:
                  xmin = 2.4800
                  xmax = 2.5500
                  text = "而"
              intervals [15]:
                  xmin = 2.5500
                  xmax = 2.7400
                  text = "要"
              intervals [16]:
                  xmin = 2.7400
                  xmax = 2.8900
                  text = "问"
              intervals [17]:
                  xmin = 2.8900
                  xmax = 3.0400
                  text = "你"
              intervals [18]:
                  xmin = 3.0400
                  xmax = 3.2000
                  text = "能"
              intervals [19]:
                  xmin = 3.2000
                  xmax = 3.3300
                  text = "为"
              intervals [20]:
                  xmin = 3.3300
                  xmax = 3.5000
                  text = "你"
              intervals [21]:
                  xmin = 3.5000
                  xmax = 3.6100
                  text = "的"
              intervals [22]:
                  xmin = 3.6100
                  xmax = 3.7700
                  text = "国"
              intervals [23]:
                  xmin = 3.7700
                  xmax = 3.9500
                  text = "家"
              intervals [24]:
                  xmin = 3.9900
                  xmax = 4.1100
                  text = "做"
              intervals [25]:
                  xmin = 4.1400
                  xmax = 4.3400
                  text = "什"
              intervals [26]:
                  xmin = 4.3400
                  xmax = 4.5500
                  text = "么"

   .. raw:: html

      </details>


.. only:: html

   .. only:: html

   .. only:: html

   Listen to each word
   --------------------
   
   Transcript: 不要问你的国家能为你做什么，而要问你能为你的国家做什么。

   Every character of the transcript, cut by the word intervals of
   ``zh-aishell3.TextGrid``.  Reproduce the clips with:

   .. code-block:: bash

      python3 tools/textgrid_to_word_wavs.py --wav mandarin/zh.wav \
        --textgrid mandarin/zh-aishell3.TextGrid --out-dir mandarin/zh-aishell3.words

   .. raw:: html

      <table>
        <tr><th>Word</th><th>Time (s)</th><th>Audio</th></tr>
          <tr><td>不</td><td>0.06 – 0.18</td><td><audio controls src="../_static/words/zh-aishell3/w01.wav"></audio></td></tr>
          <tr><td>要</td><td>0.18 – 0.32</td><td><audio controls src="../_static/words/zh-aishell3/w02.wav"></audio></td></tr>
          <tr><td>问</td><td>0.32 – 0.48</td><td><audio controls src="../_static/words/zh-aishell3/w03.wav"></audio></td></tr>
          <tr><td>你</td><td>0.48 – 0.64</td><td><audio controls src="../_static/words/zh-aishell3/w04.wav"></audio></td></tr>
          <tr><td>的</td><td>0.64 – 0.76</td><td><audio controls src="../_static/words/zh-aishell3/w05.wav"></audio></td></tr>
          <tr><td>国</td><td>0.76 – 0.95</td><td><audio controls src="../_static/words/zh-aishell3/w06.wav"></audio></td></tr>
          <tr><td>家</td><td>0.99 – 1.14</td><td><audio controls src="../_static/words/zh-aishell3/w07.wav"></audio></td></tr>
          <tr><td>能</td><td>1.14 – 1.32</td><td><audio controls src="../_static/words/zh-aishell3/w08.wav"></audio></td></tr>
          <tr><td>为</td><td>1.32 – 1.44</td><td><audio controls src="../_static/words/zh-aishell3/w09.wav"></audio></td></tr>
          <tr><td>你</td><td>1.44 – 1.60</td><td><audio controls src="../_static/words/zh-aishell3/w10.wav"></audio></td></tr>
          <tr><td>做</td><td>1.63 – 1.73</td><td><audio controls src="../_static/words/zh-aishell3/w11.wav"></audio></td></tr>
          <tr><td>什</td><td>1.76 – 1.94</td><td><audio controls src="../_static/words/zh-aishell3/w12.wav"></audio></td></tr>
          <tr><td>么</td><td>1.94 – 2.19</td><td><audio controls src="../_static/words/zh-aishell3/w13.wav"></audio></td></tr>
          <tr><td>而</td><td>2.48 – 2.55</td><td><audio controls src="../_static/words/zh-aishell3/w14.wav"></audio></td></tr>
          <tr><td>要</td><td>2.55 – 2.74</td><td><audio controls src="../_static/words/zh-aishell3/w15.wav"></audio></td></tr>
          <tr><td>问</td><td>2.74 – 2.89</td><td><audio controls src="../_static/words/zh-aishell3/w16.wav"></audio></td></tr>
          <tr><td>你</td><td>2.89 – 3.04</td><td><audio controls src="../_static/words/zh-aishell3/w17.wav"></audio></td></tr>
          <tr><td>能</td><td>3.04 – 3.20</td><td><audio controls src="../_static/words/zh-aishell3/w18.wav"></audio></td></tr>
          <tr><td>为</td><td>3.20 – 3.33</td><td><audio controls src="../_static/words/zh-aishell3/w19.wav"></audio></td></tr>
          <tr><td>你</td><td>3.33 – 3.50</td><td><audio controls src="../_static/words/zh-aishell3/w20.wav"></audio></td></tr>
          <tr><td>的</td><td>3.50 – 3.61</td><td><audio controls src="../_static/words/zh-aishell3/w21.wav"></audio></td></tr>
          <tr><td>国</td><td>3.61 – 3.77</td><td><audio controls src="../_static/words/zh-aishell3/w22.wav"></audio></td></tr>
          <tr><td>家</td><td>3.77 – 3.95</td><td><audio controls src="../_static/words/zh-aishell3/w23.wav"></audio></td></tr>
          <tr><td>做</td><td>3.99 – 4.11</td><td><audio controls src="../_static/words/zh-aishell3/w24.wav"></audio></td></tr>
          <tr><td>什</td><td>4.14 – 4.34</td><td><audio controls src="../_static/words/zh-aishell3/w25.wav"></audio></td></tr>
          <tr><td>么</td><td>4.34 – 4.55</td><td><audio controls src="../_static/words/zh-aishell3/w26.wav"></audio></td></tr>
      </table>

Large pinyin model
------------------

Run it — copy and paste:

.. code-block:: bash

   # model + dictionary (the dictionary ships inside the zip); run from the
   # repository root
   mkdir -p large_pinyin && cd large_pinyin
   wget https://github.com/lars76/forced-alignment-chinese/releases/download/model_files/large_pinyin.zip
   unzip large_pinyin.zip             # -> large_pinyin_acoustic.zip + dictionaries
   unzip large_pinyin_acoustic.zip    # -> d_pinyin_acoustic/
   cd ..

   # align
   python3 tools/align_mfa.py --wav mandarin/zh.wav \
     --text "不要问你的国家能为你做什么，而要问你能为你的国家做什么。" \
     --model-dir large_pinyin/d_pinyin_acoustic \
     --dict large_pinyin/large_pinyin_dictionary.dict \
     --phones-txt large_pinyin/d_pinyin_acoustic/phones.txt \
     --tokenize pinyin --out mandarin/zh-large.TextGrid

Expected output:

.. code-block:: text

   OK: mandarin/zh-large.TextGrid  frames=476 dur=4.76s phones=66 words=26
   timing: audio duration 4.76s, elapsed 0.67s, RTF 0.141

The resulting TextGrid
----------------------

.. only:: html

   .. raw:: html

      <details>
      <summary>Show mandarin/zh-large.TextGrid (388 lines)</summary>

   .. code-block:: text

      File type = "ooTextFile"
      Object class = "TextGrid"
      
      xmin = 0
      xmax = 4.7600
      tiers? <exists>
      size = 2
      item []:
          item [1]:
              class = "IntervalTier"
              name = "phones"
              xmin = 0
              xmax = 4.7600
              intervals: size = 66
              intervals [1]:
                  xmin = 0.0000
                  xmax = 0.0400
                  text = "sil"
              intervals [2]:
                  xmin = 0.0400
                  xmax = 0.1100
                  text = "p"
              intervals [3]:
                  xmin = 0.1100
                  xmax = 0.1800
                  text = "u"
              intervals [4]:
                  xmin = 0.1800
                  xmax = 0.2600
                  text = "j"
              intervals [5]:
                  xmin = 0.2600
                  xmax = 0.3300
                  text = "au̯"
              intervals [6]:
                  xmin = 0.3300
                  xmax = 0.3900
                  text = "w"
              intervals [7]:
                  xmin = 0.3900
                  xmax = 0.4300
                  text = "ə"
              intervals [8]:
                  xmin = 0.4300
                  xmax = 0.4800
                  text = "n"
              intervals [9]:
                  xmin = 0.4800
                  xmax = 0.5500
                  text = "n"
              intervals [10]:
                  xmin = 0.5500
                  xmax = 0.6300
                  text = "i"
              intervals [11]:
                  xmin = 0.6300
                  xmax = 0.7000
                  text = "t"
              intervals [12]:
                  xmin = 0.7000
                  xmax = 0.7600
                  text = "ɤ"
              intervals [13]:
                  xmin = 0.7600
                  xmax = 0.8500
                  text = "k"
              intervals [14]:
                  xmin = 0.8500
                  xmax = 0.8900
                  text = "w"
              intervals [15]:
                  xmin = 0.8900
                  xmax = 0.9600
                  text = "o"
              intervals [16]:
                  xmin = 0.9600
                  xmax = 1.0500
                  text = "tɕ"
              intervals [17]:
                  xmin = 1.0500
                  xmax = 1.0900
                  text = "j"
              intervals [18]:
                  xmin = 1.0900
                  xmax = 1.1400
                  text = "a"
              intervals [19]:
                  xmin = 1.1400
                  xmax = 1.2200
                  text = "n"
              intervals [20]:
                  xmin = 1.2200
                  xmax = 1.2800
                  text = "ə"
              intervals [21]:
                  xmin = 1.2800
                  xmax = 1.3200
                  text = "ŋ"
              intervals [22]:
                  xmin = 1.3200
                  xmax = 1.3700
                  text = "w"
              intervals [23]:
                  xmin = 1.3700
                  xmax = 1.4500
                  text = "ei̯"
              intervals [24]:
                  xmin = 1.4500
                  xmax = 1.5300
                  text = "n"
              intervals [25]:
                  xmin = 1.5300
                  xmax = 1.6000
                  text = "i"
              intervals [26]:
                  xmin = 1.6000
                  xmax = 1.6800
                  text = "ts"
              intervals [27]:
                  xmin = 1.6800
                  xmax = 1.7200
                  text = "w"
              intervals [28]:
                  xmin = 1.7200
                  xmax = 1.7500
                  text = "o"
              intervals [29]:
                  xmin = 1.7500
                  xmax = 1.8600
                  text = "ʂ"
              intervals [30]:
                  xmin = 1.8600
                  xmax = 1.9100
                  text = "ə"
              intervals [31]:
                  xmin = 1.9100
                  xmax = 1.9400
                  text = "n"
              intervals [32]:
                  xmin = 1.9400
                  xmax = 2.0200
                  text = "m"
              intervals [33]:
                  xmin = 2.0200
                  xmax = 2.1600
                  text = "ɤ"
              intervals [34]:
                  xmin = 2.1600
                  xmax = 2.4800
                  text = "sil"
              intervals [35]:
                  xmin = 2.4800
                  xmax = 2.5800
                  text = "aɚ̯"
              intervals [36]:
                  xmin = 2.5800
                  xmax = 2.6800
                  text = "j"
              intervals [37]:
                  xmin = 2.6800
                  xmax = 2.7400
                  text = "au̯"
              intervals [38]:
                  xmin = 2.7400
                  xmax = 2.8000
                  text = "w"
              intervals [39]:
                  xmin = 2.8000
                  xmax = 2.8400
                  text = "ə"
              intervals [40]:
                  xmin = 2.8400
                  xmax = 2.8800
                  text = "n"
              intervals [41]:
                  xmin = 2.8800
                  xmax = 2.9600
                  text = "n"
              intervals [42]:
                  xmin = 2.9600
                  xmax = 3.0300
                  text = "i"
              intervals [43]:
                  xmin = 3.0300
                  xmax = 3.1000
                  text = "n"
              intervals [44]:
                  xmin = 3.1000
                  xmax = 3.1700
                  text = "ə"
              intervals [45]:
                  xmin = 3.1700
                  xmax = 3.2100
                  text = "ŋ"
              intervals [46]:
                  xmin = 3.2100
                  xmax = 3.2600
                  text = "w"
              intervals [47]:
                  xmin = 3.2600
                  xmax = 3.3400
                  text = "ei̯"
              intervals [48]:
                  xmin = 3.3400
                  xmax = 3.4200
                  text = "n"
              intervals [49]:
                  xmin = 3.4200
                  xmax = 3.4900
                  text = "i"
              intervals [50]:
                  xmin = 3.4900
                  xmax = 3.5600
                  text = "t"
              intervals [51]:
                  xmin = 3.5600
                  xmax = 3.6100
                  text = "ɤ"
              intervals [52]:
                  xmin = 3.6100
                  xmax = 3.6800
                  text = "k"
              intervals [53]:
                  xmin = 3.6800
                  xmax = 3.7200
                  text = "w"
              intervals [54]:
                  xmin = 3.7200
                  xmax = 3.7700
                  text = "o"
              intervals [55]:
                  xmin = 3.7700
                  xmax = 3.8400
                  text = "tɕ"
              intervals [56]:
                  xmin = 3.8400
                  xmax = 3.8800
                  text = "j"
              intervals [57]:
                  xmin = 3.8800
                  xmax = 3.9600
                  text = "a"
              intervals [58]:
                  xmin = 3.9600
                  xmax = 4.0400
                  text = "ts"
              intervals [59]:
                  xmin = 4.0400
                  xmax = 4.0800
                  text = "w"
              intervals [60]:
                  xmin = 4.0800
                  xmax = 4.1100
                  text = "o"
              intervals [61]:
                  xmin = 4.1100
                  xmax = 4.2500
                  text = "ʂ"
              intervals [62]:
                  xmin = 4.2500
                  xmax = 4.3000
                  text = "ə"
              intervals [63]:
                  xmin = 4.3000
                  xmax = 4.3400
                  text = "n"
              intervals [64]:
                  xmin = 4.3400
                  xmax = 4.4300
                  text = "m"
              intervals [65]:
                  xmin = 4.4300
                  xmax = 4.5400
                  text = "ɤ"
              intervals [66]:
                  xmin = 4.5400
                  xmax = 4.7600
                  text = "sil"
          item [2]:
              class = "IntervalTier"
              name = "words"
              xmin = 0
              xmax = 4.7600
              intervals: size = 26
              intervals [1]:
                  xmin = 0.0400
                  xmax = 0.1800
                  text = "不"
              intervals [2]:
                  xmin = 0.1800
                  xmax = 0.3300
                  text = "要"
              intervals [3]:
                  xmin = 0.3300
                  xmax = 0.4800
                  text = "问"
              intervals [4]:
                  xmin = 0.4800
                  xmax = 0.6300
                  text = "你"
              intervals [5]:
                  xmin = 0.6300
                  xmax = 0.7600
                  text = "的"
              intervals [6]:
                  xmin = 0.7600
                  xmax = 0.9600
                  text = "国"
              intervals [7]:
                  xmin = 0.9600
                  xmax = 1.1400
                  text = "家"
              intervals [8]:
                  xmin = 1.1400
                  xmax = 1.3200
                  text = "能"
              intervals [9]:
                  xmin = 1.3200
                  xmax = 1.4500
                  text = "为"
              intervals [10]:
                  xmin = 1.4500
                  xmax = 1.6000
                  text = "你"
              intervals [11]:
                  xmin = 1.6000
                  xmax = 1.7500
                  text = "做"
              intervals [12]:
                  xmin = 1.7500
                  xmax = 1.9400
                  text = "什"
              intervals [13]:
                  xmin = 1.9400
                  xmax = 2.1600
                  text = "么"
              intervals [14]:
                  xmin = 2.4800
                  xmax = 2.5800
                  text = "而"
              intervals [15]:
                  xmin = 2.5800
                  xmax = 2.7400
                  text = "要"
              intervals [16]:
                  xmin = 2.7400
                  xmax = 2.8800
                  text = "问"
              intervals [17]:
                  xmin = 2.8800
                  xmax = 3.0300
                  text = "你"
              intervals [18]:
                  xmin = 3.0300
                  xmax = 3.2100
                  text = "能"
              intervals [19]:
                  xmin = 3.2100
                  xmax = 3.3400
                  text = "为"
              intervals [20]:
                  xmin = 3.3400
                  xmax = 3.4900
                  text = "你"
              intervals [21]:
                  xmin = 3.4900
                  xmax = 3.6100
                  text = "的"
              intervals [22]:
                  xmin = 3.6100
                  xmax = 3.7700
                  text = "国"
              intervals [23]:
                  xmin = 3.7700
                  xmax = 3.9600
                  text = "家"
              intervals [24]:
                  xmin = 3.9600
                  xmax = 4.1100
                  text = "做"
              intervals [25]:
                  xmin = 4.1100
                  xmax = 4.3400
                  text = "什"
              intervals [26]:
                  xmin = 4.3400
                  xmax = 4.5400
                  text = "么"

   .. raw:: html

      </details>


.. only:: html

   Listen to each word
   --------------------
   
   Transcript: 不要问你的国家能为你做什么，而要问你能为你的国家做什么。

   Every character of the transcript, cut by the word intervals of
   ``zh-large.TextGrid``.  Reproduce the clips with:

   .. code-block:: bash

      python3 tools/textgrid_to_word_wavs.py --wav mandarin/zh.wav \
        --textgrid mandarin/zh-large.TextGrid --out-dir mandarin/zh-large.words

   .. raw:: html

      <table>
        <tr><th>Word</th><th>Time (s)</th><th>Audio</th></tr>
          <tr><td>不</td><td>0.04 – 0.18</td><td><audio controls src="../_static/words/zh-large/w01.wav"></audio></td></tr>
          <tr><td>要</td><td>0.18 – 0.33</td><td><audio controls src="../_static/words/zh-large/w02.wav"></audio></td></tr>
          <tr><td>问</td><td>0.33 – 0.48</td><td><audio controls src="../_static/words/zh-large/w03.wav"></audio></td></tr>
          <tr><td>你</td><td>0.48 – 0.63</td><td><audio controls src="../_static/words/zh-large/w04.wav"></audio></td></tr>
          <tr><td>的</td><td>0.63 – 0.76</td><td><audio controls src="../_static/words/zh-large/w05.wav"></audio></td></tr>
          <tr><td>国</td><td>0.76 – 0.96</td><td><audio controls src="../_static/words/zh-large/w06.wav"></audio></td></tr>
          <tr><td>家</td><td>0.96 – 1.14</td><td><audio controls src="../_static/words/zh-large/w07.wav"></audio></td></tr>
          <tr><td>能</td><td>1.14 – 1.32</td><td><audio controls src="../_static/words/zh-large/w08.wav"></audio></td></tr>
          <tr><td>为</td><td>1.32 – 1.45</td><td><audio controls src="../_static/words/zh-large/w09.wav"></audio></td></tr>
          <tr><td>你</td><td>1.45 – 1.60</td><td><audio controls src="../_static/words/zh-large/w10.wav"></audio></td></tr>
          <tr><td>做</td><td>1.60 – 1.75</td><td><audio controls src="../_static/words/zh-large/w11.wav"></audio></td></tr>
          <tr><td>什</td><td>1.75 – 1.94</td><td><audio controls src="../_static/words/zh-large/w12.wav"></audio></td></tr>
          <tr><td>么</td><td>1.94 – 2.16</td><td><audio controls src="../_static/words/zh-large/w13.wav"></audio></td></tr>
          <tr><td>而</td><td>2.48 – 2.58</td><td><audio controls src="../_static/words/zh-large/w14.wav"></audio></td></tr>
          <tr><td>要</td><td>2.58 – 2.74</td><td><audio controls src="../_static/words/zh-large/w15.wav"></audio></td></tr>
          <tr><td>问</td><td>2.74 – 2.88</td><td><audio controls src="../_static/words/zh-large/w16.wav"></audio></td></tr>
          <tr><td>你</td><td>2.88 – 3.03</td><td><audio controls src="../_static/words/zh-large/w17.wav"></audio></td></tr>
          <tr><td>能</td><td>3.03 – 3.21</td><td><audio controls src="../_static/words/zh-large/w18.wav"></audio></td></tr>
          <tr><td>为</td><td>3.21 – 3.34</td><td><audio controls src="../_static/words/zh-large/w19.wav"></audio></td></tr>
          <tr><td>你</td><td>3.34 – 3.49</td><td><audio controls src="../_static/words/zh-large/w20.wav"></audio></td></tr>
          <tr><td>的</td><td>3.49 – 3.61</td><td><audio controls src="../_static/words/zh-large/w21.wav"></audio></td></tr>
          <tr><td>国</td><td>3.61 – 3.77</td><td><audio controls src="../_static/words/zh-large/w22.wav"></audio></td></tr>
          <tr><td>家</td><td>3.77 – 3.96</td><td><audio controls src="../_static/words/zh-large/w23.wav"></audio></td></tr>
          <tr><td>做</td><td>3.96 – 4.11</td><td><audio controls src="../_static/words/zh-large/w24.wav"></audio></td></tr>
          <tr><td>什</td><td>4.11 – 4.34</td><td><audio controls src="../_static/words/zh-large/w25.wav"></audio></td></tr>
          <tr><td>么</td><td>4.34 – 4.54</td><td><audio controls src="../_static/words/zh-large/w26.wav"></audio></td></tr>
      </table>

Notes
-----

* the two models produce slightly different phone inventories (71 vs 66
  phone intervals); both alignments are valid — choose the model that fits
  your data;
* the word tier contains the 26 characters of the transcript;
* open the TextGrids in `Praat <https://praat.org/>`_ next to
  ``mandarin/zh.wav`` to inspect the alignment;
* the official MFA Mandarin model
  (`acoustic-mandarin_mfa <https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-mandarin_mfa-v2.0.0>`_)
  is also supported: its ``meta.json`` turns pitch features on, so add
  ``--pitch``.  It needs a matching MFA pinyin dictionary — pass it with
  ``--dict`` just like the examples above.
