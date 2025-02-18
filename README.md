# DL-Py-NLP-Project-Technion
# News Classification & Headline Generation with Deep Learning
![image](https://github.com/user-attachments/assets/2b306c79-c8c5-407b-bac2-909f7f4dfdeb)

## Overview
Can AI spot bias in media coverage?

Analyzing ~13K articles from Times of Israel, BBC, CNN, Al Jazeera, and WAFA published in Oct-Nov 2023, showed interesting patterns in how these outlets frame the Israel-Hamas war. 

To classify articles by their source (Israeli, Arab, or Western), I trained a model using a FF neural network and BERT embeddings, which learned to weight titles at 28%, and achieved 90%+ accuracy on all classes. A SHAP analysis revealed the most influential words for classification:
 
𝐈𝐬𝐫𝐚𝐞𝐥𝐢: "𝘯𝘰𝘮𝘪𝘯𝘢𝘵𝘪𝘰𝘯," "𝘒𝘯𝘦𝘴𝘴𝘦𝘵," "𝘷𝘰𝘵𝘦" (politics), "𝘏𝘢𝘮𝘢𝘴’𝘴," "𝘖𝘤𝘵𝘰𝘣𝘦𝘳," "𝘗𝘰𝘴𝘵" (October 7th aftermath), and "𝘥𝘢𝘮𝘢𝘨𝘦," "𝘳𝘦𝘵𝘶𝘳𝘯𝘴" (destruction and hostage deal).
𝐀𝐫𝐚𝐛: "𝘵𝘦𝘳𝘳𝘰𝘳𝘪𝘴𝘵𝘴," "𝘵𝘦𝘳𝘳𝘪𝘵𝘰𝘳𝘺," "𝘮𝘦𝘥𝘪𝘤𝘢𝘭,", "13", "𝘴𝘵𝘢𝘵𝘦𝘮𝘦𝘯𝘵" (sovereignty and humanitarian themes), and "𝘓𝘰𝘯𝘥𝘰𝘯" (protest reference).
𝐖𝐞𝐬𝐭𝐞𝐫𝐧: "𝘬𝘪𝘭𝘭𝘦𝘥," "2,","𝘤𝘩𝘪𝘭𝘥𝘳𝘦𝘯" (an emphasis on human cost), "𝘎𝘢𝘻𝘢", and "𝘏𝘢𝘨𝘢𝘳𝘪" (IDF updates).

Israeli media was still processing October 7th's internal impact, while Arab and Western media had shifted focus to Gaza. 

To see how outlets frame the war differently, I fine-tuned a T5 transformer model to generate headlines from articles, training 3 models (1 on each category) by comparing the generated headline with the real one. Feeding in new identical neutral texts reflected the editorial bias the models had learned. Here are 2 examples: 

𝐈𝐬𝐫𝐚𝐞𝐥𝐢: "𝘐𝘴𝘳𝘢𝘦𝘭𝘪 𝘥𝘳𝘰𝘯𝘦 𝘴𝘵𝘳𝘪𝘬𝘦 𝘵𝘢𝘳𝘨𝘦𝘵𝘦𝘥 𝘞𝘰𝘳𝘭𝘥 𝘊𝘦𝘯𝘵𝘳𝘢𝘭 𝘒𝘪𝘵𝘤𝘩𝘦𝘯 𝘪𝘯 𝘎𝘢𝘻𝘢, 𝘬𝘪𝘭𝘭𝘪𝘯𝘨 7 𝘸𝘰𝘳𝘬𝘦𝘳𝘴."
𝐀𝐫𝐚𝐛: "𝘚𝘦𝘷𝘦𝘯 𝘢𝘪𝘥 𝘸𝘰𝘳𝘬𝘦𝘳𝘴 𝘬𝘪𝘭𝘭𝘦𝘥 𝘪𝘯 𝘢𝘯 𝘐𝘴𝘳𝘢𝘦𝘭𝘪 𝘴𝘵𝘳𝘪𝘬𝘦 𝘰𝘯 𝘢 𝘤𝘰𝘯𝘷𝘰𝘺 𝘪𝘯 𝘎𝘢𝘻𝘢."
𝐖𝐞𝐬𝐭𝐞𝐫𝐧: "𝘐𝘴𝘳𝘢𝘦𝘭𝘪 𝘥𝘳𝘰𝘯𝘦 𝘴𝘵𝘳𝘪𝘬𝘦 𝘵𝘢𝘳𝘨𝘦𝘵𝘦𝘥 𝘢𝘪𝘥 𝘤𝘰𝘯𝘷𝘰𝘺 𝘪𝘯 𝘎𝘢𝘻𝘢, 𝘬𝘪𝘭𝘭𝘪𝘯𝘨 𝘢𝘵 𝘭𝘦𝘢𝘴𝘵 𝘴𝘦𝘷𝘦𝘯 𝘸𝘰𝘳𝘬𝘦𝘳𝘴."

The Arab model starts with casualties, while Western media adds "at least", although the text was clear.

𝐈𝐬𝐫𝐚𝐞𝐥𝐢: "𝘐𝘋𝘍 𝘭𝘢𝘶𝘯𝘤𝘩𝘦𝘴 𝘮𝘪𝘭𝘪𝘵𝘢𝘳𝘺 𝘰𝘱𝘦𝘳𝘢𝘵𝘪𝘰𝘯 𝘪𝘯 𝘙𝘢𝘧𝘢𝘩 𝘵𝘰 𝘥𝘪𝘴𝘮𝘢𝘯𝘵𝘭𝘦 𝘏𝘢𝘮𝘢𝘴 𝘵𝘶𝘯𝘯𝘦𝘭 𝘯𝘦𝘵𝘸𝘰𝘳𝘬"
𝐀𝐫𝐚𝐛: "𝘐𝘴𝘳𝘢𝘦𝘭𝘪 𝘧𝘰𝘳𝘤𝘦𝘴 𝘭𝘢𝘶𝘯𝘤𝘩 𝘮𝘪𝘭𝘪𝘵𝘢𝘳𝘺 𝘰𝘱𝘦𝘳𝘢𝘵𝘪𝘰𝘯 𝘢𝘨𝘢𝘪𝘯𝘴𝘵 𝘏𝘢𝘮𝘢𝘴 𝘪𝘯 𝘙𝘢𝘧𝘢𝘩"
𝐀𝐫𝐚𝐛: "𝘐𝘴𝘳𝘢𝘦𝘭𝘪 𝘧𝘰𝘳𝘤𝘦𝘴 𝘭𝘢𝘶𝘯𝘤𝘩 𝘮𝘪𝘭𝘪𝘵𝘢𝘳𝘺 𝘰𝘱𝘦𝘳𝘢𝘵𝘪𝘰𝘯 𝘪𝘯 𝘙𝘢𝘧𝘢𝘩 𝘢𝘴 𝘱𝘢𝘳𝘵 𝘰𝘧 𝘰𝘯𝘨𝘰𝘪𝘯𝘨 𝘤𝘢𝘮𝘱𝘢𝘪𝘨𝘯 𝘢𝘨𝘢𝘪𝘯𝘴𝘵 𝘏𝘢𝘮𝘢𝘴"

Only the Israeli model highlights military goals. 

This project marked the end of an intensive 8-months period of learning at the Technion's Data Science program, where I gained theoretical and practical experience in design, implementation and evaluation of ML & DL models in Python and R. 
![News Classification   Headline Generation with Deep Learning - visual selection](https://github.com/user-attachments/assets/db721593-63e4-49e0-aea6-b1c6ee6c1d4b)

## 1. Data
- News articles covering the Israel-Hamas war from October-November 2023 from 5 providers: Times of Israel, BBC, CNN, Al Jazeera, and WAFA
 ![image](https://github.com/user-attachments/assets/a26fba7d-5a27-40c5-baa7-f42eda7650f3)
## 2. Pre-processing
### Categorization
- Providers categorized into 3 source-groups:
  - **Israeli**: Times of Israel, overall 6,361 articles (~51%)
  - **Western**: BBC, CNN, overall 2,030 articles (~16%)
  - **Arab**: Al Jazeera, WAFA, overall 4,086 articles (~33%)
### Dropping Outliers
- Extremely long articles removed to optimize efficiency
 ![image](https://github.com/user-attachments/assets/3bc0b22d-ff0a-4a78-9653-5d689141cd42)
## 3. Task 1: Source Classification & Word Importance
![image](https://github.com/user-attachments/assets/7567818a-c3e6-4b3d-a62b-955f0d7c1d20)
### 3.1. Classifying Articles as Israeli / Arab / Western with High Accuracy
#### 3.1.1. Tokenization
- Tokenization performed using **BERT Base** with a **512-token limit**
  - Special tokens `[CLS]` and `[SEP]`: **3 tokens**
  - Remaining tokens for titles and text: **509 tokens**
  - **Titles**: max **40 tokens**
  - **Texts**: max **469 tokens**
 
 ![image](https://github.com/user-attachments/assets/1a5aa485-6062-4f7c-9a7e-52f3acaccd12)
#### 3.1.2. Model Overview - BERT→ Feed Forward Neural Network
- **Separate BERT embedding processing** (frozen parameters) for **titles and texts**
  - **Title features**: 768 → **256 dimensions**
  - **Text features**: 768 → **256 dimensions**
  - Dropout **(0.3)** and ReLU layers for regularization
  - Concatenated features **(512 → 256 dimensions)**, with titles weighted at **30%** and texts at **70%**, but distribution dynamically adjusted during training
  - Final classification layer maps to **3 classes**, using `[CLS]` token
  ![image](https://github.com/user-attachments/assets/08f85236-b749-4858-adc6-0c4203f508a1)
#### 3.1.3. Hyperparameters
- **Loss Function**: CrossEntropyLoss (for class imbalance, punishing mistakes from smaller classes)
- **Optimizer**: Adam, **Learning Rate = 3e-5**, high since learning happens only for layers above (frozen) BERT
- **Weight Decay**: 5e-5 (L2 regularization), adding penalty for large weights
- **Epochs**: max **40** with an early stopping mechanism to break when loss plateaus
- **Batch Size**: **16**
- **LR Scheduler**: reduces LR when loss ~stops improving to prevent overshooting the optimal solution
#### 3.1.4. Results
- **Overall Accuracy**: **92%**
![image](https://github.com/user-attachments/assets/3be281dc-d2bb-426d-9113-a02e9ca16912)
- **Optimal Title Weight**: **0.28**
![image](https://github.com/user-attachments/assets/e5e34653-fc07-4357-8e1b-d3fdb09b2172)
- **Confusion Matrix**: High accuracy (~90%+ per category)

 ![image](https://github.com/user-attachments/assets/704b3151-1170-4b4e-a663-330c7ec6b0b4)
 ![image](https://github.com/user-attachments/assets/2089a455-d5f3-4951-bbbb-152f75114b12)
### 3.2. Performing SHAP Analysis on Word Importance
#### 3.2.1. SHAP Analysis on Israeli Class
- **Key Words**:
  - **"nomination," "Knesset’s," "vote"**: An emphasis on political terminology indicates coverage of governmental responses and political dynamics in reaction to the conflict.
  - **"Hamas’s," "October," "Post"**: The terms "Hamas’s", and “October” directly relate to the actions and timing of the conflict’s eruption, and “Post” could refer to publications or reports following the attacks.
  - **"damage", “returns,”**: Likely pertain to reports on the destruction in Israel (infrastructure and civilian areas), and to either hostage return efforts or Israeli civilians returning from abroad
  - **The focus is the internal political and civilian situation post October 7th**
  <img width="892" alt="image" src="https://github.com/user-attachments/assets/69bb4c48-f7fc-4fa7-85ca-ef9fe7414bcf" />
#### 3.2.2. SHAP Analysis on Arab Class
- **Key Words**:
  - **"terrorists“**: the portrayal of Israeli forces or quotation of Israeli statements.
  - **"territory“**: discussions on land and sovereignty issues.
  - **"Medical“**: a focus on healthcare crises
  - **"statement“**: official communications or declarations.
  - **"13“**: could reference specific casualty figures (YR – November 11th).
  - **"London“**: Pro-Palestinian demonstration of 300K in November 11th (YR)
  - **"After“ and "It’s“**: common words that could appear in various contexts
  - **"you're," "joining" (title)**: may indicate call to action or aim at engaging readers
  - **A focus on humanitarian concerns, territorial sovereignty, and int’l responses** 
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/78895ca4-2797-4a9d-8ed5-a69a56d77343" />
#### 3.2.3. SHAP Analysis on Western Class
- **Key Words**:
  - "killed," "2," "children" (title), "air." (text): highlight casualty reports (YR - from air bombings), focusing on children impacted.
  - "Gaza.": indicates a focus on the region (YR – more than on Israel)
  - "Hagari.": suggests reliance on official military updates.
  - "We," "is,": common in reporting, possibly reflecting direct quotations.
  - "East", and “-”: probably refer to "Middle-East,".
  -  **Western titles prominently feature casualty figures, aiming to capture attention and convey the gravity of the situation, and emphasizing the human cost**
<img width="886" alt="image" src="https://github.com/user-attachments/assets/a537682a-40b1-4d3f-a8f3-e66a9d6ba444" />

### 3.3. Task 1 Conclusions
- **Israeli media**: Focused on internal affairs post-October 7th
- **Arab & Western media**: Focused on Gaza, casualties, and international responses
- **Short time-frame** may limit generalization of key words
## 4. Task 2: Fine-tuning for Headline Generation
 ![image](https://github.com/user-attachments/assets/e3002562-2fbc-4ffe-9489-0f440d564670)
### 4.1. Approach 1: Source Token Addition
#### 4.1.1. **Mapped source categories to special tokens and created a new column with tokens prepended to texts:**
  - Israeli → `[SOURCE: ISRAELI NEWS]`
  - Arab → `[SOURCE: ARAB NEWS]`
  - Western → `[SOURCE: WESTERN NEWS]`
 ![image](https://github.com/user-attachments/assets/73602458-13f6-4b6d-8153-5c217dce8e23)
#### 4.1.2. T5-small choice rationale
 - Pre-trained on summarization (close to headlines)
 - Unified text-to-text format (encoder-decoder)
 - Fast training and inference (good performance for size)
 - Sufficient capacity (60M parameters, ~35K lexicon)

 ![image](https://github.com/user-attachments/assets/82f86acd-be88-4e8e-9f4e-09438d4549ac)
#### 4.1.3. Beam Search as generative function
 - Better than TOP K/P sampling for maintaining coherence
 - Number of Beams: 4, standard choice to balance exploration vs computation
 ![image](https://github.com/user-attachments/assets/9b0b03fd-e42d-4563-80dc-4fbf7895127a)

#### 4.1.4. Five factual texts with narratives from both sides will be fed to generate headlines with each source token (3*5 = 15 headlines)
<img width="305" alt="image" src="https://github.com/user-attachments/assets/0b5b7305-dd2f-410e-91cf-f32c8ef35056" />

#### 4.1.5. **Result**: Model failed to differentiate sources, generating similar headlines
### 4.2. Approach 2: Separate Source-based Models
- **Trained 3 models**, each on its own category with separate data
![image](https://github.com/user-attachments/assets/0770f1b6-2deb-42ff-9f29-8617805afbf7)
 #### 4.2.1. Results and Analysis:
 ![image](https://github.com/user-attachments/assets/84f86fa8-13ba-4085-8c35-0682c079173c)
![image](https://github.com/user-attachments/assets/06030c16-c7ec-4ddb-a648-9c2a4ff1f436)
![image](https://github.com/user-attachments/assets/cedc2a26-f525-49e4-bcd0-0d8aa455d772)
![image](https://github.com/user-attachments/assets/21264414-ab56-4903-b511-147d05c830e1)
![image](https://github.com/user-attachments/assets/c2361c28-da0b-4649-8a5c-86c07f00b220)
### 4.3. Task 2 Conclusions
- **Separate models** captured more source-specific nuances than token-based approach
- **Larger datasets** would further enhance headline differentiation
## 5. Future Work: Western Sentiment Evolution
![image](https://github.com/user-attachments/assets/382dd298-f3f2-4c46-b71e-cabb953cfc44)
### 5.1. Task: Classify Western Sentiment as Israeli/Arab/Neutral
- **Approach**: Train model on **Israeli & Arab** sources, classify **Western articles**
- **Goal**: Analyze sentiment shifts over time
- **Explainability**: Use **LIME** to interpret classifications
### 5.2. Current Problem
- **Western media coverage** limited to **Oct 26th - Nov 26th**, restricting time-series analysis

