# DL-Py-NLP-Project-Technion
Final Project for DL in Python @ Technion
# News Classification & Headline Generation with Deep Learning

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

