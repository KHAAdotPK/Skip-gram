### Analyzing Training Progress in [Skip-Gram](https://github.com/KHAAdotPK/Skip-gram.git) Word Embeddings: A Detailed Evaluation of Loss, Cosine Similarity, and Parameter Optimization.
---

**The model is showing stable improvement in loss and slight but consistent changes in cosine similarities, indicating that it's learning subtle word relationships with each training run.**

1. Each run shows a consistent decrease in loss per epoch, which is a good indicator that the model is learning and optimizing. Loss values decrease more gradually as training progresses, which is typical as the model finds a more stable solution.
2. Given the small learning rate (0.0001), the model is updating gradually. The incremental decrease in loss over multiple training runs suggests this cautious learning rate is preventing overshooting but might require many epochs to reach an optimal solution. If faster convergence is needed, slightly increasing the learning rate might be worth testing.

### Cosine Similarity Trends:
1. **Sunshine and Whale**: There’s a shift in cosine similarity from -0.1998 to -0.1649 to -0.1153, indicating these words are becoming less "distant" or more aligned in the embedding space across runs, even though they still aren't very closely related (negative similarity implies they're not in close semantic proximity).
2. **Sunshine and Kisses**: This pair has positive cosine similarities across all runs, hovering around 0.207 to 0.2155. The model sees them as relatively similar, and this similarity is stable with minimal change.
3. **Whale and Kisses / Whale and Cheeks**: These pairs also see slight shifts, suggesting some subtle changes in the embedding space structure, but they’re relatively stable.
4. **Self-Similarity (e.g., whale -> whale)**: Always equal to 1 as expected, indicating the cosine similarity measure and embeddings are working correctly.

### Cosine Distance Insights:
The cosine distances (1 - cosine similarity) consistently decrease across runs, even though the changes are subtle. This generally means the embeddings are becoming better aligned as the model continues to learn from the input data, though some word pairs have high distances, showing that the model still doesn’t see them as similar (e.g., sunshine -> cheeks).
    - **Shuffling Impact**: Since we are shuffling context pairs each at the begining of each epoch, this can help generalize across contexts, but for a small vocabulary, it might result in weaker reinforcement of rare or specific word pairs, especially with a small context window.

#### Small context window:
1. **Context Window Size (2)**: A context window size of 2 means the model is learning relationships based on very local word pairs. This narrow window is particularly effective in capturing syntactic relationships, such as common word pairings or small phrases, but it may limit the model’s ability to grasp broader, more semantic relationships that arise in larger contexts.
2. When the **context window is small (like 2)**, the model primarily learns strong, focused relationships between words that often appear directly next to each other, which boosts local similarity (i.e., similarity between closely related words). As training progresses with a small context window and limited vocabulary, the model might start to shif away from specific relationships (like "sunshine" and "kisses") as it tries to capture broader relationships across the dataset, and when this happens the cosine similarity of closely related words like "sunshine" and "kisses" start to decrease.

#### To address decrease in consine similarity:
1. **Increasing the embedding size** to give more space for nuanced relationships.
2. **Adding slight regularization** to maintain focus on meaningful relationships without overgeneralizing.
3. **Experimenting with a slightly larger context window** to reinforce broader relationships without overfitting.

### Conclusion: 
**Overall, the model is working**. The patterns that we are seeing, ike a drop in cosine similarity for closely related word pairs suggest that the training is indeed capturing relationships but could benefit from parameter tuning to fine-tune those similarities. The core model architecture and implementation appear to be functioning as expected but adjustments to embedding size or/and context window might help it capture both the local and broader similarities more effectively.

### RUN - 1

```BASH
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 20 loop 0 verbose --output w1p.dat w2p.dat
```
```
Corpus: ./INPUT.txt
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 20 epochs.
epoch_loss = (10637.5), 19.447
Epoch# 2 of 20 epochs.
epoch_loss = (10612.8), 19.4018
Epoch# 3 of 20 epochs.
epoch_loss = (10593.4), 19.3663
Epoch# 4 of 20 epochs.
epoch_loss = (10573.5), 19.33
Epoch# 5 of 20 epochs.
epoch_loss = (10553.5), 19.2935
Epoch# 6 of 20 epochs.
epoch_loss = (10537.6), 19.2643
Epoch# 7 of 20 epochs.
epoch_loss = (10522.3), 19.2364
Epoch# 8 of 20 epochs.
epoch_loss = (10507.8), 19.21
Epoch# 9 of 20 epochs.
epoch_loss = (10497.4), 19.1909
Epoch# 10 of 20 epochs.
epoch_loss = (10484.7), 19.1677
Epoch# 11 of 20 epochs.
epoch_loss = (10473.1), 19.1464
Epoch# 12 of 20 epochs.
epoch_loss = (10461), 19.1243
Epoch# 13 of 20 epochs.
epoch_loss = (10448.9), 19.1021
Epoch# 14 of 20 epochs.
epoch_loss = (10436.3), 19.0792
Epoch# 15 of 20 epochs.
epoch_loss = (10428.8), 19.0654
Epoch# 16 of 20 epochs.
epoch_loss = (10417), 19.0438
Epoch# 17 of 20 epochs.
epoch_loss = (10408.2), 19.0277
Epoch# 18 of 20 epochs.
epoch_loss = (10397.7), 19.0087
Epoch# 19 of 20 epochs.
epoch_loss = (10392.2), 18.9985
Epoch# 20 of 20 epochs.
epoch_loss = (10382.2), 18.9803
Trained input weights written to file: w1p.dat
Trained output weights written to file: w2p.dat
```

```BASH
F:\Chat-Bot-Skip-gram\usage>.\weights.exe sunshine whale whale kisses cheeks w2 w2p.dat
```
```
50 X 224
sunshine -> whale
Cosine Similarity = -0.199765, Cosine Distance = 0.800235
sunshine -> whale
Cosine Similarity = -0.199765, Cosine Distance = 0.800235
sunshine -> kisses
Cosine Similarity = 0.215511, Cosine Distance = 0.784489
sunshine -> cheeks
Cosine Similarity = -0.0643211, Cosine Distance = 0.935679
whale -> whale
Cosine Similarity = 1, Cosine Distance = 0
whale -> kisses
Cosine Similarity = 0.063578, Cosine Distance = 0.936422
whale -> cheeks
Cosine Similarity = 0.0842292, Cosine Distance = 0.915771
whale -> kisses
Cosine Similarity = 0.063578, Cosine Distance = 0.936422
whale -> cheeks
Cosine Similarity = 0.0842292, Cosine Distance = 0.915771
kisses -> cheeks
Cosine Similarity = 0.0962505, Cosine Distance = 0.90375
```

### RUN - 2
```BASH
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 20 loop 0 verbose --input w1p.dat w2p.dat --output w1.dat w2.dat 
```
```        
Corpus: ./INPUT.txt
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 20 epochs.
epoch_loss = (10376.4), 18.9696
Epoch# 2 of 20 epochs.
epoch_loss = (10370.1), 18.9582
Epoch# 3 of 20 epochs.
epoch_loss = (10360.1), 18.9399
Epoch# 4 of 20 epochs.
epoch_loss = (10356.9), 18.9339
Epoch# 5 of 20 epochs.
epoch_loss = (10349.4), 18.9204
Epoch# 6 of 20 epochs.
epoch_loss = (10342.7), 18.9081
Epoch# 7 of 20 epochs.
epoch_loss = (10338.1), 18.8996
Epoch# 8 of 20 epochs.
epoch_loss = (10334.1), 18.8924
Epoch# 9 of 20 epochs.
epoch_loss = (10328.3), 18.8817
Epoch# 10 of 20 epochs.
epoch_loss = (10324.2), 18.8742
Epoch# 11 of 20 epochs.
epoch_loss = (10314.3), 18.8562
Epoch# 12 of 20 epochs.
epoch_loss = (10311.7), 18.8513
Epoch# 13 of 20 epochs.
epoch_loss = (10306.2), 18.8413
Epoch# 14 of 20 epochs.
epoch_loss = (10299.4), 18.829
Epoch# 15 of 20 epochs.
epoch_loss = (10294.3), 18.8195
Epoch# 16 of 20 epochs.
epoch_loss = (10290.5), 18.8126
Epoch# 17 of 20 epochs.
epoch_loss = (10284.8), 18.8022
Epoch# 18 of 20 epochs.
epoch_loss = (10276.8), 18.7876
Epoch# 19 of 20 epochs.
epoch_loss = (10270.6), 18.7762
Epoch# 20 of 20 epochs.
epoch_loss = (10267.1), 18.7698
Trained input weights written to file: w1.dat
Trained output weights written to file: w2.dat
```
```BASH
F:\Chat-Bot-Skip-gram\usage>.\weights.exe sunshine whale whale kisses cheeks w2 w2p.dat
```
```
50 X 224
sunshine -> whale
Cosine Similarity = -0.164949, Cosine Distance = 0.835051
sunshine -> whale
Cosine Similarity = -0.164949, Cosine Distance = 0.835051
sunshine -> kisses
Cosine Similarity = 0.209901, Cosine Distance = 0.790099
sunshine -> cheeks
Cosine Similarity = -0.0975646, Cosine Distance = 0.902435
whale -> whale
Cosine Similarity = 1, Cosine Distance = 0
whale -> kisses
Cosine Similarity = 0.0706216, Cosine Distance = 0.929378
whale -> cheeks
Cosine Similarity = 0.0786869, Cosine Distance = 0.921313
whale -> kisses
Cosine Similarity = 0.0706216, Cosine Distance = 0.929378
whale -> cheeks
Cosine Similarity = 0.0786869, Cosine Distance = 0.921313
kisses -> cheeks
Cosine Similarity = 0.0820446, Cosine Distance = 0.917955
```

### RUN - 3
```BASH
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 20 loop 0 verbose --input w1p.dat w2p.dat --output w1.dat w2.dat
```
```
Corpus: ./INPUT.txt
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 20 epochs.
epoch_loss = (10261.7), 18.7599
Epoch# 2 of 20 epochs.
epoch_loss = (10256.4), 18.7503
Epoch# 3 of 20 epochs.
epoch_loss = (10251.1), 18.7406
Epoch# 4 of 20 epochs.
epoch_loss = (10244.1), 18.7278
Epoch# 5 of 20 epochs.
epoch_loss = (10239.2), 18.7188
Epoch# 6 of 20 epochs.
epoch_loss = (10232.1), 18.7058
Epoch# 7 of 20 epochs.
epoch_loss = (10228.5), 18.6993
Epoch# 8 of 20 epochs.
epoch_loss = (10220.6), 18.6848
Epoch# 9 of 20 epochs.
epoch_loss = (10214.9), 18.6745
Epoch# 10 of 20 epochs.
epoch_loss = (10210), 18.6655
Epoch# 11 of 20 epochs.
epoch_loss = (10206), 18.6581
Epoch# 12 of 20 epochs.
epoch_loss = (10199.2), 18.6456
Epoch# 13 of 20 epochs.
epoch_loss = (10193.3), 18.6349
Epoch# 14 of 20 epochs.
epoch_loss = (10183.5), 18.6171
Epoch# 15 of 20 epochs.
epoch_loss = (10179.6), 18.6098
Epoch# 16 of 20 epochs.
epoch_loss = (10170.9), 18.594
Epoch# 17 of 20 epochs.
epoch_loss = (10163), 18.5795
Epoch# 18 of 20 epochs.
epoch_loss = (10158.5), 18.5714
Epoch# 19 of 20 epochs.
epoch_loss = (10149.6), 18.5551
Epoch# 20 of 20 epochs.
epoch_loss = (10144.1), 18.5449
Trained input weights written to file: w1.dat
Trained output weights written to file: w2.dat
```

```BASH
F:\Chat-Bot-Skip-gram\usage>.\weights.exe sunshine whale whale kisses cheeks w2 w2p.dat
```
```
50 X 224
sunshine -> whale
Cosine Similarity = -0.115319, Cosine Distance = 0.884681
sunshine -> whale
Cosine Similarity = -0.115319, Cosine Distance = 0.884681
sunshine -> kisses
Cosine Similarity = 0.207155, Cosine Distance = 0.792845
sunshine -> cheeks
Cosine Similarity = -0.126842, Cosine Distance = 0.873158
whale -> whale
Cosine Similarity = 1, Cosine Distance = 2.22045e-16
whale -> kisses
Cosine Similarity = 0.0796019, Cosine Distance = 0.920398
whale -> cheeks
Cosine Similarity = 0.0648549, Cosine Distance = 0.935145
whale -> kisses
Cosine Similarity = 0.0796019, Cosine Distance = 0.920398
whale -> cheeks
Cosine Similarity = 0.0648549, Cosine Distance = 0.935145
kisses -> cheeks
Cosine Similarity = 0.066965, Cosine Distance = 0.933035
```