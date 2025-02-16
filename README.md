# Dataset
In this study, the IMDB dataset was used, consisting of 50,000 samples. The data was divided into training, validation, and test sets in an 8:1:1 ratio. The three splits were balanced. The class ratio for the train set and the val set was 50.08% / 49.92% and for the test set, it was 49.28% / 50.72%.

|                | Train | Validation | Test  |
|----------------|-------|------------|-------|
| Amount of Data | 40,000| 5,000      | 5,000 |
| Class Ratio    | Balanced | Balanced  | Balanced |
<br/>

# Models
The models used in this study were BERT-base and ModernBERT. Both models were fine-tuned using pretrained versions. The hyperparameters used are shown in the table below. Additionally, due to GPU issues, ModernBERT used regular attention instead of flash attention. The classifier used was a linear classifier, with the input being the [CLS] token.

| Model      | Epoch | max_len | batch_size | optimizer   | lr    | betas            | weight_decay |
|------------|-------|---------|------------|--------|-------|------------------|--------------|
| BERT       | 5     | 128     | 64         | Adam   | 5e-5  | (0.9, 0.999)     | 0.001        |
| ModernBERT | 5     | 128     | 32         | Adam   | 5e-5  | (0.9, 0.999)     | 0.001        |
<br/>

# Results
The best models was selected based on the validation accuracy, and testing was performed on the test set using the selected models.

| Model      | val_acc(%) | test_acc(%) |
|------------|------------|-------------|
| BERT       |96.68            |87.88             |
| ModernBERT |89.69            |75.24             |
<br/>

The trends of validation loss and accuracy for the two models are shown in the following graph. here was a significant increase in the loss around the 5400th iteration for the ㅡModernBERT model. After that, the loss gradually decreased, but there was fluctuation in terms of accuracy. In my opinion, due to the small batch size, the optimization space is noisy. It seems that around the 5400th iteration, the model oscillated near a local minimum, and eventually, gradient exploding occurred.
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/48136519-1155-405c-9756-4aeb55589658" width="60%" />
  <br />
  <img src="https://github.com/user-attachments/assets/712b35e2-881e-46b9-875e-ab3711f7fca9" width="60%" />
</div>
