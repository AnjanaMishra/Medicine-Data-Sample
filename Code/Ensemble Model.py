from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
import numpy as np


X_train_stack, X_val_stack, y_train_stack, y_val_stack = train_test_split(X_train, y_train, test_size=0.2)

# Make predictions with the base models on the validation set
lstm_val_predictions = model.predict(X_val_stack, batch_size=batch_size)
snn_val_predictions = snn_model.predict(X_val_stack, batch_size=batch_size)
cnn_val_predictions = cnn_model.predict(X_val_stack, batch_size=batch_size)

# Stack the predictions horizontally
stacked_predictions = np.hstack((lstm_val_predictions, snn_val_predictions, cnn_val_predictions))

# Train a meta-learner (e.g., Random Forest) on the stacked predictions
meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
meta_learner.fit(stacked_predictions, y_val_stack)

# Make predictions with the base models on the test set
lstm_test_predictions = model.predict(X_test, batch_size=batch_size)
snn_test_predictions = snn_model.predict(X_test, batch_size=batch_size)
cnn_test_predictions = cnn_model.predict(X_test, batch_size=batch_size)

# Stack the test set predictions
stacked_test_predictions = np.hstack((lstm_test_predictions, snn_test_predictions, cnn_test_predictions))

# Use the meta-learner to make ensemble predictions on the test set
ensemble_predictions = meta_learner.predict(stacked_test_predictions)

# Convert ensemble_predictions to one-hot encoded format
ensemble_predictions_onehot = label_binarize(ensemble_predictions, classes=np.unique(labels))

# Ensure both y_test and ensemble_predictions have the same format
print(classification_report(y_test, ensemble_predictions_onehot))



ensemble_pred_classes = np.argmax(ensemble_predictions, axis=1)
plot_conf_matrix(labels, ensemble_pred_classes, title="Ensemble")

# AUC ROC using probabilities (if binary classification)
fpr, tpr, _ = roc_curve(labels, ensemble_predictions[:, 1])  # Probabilities for class 1
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Ensemble ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


lstm_probs = lstm_test_predictions
snn_probs = snn_test_predictions
cnn_probs = cnn_test_predictions
ensemble_probs = meta_learner.predict_proba(stacked_test_predictions)


ensemble_probs = np.array(ensemble_probs)
if ensemble_probs.ndim == 3:
    ensemble_probs = ensemble_probs[-1]


print("Generating the ROC Curve Comparison plot for all models...")
roc_sources = [
    ("Ensemble", ensemble_probs),
    ("RNN_LSTM", lstm_probs),
    ("CNN", cnn_probs),
    ("SNN", snn_probs),
]

plt.figure(figsize=(7, 6))
for name, prob in roc_sources:
    fpr, tpr, _ = roc_curve(labels, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, ls=':')
plt.show()



# Calculate ROC and AUC for the ensemble model specifically
fpr_ens, tpr_ens, _ = roc_curve(labels, ensemble_probs[:, 1])
roc_auc_ens = auc(fpr_ens, tpr_ens)

# Create the new plot
plt.figure(figsize=(7, 6))

# Plot the ensemble ROC curve
plt.plot(fpr_ens, tpr_ens, lw=2, label=f"Ensemble (AUC = {roc_auc_ens:.2f})", color='C0') # C0 is the default blue

# Plot the random chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random')

# Set plot limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Ensemble ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, ls=':')
plt.show()




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

lstm_probs      = lstm_test_predictions
snn_probs       = snn_test_predictions
cnn_probs       = cnn_test_predictions
ensemble_probs = meta_learner.predict_proba(stacked_test_predictions)

ensemble_probs = np.array(ensemble_probs)
print("Before reshape:", ensemble_probs.shape)

if ensemble_probs.ndim == 3 and ensemble_probs.shape[1] == len(labels):
    ensemble_probs = ensemble_probs[-1]  # shape now (114237, 2)

print("After reshape:", ensemble_probs.shape)



roc_sources = [
    ("Ensemble", ensemble_probs),
    ("RNN_LSTM", lstm_probs),
    ("CNN", cnn_probs),
    ("SNN", snn_probs),
]


plt.figure(figsize=(7,6))

for name, prob in roc_sources:
  fpr, tpr, _ = roc_curve(labels, prob[:, 1])
  roc_auc       = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, ls=':')
plt.show()






num_words = 50000
maxlen = 200
test_feat = req_df['review_clean']
tokenizer = Tokenizer(num_words=num_words, split=" ", lower=False)
tokenizer.fit_on_texts(test_feat)
test_feat = tokenizer.texts_to_sequences(test_feat)
test_feat = pad_sequences(test_feat, maxlen=maxlen)





import pandas as pd

# Assuming req.df contains your original dataset

# Make predictions with the base models on the original dataset
lstm_predictions = model.predict(test_feat, batch_size=batch_size)
snn_predictions = snn_model.predict(test_feat, batch_size=batch_size)
cnn_predictions = cnn_model.predict(test_feat, batch_size=batch_size)

# Stack the predictions horizontally
stacked_predictions = np.hstack((lstm_predictions, snn_predictions, cnn_predictions))

# Use the meta-learner to make ensemble predictions on the original dataset
ensemble_predictions = meta_learner.predict(stacked_predictions)



