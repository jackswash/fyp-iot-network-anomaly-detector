import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandarallel import pandarallel
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data settings.
DATA_PATH = "data/MERGED_CSV"
SAMPLES_PER_CLASS_PER_FILE = 1000
GLOBAL_CAP_PER_CLASS = 20000
NUM_WORKERS = 16

# Training settings.
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 5
RANDOM_STATE = 123

# Output paths.
OUTPUT_DIR = "outputs"
TRAINING_CURVES_PATH = f"{OUTPUT_DIR}/training_curves.pdf"
CONFUSION_MATRIX_PATH = f"{OUTPUT_DIR}/confusion_matrix.pdf"
CLASSIFICATION_REPORT_PATH = f"{OUTPUT_DIR}/classification_report.txt"

pandarallel.initialize(nb_workers=NUM_WORKERS, progress_bar=False, verbose=0)

# TODO: Refactor codebase into smaller subfunctions as right now it looks
#  horrid. preprocess_data needs shortening as many functions within it
#  should be placed elsewhere in the code. Technically the code is fully
#  functional however it needs TLC.


def preprocess_data(data_path, max_per_class):
    """
    Here we begin to preprocess the data. Firstly, the CSVs are loaded using
    pandarallel to speed up the process compared to a single-threaded
    approach. Then we load each class within a file up to the limit dictated in
    "SAMPLES_PER_CLASS_PER_FILE", as well as using a capped limit across all
    files dictated in "GLOBAL_CAP_PER_CLASS". The global cap is used to
    prevent classes with large amounts of data samples from biasing the
    model. Classes that do not reach the maximum threshold are loaded as
    normal under the limit. Both NaN and infinite values are removed.

    Args:
        data_path: Path to the directory containing the merged CSV files.
        max_per_class: Maximum number of records per class after combining.

    Returns:
        A tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
         class_names).
    """

    # Find all CSV files within the directory stored in data_path that contain
    # "Merged" and any other word or number. In this case, each CSV file
    # contains its corresponding number indicating which CSV it is, and we have
    # a total of 63. This could be hardcoded to just search for a number of 63
    # but this way is more flexible.
    csv_files = sorted(glob.glob(os.path.join(data_path, "Merged*.csv")))

    if not csv_files:
        print(f"Error: No CSV files were found in '{data_path}'.")
        sys.exit(1)

    total_files = len(csv_files)
    print(f"Found {total_files} CSV files, loading...\n")

    # Each CSV file is loaded in parallel using pandarallel. This is used
    # because loading CSVs with pandas is single-threaded, meaning each one
    # loads one by one and depending on the size of the CSV files, this can
    # take minutes. By running multiple workers at once, multiple CSVs can
    # be loaded at the same time in parallel. Sampling per class within each
    # file is done so that one class cannot dominate the loaded data, which
    # would cause the model to develop a heavy bias towards the majority
    # class during training.
    def load_and_sample(file):
        df = pd.read_csv(file)

        # For each unique class label within the current file, take up to
        # the number of samples specified in SAMPLES_PER_CLASS_PER_FILE. If
        # the class has fewer rows than this within the file, loading
        # proceeds normally and the cap is ignored.
        per_class_samples = []
        # Loop to get all of the individual class labels.
        for current_class in df['Label'].unique():
            # Here we use boolean indexing from pandas. df['Label'] ==
            # current_class outputs a column of true/false values,
            # and wrapping that inside df[] returns only the rows where the
            # condition is true.
            class_rows = df[df['Label'] == current_class]

            # Use SAMPLES_PER_CLASS_PER_FILE to enforce a sample cap,
            # however if the class has fewer entries than the cap, continue
            # as normal.
            sample_size = min(len(class_rows), SAMPLES_PER_CLASS_PER_FILE)

            # Here we take a sample of the rows from class_rows according to
            # the SAMPLES_PER_CLASS_PER_FILE referenced through sample_size,
            # and then use the random_state argument so it can be reproduced
            # at runtime. The random state integer is referenced above.
            sampled = class_rows.sample(sample_size, random_state=RANDOM_STATE)
            per_class_samples.append(sampled)

        # Merge all of the dataframes stored inside per_class_samples into
        # one dataframe. Argument ignore_index=True is used to ensure that
        # the row numbers are not carried over from the original dataframes.
        return pd.concat(per_class_samples, ignore_index=True)

    # Here we use pandarallel to load and sample each CSV in parallel.
    # pandarallel adds parallel versions of pandas methods. We use a series
    # since the CSV paths are just a one dimensional list.
    paths = pd.Series(csv_files)
    sampled_dfs = paths.parallel_apply(load_and_sample).tolist()

    print("\nCombining files...")
    data = pd.concat(sampled_dfs, ignore_index=True)
    del sampled_dfs
    print(f"Combined shape: {data.shape}")

    # After all of the files have been combined, an additional global cap is
    # applied per class across the entire dataset. The reasoning behind this
    # is that even with the per-file sampling done earlier, classes that
    # appear in many files will still have far more rows than classes that
    # only appear in a few. This step enforces a hard upper limit per class
    # to keep the final dataset balanced.
    capped_groups = []
    # Loop to retrieve all unique class labels.
    for label in data['Label'].unique():
        # Here we use boolean indexing from pandas. data['Label'] ==
        # label outputs a column of true/false values, and wrapping that
        # inside data[] returns only the rows where the condition is true.
        class_rows = data[data['Label'] == label]

        # Use max_per_class to enforce a sample cap, however if the class
        # has fewer entries than the cap, continue as normal.
        sample_size = min(len(class_rows), max_per_class)

        # Take the random sample and add it to the capped_groups list,
        # which collects one capped sample per class before they are combined.
        capped_groups.append(class_rows.sample(sample_size,
                                                random_state=RANDOM_STATE))

    # Merge all the dataframes stored inside capped_groups into one
    # dataframe. Argument ignore_index=True is used to ensure that the row
    # numbers are not carried over from the original dataframes.
    data = pd.concat(capped_groups, ignore_index=True)

    print(f"After global cap of {max_per_class:,} per class: "
          f"{data.shape[0]:,} records")

    # Here, any values that contain "NaN" or infinite values are removed
    # entirely from the dataset, since these numbers fail to represent
    # accurate data. An alternative approach would've been to set the values
    # to 0 or calculate the mean based on the other values within their
    # respective classes however adding artificial values doesn't represent
    # the true information contained within the dataset.
    rows_before = len(data)

    # No dropinf() method in pandas since technically inf values are correct
    # but for this case they are not, so we replace all positive and
    # negative inf values with NaN and then call dropna() to remove them.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Dropped {rows_before - len(data):,} rows with invalid data "
          f"(NaN, inf. values) ({len(data):,} remaining)")

    # Display the class distribution to the console. Use value_counts to go
    # through the 'Label column and count how many times each label appears.
    print("\nClass distribution:")
    print(data['Label'].value_counts())

    # Split dataframe into X and y, where X is the features and y is the
    # labels. Drop 'Label' column so that X only contains features, and use
    # .values to convert the objects into a NumPy array to be parsed into
    # Keras.
    X = data.drop('Label', axis=1).values
    y = data['Label'].values

    # Need to convert the strings to integers as DL algorithms cannot work
    # with strings. Use fit_transform to perform both fit and transform,
    # where fit builds the numeric mapping from the label names to integers,
    # and transform applies said mappings.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)

    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Number of features: {X.shape[1]}")

    # Here we split the data into 70/15/15, where 70% is training, 15% is
    # validation, and the last 15% is test. First we pass in X and
    # y_encoded into train_test_split, where X has the feature data and
    # y_encoded has the integer labels. 70% of X is stored inside of X_train,
    # and the remaining 30% is stored inside of X_temp, then 70% of y_encoded
    # is stored inside of y_train, and the remaining 30% is stored inside of
    # y_temp. stratify=y_encoded is used to keep the proportions of the
    # classes balanced, so no class ends up underrepresented in any split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, train_size=0.7, random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    # Here we split the remaining 30% into 15/15, where 15% is validation
    # and the last 15% is test. We pass in X_temp and y_temp into
    # train_test_split, where X_temp has the feature data and y_temp has
    # the integer labels. Half of X_temp is stored inside of X_val, and
    # the remaining half is stored inside of X_test, then half of y_temp
    # is stored inside of y_val, and the remaining half is stored inside
    # of y_test. stratify=y_temp is used to keep the proportions of the
    # classes balanced, so no class ends up underrepresented in any split.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Here we standardise the features so every column has a mean of 0 and
    # standard deviation of 1. NN models are sensitive to inputs that are on
    # vastly different scales, and standardisation ensures that all features
    # are on the same level. fit_transform is only used for X_train so that
    # the scaler learns the mean and standard deviation from the training
    # data alone. X_val and X_test are then transformed with those same
    # training statistics, which is to prevent information from X_val and
    # X_test from leaking.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Here we need to reshape the sets from 2D to 3D as Conv1D in Keras
    # expects a 3D input where the last input required that we previously
    # didn't have is channels. Since our data is tabular and doesn't have
    # multiple channels like an image would, we set the number of channels
    # to 1.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"\nSplit: {len(X_train):,} train, {len(X_val):,} val, "
          f"{len(X_test):,} test")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


def build_model(input_shape, num_classes):
    """
    Here we construct the 1D-CNN model used for multi-class classification.
    Batch normalisation is applied after each layer to stabilise training, and
    dropout is used in the dense layers to reduce overfitting. No pooling is
     applied as the input sequence is already short and pooling would discard
     useful spatial information. The final softmax layer outputs
     probability scores across all classes.

    Args:
        input_shape: Shape of input data as (features, channels).
        num_classes: Number of output classes for the softmax layer.

    Returns:
        A compiled Keras Sequential model.
    """
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),

        # Convolutional block with two stacked Conv1D layers at 256 filters.
        # Batch normalisation is used after each layer to ensure that the
        # outputs of each layer are steady, which leads to the model being
        # able to train more effectively.
        keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        # The output of the convolutional block is flattened, transforming
        # the multi-dimensional output into a one-dimensional vector ready
        # for the dense classification layers.
        keras.layers.Flatten(),

        # Two dense layers are used to combine the patterns learned by the
        # convolutional block and produce the classification decision.
        # Dropout is applied to reduce overfitting during training.
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # The final softmax layer converts the raw output values into
        # probability scores. Each score lies between 0 and 1, and the
        # combined scores sum to 1.
        keras.layers.Dense(num_classes, activation='softmax'),
    ])

    # sparse_categorical_crossentropy is used as the loss function,
    # as it works directly with integer-encoded labels rather than requiring
    # one-hot encoded vectors.
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy'],
    )

    return model


def train(model, X_train, y_train, X_val, y_val):
    """
    Train the model with early stopping and a learning rate scheduler.

    Early stopping monitors the validation loss and restores the best
    weights once training ends, which prevents the model from being
    saved in a degraded state. The learning rate scheduler halves the
    learning rate when the validation loss plateaus, allowing the
    model to settle into a better minimum once initial progress slows.
    """
    callbacks = [
        # Early stopping is used to prevent overfitting, where the model learns
        # patterns specific to the training data that do not generalise to
        # unseen data. Training stops when the validation loss has not improved
        # for EARLY_STOP_PATIENCE epochs, which gives the model some leniency
        # in case the validation loss temporarily plateaus before improving.
        # restore_best_weights ensures the final model is the best version
        # seen during training, not the most recent.
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),

        # The learning rate scheduler dynamically adjusts the learning rate
        # based on training performance. A learning rate that is too high
        # causes the model to overshoot optimal weights, and one that is too
        # low causes slow convergence and a higher chance of settling into
        # a poor local minimum. When the validation loss plateaus for
        # LR_PATIENCE epochs, the learning rate is halved, allowing the
        # model to settle into a better minimum. min_lr prevents the
        # learning rate from dropping so low that training effectively halts.
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Here we train the model on X_train/y_train, evaluating on X_val/y_val
    # at the end of each epoch so the callbacks can monitor val_loss.
    # Returns a history object containing the loss and accuracy values for
    # each epoch, used later to plot the training curves.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    return history


def plot_history(history):
    """
    Plot the training and validation loss and accuracy curves and save
    them to file. Plotting both training and validation curves on the
    same axes makes it easy to spot overfitting: if the training curve
    keeps improving while the validation curve flattens or degrades,
    the model is memorising the training data rather than generalising.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot training and validation loss on the top subplot. Training loss is
    # shown in blue and validation loss in red.
    axes[0].plot(history.history['loss'], color='b', label='Training Loss')
    axes[0].plot(history.history['val_loss'], color='r', label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot training and validation accuracy on the bottom subplot.
    axes[1].plot(history.history['accuracy'], color='b',
                 label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], color='r',
                 label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    # Save the figure and also display it as a plot.
    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=600)
    plt.show()

    # Display the best validation accuracy.
    best_val = np.max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    print(f"\nBest validation accuracy: {best_val:.3f} (epoch {best_epoch})")


def evaluate(model, X_test, y_test, class_names):
    """Run predictions on the test set and save a confusion matrix."""

    # Evaluate the model on the test set to get the final test loss and
    # accuracy.
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.3f}")
    print(f"Test accuracy: {test_acc:.3f}\n")

    # We use the model to predict classes for X_test. By using model.predict
    # we get the raw probability scores from the softmax layer, output as a
    # 2D array. np.argmax with axis=1 is used to pick the index of the
    # highest probability score.
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Used to make the confusion matrix. The diagonals represent correct
    # predictions.
    cm = confusion_matrix(y_test, y_pred)

    # Code to plot the confusion matrix as a heatmap, with the counts inside
    # of each square and the numbers formatted to integers.
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: IoT Intrusion Detection CNN')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=600)
    plt.show()

    # Build the classification report, which shows precision, recall and
    # F1-score for each class. zero_division=0 prevents warnings when a
    # class doesn't have any predicted samples. The report is saved to
    # disk as well as printed so it can be referenced from the dissertation
    # without needing to re-run the training pipeline.
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   zero_division=0)
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(report)


def main():
    """Main function to run the program."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = \
        preprocess_data(DATA_PATH, GLOBAL_CAP_PER_CLASS)
    model = build_model(input_shape=(X_train.shape[1], 1),
                        num_classes=len(class_names))
    model.summary()
    history = train(model, X_train, y_train, X_val, y_val)
    plot_history(history)
    print(f"Training curves saved to '{TRAINING_CURVES_PATH}'")
    evaluate(model, X_test, y_test, class_names)
    print(f"Confusion matrix saved to '{CONFUSION_MATRIX_PATH}'")
    print(f"Classification report saved to '{CLASSIFICATION_REPORT_PATH}'")


if __name__ == "__main__":
    main()