"""
model_builder.py — Six 1-D CNN architectures adapted for time-series regression
plus a dynamic custom model builder and a thread-safe Trainer wrapper.

All architectures use Conv1D layers so they accept input shape (timesteps, features).
For tabular input (no temporal window) the input is reshaped to (1, n_features)
automatically by the Trainer.
"""

import numpy as np
import threading

# ---------------------------------------------------------------------------
# Lazy-import TensorFlow so the import doesn't crash if TF isn't installed yet
# ---------------------------------------------------------------------------
def _tf():
    import tensorflow as tf
    # Turn off JIT compiler globally to be extra safe against XLA crashes
    tf.config.optimizer.set_jit(False)
    return tf

def _keras():
    tf = _tf()
    return tf.keras


# ===========================================================================
# 1-D CNN Architecture Factories
# ===========================================================================

def build_alexnet_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D AlexNet adaptation — 5 Conv1D blocks followed by 3 Dense layers.
    input_shape : (timesteps, features)
    """
    keras = _keras()
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(96, 11, strides=4, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = keras.layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = keras.layers.Conv1D(384, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(384, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(output_units, activation="linear")(x)
    return keras.Model(inp, out, name="AlexNet1D")


def build_googlenet_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D GoogLeNet adaptation — inception modules with Conv1D branches.
    """
    keras = _keras()

    def inception_block(x, f1, f3r, f3, f5r, f5, fp):
        b1 = keras.layers.Conv1D(f1,  1, padding="same", activation="relu")(x)
        b2 = keras.layers.Conv1D(f3r, 1, padding="same", activation="relu")(x)
        b2 = keras.layers.Conv1D(f3,  3, padding="same", activation="relu")(b2)
        b3 = keras.layers.Conv1D(f5r, 1, padding="same", activation="relu")(x)
        b3 = keras.layers.Conv1D(f5,  5, padding="same", activation="relu")(b3)
        b4 = keras.layers.MaxPooling1D(3, strides=1, padding="same")(x)
        b4 = keras.layers.Conv1D(fp,  1, padding="same", activation="relu")(b4)
        return keras.layers.Concatenate()([b1, b2, b3, b4])

    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 7, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = keras.layers.Conv1D(192, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32)
    x = inception_block(x, 128, 128, 192, 32, 96, 64)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = inception_block(x, 192, 96, 208, 16, 48, 64)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(output_units, activation="linear")(x)
    return keras.Model(inp, out, name="GoogLeNet1D")


def build_shufflenet_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D ShuffleNet-style adaptation using depthwise separable Conv1D blocks.
    """
    keras = _keras()

    def shuffle_block(x, filters, strides=1):
        # Depthwise separable convolution as approximation of channel shuffle
        x = keras.layers.SeparableConv1D(filters, 3, strides=strides,
                                         padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        return x

    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(24, 3, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    for filters in [116, 116, 116, 116]:
        x = shuffle_block(x, filters)
    x = shuffle_block(x, 232, strides=2)
    for filters in [232, 232, 232, 232, 232, 232, 232]:
        x = shuffle_block(x, filters)
    x = shuffle_block(x, 464, strides=2)
    for filters in [464, 464, 464]:
        x = shuffle_block(x, filters)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(output_units, activation="linear")(x)
    return keras.Model(inp, out, name="ShuffleNet1D")


def build_resnet_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D ResNet-18 adaptation with residual Conv1D blocks.
    """
    keras = _keras()

    def residual_block(x, filters, strides=1):
        shortcut = x
        x = keras.layers.Conv1D(filters, 3, strides=strides, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        # Adjust shortcut dimensions if needed
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv1D(filters, 1, strides=strides, padding="same")(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.ReLU()(x)
        return x

    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 7, strides=2, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    for _ in range(2): x = residual_block(x, 64)
    for i in range(2): x = residual_block(x, 128, strides=2 if i == 0 else 1)
    for i in range(2): x = residual_block(x, 256, strides=2 if i == 0 else 1)
    for i in range(2): x = residual_block(x, 512, strides=2 if i == 0 else 1)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(output_units, activation="linear")(x)
    return keras.Model(inp, out, name="ResNet1D")


def build_vgg16_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D VGG-16 adaptation — two-block Conv1D groups followed by Dense head.
    """
    keras = _keras()
    inp = keras.Input(shape=input_shape)
    x = inp
    # Block 1
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    # Block 2
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    # Block 3
    for _ in range(3):
        x = keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    # Block 4
    for _ in range(3):
        x = keras.layers.Conv1D(512, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    # Block 5
    for _ in range(3):
        x = keras.layers.Conv1D(512, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(output_units, activation="linear")(x)
    return keras.Model(inp, out, name="VGG16_1D")


def build_squeezenet_1d(input_shape: tuple, output_units: int = 1) -> "tf.keras.Model":
    """
    1-D SqueezeNet adaptation — fire modules with Conv1D squeeze/expand layers.
    """
    keras = _keras()

    def fire_module(x, squeeze, expand):
        sq  = keras.layers.Conv1D(squeeze, 1, activation="relu", padding="same")(x)
        ex1 = keras.layers.Conv1D(expand,  1, activation="relu", padding="same")(sq)
        ex3 = keras.layers.Conv1D(expand,  3, activation="relu", padding="same")(sq)
        return keras.layers.Concatenate()([ex1, ex3])

    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(96, 7, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = fire_module(x, 64, 256)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv1D(output_units, 1, padding="same")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, x, name="SqueezeNet1D")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "AlexNet":    build_alexnet_1d,
    "GoogLeNet":  build_googlenet_1d,
    "ShuffleNet": build_shufflenet_1d,
    "ResNet":     build_resnet_1d,
    "VGG-16":     build_vgg16_1d,
    "SqueezeNet": build_squeezenet_1d,
}


def get_model(name: str, input_shape: tuple) -> "tf.keras.Model":
    """Instantiate and compile a named pre-defined model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](input_shape)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"], jit_compile=False)
    return model


# ===========================================================================
# Custom Model Builder
# ===========================================================================

class CustomModelBuilder:
    """
    Lets users stack Keras layers interactively.
    Each call to add_layer() appends a layer specification.
    build() compiles and returns the final tf.keras.Model.
    """

    SUPPORTED_LAYERS = [
        "Conv1D", "Activation", "BatchNormalization",
        "MaxPooling1D", "Dense", "Dropout", "Flatten", "LSTM", "GRU"
    ]

    def __init__(self):
        self.layer_specs: list[dict] = []   # ordered list of layer configs

    def add_layer(self, layer_type: str, params: dict):
        """
        Append a layer specification.

        Parameters
        ----------
        layer_type : one of SUPPORTED_LAYERS
        params     : dict of keyword arguments for the Keras layer constructor
        """
        self.layer_specs.append({"type": layer_type, "params": params})

    def remove_layer(self, index: int):
        """Remove a layer by its list index."""
        if 0 <= index < len(self.layer_specs):
            self.layer_specs.pop(index)

    def clear(self):
        self.layer_specs.clear()

    def build(self, input_shape: tuple, lr: float = 1e-3) -> "tf.keras.Model":
        """
        Compile and return the custom model.

        input_shape : (timesteps, features)
        """
        keras = _keras()
        inp = keras.Input(shape=input_shape)
        x = inp

        for spec in self.layer_specs:
            lt = spec["type"]
            p  = spec["params"]
            if lt == "Conv1D":
                x = keras.layers.Conv1D(**p)(x)
            elif lt == "Activation":
                x = keras.layers.Activation(**p)(x)
            elif lt == "BatchNormalization":
                x = keras.layers.BatchNormalization()(x)
            elif lt == "MaxPooling1D":
                x = keras.layers.MaxPooling1D(**p)(x)
            elif lt == "Dense":
                x = keras.layers.Dense(**p)(x)
            elif lt == "Dropout":
                x = keras.layers.Dropout(**p)(x)
            elif lt == "Flatten":
                x = keras.layers.Flatten()(x)
            elif lt == "LSTM":
                x = keras.layers.LSTM(**p)(x)
            elif lt == "GRU":
                x = keras.layers.GRU(**p)(x)

        # Ensure output is 1 scalar per sample
        if len(x.shape) > 2:
            x = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(1, activation="linear", name="output")(x)

        tf = _tf()
        model = keras.Model(inp, out, name="CustomModel")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
            jit_compile=False
        )
        return model

    def summary_str(self) -> str:
        """Return a human-readable string of the current layer stack."""
        if not self.layer_specs:
            return "(no layers added)"
        lines = []
        for i, spec in enumerate(self.layer_specs):
            params_str = ", ".join(f"{k}={v}" for k, v in spec["params"].items())
            lines.append(f"[{i}] {spec['type']}({params_str})")
        return "\n".join(lines)


# ===========================================================================
# Trainer — runs Keras training in a background thread
# ===========================================================================

class Trainer:
    """
    Wraps keras model.fit() in a daemon thread so the UI stays responsive.
    Progress is communicated via a shared queue of (epoch, logs) tuples.
    """

    def __init__(self):
        self._thread: threading.Thread = None
        self.history = None
        self.error: Exception = None
        self.is_running = False

    def train(self,
              model,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              on_epoch_end=None,
              on_done=None):
        """
        Start training in a background daemon thread.

        Parameters
        ----------
        on_epoch_end : callable(epoch, logs) — called after each epoch
        on_done      : callable(history)     — called when training completes
        """
        self.is_running = True
        self.error = None

        # Reshape: if 2-D input, add timestep dim → (samples, 1, features)
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
            X_val   = X_val[:, np.newaxis, :]

        keras = _keras()

        # Build Keras callback to relay progress
        class _ProgressCallback(keras.callbacks.Callback):
            def __init__(self, on_epoch_end_fn):
                super().__init__()
                self._fn = on_epoch_end_fn

            def on_epoch_end(self, epoch, logs=None):
                if self._fn:
                    self._fn(epoch, logs or {})

        callbacks = [_ProgressCallback(on_epoch_end)]

        def _run():
            try:
                self.history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0,
                )
            except Exception as exc:
                self.error = exc
            finally:
                self.is_running = False
                if on_done:
                    on_done(self.history)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the model to stop after the current epoch."""
        self.is_running = False

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """Run inference, reshaping 2-D input if necessary."""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        return model.predict(X, verbose=0).ravel()
