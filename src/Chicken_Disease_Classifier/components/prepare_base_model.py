import tensorflow as tf
from pathlib import Path
from Chicken_Disease_Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: tf.keras.Model | None = None
        self.full_model: tf.keras.Model | None = None

    def get_base_model(self):
        """Build the base VGG16 and save base_model.keras"""
        try:
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,          # e.g., "imagenet" or None
                include_top=self.config.params_include_top   # should be False for custom head
            )
        except Exception as e:
            # Optional: fallback if ImageNet weights can’t be downloaded
            if str(self.config.params_weights).lower() == "imagenet":
                print("Warning: could not load ImageNet weights, falling back to weights=None\n", e)
                self.model = tf.keras.applications.vgg16.VGG16(
                    input_shape=self.config.params_image_size,
                    weights=None,
                    include_top=self.config.params_include_top
                )
            else:
                raise

        self.save_model(self.config.base_model_path, self.model)
        return self.model

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Add classification head, apply freezing, then compile.
        """
        # --- FIXED: freeze layers properly ---
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = tf.keras.layers.Flatten()(model.output)
        outputs = tf.keras.layers.Dense(units=classes, activation="softmax")(x)

        full_model = tf.keras.Model(inputs=model.input, outputs=outputs)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Build the full head-on-top model and save updated_model.keras
        """
        if self.model is None:
            self.get_base_model()

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)
        return self.full_model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save in Keras v3 format (.keras). Ensure parent dir exists and cast Path -> str.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(p))  # important: str(path) for cross-platform safety
