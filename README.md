# Code used to train diffusion models

## Training a model

## Using a trained model
An example of how to use a trained diffusion model to transform traces is provided in test_diffusion.py and cpa.py. Adapting code for any existing attacks is generally straightforward. Just make sure to standardize the data the using the scale_dataset function in utils.py
```
#Loading a model
dif_folder ="diffusion_ascadv2_22_11_2023_07_22_04_1057297"
model = tf.keras.models.load_model(f"{results_root_path}/{dif_folder}/trained_model.keras")
#Remove noise from traces witt t = i
dataset.x_profiling= model.predict([orig_x_prof, np.ones(dataset.x_profiling.shape[0])*i])
```