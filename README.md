# Code used to train diffusion models for paper: "Diffuse some Noise: Diffusion Models for1Measurement Noise Removal in Side-channel Analysis"


## Installation
We recommend setting up a virtual environment using conda and then installing the requirements in it using:
```
pip install -r requirements.txt
```
Datasets can be downloaded from the links provided in the paper, and we provide scripts to adapt eshard and aes_hd_mm to h5 files in scripts

## Training a model
Make sure to update the default paths in main.py to where datasets are stored and check/adapt the dataset locations in src/datasets/paths.py to match the actual locations. The default location for the results can also be adapted. 
#### Example usage
```
python main.py --dataset=eshard --dataset_dim=1400 --epochs=20
```



## Using a trained model
An example of how to use a trained diffusion model to transform traces is provided in test_diffusion.py and cpa.py. Adapting code for any existing attacks is generally straightforward. Just make sure to standardize the data the using the scale_dataset function in utils.py. 
```
#Loading a model
dif_folder ="diffusion_ascadv2_22_02_2024_07_22_04_1057297"
orig_x_prof = dataset.x_profiling.copy()
model = tf.keras.models.load_model(f"{results_root_path}/{dif_folder}/trained_model.keras")
#Remove noise from traces with t = i
dataset.x_profiling= model.predict([orig_x_prof, np.ones(dataset.x_profiling.shape[0])*i])
```
## Example analyses
We provided some of the analyses in gradients.py and gaussians.py 