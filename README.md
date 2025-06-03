# Towards Fast and Transferable Sparse Adversarial Attacks
This repository provides the official implementation of our proposed sparse attack, i.e., WI-FGSM. Specifically, this repository includes two demo experiments. First, we evaluate the sparsity of our approach by exhibiting that perturbing only 4 pixels, our approach is sufficient to mislead deep neural networks (DNNs) to predict an incorrect label of "mink" on an image of "giant panda". Second, resorting to Grad-CAM visualizations, we interpret the adversarial perturbation by discovering two types of noises, i.e., "obscuring noise" and "leading noise".



# Requirement

Our algorithm is based on the following libraries:

- torch
- torchvision
- matplotlib
- tqdm
- Pillow
- numpy
- grad-cam

You could use the following instruction to install all the requirements:

```python
# install requirements
pip install -r requirements.txt
```



## Evaluation on Sparsity

### Run

To evaluate the sparsity, please first move to  `src` directory, and then directly run `main.py`.

```python
# move to src
cd src

# run
python main.py
```

### Result

The following figure illustrates an example of our sparse attack, where our approach misleads the classifier to predict "mink" on the "giant panda" with only 4 perturbed pixels.

![result](output/readme.png)


## Interpret the Adversarial Perturbation

### Run

To exhibit two types of noise, please first move to  `src` directory, and then directly run `viz_noise.py`.

```python
# move to src
cd src

# run
python viz_noise.py
```



### Result

- Grad-CAM and Guided Grad-CAM visualizations on a clean image of "candle"

![clean](output/viz_clean.png)

- Grad-CAM and Guided Grad-CAM visualizations on an adversarial example of "toilet tissue"

![clean](output/viz_adv.png)

