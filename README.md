# Synthetic-data-thesis
Case study which investigates the opportunities of synthetic data at governmental institution UWV. Currently, they face privacy challenges concerning their client data.
For now, this is solved by anonimising the data. However, this comes with its risks and might not be useful. Therefore, synthetic data generation is considered and implemented as another anonymization technique to discover its benefits and disadvantages for our client data. In this way, the privacy of our clients is still maintained while analyzing their information using several AI models.

# Methodology
## Part 1: Generating synthetic data from a use case WW naar Bijstand and using company's IT server
## Part 2: Generating synthetic data from one or more public datasets and using LISA.

Two seperate researches will run in parallel and apply the following techniques:
- Decision tree based models: Adversarial Random Forest & CART
- Deep generative models: GAN, VAE & DDPM

The evaluation procedure is similar for both and contains the following metrics:
- Applying Machine Learning model to both original and synthetic data and compare performance.
- Comparison based on statistical properties such as distribution, KStest, etcera.

