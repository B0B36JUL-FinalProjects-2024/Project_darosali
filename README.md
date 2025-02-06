# **Latent Space Clustering**  

## **Project Description**  
This project implements a **β-VAE (Beta Variational Autoencoder)** in Julia and applies **k-means clustering** to the latent space. The goal is to **explore the latent space and analyze whether it captures meaningful structure in the data**. By using a **β-VAE**, we also investigate how the **β parameter** influences the balance between **reconstruction quality** and **disentanglement** in the latent space.  

## **Main Features**  

### **1️⃣ β-VAE (Beta Variational Autoencoder)**  
- **Training**: The `vae.jl` module includes an encoder, decoder, and a loss function combining **reconstruction error** and **KL divergence**, scaled by β.  
- **Latent Space Sampling**: Uses the **reparameterization trick** for stable latent space learning.  
- **Data Reconstruction**: Allows decoding of latent vectors back into original data.  
- **β Parameter Influence**: Investigates how different values of β affect latent space structure and clustering.  

### **2️⃣ K-Means Clustering in Latent Space**  
- **k-means++ Initialization**: The `clustering.jl` module implements k-means clustering with optimized centroid initialization.  
- **Clustering of Latent Representations**: Groups VAE-encoded data into **k clusters**.  
- **Visualization of Clusters**: Uses **t-SNE** to reduce dimensionality and analyze cluster separation.  

### **3️⃣ Visualization and Analysis**  
- **t-SNE visualization of the latent space**  
- **Decoding cluster centroids back into original data space**  
- **Exploring the impact of β on the learned representations**  

## **Use Cases**  
This package is useful for **generative modeling, dimensionality reduction, and clustering**, such as:  
✅ Generating new data (e.g., images)  
✅ Understanding latent space structure  
✅ Discovering natural groupings in high-dimensional data  