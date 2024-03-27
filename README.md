# DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment

Authors official PyTorch implementation of the **[DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment](https://arxiv.org/abs/2403.17217)**. If you use this code for your research, please [**cite**](#citation) our paper.

<p align="center">
<img src="images/architecture.png" style="width: 750px"/>
</p>

>**DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment**<br>
> Stella Bounareli, Christos Tzelepis, Vasileios Argyriou, Ioannis Patras, Georgios Tzimiropoulos<br>
>
> **Abstract**: Video-driven neural face reenactment aims to synthesize realistic facial images that successfully preserve the identity and appearance of a source face, while transferring the target head pose and facial expressions. 
Existing GAN-based methods suffer from either distortions and visual artifacts or poor reconstruction quality, i.e., the background and several important appearance details, such as hair style/color, glasses and accessories, 
are not faithfully reconstructed. Recent advances in Diffusion Probabilistic Models (DPMs) enable the generation of high-quality realistic images. To this end, in this paper we present DiffusionAct, a novel method that leverages the photo-realistic 
image generation of diffusion models to perform neural face reenactment. Specifically, we propose to control the semantic space of a Diffusion Autoencoder (DiffAE), in order to edit the facial pose of the input images, defined as the head pose 
orientation and the facial expressions. Our method allows one-shot, self, and cross-subject reenactment, without requiring subject-specific fine-tuning. 
We compare against state-of-the-art GAN-, StyleGAN2-, and diffusion-based methods, showing better or on-par reenactment performance. 


<a href="https://arxiv.org/abs/2403.17217"><img src="https://img.shields.io/badge/arXiv-2403.17217-b31b1b.svg" height=22.5></a>
<a href="https://stelabou.github.io/diffusionact/"><img src="https://img.shields.io/badge/Page-Demo-darkgreen.svg" height=22.5></a>


<p align="center">
<img src="images/teaser.png" style="width: 750px"/>
</p>


# Code is coming soon


## Citation

```bibtex
@InProceedings{bounareli2024diffusionact,
    author    = {Bounareli, Stella and Tzelepis, Christos and Argyriou, Vasileios and Patras, Ioannis and   Tzimiropoulos, Georgios},
    title     = {DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment},
    journal   = {arXiv},
    year      = {2024},
}
```


