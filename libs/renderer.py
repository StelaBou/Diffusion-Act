from libs.config import *

from torch.cuda import amp


def render_uncondition(conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler,
                       latent_sampler: Sampler,
                       conds_mean=None,
                       conds_std=None,
                       clip_latent_noise: bool = False):
    device = x_T.device
    
    if conf.train_mode == TrainMode.diffusion:
        
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T)
        
    elif conf.train_mode.is_latent_diffusion():
        model: BeatGANsAutoencModel
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
            # print('latent_noise', latent_noise.shape)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        print('Sample z_sem')
        cond = latent_sampler.sample(  # -> z_sem: start with random latent_noise and use latent_DDIM to generate z_sem (sampled from the distribution)
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
        )
        # print('z_sem', cond.shape)
        
        if conf.latent_znormalize:
            cond = cond * conds_std.to(device) + conds_mean.to(device)

        # the diffusion on the model
        print('Sample image')
        
        return sampler.sample(model=model, noise=x_T, cond=cond)    # -> SpacedDiffusionBeatGans -> GaussianDiffusionBeatGans base.py, 
                                                                    # self.conf.gen_type == GenerativeType.ddim
    else:
        raise NotImplementedError()


def render_condition_inter(conf: TrainConfig,
						    model: BeatGANsAutoencModel,
							x_T,
							sampler: Sampler,
                            x_start=None,
							z_edit=None, z_source = None
):

    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()        
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': z_edit}, z_source = z_source)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    sampler: Sampler,
    x_start=None,
    cond=None, shift_h = None, return_all_steps = False,
):
    
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': cond}, shift_h = shift_h, return_all_steps = return_all_steps)
    else:
        raise NotImplementedError()
