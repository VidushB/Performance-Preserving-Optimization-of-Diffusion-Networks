from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8) #32 16 8 4
).cuda()
#There is also an issue with the sampling step too
diffusion = GaussianDiffusion(
    model,
    image_size = 32,           #Try the other set instead
    timesteps = 1000,           # number of steps 
    sampling_timesteps = 500,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
#    #Not sure what sampling timesteps
    #How much sample when not train?
    #beta_schedule="linear",  # We are setting this, is cosine by default
    loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/scratch/vb2184/cifar10/train', #Chexpert is a Grayscale Image
    train_batch_size = 16*3,
    train_lr = 8e-5,
    train_num_steps = 30000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainin_losses=trainer.train()
#To test

sampled_images = diffusion.sample(batch_size = 9) #Figure out testing
import matplotlib.pyplot as plt
f = plt.figure()
for i in range(9):
  f.add_subplot(3, 3, i + 1)
  temp=sampled_images[i]
  temp=temp.detach().cpu()
  temp = temp.swapaxes(0,1)
  temp = temp.swapaxes(1,2)
  plt.imshow(temp)

plt.show(block=True)