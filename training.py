#!/usr/bin/env python
# coding: utf-8


# In[ ]:


"""
Here, we first use torch.jit.trace() to trace the model using a dummy input and 
time step value, and store the result in traced_model.
Next, we wrap the trainer.train() function with a with torch.profiler.profile()
context manager. Inside the context manager, we specify a profiling schedule 
with torch.profiler.schedule() and a callback function to handle the profiling 
results with torch.profiler.tensorboard_trace_handler(). We also pass the 
profiler object to the trainer.train() function so that it can record the 
necessary profiling information.
When you run this code, the profiler will wait for 2 seconds, warm up for 1 
second, and then perform profiling for 3 seconds, repeated twice. The profiling 
results will be saved to the log_dir directory in the TensorBoard format.
"""
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,
    sampling_timesteps = 250,
    loss_type = 'l1'
).cuda()

trainer = Trainer(
    diffusion,
    '/content/hpml_project/train',
    train_batch_size = 128,
    train_lr = 2e-4,
    train_num_steps = 1000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = False
)

def trace_handler(p):
   # called automatically 
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/content/hpml_project/results/trace_" + str(p.step_num) + ".json")

with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=trace_handler
) as profiler:
    trainer.train()

# In[ ]:


# trainin_losses=trainer.train() #Training original model without any changes or jit capabilites



# In[ ]:


# import matplotlib.pyplot as plt
# plt.plot(trainin_losses)


# In[ ]:


#To test
# diffusion = GaussianDiffusion(
#     model,
#     image_size = 32,           #Try the other set instead
#     timesteps = 1000,           # number of steps 
#     sampling_timesteps = 1000,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
#     loss_type = 'l1'            # L1 or L2
# ).cuda()


# sampled_images = diffusion.sample(batch_size = 9)



# f = plt.figure()
# for i in range(9):
#   f.add_subplot(3, 3, i + 1)
#   temp=sampled_images[i]
#   temp=temp.detach().cpu()
#   print(temp.shape)
#   temp = temp.swapaxes(0,1)
#   temp = temp.swapaxes(1,2)
  
#   plt.imshow(temp)

# plt.show(block=True)


# # In[ ]:


# # # TODO  
# # # CHECK THIS OUT https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs
#  def trace_handler(p):
#     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
#     print(output)
#     p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=2),
#     on_trace_ready=trace_handler
# ) as p:
#     for idx in range(8):
#         model(inputs)
#         p.step()

