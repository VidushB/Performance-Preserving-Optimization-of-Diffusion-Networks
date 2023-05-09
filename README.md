# Performance-Preserving-Optimization-of-Diffusion-Networks
Final project for HPML Spring 23 at NYU


********** Running on Google Colab, testing purposes, single GPU **************************
!python3 /content/hpml_project/training.py
# specs->
# trainer = Trainer(
#     diffusion,
#     '/content/hpml_project/train',
#     train_batch_size = 128,
#     train_lr = 2e-4,
#     train_num_steps = 1000,
#     gradient_accumulate_every = 2,
#     ema_decay = 0.995,
#     amp = False
# )

loss: 0.1589: 100% 1000/1000 [12:29<00:00,  1.33it/s]
training complete
____________________________________________________________________

!python3 /content/hpml_project/training.py
# amp true ( faster by -4 mins)
# specs->
# trainer = Trainer(
#     diffusion,
#     '/content/hpml_project/train',
#     train_batch_size = 128,
#     train_lr = 2e-4,
#     train_num_steps = 1000,
#     gradient_accumulate_every = 2,
#     ema_decay = 0.995,
#     amp = True
# )

loss: 0.1690: 100% 1000/1000 [08:32<00:00,  1.95it/s]
training complete

____________________________________________________________________
# # Trying profiler :)
# import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# model = Unet(
#     dim=64,
#     dim_mults=(1, 2, 4)
# ).cuda()

# diffusion = GaussianDiffusion(
#     model,
#     image_size=32,
#     timesteps=1000,
#     sampling_timesteps=250,
#     loss_type='l1'
# ).cuda()

# trainer = Trainer(
#     diffusion,
#     '/content/hpml_project/train',
#     train_batch_size=128,
#     train_lr=2e-4,
#     train_num_steps=1000,
#     gradient_accumulate_every=2,
#     ema_decay=0.995,
#     amp=True
# )

# def trace_handler(p):
#     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
#     print(output)
#     p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(
#         wait=2,
#         warmup=1,
#         active=3,
#         repeat=2
#     ),
#     on_trace_ready=trace_handler
# ) as profiler:
#     trainer.train()

************************ Important TEST *******************************************
->>>>>>>>>  No output in the directory, code runs tho!

