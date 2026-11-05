# Speed on this machine (AWS, 1x Tesla T4 16 GB, 4 vCPU), measured 2026-11-05

`scripts/aws/benchmark.sh` (200 L0 demos, batch 64, evaluation on the L4b test scenes with an untrained policy, so
every episode runs all 120 steps):

| variant | training | evaluation |
|---|---|---|
| rgb | 1.07-1.14 s/it | 2 episodes: 1:20 |
| rgbdt | 1.28 s/it | 10 episodes: 1:57 |

GPU utilization was 98-100% and CPU about 21% while training, so the GPU (the 512-1024-2048 conv UNet), not data
loading, sets the speed. For comparison, the V100 runs on Bridges-2 trained at about 6 it/s.
Peak GPU memory while training rgbdt: 13.3 GB of 15.4 GB. The envs' EGL framebuffers did not fit next to torch's
cache ("Offscreen framebuffer is not complete") until `GlassPickRunner.run` empties the cache before the rollouts.
