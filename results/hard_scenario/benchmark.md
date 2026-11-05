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

## g5.xlarge (1x A10G 24 GB, 4 vCPU), measured 2026-11-05, with cudnn.benchmark

| variant | training | evaluation (L4b test scenes, untrained policy: all 120 steps) |
|---|---|---|
| rgb | 7.3 it/s | 10 episodes: 0:26 |
| rgbdt | 4.1 it/s | 10 episodes: 0:27 |

On the T4 with cudnn.benchmark, rgb trained at about 2.5 it/s. Estimated sweep run (200 demos, 101 epochs = 16,766 steps,
5 evaluations of 50 episodes plus test sets a and g): rgb about 55 min, rgbd about 67 min, rgbdt about 85 min.
