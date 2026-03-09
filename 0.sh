# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29503 run.py --data MME --model bagel_gpt --verbose --batch-size 16
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29503 run.py --data HallusionBench --model bagel_gpt --verbose --batch-size 12
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29502 run.py --data MMVet --model bagel_gpt --verbose --batch-size 12
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29502 run.py --data MMStar --model bagel_gpt --verbose --batch-size 12
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29503 run.py --data MMBench_DEV_EN --model bagel_gpt --verbose --batch-size 16



CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29500 run.py --data AI2D_TEST --model bagel_zoomin --verbose --batch-size 6

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29500 run.py --data MathVista_MINI --model bagel_zoomin --verbose --batch-size 12

CUDA_VISIBLE_DEVICES=2 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29502 run.py --data MMBench --model bagel_zoomin --verbose --batch-size 8

