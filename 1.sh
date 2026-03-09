
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29508 run.py --data MME --model bagel_zoomin --verbose --batch-size 12

CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29508 run.py --data MMVet --model bagel_zoomin --verbose --batch-size 8 --reuse

# CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29508 run.py --data MMStar --model bagel_zoomin --verbose --batch-size 12

CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --rdzv_endpoint=localhost:29508 run.py --data HallusionBench --model bagel_zoomin --verbose --batch-size 8 --reuse
