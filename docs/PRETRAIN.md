To pre-train `CIM` using `ViT-Base` as the backbone, run the following on 8 A100 GPUs with port 8888:
```shell
sh scripts/dist_pretrain.sh 8 8888 <path-to-imagenet> cim base none <job-name>
```